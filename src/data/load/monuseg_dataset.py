import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from cellpose import io
from skimage.draw import polygon as draw_polygon

from src.utils.logger import logger

from .base_dataset import BaseDataset


def read_xml_regions(xml_path: str) -> List[np.ndarray]:
    """Lê os contornos de todas as regiões (núcleos) de um XML do MoNuSeg.

    Percorre todos os blocos ``<Annotation>`` (``TCGA-HE-7129`` tem dois).

    Args:
        xml_path: Caminho do arquivo ``.xml`` (formato do Aperio ImageScope).

    Returns:
        Lista com um array ``(K, 2)`` float64 de vértices ``(x, y)`` por região, na ordem do XML.
    """
    regions = []
    for region in ET.parse(xml_path).getroot().iter("Region"):
        vertices = [
            (float(vertex.attrib["X"]), float(vertex.attrib["Y"])) for vertex in region.iter("Vertex")
        ]
        regions.append(np.asarray(vertices, dtype=np.float64).reshape(-1, 2))
    return regions


def polygon_area(vertices: np.ndarray) -> float:
    """Área de um polígono pela fórmula do laço (*shoelace*).

    Args:
        vertices: Array ``(K, 2)`` de vértices ``(x, y)``.

    Returns:
        Área em px². É igual ao atributo ``Area`` das regiões do XML.
    """
    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def xml_to_instance_mask(xml_path: str, shape: Tuple[int, int]) -> np.ndarray:
    """Rasteriza os núcleos de um XML do MoNuSeg como máscara por instância.

    Regras (base medida em ``docs/estudo/09-correcoes-pontuais.md`` §13; PD-25 e PD-07):

    - **Convenção de canto do pixel:** ``X = 0`` é a borda esquerda do pixel 0, então o centro do
      pixel ``(r, c)`` fica em ``(c + 0,5, r + 0,5)``. Um pixel pertence ao núcleo se o seu centro
      cai dentro do polígono. Assim a área rasterizada bate com a área anotada (o ``cv2.fillPoly``
      usado antes pintava todo pixel tocado pela borda e deixava o GT ~10% maior).
    - Regiões com menos de 3 vértices ou área zero (cliques soltos na anotação) são descartadas.
    - Na sobreposição entre polígonos, **o menor vence**: os núcleos são pintados do maior para o
      menor, e nenhum núcleo pequeno some por inteiro.
    - Os rótulos são ``1..N`` consecutivos, na ordem do XML, só para as regiões que ocupam pelo
      menos um pixel; ``0`` é fundo.

    Args:
        xml_path: Caminho do arquivo ``.xml``.
        shape: ``(altura, largura)`` da imagem correspondente.

    Returns:
        Array ``int32`` com formato ``shape`` e um rótulo por núcleo.
    """
    height, width = shape[:2]
    rasterized = []
    n_degenerate = 0
    n_empty = 0
    for vertices in read_xml_regions(xml_path):
        if len(vertices) < 3 or polygon_area(vertices) <= 0.0:
            n_degenerate += 1
            continue
        rows, cols = draw_polygon(vertices[:, 1] - 0.5, vertices[:, 0] - 0.5, shape=(height, width))
        if rows.size == 0:
            n_empty += 1
            continue
        rasterized.append((rows, cols))

    instances = np.zeros((height, width), dtype=np.int32)
    # Do maior para o menor (ordenação estável: empates seguem a ordem do XML).
    for label in sorted(range(len(rasterized)), key=lambda k: -rasterized[k][0].size):
        rows, cols = rasterized[label]
        instances[rows, cols] = label + 1

    if n_degenerate or n_empty:
        logger.info(
            f"{os.path.basename(xml_path)}: {n_degenerate} região(ões) degenerada(s) descartada(s) "
            f"(< 3 vértices ou área 0); {n_empty} sem nenhum pixel."
        )
    return instances


class MonusegDataset(BaseDataset):
    def __init__(
      self,
      dataset_name: str,
      config_key: str,
      transform: Optional[Any] = None,
      yaml_path: str = BaseDataset.YAML_CONFIG_PATH
    ) -> None:
        super().__init__(dataset_name, config_key, transform, yaml_path)
        self.image_dir = self.get_image_dir()
        self.mask_dir = self.get_mask_dir()
        self.file_pairs = self._get_file_pairs()

    def __len__(self) -> int:
        """Retorna o número de amostras no dataset."""
        return len(self.file_pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Retorna a amostra (imagem e máscara) no índice especificado.

        Com anotações em ``.xml``, a amostra também traz ``ground_truth_instances`` (``int32``, um
        rótulo por núcleo), e ``ground_truth`` é a versão binária dela (``uint8`` 0/1).
        """
        image_path, mask_path = self.file_pairs[idx]
        image = self._load_image(image_path)
        instances = None
        if mask_path.lower().endswith(".xml"):
            instances = xml_to_instance_mask(mask_path, image.shape[:2])
            mask = (instances > 0).astype(np.uint8)
        else:
            mask = self._load_mask(mask_path, image_shape=image.shape[:2])

        sample = {
            'id': os.path.splitext(os.path.basename(image_path))[0],
            'image': image,
            'ground_truth': mask,
            'meta': {
                'image_path': image_path,
                'mask_path': mask_path
            }
        }
        if instances is not None:
            sample['ground_truth_instances'] = instances

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _load_image(self, image_path: str) -> np.ndarray:
        """Carrega uma imagem a partir do caminho usando o mesmo leitor do Cellpose."""
        logger.debug(f"Loading image: {image_path}")
        image = io.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        return image

    def _xml_to_mask(self, xml_path: str, shape: Tuple[int, int]) -> np.ndarray:
        """Máscara binária ``uint8`` 0/1 do XML (ver :func:`xml_to_instance_mask`)."""
        return (xml_to_instance_mask(xml_path, shape) > 0).astype(np.uint8)

    def _load_mask(self, mask_path: str, image_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Carrega uma máscara do disco. Suporta formatos .xml, .npy e baseados em imagem."""
        logger.debug(f"Loading mask: {mask_path}")
        extension = os.path.splitext(mask_path)[1].lower()

        if extension == '.npy':
            mask = np.load(mask_path)
        elif extension == '.xml':
            if image_shape is None:
                stem = os.path.splitext(os.path.basename(mask_path))[0]
                image_path = os.path.join(self.image_dir, stem + self.config['image_extension'])
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Corresponding image not found for {mask_path}: expected {image_path}")
                image_shape = self._load_image(image_path).shape[:2]
            mask = self._xml_to_mask(mask_path, image_shape)
        else:
            mask = io.imread(mask_path)

        if mask is None:
            raise FileNotFoundError(f"Failed to load mask: {mask_path}")

        return mask

    def _get_file_pairs(self) -> List[Tuple[str, str]]:
        """Retorna a lista de pares (imagem, máscara) a partir das pastas configuradas."""
        logger.debug(f"Getting file pairs from: image_dir={self.image_dir}, mask_dir={self.mask_dir}")
        image_ext = self.config['image_extension'].lower()
        mask_ext = self.config['mask_extension'].lower()

        image_files = sorted(
            [f for f in os.listdir(self.image_dir) if f.lower().endswith(image_ext)]
        )

        if not image_files:
            raise FileNotFoundError(
                f"No images found in {self.image_dir} with extension {image_ext}"
            )

        pairs: List[Tuple[str, str]] = []
        for image_name in image_files:
            image_path = os.path.join(self.image_dir, image_name)
            stem = os.path.splitext(image_name)[0]
            mask_name = stem + mask_ext
            mask_path = os.path.join(self.mask_dir, mask_name)

            if not os.path.exists(mask_path):
                raise FileNotFoundError(
                    f"Mask not found for {image_name}: expected {mask_path}"
                )

            pairs.append((image_path, mask_path))

        return pairs
