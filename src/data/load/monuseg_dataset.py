import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from cellpose import io

from .base_dataset import BaseDataset


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
        """Retorna a amostra (imagem e máscara) no índice especificado."""
        if idx < 0 or idx >= len(self.file_pairs):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        image_path, mask_path = self.file_pairs[idx]
        image = self._load_image(image_path)
        mask = self._load_mask(mask_path)

        sample = {
            'image': image,
            'mask': mask,
            'image_name': os.path.basename(image_path),
            'mask_name': os.path.basename(mask_path),
            'image_path': image_path,
            'mask_path': mask_path,
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _load_image(self, image_path: str) -> np.ndarray:
        """Carrega uma imagem a partir do caminho usando o mesmo leitor do Cellpose."""
        image = io.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Falha ao carregar a imagem: {image_path}")
        return image

    def _load_mask(self, mask_path: str) -> np.ndarray:
        """Carrega máscara do disco. Suporta .npy e leitura por imagem."""
        extension = os.path.splitext(mask_path)[1].lower()

        if extension == '.npy':
            mask = np.load(mask_path)
        else:
            mask = io.imread(mask_path)

        if mask is None:
            raise FileNotFoundError(f"Falha ao carregar a máscara: {mask_path}")

        return mask

    def _get_file_pairs(self) -> List[Tuple[str, str]]:
        """Retorna a lista de pares (imagem, máscara) a partir das pastas configuradas."""
        image_ext = self.config['image_extension'].lower()
        mask_ext = self.config['mask_extension'].lower()

        image_files = sorted(
            [f for f in os.listdir(self.image_dir) if f.lower().endswith(image_ext)]
        )

        if not image_files:
            raise FileNotFoundError(
                f"Nenhuma imagem encontrada em {self.image_dir} com extensão {image_ext}"
            )

        pairs: List[Tuple[str, str]] = []
        for image_name in image_files:
            image_path = os.path.join(self.image_dir, image_name)
            stem = os.path.splitext(image_name)[0]
            mask_name = stem + mask_ext
            mask_path = os.path.join(self.mask_dir, mask_name)

            if not os.path.exists(mask_path):
                raise FileNotFoundError(
                    f"Máscara não encontrada para {image_name}: esperado {mask_path}"
                )

            pairs.append((image_path, mask_path))

        return pairs
      