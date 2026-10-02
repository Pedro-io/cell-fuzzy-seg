from typing import Any, Dict

import numpy as np

from src.utils.image_utils import to_float32_rgb
from src.utils.logger import logger

from ..base_step import PipelineStep


class RGBAStep(PipelineStep):
    """Passo do pipeline que une a imagem RGB e a saída do Cellpose numa imagem RGBA.

    O canal alpha vem do Cellpose, de uma de duas formas (parâmetro ``alpha``):

    - ``"mask"``: máscara binária, 1.0 nos pixels de algum segmento e 0.0 no fundo
      (lida de ``data["segmentation"]``);
    - ``"prob"``: probabilidade de cada pixel ser célula, em ``[0, 1]`` (lida de
      ``data["cellpose_prob"]``). Preserva a incerteza que a máscara binária descarta.

    Atributos:
        name: Identificador deste passo no pipeline.
        alpha: Origem do canal alpha (``"mask"`` ou ``"prob"``).
    """

    ALPHA_SOURCES = {"mask": "segmentation", "prob": "cellpose_prob"}

    def __init__(self, name: str = "RGBAStep", alpha: str = "mask"):
        """Inicializa RGBAStep.

        Args:
            name: Identificador deste passo no pipeline.
            alpha: Origem do canal alpha: ``"mask"`` (padrão) ou ``"prob"``.

        Raises:
            ValueError: Se ``alpha`` não for uma das opções.
        """
        super().__init__(name)
        if alpha not in self.ALPHA_SOURCES:
            raise ValueError(
                f"alpha deve ser um de {sorted(self.ALPHA_SOURCES)}, recebido {alpha!r}."
            )
        self.alpha = alpha

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Constrói uma imagem RGBA e armazena-a no dicionário de dados.

        Args:
            data: Dicionário contendo:
                - ``"image"``: array NumPy com formato ``(H, W)`` ou ``(H, W, C)``.
                - ``"segmentation"`` (com ``alpha="mask"``): máscara inteira ``(H, W)``, em que
                  0 representa o fundo.
                - ``"cellpose_prob"`` (com ``alpha="prob"``): probabilidade ``(H, W)`` em ``[0, 1]``.

        Returns:
            O mesmo dicionário ``data`` estendido com:
                - ``"rgba"``: array float32 com formato ``(H, W, 4)`` em [0, 1].

        Raises:
            KeyError: Se ``"image"`` ou a chave da origem do alpha estiverem ausentes em ``data``.
        """
        if "image" not in data:
            raise KeyError("Missing 'image' in data")

        source_key = self.ALPHA_SOURCES[self.alpha]
        if source_key not in data:
            raise KeyError(f"Missing '{source_key}' in data (alpha={self.alpha!r})")

        data["rgba"] = self._build_rgba(data["image"], data[source_key])
        return data

    def _build_rgba(self, image: np.ndarray, alpha_source: np.ndarray) -> np.ndarray:
        """Constrói um array RGBA a partir da imagem e da origem do canal alpha.

        Imagens em tons de cinza são expandidas para 3 canais. Imagens que não são
        float32 são convertidas para float32 e normalizadas para [0, 1] antes da concatenação.

        Args:
            image: Imagem de entrada com formato ``(H, W)`` ou ``(H, W, C)``.
            alpha_source: Com ``alpha="mask"``, máscara inteira ``(H, W)`` em que valores
                diferentes de zero indicam o primeiro plano; com ``alpha="prob"``, probabilidade
                ``(H, W)`` em ``[0, 1]``.

        Returns:
            Array float32 com formato ``(H, W, 4)`` e valores em [0, 1].

        Raises:
            ValueError: Se ``alpha_source`` não tiver o formato espacial da imagem, ou se a
                probabilidade sair de ``[0, 1]``.
        """
        logger.debug(
            f"Building RGBA image (alpha={self.alpha}): image shape {image.shape}, "
            f"alpha source shape {alpha_source.shape}"
        )

        if alpha_source.shape != image.shape[:2]:
            raise ValueError(
                f"Origem do alpha com formato {alpha_source.shape}, diferente da imagem {image.shape[:2]}."
            )

        image = to_float32_rgb(image)

        if self.alpha == "mask":
            alpha = (alpha_source > 0).astype(np.float32)
        else:
            alpha = np.asarray(alpha_source, dtype=np.float32)
            if alpha.min() < 0.0 or alpha.max() > 1.0:
                raise ValueError(
                    f"Probabilidade fora de [0, 1]: mín {alpha.min()}, máx {alpha.max()}."
                )

        rgba = np.concatenate([image, alpha[..., None]], axis=-1)

        return rgba.astype(np.float32)
