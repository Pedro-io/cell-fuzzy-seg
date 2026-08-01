from typing import Any, Dict

import numpy as np

from src.utils.image_utils import to_float32_rgb
from src.utils.logger import logger

from ..base_step import PipelineStep


class RGBAStep(PipelineStep):
    """Passo do pipeline que une uma imagem RGB com uma máscara de segmentação em RGBA.

    O canal alpha é derivado da máscara de segmentação: pixels pertencentes a
    algum segmento são totalmente opacos (1.0) e pixels de fundo são totalmente
    transparentes (0.0).

    Atributos:
        name: Identificador deste passo no pipeline.
    """

    def __init__(self, name: str = "RGBAStep"):
        """Inicializa RGBAStep.

        Args:
            name: Identificador deste passo no pipeline.
        """
        super().__init__(name)

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Constrói uma imagem RGBA e armazena-a no dicionário de dados.

        Args:
            data: Dicionário contendo:
                - ``"image"``: array NumPy com formato ``(H, W)`` ou ``(H, W, C)``.
                - ``"segmentation"``: array de máscara inteira com formato ``(H, W)``,
                  em que 0 representa o fundo.

        Returns:
            O mesmo dicionário ``data`` estendido com:
                - ``"rgba"``: array float32 com formato ``(H, W, 4)`` em [0, 1].

        Raises:
            KeyError: Se ``"image"`` ou ``"segmentation"`` estiverem ausentes em ``data``.
        """
        if "image" not in data:
            raise KeyError("Missing 'image' in data")

        if "segmentation" not in data:
            raise KeyError("Missing 'segmentation' in data")

        image = data["image"]
        segmentation = data["segmentation"]

        rgba = self._build_rgba(image, segmentation)

        data["rgba"] = rgba
        return data

    def _build_rgba(self, image: np.ndarray, segmentation: np.ndarray) -> np.ndarray:
        """Constrói um array RGBA a partir de uma imagem e de uma máscara de segmentação.

        Imagens em tons de cinza são expandidas para 3 canais. Imagens que não são
        float32 são convertidas para float32 e normalizadas para [0, 1] antes da concatenação.

        Args:
            image: Imagem de entrada com formato ``(H, W)`` ou ``(H, W, C)``.
            segmentation: Máscara inteira com formato ``(H, W)`` em que valores diferentes
                de zero indicam o primeiro plano.

        Returns:
            Array float32 com formato ``(H, W, 4)`` com o canal alpha derivado da máscara
            de segmentação e valores em [0, 1].
        """
        logger.debug(f"Building RGBA image: image shape {image.shape}, segmentation shape {segmentation.shape}")

        image = to_float32_rgb(image)

        # Binary alpha: fully opaque for segmented pixels, transparent for background.
        alpha = (segmentation > 0).astype(np.float32)

        alpha = np.expand_dims(alpha, axis=-1)

        rgba = np.concatenate([image, alpha], axis=-1)

        return rgba.astype(np.float32)
