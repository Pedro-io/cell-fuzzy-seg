import numpy as np
from .base_step import PipelineStep
from src.utils.logger import logger


class RGBAStep(PipelineStep):
    """
    Combines RGB image with segmentation mask into RGBA image.

    Output:
        data["rgba"] -> shape (H, W, 4), dtype uint8
    """

    def forward(self, data: dict) -> dict:
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
        logger.debug(f"Building RGBA image: image shape {image.shape}, segmentation shape {segmentation.shape}")  
        if image.ndim == 2:
            image = np.stack([image]*3, axis=-1)

        # 🔹 garantir uint8 (0–255)
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            if image.max() > 0:
                image = image / image.max() * 255
            image = image.astype(np.uint8)

        # 🔹 alpha binário
        alpha = (segmentation > 0).astype(np.uint8) * 255

        # 🔹 garantir shape (H, W, 1)
        alpha = np.expand_dims(alpha, axis=-1)

        # 🔹 concatena canais
        rgba = np.concatenate([image, alpha], axis=-1)

        return rgba