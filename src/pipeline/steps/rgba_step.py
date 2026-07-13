from typing import Dict, Any

import numpy as np
from .base_step import PipelineStep
from src.utils.logger import logger
from src.utils.image_utils import to_float32_rgb


class RGBAStep(PipelineStep):
    """Pipeline step that merges an RGB image with a segmentation mask into RGBA.

    The alpha channel is derived from the segmentation mask: pixels belonging
    to any segment are fully opaque (1.0) and background pixels are fully
    transparent (0.0).

    Attributes:
        name: Identifier for this step in the pipeline.
    """

    def __init__(self, name: str = "RGBAStep"):
        """Initializes RGBAStep.

        Args:
            name: Identifier for this step in the pipeline.
        """
        super().__init__(name)

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Builds an RGBA image and stores it in the data dictionary.

        Args:
            data: Dictionary containing:
                - ``"image"``: NumPy array of shape ``(H, W)`` or ``(H, W, C)``.
                - ``"segmentation"``: Integer mask array of shape ``(H, W)``,
                  where 0 represents background.

        Returns:
            The same ``data`` dictionary extended with:
                - ``"rgba"``: float32 array of shape ``(H, W, 4)`` in [0, 1].

        Raises:
            KeyError: If ``"image"`` or ``"segmentation"`` are absent from ``data``.
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
        """Constructs an RGBA array from an image and a segmentation mask.

        Grayscale images are expanded to 3 channels. Non-float32 images are
        converted to float32 and normalized to [0, 1] before concatenation.

        Args:
            image: Input image of shape ``(H, W)`` or ``(H, W, C)``.
            segmentation: Integer mask of shape ``(H, W)`` where non-zero values
                indicate foreground.

        Returns:
            float32 array of shape ``(H, W, 4)`` with the alpha channel derived
            from the segmentation mask and values in [0, 1].
        """
        logger.debug(f"Building RGBA image: image shape {image.shape}, segmentation shape {segmentation.shape}")

        image = to_float32_rgb(image)

        # Binary alpha: fully opaque for segmented pixels, transparent for background.
        alpha = (segmentation > 0).astype(np.float32)

        alpha = np.expand_dims(alpha, axis=-1)

        rgba = np.concatenate([image, alpha], axis=-1)

        return rgba.astype(np.float32)