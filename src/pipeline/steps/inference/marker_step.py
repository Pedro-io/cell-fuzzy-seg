from typing import Any, Dict, Optional

import cv2
import numpy as np
import torch

from src.utils.logger import logger

from ..base_step import PipelineStep


class MarkerStep(PipelineStep):
    """Pipeline step that generates cell markers using a neural network.

    Uses a pre-trained MarkerNet model to refine segmentation masks into
    precise cell markers. Expects 4-channel RGBA input (RGB image + segmentation).

    Attributes:
        model: Loaded MarkerNet model for inference.
        target_size: Size to resize input to for model inference (default: 256).
        threshold: Probability threshold for binarizing marker output (default: 0.5).
        device: PyTorch device for inference (cpu or cuda).
        name: Identifier for this step in the pipeline.
    """

    def __init__(
        self,
        model: Optional[Any] = None,
        target_size: int = 256,
        threshold: float = 0.5,
        device: Optional[str] = None,
        name: str = "MarkerStep"
    ):
        """Initializes MarkerStep.

        Args:
            model: Pre-trained MarkerNet model. If None, inference is skipped.
            target_size: Height/width to resize input to for inference.
            threshold: Probability threshold for binarization (0.0 to 1.0).
            device: PyTorch device. If None, uses GPU if available.
            name: Identifier for this step in the pipeline.
        """
        super().__init__(name=name)
        self.model = model
        self.target_size = target_size
        self.threshold = threshold

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if self.model is not None:
            self.model.model.to(self.device)
            self.model.model.eval()

        logger.info(f"[{self.name}] Initialized on {self.device}")

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generates markers from RGBA image using MarkerNet.

        Args:
            data: Dictionary containing:
                - ``"rgba"``: uint8 array of shape ``(H, W, 4)`` (RGB + alpha).
                - Optionally ``"image"`` and ``"segmentation"`` for fallback.

        Returns:
            The same ``data`` dictionary extended with:
                - ``"markers"``: float array of shape ``(H, W)`` in [0, 1].

        Raises:
            KeyError: If required keys are missing and model is loaded.
            RuntimeError: If model inference fails.
        """
        if "rgba" not in data:
            raise KeyError("Missing 'rgba' in data. Ensure RGBAStep runs before MarkerStep.")

        # Skip if model is not provided
        if self.model is None:
            logger.warning(f"[{self.name}] Model not provided. Markers will be computed from segmentation.")
            # Fallback: use segmentation as binary markers
            if "segmentation" not in data:
                raise KeyError("Fallback requires 'segmentation' key")
            data["markers"] = (data["segmentation"] > 0).astype(np.float32)
            return data

        rgba = data["rgba"]
        original_shape = rgba.shape[:2]

        try:
            # Preprocess: resize and normalize
            rgba_resized = self._preprocess(rgba)

            # Convert to tensor and add batch dimension
            rgba_tensor = torch.from_numpy(rgba_resized).float().to(self.device)
            rgba_tensor = rgba_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 4, 256, 256)

            # Inference
            with torch.no_grad():
                markers_pred = self.model.predict(rgba_tensor)  # (1, 1, 256, 256)

            # Post-process: resize back and threshold
            markers = self._postprocess(markers_pred.squeeze().cpu().numpy(), original_shape)

            data["markers"] = markers

            logger.debug(f"[{self.name}] Generated markers with shape {markers.shape}")

        except Exception as e:
            logger.error(f"[{self.name}] Inference failed: {e}")
            raise RuntimeError(f"MarkerNet inference failed: {e}") from e

        return data

    def _preprocess(self, rgba: np.ndarray) -> np.ndarray:
        """Resizes RGBA image to target size and normalizes to [0, 1].

        Args:
            rgba: uint8 array of shape ``(H, W, 4)``.

        Returns:
            float32 array of shape ``(256, 256, 4)`` normalized to [0, 1].
        """
        # Resize to target size
        rgba_resized = cv2.resize(
            rgba,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR
        )

        # Normalize to [0, 1] if the input is still in [0, 255]
        rgba_norm = rgba_resized.astype(np.float32)
        if rgba_norm.max() > 1.0:
            rgba_norm = rgba_norm / 255.0

        return rgba_norm

    def _postprocess(self, markers_pred: np.ndarray, original_shape: tuple) -> np.ndarray:
        """Resizes predictions back to original shape and applies threshold.

        Args:
            markers_pred: float32 array of shape ``(256, 256)`` with values in [0, 1].
            original_shape: Target output shape ``(H, W)``.

        Returns:
            float32 array of shape ``original_shape`` with thresholded probabilities.
        """
        # Resize back to original shape
        markers_resized = cv2.resize(
            markers_pred,
            (original_shape[1], original_shape[0]),  # cv2.resize uses (W, H)
            interpolation=cv2.INTER_LINEAR
        )

        # Apply threshold
        markers_binary = (markers_resized > self.threshold).astype(np.float32)

        return markers_binary
