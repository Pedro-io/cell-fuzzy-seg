"""Output writer utilities for segmentation and overlay export."""

import os
from typing import Optional, Tuple

import cv2
import numpy as np


class OutputWriter:
    """Write segmentation outputs to disk."""

    def __init__(self, output_dir: str):
        """Initialize output directories.

        Args:
            output_dir: Base directory where output files will be saved.
        """
        self.output_dir = output_dir
        self.seg_dir = os.path.join(output_dir, "segmentations")
        self.marker_dir = os.path.join(output_dir, "markers")
        self.overlay_dir = os.path.join(output_dir, "overlays")

        os.makedirs(self.seg_dir, exist_ok=True)
        os.makedirs(self.marker_dir, exist_ok=True)
        os.makedirs(self.overlay_dir, exist_ok=True)

    def save_all(self, data: dict) -> None:
        """Save all available outputs from a data dictionary.

        Args:
            data: Dictionary containing optional keys 'id', 'image', 'segmentation',
                'markers', and 'ground_truth'.
        """
        image_id = data.get("id", "sample")

        if "segmentation" in data:
            self.save_segmentation(image_id, data["segmentation"])

        if "markers" in data:
            self.save_markers(image_id, data["markers"])

        if "image" in data and "segmentation" in data:
            self.save_overlay(
                image_id,
                data["image"],
                data["segmentation"],
                data.get("ground_truth"),
            )

    def save_segmentation(self, image_id: str, seg: np.ndarray) -> None:
        """Save a segmentation mask as a PNG file.

        Args:
            image_id: Identifier for the image used in the filename.
            seg: Segmentation mask array.
        """
        path = os.path.join(self.seg_dir, f"{image_id}_seg.png")

        seg = self._to_uint8(seg)

        cv2.imwrite(path, seg)

    def save_markers(self, image_id: str, markers: np.ndarray) -> None:
        """Save marker annotations as a PNG file.

        Args:
            image_id: Identifier for the image used in the filename.
            markers: Marker mask array.
        """
        path = os.path.join(self.marker_dir, f"{image_id}_markers.png")

        markers = self._to_uint8(markers)

        cv2.imwrite(path, markers)

    def save_overlay(
        self,
        image_id: str,
        image: np.ndarray,
        segmentation: np.ndarray,
        ground_truth: Optional[np.ndarray] = None,
    ) -> None:
        path = os.path.join(self.overlay_dir, f"{image_id}_overlay.png")

        image = self._to_rgb(image)

        mask = segmentation > 0
        overlay = image.copy()

        overlay[mask] = (
            0.7 * image[mask] +
            0.3 * np.array([0, 255, 0])
        ).astype(np.uint8)

        if ground_truth is not None:
            gt_mask = ground_truth > 0

            overlay[gt_mask] = (
                0.7 * overlay[gt_mask] +
                0.3 * np.array([255, 0, 0])
            ).astype(np.uint8)

        overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        cv2.imwrite(path, overlay)

    def save_rgba(self, image_id, rgba):
      path = f"{self.output_dir}/{image_id}_rgba.png"
      cv2.imwrite(path, rgba)

    def _to_uint8(self, img: np.ndarray) -> np.ndarray:
        """Convert an image or mask to uint8.

        Args:
            img: Input image array.

        Returns:
            Image array with dtype uint8.
        """
        if img.dtype == np.uint8:
            return img

        img = img.astype(np.float32)

        if img.max() > 0:
            img = img / img.max()

        img = (img * 255).clip(0, 255).astype(np.uint8)
        return img

    def _to_rgb(self, img: np.ndarray) -> np.ndarray:
        """Convert a grayscale image to RGB if needed.

        Args:
            img: Input image array.

        Returns:
            RGB image array.
        """
        if img.ndim == 2:
            return cv2.cvtColor(self._to_uint8(img), cv2.COLOR_GRAY2BGR)

        if img.shape[2] == 3:
            return self._to_uint8(img)

        return img

    def _colorize(
        self,
        mask: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """Create a colored overlay for a mask.

        Args:
            mask: Binary or label mask.
            color: RGB color used for the overlay.

        Returns:
            Colorized mask overlay.
        """
        mask = self._to_uint8(mask)

        colored = np.zeros((*mask.shape, 3), dtype=np.uint8)

        for i in range(3):
            colored[:, :, i] = mask * (color[i] / 255)

        return colored.astype(np.uint8)
