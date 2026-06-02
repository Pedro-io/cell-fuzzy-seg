from typing import Dict, Any
import numpy as np
import cv2
from scipy import ndimage

from .base_step import PipelineStep
from src.utils.logger import logger


class SegmentationStep(PipelineStep):
    """Pipeline step that refines initial segmentation using cell markers.

    Uses marker-based watershed segmentation to refine the initial Cellpose
    segmentation. Markers guide the watershed algorithm to separate touching
    cells and improve overall segmentation quality.

    Attributes:
        name: Identifier for this step in the pipeline.
        use_distance_map: Whether to use distance map for watershed (default: True).
    """

    def __init__(self, name: str = "SegmentationStep", use_distance_map: bool = True):
        """Initializes SegmentationStep.

        Args:
            name: Identifier for this step in the pipeline.
            use_distance_map: If True, uses distance map for watershed.
                If False, uses binary opening/closing with markers.
        """
        super().__init__(name=name)
        self.use_distance_map = use_distance_map

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Applies watershed-based segmentation refinement.

        Args:
            data: Dictionary containing at least:
                - ``"image"``: NumPy array of shape ``(H, W)`` or ``(H, W, C)``.
                - ``"segmentation"``: integer mask array of shape ``(H, W)``
                  (initial Cellpose output).
                - ``"markers"``: float array of shape ``(H, W)`` with marker seeds
                  (values in [0, 1]).

        Returns:
            The same ``data`` dictionary with updated:
                - ``"segmentation"``: refined integer mask of shape ``(H, W)``.

        Raises:
            KeyError: If required keys are absent from ``data``.
            RuntimeError: If watershed fails.
        """
        required_keys = ["image", "segmentation", "markers"]
        missing = [k for k in required_keys if k not in data]
        
        if missing:
            raise KeyError(f"Missing required keys for SegmentationStep: {missing}")

        image = data["image"]
        segmentation = data["segmentation"].astype(np.uint8)
        markers = data["markers"].astype(np.float32)

        try:
            # Apply watershed refinement
            refined_seg = self._watershed_refinement(image, segmentation, markers)
            
            data["segmentation"] = refined_seg
            
            logger.debug(f"[{self.name}] Refined segmentation from {segmentation.shape}")
            
        except Exception as e:
            logger.error(f"[{self.name}] Refinement failed: {e}")
            raise RuntimeError(f"Segmentation refinement failed: {e}") from e

        return data

    def _watershed_refinement(
        self, 
        image: np.ndarray, 
        segmentation: np.ndarray, 
        markers: np.ndarray
    ) -> np.ndarray:
        """Applies watershed algorithm to refine segmentation using markers.

        Strategy:
        1. Binarize the initial segmentation
        2. Label the marker regions
        3. Use watershed on inverted distance map with markers as seeds
        4. Re-label the output

        Args:
            image: Original image of shape ``(H, W)`` or ``(H, W, C)``.
            segmentation: Initial segmentation mask of shape ``(H, W)``.
            markers: Marker seeds of shape ``(H, W)`` in [0, 1].

        Returns:
            Refined segmentation mask as integer array.
        """
        # Binarize initial segmentation (ensure background is 0, foreground is non-zero)
        binary_seg = (segmentation > 0).astype(np.uint8)
        
        # Binarize markers and label them
        binary_markers = (markers > 0.5).astype(np.uint8)
        labeled_markers, num_markers = ndimage.label(binary_markers)
        
        logger.debug(f"[{self.name}] Found {num_markers} marker regions")
        
        if num_markers == 0:
            logger.warning(f"[{self.name}] No markers found. Returning original segmentation.")
            return segmentation
        
        if self.use_distance_map:
            # Use distance map from segmentation as watershed input
            dist_map = ndimage.distance_transform_edt(binary_seg)
            
            # Invert distance map for watershed (watershed floods from minima)
            watershed_input = -dist_map
        else:
            # Alternative: use negative of image gradient
            if image.ndim == 3:
                image_gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                image_gray = image.astype(np.uint8)
            
            gradient = cv2.morphologyEx(image_gray, cv2.MORPH_GRADIENT, 
                                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            watershed_input = gradient.astype(np.float32)
        
        # Apply watershed with markers as seeds
        refined_seg = cv2.watershed(
            cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
            if image.ndim == 2
            else image.astype(np.uint8),
            labeled_markers.copy()
        )
        
        # -1 values are borders; set them to 0 (background)
        refined_seg[refined_seg == -1] = 0
        refined_seg[refined_seg == 1] = 0  # Marker label 1 is usually background
        
        # Re-map labels to consecutive integers
        unique_labels = np.unique(refined_seg)
        unique_labels = unique_labels[unique_labels > 0]
        
        remapped_seg = np.zeros_like(refined_seg)
        for new_id, old_id in enumerate(unique_labels, start=1):
            remapped_seg[refined_seg == old_id] = new_id
        
        return remapped_seg.astype(np.uint8)
