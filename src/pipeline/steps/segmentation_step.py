from typing import Dict, Any

from .base_step import PipelineStep


class SegmentationStep(PipelineStep):
    """Pipeline step that applies fuzzy segmentation using cell markers.

    Refines the initial segmentation mask produced by a prior step using
    marker-based techniques (e.g. watershed). Requires both ``"image"`` and
    ``"markers"`` keys to be present in the data dictionary.

    Attributes:
        name: Identifier for this step in the pipeline.
    """

    def __init__(self, name: str = "SegmentationStep"):
        """Initializes SegmentationStep.

        Args:
            name: Identifier for this step in the pipeline.
        """
        super().__init__(name=name)

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Applies fuzzy segmentation and stores the result in data.

        Args:
            data: Dictionary containing at least:
                - ``"image"``: NumPy array of shape ``(H, W)`` or ``(H, W, C)``.
                - ``"markers"``: array of shape ``(H, W)`` with marker seeds.

        Returns:
            The same ``data`` dictionary extended with:
                - ``"segmentation"``: refined integer mask of shape ``(H, W)``.

        Raises:
            KeyError: If ``"image"`` or ``"markers"`` are absent from ``data``.
        """
        raise NotImplementedError("SegmentationStep.forward() is not implemented yet.")
