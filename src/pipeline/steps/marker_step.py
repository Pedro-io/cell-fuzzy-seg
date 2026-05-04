from typing import Dict, Any

from .base_step import PipelineStep


class MarkerStep(PipelineStep):
    """Pipeline step that generates cell markers from a segmentation mask.

    Markers are used as seeds for downstream watershed or fuzzy segmentation.
    Operates on the ``"segmentation"`` key produced by a prior segmentation step.

    Attributes:
        name: Identifier for this step in the pipeline.
    """

    def __init__(self, name: str = "MarkerStep"):
        """Initializes MarkerStep.

        Args:
            name: Identifier for this step in the pipeline.
        """
        super().__init__(name=name)

    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generates markers from the segmentation mask.

        Args:
            data: Dictionary containing at least:
                - ``"segmentation"``: integer mask array of shape ``(H, W)``.

        Returns:
            The same ``data`` dictionary extended with:
                - ``"markers"``: array of shape ``(H, W)`` with marker seeds.

        Raises:
            KeyError: If ``"segmentation"`` is absent from ``data``.
        """
        raise NotImplementedError("MarkerStep.forward() is not implemented yet.")
