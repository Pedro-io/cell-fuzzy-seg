from typing import Any, Dict, List

from src.utils.logger import logger

from .steps.base_step import PipelineStep


class ModelPipeline:
    """
    EXCLUIR SCRIPT
    Orchestrates execution of a sequence of pipeline steps.
    """

    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps

    def forward(self, data: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
        """Run all steps sequentially, threading ``data`` through each one.

        The minimum required structure of ``data`` at pipeline entry depends on
        the configured steps. The standard pipeline expects:

        - ``"image"`` (``np.ndarray``): input image of shape ``(H, W)`` or
          ``(H, W, C)``. Required by ``CellposeStep``, ``MarkerStep``, and
          ``SegmentationStep``.

        Each step may add new keys that subsequent steps depend on:

        - ``CellposeStep`` adds ``"segmentation"``, ``"flows"``, ``"styles"``.
        - ``MarkerStep`` adds ``"markers"`` (when implemented).
        - ``SegmentationStep`` updates ``"segmentation"`` (when implemented).
        - ``RGBAStep`` adds ``"rgba"``.

        Args:
            data: Input data dictionary. Must be a ``dict``; raises
                ``TypeError`` otherwise.
            verbose: Log the name of each step before it runs.

        Returns:
            Updated data dictionary after all steps have been applied.

        Raises:
            TypeError: If ``data`` is not a ``dict``.
            Exception: Re-raises any exception thrown by a step, after logging
                which step failed.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Pipeline input must be a dict, got {type(data).__name__}.")

        for step in self.steps:
            if verbose:
                logger.info(f"[Pipeline] Running: {step.name}")
            try:
                data = step(data)
            except Exception as e:
                logger.error(f"[Pipeline] Step '{step.name}' failed: {e}")
                raise

        return data
