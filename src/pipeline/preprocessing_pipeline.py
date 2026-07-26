"""Preprocessing pipeline: executes a sequence of non-trainable steps once per sample.

This module provides :class:`PreprocessingPipeline`, the canonical orchestrator
for preprocessing steps (e.g. Cellpose, RGBA conversion) that are run **once**
per image, outside the training loop.  Results are typically persisted by a
:class:`SaveResultsStep` and later consumed by :class:`MonusegPreprocessedDataset`
during training, avoiding recomputation of expensive steps (e.g. Cellpose) on
every epoch.
"""

from typing import Any, Dict, List

from src.pipeline.steps.base_step import PipelineStep
from src.utils.logger import logger


class PreprocessingPipeline:
    """Orchestrates a sequence of non-trainable preprocessing steps.

    Each step is executed in order, passing the same data dictionary through
    the chain.  Steps should **add** keys to the dictionary, never remove
    them.

    The canonical entry point is :meth:`run`, which matches the interface
    expected by :class:`~src.data.load.monuseg_preprocessed_dataset.MonusegPreprocessedDataset`.

    Typical usage::

        pipeline = PreprocessingPipeline([
            CellposeStep(),
            RGBAStep(),
            SaveResultsStep(output_dir="data/preprocessed"),
        ])
        data = pipeline.run({"image": image, "ground_truth": mask})

    Args:
        steps: Ordered list of preprocessing :class:`PipelineStep` instances.

    Raises:
        TypeError: If ``data`` passed to :meth:`run` is not a ``dict``.
    """

    def __init__(self, steps: List[PipelineStep]) -> None:
        self.steps = steps

    def run(self, data: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
        """Execute all steps sequentially on the data dictionary.

        Each step receives the current data dictionary and returns an updated
        version.  The output of step *i* becomes the input of step *i+1*.

        Args:
            data: Input data dictionary.  Must be a ``dict``.
            verbose: If ``True``, log each step's name before execution.

        Returns:
            Updated data dictionary after all steps have been applied.

        Raises:
            TypeError: If ``data`` is not a ``dict``.
            Exception: Re-raises any exception raised by a step, after logging
                the failing step name.
        """
        if not isinstance(data, dict):
            raise TypeError(f"Pipeline input must be a dict, got {type(data).__name__}.")

        for step in self.steps:
            if verbose:
                logger.info(f"[PreprocessingPipeline] Running: {step.name}")
            try:
                data = step(data)
            except Exception as e:
                logger.error(f"[PreprocessingPipeline] Step '{step.name}' failed: {e}")
                raise

        return data

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience alias for :meth:`run` (non-verbose).

        Allows the pipeline to be passed as a callable to
        :class:`~src.data.load.monuseg_preprocessed_dataset.MonusegPreprocessedDataset`
        without requiring ``.run()`` or ``.forward()`` explicitly.
        """
        return self.run(data, verbose=False)
