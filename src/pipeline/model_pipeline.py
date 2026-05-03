from typing import List, Dict, Any

from .steps.base_step import PipelineStep
from src.utils.logger import logger


class ModelPipeline:
    """
    Orchestrates execution of a sequence of pipeline steps.
    """

    def __init__(self, steps: List[PipelineStep]):
        self.steps = steps

    def forward(self, data: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
        """
        Run all steps sequentially.

        Args:
            data: Input data dictionary.
            verbose: Whether to print step names.

        Returns:
            Updated data dictionary after all steps.
        """
        for step in self.steps:
            if verbose:
                logger.info(f"[Pipeline] Running: {step.name}")
            data = step(data)

        return data