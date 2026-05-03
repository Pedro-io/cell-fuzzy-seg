from abc import ABC, abstractmethod
from typing import Dict, Any


class PipelineStep(ABC):
    """
    Base class for all pipeline steps.

    Each step:
    - Receives a data dictionary
    - Processes it
    - Returns the updated dictionary
    """

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Allows the step to be called like a function.
        """
        return self.forward(data)

    @abstractmethod
    def forward(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main logic of the step.
        Must be implemented by subclasses.
        """
        pass