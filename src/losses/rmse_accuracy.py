"""Root Mean Square Error accuracy module."""

import torch
import torch.nn as nn


class RMSEAccuracy(nn.Module):
    """Accuracy metric derived from RMSE as ``1 - RMSE(y_pred, y_true)``.

    Higher values indicate better predictions. The metric is bounded above by
    1.0 and has no lower bound.
    """

    def __init__(self) -> None:
        """Initializes RMSEAccuracy with an internal MSELoss."""
        super().__init__()
        self._mse = nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Computes RMSE-based accuracy.

        Args:
            y_pred: Predicted tensor of any shape.
            y_true: Ground truth tensor of the same shape as ``y_pred``.

        Returns:
            Scalar accuracy value in the range ``(-inf, 1]``.
        """
        return 1 - torch.sqrt(self._mse(y_pred, y_true))
