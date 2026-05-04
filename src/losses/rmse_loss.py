"""Root Mean Square Error loss module."""

import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    """Root Mean Square Error loss.

    Computes RMSE as ``sqrt(MSE(y_pred, y_true))``.
    """

    def __init__(self) -> None:
        """Initializes RMSELoss with an internal MSELoss."""
        super().__init__()
        self._mse = nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Computes RMSE between prediction and target.

        Args:
            y_pred: Predicted tensor of any shape.
            y_true: Ground truth tensor of the same shape as ``y_pred``.

        Returns:
            Scalar RMSE loss value.
        """
        return torch.sqrt(self._mse(y_pred, y_true))
