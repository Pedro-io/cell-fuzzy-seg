"""Object size regularization loss module."""

import torch
import torch.nn as nn


class ObjectSizeLoss(nn.Module):
    """Penalizes predicted mask size relative to ground truth mass.

    Computes the ratio of the total predicted activation to the total ground
    truth activation, weighted by a scalar. Encourages predictions to match
    the overall size of annotated objects.

    Expected input shape: ``(N, C, H, W)``.
    """

    def __init__(self, weight: float = 0.1) -> None:
        """Initializes ObjectSizeLoss.

        Args:
            weight: Scalar multiplier applied to the size ratio loss.
        """
        super().__init__()
        self.weight = weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Computes the object size loss.

        Args:
            y_pred: Predicted tensor of shape ``(N, C, H, W)``.
            y_true: Ground truth tensor of shape ``(N, C, H, W)``.

        Returns:
            Scalar weighted size ratio loss.
        """
        ratio = y_pred.sum() / y_true.sum()
        return self.weight * ratio
