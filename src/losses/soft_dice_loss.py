"""Soft Dice loss module."""

import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    """Soft Dice loss for segmentation tasks.

    Computes the differentiable Dice loss averaged over the batch and all
    channels.

    Expected input shape: ``(B, C, H, W)``.
    """

    def __init__(self, epsilon: float = 1e-9) -> None:
        """Initializes SoftDiceLoss.

        Args:
            epsilon: Small constant added to numerator and denominator for
                numerical stability.
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Computes the soft Dice loss.

        Args:
            y_pred: Predicted segmentation tensor of shape ``(B, C, H, W)``.
            y_true: Ground truth segmentation tensor of shape ``(B, C, H, W)``.

        Returns:
            Scalar soft Dice loss value in the range ``[0, 1]``.
        """
        axes = tuple(range(2, y_pred.ndim))
        numerator = 2.0 * torch.sum(y_pred * y_true, dim=axes)
        denominator = torch.sum(y_pred**2 + y_true**2, dim=axes)
        dice_per_channel = (numerator + self.epsilon) / (denominator + self.epsilon)
        return 1 - torch.mean(dice_per_channel)
