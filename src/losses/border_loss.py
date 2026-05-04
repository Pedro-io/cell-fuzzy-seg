"""Border regularization loss module."""

import torch
import torch.nn as nn


class BorderLoss(nn.Module):
    """Penalizes activations near the image border.

    Builds a binary border mask of configurable width and computes a weighted,
    spatially-normalized inner product between the prediction and the mask.
    Useful for suppressing spurious detections at image edges.

    Expected input shape: ``(N, 1, H, W)``.
    """

    def __init__(self, weight: float = 1.0, border_size: int = 50) -> None:
        """Initializes BorderLoss.

        Args:
            weight: Scalar multiplier applied to the loss.
            border_size: Width in pixels of the border region to penalize.
        """
        super().__init__()
        self.weight = weight
        self.border_size = border_size

    def forward(self, y_pred: torch.Tensor) -> torch.Tensor:
        """Computes the border loss.

        Args:
            y_pred: Predicted tensor of shape ``(N, 1, H, W)``.

        Returns:
            Scalar weighted border loss.
        """
        n, c, h, w = y_pred.shape
        b = self.border_size
        border_mask = torch.zeros_like(y_pred)
        border_mask[:, :, :b, :] = 1
        border_mask[:, :, :, :b] = 1
        border_mask[:, :, h - b:, :] = 1
        border_mask[:, :, :, w - b:] = 1
        loss = torch.mul(y_pred, border_mask).sum() / (n * c * h * w)
        return self.weight * loss
