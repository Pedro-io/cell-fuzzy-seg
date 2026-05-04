"""Total variation regularization loss module."""

import torch
import torch.nn as nn


class TotalVariationLoss(nn.Module):
    """Anisotropic total variation loss normalized by ground truth mass.

    Encourages spatial smoothness in predictions by penalizing differences
    between adjacent pixels, normalized by the square root of the ground truth
    sum to remain invariant to object density.

    Expected input shape: ``(N, C, H, W)``.
    """

    def __init__(self, weight: float = 0.1, power: int = 1) -> None:
        """Initializes TotalVariationLoss.

        Args:
            weight: Scalar multiplier applied to the loss.
            power: Exponent applied to absolute pixel differences.
        """
        super().__init__()
        self.weight = weight
        self.power = power

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Computes the total variation loss.

        Args:
            y_pred: Predicted tensor of shape ``(N, C, H, W)``.
            y_true: Ground truth tensor used as a normalization reference.

        Returns:
            Scalar weighted total variation loss.
        """
        normalization = torch.sqrt(y_true.sum())
        h_tv = torch.sum(torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]) ** self.power)
        w_tv = torch.sum(torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]) ** self.power)
        return self.weight * (h_tv + w_tv) / normalization
