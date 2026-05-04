"""Distance map regularization loss module."""

import torch
import torch.nn as nn


class DistanceMapLoss(nn.Module):
    """Penalizes predictions in high-distance-map regions.

    Computes a weighted element-wise product between the prediction and a
    precomputed distance map, normalized by the total ground truth mass. This
    pushes predictions away from inter-object boundaries encoded in the map.

    Expected input shape: ``(N, C, H, W)``.
    """

    def __init__(self, weight: float = 0.1) -> None:
        """Initializes DistanceMapLoss.

        Args:
            weight: Scalar multiplier applied to the loss.
        """
        super().__init__()
        self.weight = weight

    def forward(
        self,
        y_pred: torch.Tensor,
        distance_map: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """Computes the distance map loss.

        Args:
            y_pred: Predicted tensor of shape ``(N, C, H, W)``.
            distance_map: Distance map tensor with the same shape as ``y_pred``.
            y_true: Ground truth tensor used as a normalization reference.

        Returns:
            Scalar weighted distance map loss.

        Raises:
            AssertionError: If ``y_pred`` and ``distance_map`` shapes differ.
        """
        assert y_pred.size() == distance_map.size(), (
            f"y_pred {y_pred.size()} and distance_map {distance_map.size()} must have the same shape."
        )
        normalization = y_true.sum()
        loss = torch.mul(y_pred, distance_map).sum() / normalization
        return self.weight * loss
