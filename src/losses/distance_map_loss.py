"""Módulo de perda de regularização por mapa de distância."""

import torch
import torch.nn as nn


class DistanceMapLoss(nn.Module):
    """Penaliza previsões em regiões de alto valor no mapa de distância.

    Calcula um produto elementwise ponderado entre a previsão e um mapa de
    distância pré-computado, normalizado pela massa total do ground truth.
    Isso empurra as previsões para longe das fronteiras entre objetos codificadas
    no mapa.

    Formato esperado de entrada: ``(N, C, H, W)``.
    """

    def __init__(self, weight: float = 0.1) -> None:
        """Inicializa DistanceMapLoss.

        Args:
            weight: Multiplicador escalar aplicado à perda.
        """
        super().__init__()
        self.weight = weight

    def forward(
        self,
        y_pred: torch.Tensor,
        distance_map: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """Calcula a perda do mapa de distância.

        Args:
            y_pred: Tensor previsto com formato ``(N, C, H, W)``.
            distance_map: Tensor do mapa de distância com o mesmo formato de ``y_pred``.
            y_true: Tensor de ground truth usado como referência de normalização.

        Returns:
            Perda do mapa de distância ponderada e escalar.

        Raises:
            AssertionError: Se os formatos de ``y_pred`` e ``distance_map`` forem diferentes.
        """
        assert y_pred.size() == distance_map.size(), (
            f"y_pred {y_pred.size()} and distance_map {distance_map.size()} must have the same shape."
        )
        normalization = y_true.sum()
        loss = torch.mul(y_pred, distance_map).sum() / normalization
        return self.weight * loss
