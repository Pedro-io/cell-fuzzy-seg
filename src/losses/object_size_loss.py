"""Módulo de perda de regularização por tamanho de objeto."""

import torch
import torch.nn as nn


class ObjectSizeLoss(nn.Module):
    """Penaliza o tamanho da máscara prevista em relação à massa do ground truth.

    Calcula a razão entre a ativação prevista total e a ativação total do
    ground truth, ponderada por um escalar. Incentiva as previsões a corresponderem
    ao tamanho geral dos objetos anotados.

    Formato esperado de entrada: ``(N, C, H, W)``.
    """

    def __init__(self, weight: float = 0.1) -> None:
        """Inicializa ObjectSizeLoss.

        Args:
            weight: Multiplicador escalar aplicado à perda de razão de tamanho.
        """
        super().__init__()
        self.weight = weight

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calcula a perda de tamanho do objeto.

        Args:
            y_pred: Tensor previsto com formato ``(N, C, H, W)``.
            y_true: Tensor de ground truth com formato ``(N, C, H, W)``.

        Returns:
            Perda de razão de tamanho ponderada e escalar.
        """
        ratio = y_pred.sum() / y_true.sum()
        return self.weight * ratio
