"""Módulo de perda de regularização de borda."""

import torch
import torch.nn as nn


class BorderLoss(nn.Module):
    """Penaliza ativações próximas à borda da imagem.

    Constrói uma máscara binária de borda com largura configurável e calcula
    um produto interno ponderado, normalizado espacialmente, entre a previsão
    e a máscara. É útil para suprimir detecções espúrias nas bordas da imagem.

    Formato esperado de entrada: ``(N, 1, H, W)``.
    """

    def __init__(self, weight: float = 1.0, border_size: int = 50) -> None:
        """Inicializa BorderLoss.

        Args:
            weight: Multiplicador escalar aplicado à perda.
            border_size: Largura, em pixels, da região de borda a ser penalizada.
        """
        super().__init__()
        self.weight = weight
        self.border_size = border_size

    def forward(self, y_pred: torch.Tensor) -> torch.Tensor:
        """Calcula a perda de borda.

        Args:
            y_pred: Tensor previsto com formato ``(N, 1, H, W)``.

        Returns:
            Perda de borda ponderada e escalar.
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
