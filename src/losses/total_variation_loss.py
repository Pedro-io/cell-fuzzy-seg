"""Módulo de perda de regularização por variação total."""

import torch
import torch.nn as nn


class TotalVariationLoss(nn.Module):
    """Perda anisotrópica de variação total normalizada pela massa do ground truth.

    Incentiva suavidade espacial nas previsões penalizando diferenças entre
    pixels adjacentes, normalizadas pela raiz quadrada da soma do ground truth
    para permanecer invariantes à densidade de objetos.

    Formato esperado de entrada: ``(N, C, H, W)``.
    """

    def __init__(self, weight: float = 0.1, power: int = 1) -> None:
        """Inicializa TotalVariationLoss.

        Args:
            weight: Multiplicador escalar aplicado à perda.
            power: Expoente aplicado às diferenças absolutas entre pixels.
        """
        super().__init__()
        self.weight = weight
        self.power = power

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calcula a perda de variação total.

        Args:
            y_pred: Tensor previsto com formato ``(N, C, H, W)``.
            y_true: Tensor de ground truth usado como referência de normalização.

        Returns:
            Perda de variação total ponderada e escalar.
        """
        normalization = torch.sqrt(y_true.sum())
        h_tv = torch.sum(torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]) ** self.power)
        w_tv = torch.sum(torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1]) ** self.power)
        return self.weight * (h_tv + w_tv) / normalization
