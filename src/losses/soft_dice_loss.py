"""Módulo de perda Soft Dice."""

import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    """Perda Soft Dice para tarefas de segmentação.

    Calcula a perda diferenciável de Dice média sobre o batch e todos os canais.

    Formato esperado de entrada: ``(B, C, H, W)``.
    """

    def __init__(self, epsilon: float = 1e-9) -> None:
        """Inicializa SoftDiceLoss.

        Args:
            epsilon: Constante pequena adicionada ao numerador e ao denominador
                para estabilidade numérica.
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calcula a perda soft Dice.

        Args:
            y_pred: Tensor de segmentação previsto com formato ``(B, C, H, W)``.
            y_true: Tensor de segmentação de ground truth com formato ``(B, C, H, W)``.

        Returns:
            Valor escalar da perda soft Dice na faixa ``[0, 1]``.
        """
        axes = tuple(range(2, y_pred.ndim))
        numerator = 2.0 * torch.sum(y_pred * y_true, dim=axes)
        denominator = torch.sum(y_pred**2 + y_true**2, dim=axes)
        dice_per_channel = (numerator + self.epsilon) / (denominator + self.epsilon)
        return 1 - torch.mean(dice_per_channel)
