"""Módulo de perda de regularização por tamanho de objeto."""

import torch
import torch.nn as nn


class ObjectSizeLoss(nn.Module):
    """Penaliza o desvio de tamanho entre a máscara prevista e o ground truth.

    Calcula a razão entre a ativação prevista total e a ativação total do
    ground truth e penaliza o **desvio** dessa razão em relação a 1.0
    (``|ratio - 1|``). Diferente da formulação anterior (``weight * ratio``,
    que só empurrava a massa para baixo e tinha gradiente constante), esta
    versão é **simétrica**: tanto superestimação quanto subestimação do
    tamanho são penalizadas, e o ótimo ocorre em ``ratio == 1`` (massa prevista
    igual à do ground truth). O gradiente se anula no ponto ótimo, eliminando
    o "cabo de guerra" com gradiente constante contra os demais termos
    (investigação, P10).

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
        gt_sum = y_true.sum()
        if gt_sum == 0:
            # GT vazio: a massa prevista deve ser (idealmente) zero também.
            return self.weight * y_pred.sum().abs()
        ratio = y_pred.sum() / gt_sum
        return self.weight * (ratio - 1.0).abs()
