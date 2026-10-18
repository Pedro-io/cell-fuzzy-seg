"""Módulo de perda de entropia cruzada binária (BCE) sobre probabilidades."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryCrossEntropyLoss(nn.Module):
    """Entropia cruzada binária média por pixel, calculada sobre probabilidades.

    ``−mean(y·log(p) + (1 − y)·log(1 − p))``. Diferente do soft Dice, que olha a sobreposição das
    regiões, ela pune cada pixel isolado: um pixel de fundo com probabilidade alta custa caro mesmo
    quando a sobreposição já é boa, o que empurra a borda prevista para o limiar de binarização.

    A entrada são probabilidades (a saída da rede final já passa por uma sigmoide), cortadas em
    ``[eps, 1 − eps]`` para o logaritmo não divergir.

    Formato esperado de entrada: ``(N, C, H, W)``.
    """

    def __init__(self, weight: float = 1.0, eps: float = 1e-6) -> None:
        """Inicializa BinaryCrossEntropyLoss.

        Args:
            weight: Multiplicador escalar aplicado à perda.
            eps: Margem do corte das probabilidades.
        """
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calcula a perda.

        Args:
            y_pred: Probabilidades previstas, em ``[0, 1]``.
            y_true: Ground truth binário com o mesmo formato.

        Returns:
            Perda escalar ponderada.
        """
        prob = y_pred.clamp(self.eps, 1.0 - self.eps)
        return self.weight * F.binary_cross_entropy(prob, y_true.to(prob.dtype))
