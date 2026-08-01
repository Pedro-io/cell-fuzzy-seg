"""Módulo de métrica de acurácia baseada em RMSE."""

import torch
import torch.nn as nn


class RMSEAccuracy(nn.Module):
    """Métrica de acurácia derivada do RMSE como ``1 - RMSE(y_pred, y_true)``.

    Valores maiores indicam melhores previsões. A métrica é limitada
    superiormente por 1.0 e não possui limite inferior.
    """

    def __init__(self) -> None:
        """Inicializa RMSEAccuracy com um MSELoss interno."""
        super().__init__()
        self._mse = nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calcula a acurácia baseada em RMSE.

        Args:
            y_pred: Tensor previsto de qualquer formato.
            y_true: Tensor de ground truth com o mesmo formato de ``y_pred``.

        Returns:
            Valor escalar de acurácia na faixa ``(-inf, 1]``.
        """
        return 1 - torch.sqrt(self._mse(y_pred, y_true))
