"""Módulo de perda de erro quadrático médio (RMSE)."""

import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    """Perda de erro quadrático médio (RMSE).

    Calcula o RMSE como ``sqrt(MSE(y_pred, y_true))``.
    """

    def __init__(self) -> None:
        """Inicializa RMSELoss com um MSELoss interno."""
        super().__init__()
        self._mse = nn.MSELoss()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Calcula o RMSE entre a previsão e o alvo.

        Args:
            y_pred: Tensor previsto de qualquer formato.
            y_true: Tensor de ground truth com o mesmo formato de ``y_pred``.

        Returns:
            Valor escalar da perda RMSE.
        """
        return torch.sqrt(self._mse(y_pred, y_true))
