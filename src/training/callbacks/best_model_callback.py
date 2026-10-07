"""Callback que acompanha o Dice de validação, guarda o melhor modelo e para o treino cedo.

O Dice é calculado a partir da saída que o ``Trainer`` já produz na validação (o ``data`` recebido
em ``on_validation_step_end``), sem um forward a mais. O melhor estado é guardado em memória; nada
é gravado em disco.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.evaluation.metrics import binary_metrics
from src.training.trainer import Trainer, TrainerCallback
from src.utils.logger import logger


class BestModelCallback(TrainerCallback):
    """Guarda o estado do modelo na época de melhor Dice de validação e, opcionalmente, para cedo.

    A cada época com validação, calcula o Dice binário médio por imagem das predições de validação.
    Quando ele supera o melhor anterior em mais de ``min_delta``, guarda uma cópia (em CPU) do
    ``state_dict`` de ``model``. Com ``patience``, marca ``trainer.stop_training = True`` depois de
    ``patience`` validações seguidas sem melhora. Ao fim, :meth:`restore_best` carrega o melhor estado.

    Args:
        model: Módulo cujo estado é guardado (a MarkerUNet).
        patience: Validações seguidas sem melhora até a parada. ``None``: nunca para.
        min_delta: Melhora mínima do Dice para contar como melhora.
        threshold: Limiar de binarização da predição.
        prediction_key: Chave da predição no ``data`` da validação.
        ground_truth_key: Chave do ground truth no ``data`` da validação.

    Atributos:
        history: Uma entrada ``{"epoch", "dice"}`` por época com validação.
        best_epoch: Época do melhor Dice (``None`` antes da primeira validação).
        best_dice: Melhor Dice médio de validação.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        patience: Optional[int] = None,
        min_delta: float = 0.0,
        threshold: float = 0.5,
        prediction_key: str = "segmentation",
        ground_truth_key: str = "ground_truth",
    ) -> None:
        super().__init__()
        if patience is not None and patience <= 0:
            raise ValueError(f"patience deve ser positivo ou None, recebido {patience}")
        self.model = model
        self.patience = patience
        self.min_delta = min_delta
        self.threshold = threshold
        self.prediction_key = prediction_key
        self.ground_truth_key = ground_truth_key

        self.history: List[Dict[str, float]] = []
        self.best_epoch: Optional[int] = None
        self.best_dice = -np.inf
        self.best_state: Optional[Dict[str, torch.Tensor]] = None
        self._epoch_dices: List[float] = []
        self._since_best = 0

    def on_train_step_end(self, trainer: Trainer, data: Dict[str, Any], loss: torch.Tensor,
                          loss_log: Dict[str, torch.Tensor]) -> None:
        """Sem ação no treino."""

    def on_validation_step_end(self, trainer: Trainer, data: Dict[str, Any], loss: torch.Tensor,
                               loss_log: Dict[str, torch.Tensor]) -> None:
        """Acumula o Dice de cada imagem do batch de validação."""
        prediction = data[self.prediction_key].detach().cpu()
        ground_truth = data[self.ground_truth_key].detach().cpu()
        for b in range(prediction.shape[0]):
            self._epoch_dices.append(
                binary_metrics(prediction[b, 0], ground_truth[b, 0], self.threshold)["dice"]
            )

    def on_epoch_end(self, trainer: Trainer, epoch: int, train_metrics: Dict[str, Any],
                     val_metrics: Optional[Dict[str, Any]]) -> None:
        """Atualiza o melhor estado e o contador de paciência ao fim de uma época com validação."""
        if val_metrics is None or not self._epoch_dices:
            return

        dice = float(np.mean(self._epoch_dices))
        self._epoch_dices = []
        self.history.append({"epoch": epoch, "dice": dice})

        if dice > self.best_dice + self.min_delta:
            self.best_dice = dice
            self.best_epoch = epoch
            self.best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            self._since_best = 0
        else:
            self._since_best += 1
            if self.patience is not None and self._since_best >= self.patience:
                trainer.stop_training = True
                logger.info(
                    f"[BestModelCallback] {self._since_best} validações sem melhora; melhor Dice "
                    f"{self.best_dice:.4f} na época {self.best_epoch}."
                )

    def restore_best(self) -> None:
        """Carrega no modelo o estado da época de melhor Dice.

        Raises:
            RuntimeError: Se nenhuma validação foi registrada.
        """
        if self.best_state is None:
            raise RuntimeError("Nenhum estado guardado: o treino não teve nenhuma validação.")
        self.model.load_state_dict(self.best_state)
