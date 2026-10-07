"""Callbacks de treinamento.

Callbacks observam o progresso do treinamento implementando a interface
:class:`~src.training.trainer.TrainerCallback` (logging, early stopping,
checkpointing, monitoramento de gradientes etc.).
"""

from .best_model_callback import BestModelCallback
from .grad_norm_callback import GradNormCallback

__all__ = ["BestModelCallback", "GradNormCallback"]
