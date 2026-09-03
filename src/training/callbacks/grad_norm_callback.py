"""Callback de monitoramento de gradientes.

Este módulo fornece :class:`GradNormCallback`, um :class:`TrainerCallback` que
calcula e registra, ao final de cada passo de treinamento, a norma dos
gradientes das redes treináveis presentes no ``TrainingPipeline`` (na prática,
a MarkerNet — a única rede com ``requires_grad=True``).

O objetivo é o diagnóstico da investigação do experimento 1 (problema P1): a
loss ficava perfeitamente plana porque o gradiente que chegava à MarkerNet era
efetivamente zero em precisão float32. Com este callback é possível confirmar
(ou descartar) a hipótese observando ``total``/``max_abs`` ao longo das épocas:
valores abaixo de ~1e-6 (ou exatamente 0) indicam gradiente morto.
"""

from typing import Any, Dict, List

import torch

from src.training.trainer import Trainer, TrainerCallback
from src.utils.logger import logger


class GradNormCallback(TrainerCallback):
    """Calcula e registra a norma L2 dos gradientes das redes treináveis.

    Para cada passo de treinamento, percorre os steps do ``TrainingPipeline``,
    coleta os parâmetros com ``requires_grad=True`` e computa:

    - ``total``: norma L2 combinada de todos os gradientes;
    - ``mean_abs``: média do valor absoluto dos gradientes;
    - ``max_abs``: maior valor absoluto entre os gradientes;
    - ``nonzero_frac``: fração de parâmetros com gradiente não-``None``
      (detecta parâmetros "mortos").

    Os valores são acumulados em :attr:`history` (para plotagem) e registrados
    no logger a cada ``log_every`` passos.

    Args:
        log_every: A cada quantos passos de treinamento registrar no logger.
            Padrão: 1 (todos os passos).
    """

    def __init__(self, log_every: int = 1) -> None:
        super().__init__()
        if log_every <= 0:
            raise ValueError(f"log_every must be positive, got {log_every}")
        self.log_every = log_every
        self._step = 0
        self.history: Dict[str, List[float]] = {
            "total": [],
            "mean_abs": [],
            "max_abs": [],
            "nonzero_frac": [],
        }

    def on_train_step_end(
        self,
        trainer: Trainer,
        data: Dict[str, Any],
        loss: torch.Tensor,
        loss_log: Dict[str, torch.Tensor],
    ) -> None:
        """Calcula as estatísticas de gradiente após o passo de treinamento.

        Args:
            trainer: O :class:`Trainer` associado (contém o ``TrainingPipeline``).
            data: Dicionário de dados do batch.
            loss: Valor escalar da loss do passo.
            loss_log: Log dos termos de perda do passo.
        """
        self._step += 1
        stats = self._compute_stats(trainer)

        for key, value in stats.items():
            self.history[key].append(value)

        if self._step % self.log_every == 0:
            logger.info(
                f"[GradNorm] step={self._step} | total={stats['total']:.3e} | "
                f"mean_abs={stats['mean_abs']:.3e} | max_abs={stats['max_abs']:.3e} | "
                f"nonzero_frac={stats['nonzero_frac']:.3f}"
            )

    def on_validation_step_end(
        self,
        trainer: Trainer,
        data: Dict[str, Any],
        loss: torch.Tensor,
        loss_log: Dict[str, torch.Tensor],
    ) -> None:
        """Sem ação na validação (não há backward)."""

    def on_epoch_end(
        self,
        trainer: Trainer,
        epoch: int,
        train_metrics: Dict[str, Any],
        val_metrics: Dict[str, Any],
    ) -> None:
        """Sem ação no fim da época."""

    def _compute_stats(self, trainer: Trainer) -> Dict[str, float]:
        """Computa as estatísticas de gradiente dos parâmetros treináveis.

        Args:
            trainer: O :class:`Trainer` associado.

        Returns:
            Dicionário com ``total``, ``mean_abs``, ``max_abs`` e
            ``nonzero_frac``.
        """
        params = self._collect_trainable_params(trainer)

        if not params:
            return {"total": 0.0, "mean_abs": 0.0, "max_abs": 0.0, "nonzero_frac": 0.0}

        grads = [p.grad for p in params if p.grad is not None]
        nonzero_frac = len(grads) / len(params)

        if not grads:
            return {"total": 0.0, "mean_abs": 0.0, "max_abs": 0.0, "nonzero_frac": nonzero_frac}

        total_sq = torch.stack([g.detach().pow(2).sum() for g in grads]).sum()
        total = torch.sqrt(total_sq).item()
        mean_abs = torch.stack([g.detach().abs().mean() for g in grads]).mean().item()
        max_abs = torch.stack([g.detach().abs().max() for g in grads]).max().item()

        return {
            "total": float(total),
            "mean_abs": float(mean_abs),
            "max_abs": float(max_abs),
            "nonzero_frac": float(nonzero_frac),
        }

    @staticmethod
    def _collect_trainable_params(trainer: Trainer) -> List[torch.nn.Parameter]:
        """Coleta os parâmetros treináveis dos steps do pipeline.

        Percorre os steps do ``TrainingPipeline`` e coleta os parâmetros com
        ``requires_grad=True`` dos atributos de modelo conhecidos
        (``model``/``final_network``). Redes congeladas (``requires_grad=False``)
        são naturalmente excluídas.

        Args:
            trainer: O :class:`Trainer` associado.

        Returns:
            Lista de parâmetros treináveis.
        """
        params: List[torch.nn.Parameter] = []
        pipeline = getattr(trainer, "training_pipeline", None)
        steps = getattr(pipeline, "steps", []) or []

        for step in steps:
            for attr in ("model", "final_network"):
                module = getattr(step, attr, None)
                if module is not None and hasattr(module, "parameters"):
                    params.extend(p for p in module.parameters() if p.requires_grad)

        return params
