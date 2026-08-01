"""Treinador: orquestra o passo de otimização consumindo o ``TrainingPipeline``.

Este módulo fornece :class:`Trainer` e a interface :class:`TrainerCallback`.

O :class:`Trainer` é a **única** classe do sistema autorizada a chamar
``backward()`` e ``optimizer.step()`` (regra arquitetural 23). Ele nunca
implementa redes neurais: recebe o :class:`TrainingPipeline` já configurado com
as redes e o trata como uma caixa-preta diferenciável, delegando o cálculo da
loss ao ``LossComposer``.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.pipeline.training_pipeline import TrainingPipeline
from src.utils.logger import logger


class TrainerCallback(ABC):
    """Interface base para callbacks de treinamento.

    Subclasses devem implementar :meth:`on_train_step_end`,
    :meth:`on_validation_step_end` e :meth:`on_epoch_end` para observar o
    progresso do treinamento (logging, early stopping, checkpointing, etc.).
    """

    @abstractmethod
    def on_train_step_end(
        self,
        trainer: "Trainer",
        data: Dict[str, Any],
        loss: torch.Tensor,
        loss_log: Dict[str, torch.Tensor],
    ) -> None:
        """Chamado após um passo de treinamento ser concluído."""
        raise NotImplementedError

    @abstractmethod
    def on_validation_step_end(
        self,
        trainer: "Trainer",
        data: Dict[str, Any],
        loss: torch.Tensor,
        loss_log: Dict[str, torch.Tensor],
    ) -> None:
        """Chamado após um passo de validação ser concluído."""
        raise NotImplementedError

    @abstractmethod
    def on_epoch_end(
        self,
        trainer: "Trainer",
        epoch: int,
        train_metrics: Dict[str, Any],
        val_metrics: Optional[Dict[str, Any]],
    ) -> None:
        """Chamado ao final de cada época de treinamento.

        Args:
            trainer: O :class:`Trainer` associado.
            epoch: Número da época concluída (1-based).
            train_metrics: Dicionário com métricas médias da época de treino
                (ex.: ``{"loss": ..., "terms": {...}}``).
            val_metrics: Dicionário com métricas médias da época de validação,
                ou ``None`` se a validação não foi executada nesta época.
        """
        raise NotImplementedError


class Trainer:
    """Orquestra um passo de otimização de treinamento.

    Para cada batch, o :class:`Trainer`:

    1. move os tensores do batch para o device configurado;
    2. executa o *forward* delegando ao :class:`TrainingPipeline`;
    3. calcula a loss delegando ao ``loss_composer``;
    4. executa ``backward()``, ``optimizer.step()`` e ``scheduler.step()``;
    5. notifica os callbacks.

    O :class:`Trainer` **não implementa redes** e **não conhece detalhes
    internos** de ``MarkerNet`` ou da rede final — apenas consome o pipeline
    já configurado.

    Uso típico::

        trainer = Trainer(
            training_pipeline=training_pipeline,
            loss_composer=loss_composer,
            optimizer=optimizer,
            scheduler=scheduler,
            callbacks=[EarlyStoppingCallback(), CheckpointCallback()],
        )

        loss, loss_log = trainer.train_step(batch)

    Args:
        training_pipeline: Pipeline de treinamento já configurado com as redes.
        loss_composer: Compositor de perdas (``LossComposer`` ou objeto com
            ``forward(prediction, distance_maps, gt_masks)``).
        optimizer: Otimizador PyTorch.
        scheduler: Scheduler PyTorch opcional (executado a cada passo de treino).
        callbacks: Lista opcional de :class:`TrainerCallback`.
        device: Device usado para mover os tensores do batch. Se ``None``, usa
            GPU quando disponível.
        prediction_key: Chave do dicionário com a predição usada na loss.
            Padrão ``"segmentation"`` (adicionada pelo ``FrozenSegmentationStep``).
        distance_map_key: Chave com o mapa de distância. Padrão ``"distance_map"``.
        ground_truth_key: Chave com o ground truth. Padrão ``"ground_truth"``.
    """

    def __init__(
        self,
        training_pipeline: TrainingPipeline,
        loss_composer: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        device: Optional[str] = None,
        prediction_key: str = "segmentation",
        distance_map_key: str = "distance_map",
        ground_truth_key: str = "ground_truth",
    ) -> None:
        self.training_pipeline = training_pipeline
        self.loss_composer = loss_composer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.callbacks = list(callbacks or [])

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.prediction_key = prediction_key
        self.distance_map_key = distance_map_key
        self.ground_truth_key = ground_truth_key

        if isinstance(self.loss_composer, nn.Module):
            self.loss_composer.to(self.device)

        logger.info(f"[Trainer] Initialized on {self.device}")

    def train_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Executa um passo completo de treinamento para um batch.

        Orquestra forward (via :class:`TrainingPipeline`), cálculo da loss (via
        ``loss_composer``), ``backward()``, ``optimizer.step()``,
        ``scheduler.step()`` e notificação dos callbacks.

        Args:
            batch: Dicionário com o batch. Deve conter as chaves de mapa de
                distância e ground truth configuradas no construtor. O pipeline
                produz a predição (``prediction_key``).

        Returns:
            Tupla ``(loss, loss_log)`` em que ``loss_log`` mapeia o nome de cada
            termo de perda ao seu valor escalar.

        Raises:
            KeyError: Se alguma chave esperada pelo cálculo da loss estiver ausente.
        """
        data = self._to_device(batch)

        self.optimizer.zero_grad()
        data = self.training_pipeline.run(data, verbose=False)

        loss, loss_log = self._compute_loss(data)
        loss.backward()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()

        for callback in self.callbacks:
            callback.on_train_step_end(self, data, loss, loss_log)

        return loss, loss_log

    @torch.no_grad()
    def eval_step(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Executa um passo de validação/inferência sem otimização.

        Não chama ``backward()`` nem ``optimizer.step()``. Útil para avaliação
        no loop de validação durante o treinamento.

        Args:
            batch: Dicionário com o batch.

        Returns:
            Tupla ``(loss, loss_log)`` em que ``loss_log`` mapeia o nome de cada
            termo de perda ao seu valor escalar.

        Raises:
            KeyError: Se alguma chave esperada pelo cálculo da loss estiver ausente.
        """
        data = self._to_device(batch)
        data = self.training_pipeline.run(data, verbose=False)

        loss, loss_log = self._compute_loss(data)

        for callback in self.callbacks:
            callback.on_validation_step_end(self, data, loss, loss_log)

        return loss, loss_log

    def _compute_loss(self, data: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Delega o cálculo da loss ao ``loss_composer``.

        Extrai do dicionário de dados as chaves de predição, mapa de distância e
        ground truth configuradas no construtor e repassa ao compositor.

        Args:
            data: Dicionário produzido pelo pipeline após o forward.

        Returns:
            Tupla ``(loss, loss_log)``.

        Raises:
            KeyError: Se alguma chave esperada estiver ausente em ``data``.
        """
        prediction = data[self.prediction_key]
        distance_maps = data[self.distance_map_key]
        gt_masks = data[self.ground_truth_key]
        return self.loss_composer(prediction, distance_maps, gt_masks)

    def _to_device(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Move tensores do dicionário para o device configurado.

        Valores que não são tensores são mantidos inalterados.

        Args:
            data: Dicionário com o batch.

        Returns:
            Novo dicionário com os tensores movidos para ``self.device``.
        """
        return {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for key, value in data.items()
        }
