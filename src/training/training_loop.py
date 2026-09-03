"""Laço de treinamento: orquestra épocas, batches e validação periódica.

Este módulo fornece :class:`TrainingLoop`, o laço de mais alto nível do
treinamento. Ele itera sobre os ``DataLoader`` de treino e validação por
múltiplas épocas, invocando :meth:`Trainer.train_step` para cada batch de treino
e :meth:`Trainer.eval_step` para cada batch de validação, acumulando métricas e
acionando os callbacks do :class:`Trainer` ao final de cada época.
"""

from typing import Any, Dict, List, Optional, Tuple

from src.training.trainer import Trainer
from src.utils.logger import logger


class TrainingLoop:
    """Organiza o laço de mais alto nível do treinamento.

    Responsabilidades:
    - Iterar sobre o ``DataLoader`` de treino por múltiplas épocas;
    - Invocar :meth:`Trainer.train_step` para cada batch de treino;
    - Executar validação (:meth:`Trainer.eval_step`) nos intervalos configurados;
    - Acionar os callbacks do :class:`Trainer` (via ``on_epoch_end``) ao final
      de cada época;
    - Acumular e retornar um histórico com as métricas médias por época.

    **Não implementa** a lógica de um único passo de otimização (isso é
    responsabilidade do :class:`Trainer`) e **não conhece** detalhes de
    arquitetura de rede. O scheduler é gerenciado pelo :class:`Trainer` (a cada
    passo de treino, conforme a arquitetura) — este laço nunca o aciona.

    Uso típico::

        training_loop = TrainingLoop(
            trainer=trainer,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=100,
        )

        history = training_loop.run()

    Args:
        trainer: Instância de :class:`Trainer` já configurada.
        train_loader: Iterável com os batches de treino (``DataLoader``).
        val_loader: Iterável com os batches de validação (``DataLoader``).
            Opcional — se ``None``, a validação é ignorada.
        num_epochs: Número de épocas a executar. Padrão: 1.
        validate_every: Intervalo (em épocas) entre execuções de validação.
            Padrão: 1 (valida em todas as épocas, se ``val_loader`` existir).
        log_every: Intervalo (em épocas) entre logs de progresso. Padrão: 1.

    Raises:
        ValueError: Se ``validate_every`` ou ``log_every`` não forem positivos.
    """

    def __init__(
        self,
        trainer: Trainer,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        num_epochs: int = 1,
        validate_every: int = 1,
        log_every: int = 1,
    ) -> None:
        self.trainer = trainer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.validate_every = validate_every
        self.log_every = log_every

        if self.validate_every <= 0:
            raise ValueError(f"validate_every must be positive, got {self.validate_every}")
        if self.log_every <= 0:
            raise ValueError(f"log_every must be positive, got {self.log_every}")

    def run(self) -> Dict[str, Any]:
        """Executa o treinamento completo por ``num_epochs`` épocas.

        Returns:
            Dicionário com o histórico de treinamento:
                - ``"train_loss"``: lista com a loss média por época de treino;
                - ``"val_loss"``: lista com a loss média por época de validação
                  executada;
                - ``"train_terms"``: dict ``{termo: lista de médias por época}``
                  para os termos de perda do treino;
                - ``"val_terms"``: dict ``{termo: lista de médias por época}``
                  para os termos de perda da validação executada.

            Observação: quando ``validate_every > 1``, as listas ``"val_loss"`` e
            ``"val_terms"`` possuem menos entradas do que ``"train_loss"`` — elas
            registram apenas as épocas em que a validação foi executada.
        """
        history: Dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "train_terms": {},
            "val_terms": {},
        }

        for epoch in range(1, self.num_epochs + 1):
            train_loss, train_terms = self._run_epoch(self.train_loader, training=True)
            history["train_loss"].append(train_loss)
            self._accumulate_terms(history["train_terms"], train_terms)

            val_metrics: Optional[Dict[str, Any]] = None
            should_validate = (
                self.val_loader is not None and epoch % self.validate_every == 0
            )
            if should_validate:
                val_loss, val_terms = self._run_epoch(self.val_loader, training=False)
                history["val_loss"].append(val_loss)
                self._accumulate_terms(history["val_terms"], val_terms)
                val_metrics = {"loss": val_loss, "terms": val_terms}

            train_metrics: Dict[str, Any] = {"loss": train_loss, "terms": train_terms}
            self._notify_epoch_end(epoch, train_metrics, val_metrics)

            if epoch % self.log_every == 0:
                self._log_epoch(epoch, train_metrics, val_metrics)

        return history

    def _run_epoch(
        self, loader: Any, training: bool
    ) -> Tuple[float, Dict[str, float]]:
        """Executa uma época completa sobre um loader.

        Args:
            loader: Iterável de batches.
            training: Se ``True``, usa :meth:`Trainer.train_step`; caso
                contrário, usa :meth:`Trainer.eval_step`.

        Returns:
            Tupla ``(loss_media, termos_medios)`` em que ``termos_medios`` mapeia
            o nome de cada termo de perda ao seu valor médio na época.
        """
        total_loss = 0.0
        terms_accum: Dict[str, float] = {}
        n_batches = 0

        for batch in loader:
            if training:
                loss, loss_log = self.trainer.train_step(batch)
            else:
                loss, loss_log = self.trainer.eval_step(batch)

            total_loss += float(loss.item())
            for name, value in loss_log.items():
                terms_accum[name] = terms_accum.get(name, 0.0) + float(value.item())
            n_batches += 1

        if n_batches == 0:
            logger.warning("[TrainingLoop] Loader vazio: nenhum batch processado.")
            return 0.0, {}

        avg_loss = total_loss / n_batches
        avg_terms = {name: total / n_batches for name, total in terms_accum.items()}
        return avg_loss, avg_terms

    def _accumulate_terms(self, history_terms: Dict[str, List[float]], terms: Dict[str, float]) -> None:
        """Acumula os termos médios de uma época no histórico.

        Args:
            history_terms: Dicionário do histórico que acumula listas por termo.
            terms: Termos médios da época corrente.
        """
        for name, value in terms.items():
            history_terms.setdefault(name, []).append(value)

    def _notify_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, Any],
        val_metrics: Optional[Dict[str, Any]],
    ) -> None:
        """Aciona ``on_epoch_end`` em todos os callbacks do Trainer.

        Args:
            epoch: Número da época concluída.
            train_metrics: Métricas médias da época de treino.
            val_metrics: Métricas médias da época de validação ou ``None``.
        """
        for callback in self.trainer.callbacks:
            callback.on_epoch_end(self.trainer, epoch, train_metrics, val_metrics)

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, Any],
        val_metrics: Optional[Dict[str, Any]],
    ) -> None:
        """Registra o progresso da época no logger.

        Args:
            epoch: Número da época concluída.
            train_metrics: Métricas médias da época de treino.
            val_metrics: Métricas médias da época de validação ou ``None``.
        """
        train_loss = train_metrics["loss"]
        message = f"[TrainingLoop] Epoch {epoch}/{self.num_epochs} | train_loss={train_loss:.4f}"

        if val_metrics is not None:
            message += f" | val_loss={val_metrics['loss']:.4f}"

        logger.info(message)
