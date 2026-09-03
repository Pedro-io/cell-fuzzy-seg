from src.training.trainer import TrainerCallback
from src.training.training_loop import TrainingLoop


class Scalar:
    """Mínimo suficiente para simular um tensor de perda (``loss.item()``)."""

    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeLossLog:
    """Simula o ``loss_log`` retornado pelo Trainer: dict de tensores escalares."""

    def __init__(self, values):
        self._values = values

    def items(self):
        return {k: Scalar(v) for k, v in self._values.items()}.items()


class FakeTrainer:
    """Trainer dummy que registra as chamadas e retorna losses controladas."""

    def __init__(self, train_loss=1.0, val_loss=2.0, terms=None, callbacks=None):
        self.train_calls = 0
        self.val_calls = 0
        self.train_loss = train_loss
        self.val_loss = val_loss
        self.terms = terms or {"mse": 0.5}
        self.callbacks = list(callbacks or [])

    def train_step(self, batch):
        self.train_calls += 1
        return Scalar(self.train_loss), FakeLossLog(self.terms)

    def eval_step(self, batch):
        self.val_calls += 1
        return Scalar(self.val_loss), FakeLossLog(self.terms)


class EpochCallback(TrainerCallback):
    """Callback que registra as épocas e métricas recebidas."""

    def __init__(self):
        self.epochs = []
        self.metrics = []

    def on_train_step_end(self, trainer, data, loss, loss_log):
        pass

    def on_validation_step_end(self, trainer, data, loss, loss_log):
        pass

    def on_epoch_end(self, trainer, epoch, train_metrics, val_metrics):
        self.epochs.append(epoch)
        self.metrics.append((train_metrics, val_metrics))


def make_loader(n_batches=2):
    """Retorna um loader fake com ``n_batches`` batches."""
    return [{"batch": i} for i in range(n_batches)]


def test_run_iterates_all_epochs_and_batches():
    trainer = FakeTrainer()
    loop = TrainingLoop(trainer=trainer, train_loader=make_loader(), num_epochs=3)

    history = loop.run()

    assert len(history["train_loss"]) == 3
    assert trainer.train_calls == 3 * 2  # 3 épocas × 2 batches
    assert all(loss == 1.0 for loss in history["train_loss"])


def test_run_populates_train_terms_history():
    trainer = FakeTrainer(terms={"mse": 0.5})
    loop = TrainingLoop(trainer=trainer, train_loader=make_loader(), num_epochs=2)

    history = loop.run()

    assert history["train_terms"] == {"mse": [0.5, 0.5]}


def test_validation_runs_by_default_with_val_loader():
    trainer = FakeTrainer()
    loop = TrainingLoop(
        trainer=trainer,
        train_loader=make_loader(),
        val_loader=make_loader(),
        num_epochs=2,
    )

    history = loop.run()

    assert history["val_loss"] == [2.0, 2.0]
    assert trainer.val_calls == 2 * 2  # 2 épocas × 2 batches de validação


def test_validation_respects_validate_every():
    trainer = FakeTrainer()
    loop = TrainingLoop(
        trainer=trainer,
        train_loader=make_loader(),
        val_loader=make_loader(),
        num_epochs=4,
        validate_every=2,
    )

    history = loop.run()

    assert len(history["val_loss"]) == 2  # épocas 2 e 4 apenas
    assert trainer.val_calls == 2 * 2  # 2 validações × 2 batches


def test_no_validation_without_val_loader():
    trainer = FakeTrainer()
    loop = TrainingLoop(trainer=trainer, train_loader=make_loader(), num_epochs=2)

    history = loop.run()

    assert history["val_loss"] == []
    assert trainer.val_calls == 0


def test_epoch_callbacks_are_notified():
    callback = EpochCallback()
    trainer = FakeTrainer(callbacks=[callback])
    loop = TrainingLoop(
        trainer=trainer,
        train_loader=make_loader(),
        val_loader=make_loader(),
        num_epochs=2,
    )

    loop.run()

    assert callback.epochs == [1, 2]
    train_metrics, val_metrics = callback.metrics[0]
    assert train_metrics["loss"] == 1.0
    assert val_metrics is not None
    assert val_metrics["loss"] == 2.0


def test_epoch_callback_receives_none_when_no_validation():
    callback = EpochCallback()
    trainer = FakeTrainer(callbacks=[callback])
    loop = TrainingLoop(trainer=trainer, train_loader=make_loader(), num_epochs=1)

    loop.run()

    _, val_metrics = callback.metrics[0]
    assert val_metrics is None


def test_empty_loader_yields_zero_loss():
    trainer = FakeTrainer()
    loop = TrainingLoop(trainer=trainer, train_loader=[], num_epochs=1)

    history = loop.run()

    assert history["train_loss"] == [0.0]


def test_non_positive_validate_every_raises():
    trainer = FakeTrainer()

    try:
        TrainingLoop(trainer=trainer, train_loader=[], num_epochs=1, validate_every=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for validate_every <= 0")


def test_non_positive_log_every_raises():
    trainer = FakeTrainer()

    try:
        TrainingLoop(trainer=trainer, train_loader=[], num_epochs=1, log_every=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for log_every <= 0")
