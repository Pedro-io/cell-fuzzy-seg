"""Testes do BestModelCallback (melhor Dice de validação + parada antecipada) e da parada no TrainingLoop."""

import pytest
import torch
import torch.nn as nn

from src.training.callbacks import BestModelCallback
from src.training.training_loop import TrainingLoop
from tests.test_training_integration import make_batches, make_pipeline


class Scalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class ScheduledTrainer:
    """Trainer de mentira: na época e, o "peso" do modelo vira e e a validação acerta ``hits[e-1]`` pixels."""

    def __init__(self, model, hits, callbacks):
        self.model = model
        self.hits = hits
        self.callbacks = callbacks
        self.stop_training = False
        self.epoch = 0

    def train_step(self, batch):
        self.epoch += 1
        with torch.no_grad():
            self.model.weight.fill_(float(self.epoch))
        return Scalar(1.0), {}

    def eval_step(self, batch):
        # GT com 10 px de núcleo; a predição acerta p deles e não marca fundo: Dice = 2p / (p + 10).
        p = self.hits[self.epoch - 1]
        gt = torch.ones(1, 1, 1, 10)
        pred = torch.zeros(1, 1, 1, 10)
        pred[..., :p] = 0.9
        data = {"segmentation": pred, "ground_truth": gt}
        for callback in self.callbacks:
            callback.on_validation_step_end(self, data, Scalar(1.0), {})
        return Scalar(1.0), {}


def run(hits, num_epochs=None, validate_every=1, **kwargs):
    model = nn.Linear(1, 1, bias=False)
    callback = BestModelCallback(model, **kwargs)
    trainer = ScheduledTrainer(model, hits, [callback])
    loop = TrainingLoop(trainer, train_loader=[{}], val_loader=[{}],
                        num_epochs=num_epochs or len(hits), validate_every=validate_every)
    return model, callback, loop.run()


def dice(p):
    return 2 * p / (p + 10)


def test_tracks_best_epoch_and_restores_its_weights():
    model, cb, history = run([3, 8, 6, 7])

    assert cb.best_epoch == 2
    assert cb.best_dice == pytest.approx(dice(8))
    assert [h["epoch"] for h in cb.history] == [1, 2, 3, 4]
    assert history["epochs_run"] == 4                 # sem paciência: roda todas as épocas
    assert model.weight.item() == 4.0
    cb.restore_best()
    assert model.weight.item() == 2.0                 # pesos da época 2


def test_stops_after_patience_validations_without_improvement():
    _, cb, history = run([3, 8, 6, 7, 7, 9, 9], patience=3)

    assert history["epochs_run"] == 5                 # melhor na 2; sem melhora em 3, 4 e 5
    assert cb.best_epoch == 2


def test_min_delta_ignores_tiny_improvements():
    # Dice 0,667 → 0,750 → 0,824. Com min_delta 0,1, a época 2 (+0,083 sobre o melhor) não conta; a 3
    # (+0,157 sobre o melhor, que ainda é o da época 1) conta.
    _, cb, _ = run([5, 6, 7], min_delta=0.1)

    assert cb.best_epoch == 3
    assert cb.best_dice == pytest.approx(dice(7))


def test_epochs_without_validation_are_not_counted():
    _, cb, history = run([3, 8, 6, 7, 5, 4], validate_every=2, patience=2)

    assert [h["epoch"] for h in cb.history] == [2, 4, 6]
    assert history["epochs_run"] == 6                 # só 2 validações sem melhora (4 e 6)
    assert cb.best_epoch == 2


def test_restore_without_validation_and_invalid_patience_raise():
    cb = BestModelCallback(nn.Linear(1, 1))
    with pytest.raises(RuntimeError, match="validação"):
        cb.restore_best()
    with pytest.raises(ValueError, match="patience"):
        BestModelCallback(nn.Linear(1, 1), patience=0)


def test_training_loop_resets_the_stop_flag_at_start():
    model = nn.Linear(1, 1, bias=False)
    trainer = ScheduledTrainer(model, [1, 2], callbacks=[])
    trainer.stop_training = True

    history = TrainingLoop(trainer, train_loader=[{}], num_epochs=2).run()

    assert history["epochs_run"] == 2


def test_works_with_the_real_trainer():
    from src.losses.loss_composer import LossComposer
    from src.losses.terms import DiceTerm
    from src.training.trainer import Trainer

    marker_net, _, pipeline = make_pipeline()
    callback = BestModelCallback(marker_net, patience=5)
    trainer = Trainer(pipeline, LossComposer([DiceTerm()]),
                      torch.optim.Adam(marker_net.parameters(), lr=1e-3),
                      callbacks=[callback], device="cpu")
    TrainingLoop(trainer, make_batches(2, 2), make_batches(1, 2, seed=1), num_epochs=3).run()

    assert len(callback.history) == 3
    assert 0.0 <= callback.best_dice <= 1.0
    callback.restore_best()
    for name, value in marker_net.state_dict().items():
        assert torch.equal(value.cpu(), callback.best_state[name])
