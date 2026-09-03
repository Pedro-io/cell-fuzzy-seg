import torch
import torch.nn as nn

from src.pipeline.training_pipeline import TrainingPipeline
from src.training.trainer import Trainer, TrainerCallback


class DummyNetwork(nn.Module):
    """Rede dummy diferenciável usada nos testes."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class IdentityStep:
    """Step dummy que simula o output de um pipeline treinável."""

    name = "IdentityStep"

    def __init__(self, net):
        self.net = net

    def __call__(self, data):
        return self.forward(data)

    def forward(self, data):
        data["markers"] = self.net(data["image"])
        data["segmentation"] = data["markers"]
        return data


class DummyLossComposer(nn.Module):
    """Compositor de perda dummy: MSE entre predição e ground truth."""

    def forward(self, prediction, distance_maps, gt_masks, markers=None):
        loss = ((prediction - gt_masks) ** 2).mean()
        return loss, {"mse": loss}


class RecordingCallback(TrainerCallback):
    """Callback que registra as chamadas recebidas."""

    def __init__(self):
        self.train_ends = 0
        self.val_ends = 0
        self.epoch_ends = 0

    def on_train_step_end(self, trainer, data, loss, loss_log):
        self.train_ends += 1

    def on_validation_step_end(self, trainer, data, loss, loss_log):
        self.val_ends += 1

    def on_epoch_end(self, trainer, epoch, train_metrics, val_metrics):
        self.epoch_ends += 1


def make_batch(device="cpu"):
    return {
        "image": torch.randn(2, 1, 4, 4).to(device),
        "distance_map": torch.zeros(2, 1, 4, 4).to(device),
        "ground_truth": torch.ones(2, 1, 4, 4).to(device),
    }


def make_trainer(net=None, scheduler=None, callbacks=None, optimizer=None):
    net = net or DummyNetwork()
    pipeline = TrainingPipeline([IdentityStep(net)])
    composer = DummyLossComposer()
    optimizer = optimizer or torch.optim.SGD(net.parameters(), lr=0.1)
    return Trainer(
        training_pipeline=pipeline,
        loss_composer=composer,
        optimizer=optimizer,
        scheduler=scheduler,
        callbacks=callbacks,
        device="cpu",  # testes determinísticos independentes da GPU disponível
    )


def test_train_step_returns_loss_and_log():
    trainer = make_trainer()

    loss, loss_log = trainer.train_step(make_batch())

    assert loss.ndim == 0
    assert "mse" in loss_log
    assert loss_log["mse"].ndim == 0


def test_train_step_optimizes_parameters():
    net = DummyNetwork()
    trainer = make_trainer(net=net)
    weight_before = net.conv.weight.clone()

    trainer.train_step(make_batch())

    assert not torch.equal(net.conv.weight, weight_before)


def test_eval_step_does_not_update_parameters():
    net = DummyNetwork()
    trainer = make_trainer(net=net)
    weight_before = net.conv.weight.clone()

    loss, loss_log = trainer.eval_step(make_batch())

    assert torch.equal(net.conv.weight, weight_before)
    assert loss.ndim == 0
    assert "mse" in loss_log


def test_train_step_requires_grad_flow():
    net = DummyNetwork()
    trainer = make_trainer(net=net)

    trainer.train_step(make_batch())

    assert net.conv.weight.grad is not None


def test_callbacks_are_notified():
    callback = RecordingCallback()
    trainer = make_trainer(callbacks=[callback])

    trainer.train_step(make_batch())
    trainer.eval_step(make_batch())

    assert callback.train_ends == 1
    assert callback.val_ends == 1


def test_scheduler_is_called():
    net = DummyNetwork()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    # O scheduler deve usar o MESMO otimizador do Trainer (senão o passo do
    # scheduler não tem efeito no treinamento e o torch emite warning).
    trainer = make_trainer(net=net, scheduler=scheduler, optimizer=optimizer)
    lr_before = optimizer.param_groups[0]["lr"]

    trainer.train_step(make_batch())

    assert optimizer.param_groups[0]["lr"] < lr_before


def test_missing_loss_keys_raise_key_error():
    trainer = make_trainer()
    batch = {"image": torch.randn(2, 1, 4, 4)}  # sem distance_map / ground_truth

    try:
        trainer.train_step(batch)
    except KeyError:
        pass
    else:
        raise AssertionError("Expected KeyError when loss keys are missing")


def test_non_tensor_values_are_preserved():
    trainer = make_trainer()
    batch = make_batch()
    batch["id"] = "sample-1"

    loss, _ = trainer.train_step(batch)

    assert loss.ndim == 0
