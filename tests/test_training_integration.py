"""Teste de integração do fluxo de treinamento corrigido (investigação do experimento 1).

Valida, em escala reduzida, a cadeia completa que o notebook ``experiment_1.ipynb``
executa após as correções:

- ``MarkerStep`` diferenciável mantém a MarkerNet em ``train()`` (P3);
- o gradiente flui até a MarkerNet através da rede final congelada (P1);
- o ``LossComposer`` recebe ``prediction`` (segmentação) e ``markers`` separados,
  permitindo que o ``DMapTerm`` supervisione os marcadores diretamente (C3/C4);
- o ``Trainer`` aplica grad clipping (P5) e o ``GradNormCallback`` registra a
  magnitude dos gradientes (E1).
"""

import numpy as np
import torch
import torch.nn as nn

from src.losses.loss_composer import LossComposer
from src.losses.terms import DiceTerm, DMapTerm, RMSETerm, SizeTerm
from src.models.networks.final_segmentation.base_final_segmentation import BaseFinalSegmentation
from src.pipeline.steps.inference.frozen_segmentation_step import FrozenSegmentationStep
from src.pipeline.steps.inference.marker_step import MarkerStep
from src.pipeline.training_pipeline import TrainingPipeline
from src.training.callbacks import GradNormCallback
from src.training.trainer import Trainer
from src.training.training_loop import TrainingLoop


class SmallMarkerNet(nn.Module):
    """MarkerNet mínima com a mesma interface esperada pelo MarkerStep (attr ``model``)."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


class DummyFinalNetwork(BaseFinalSegmentation):
    """Rede final congelada: conv sobre [imagem, scribbles]."""

    def __init__(self):
        super().__init__(config={})
        self.conv = nn.Sequential(
            nn.Conv2d(4, 8, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),
        )
        self.conv.requires_grad_(False)

    def forward(self, data):
        img = data["image"]
        scribbles = data["scribbles"]
        img = img.float()
        if img.ndim == 3:
            img = img[None]
        if img.max() > 1.0:
            img = img / 255.0
        s = scribbles.float()
        if s.ndim == 3:
            s = s[None]
        x = torch.cat([img, s], dim=1)
        return torch.sigmoid(self.conv(x))

    def predict(self, data):
        with torch.no_grad():
            return self.forward(data)


def make_batches(n_batches=2, batch_size=2, size=32, seed=0):
    """Gera batches sintéticos (imagem, rgba, gt, dmap) na resolução de trabalho."""
    rng = np.random.default_rng(seed)
    batches = []
    for _ in range(n_batches):
        images, rgbs, gts, dmaps = [], [], [], []
        for _ in range(batch_size):
            img = (rng.random((size, size, 3)) * 255).astype(np.float32)
            gt = (rng.random((size, size)) > 0.7).astype(np.float32)
            rgba = np.concatenate([img / 255.0, gt[..., None]], axis=-1).astype(np.float32)
            dmap = 1.0 - gt
            images.append(torch.from_numpy(img.transpose(2, 0, 1)))
            rgbs.append(torch.from_numpy(rgba.transpose(2, 0, 1)))
            gts.append(torch.from_numpy(gt)[None])
            dmaps.append(torch.from_numpy(dmap)[None])
        batches.append(
            {
                "id": [f"img-{seed}-{i}" for i in range(batch_size)],
                "image": torch.stack(images),
                "rgba": torch.stack(rgbs),
                "ground_truth": torch.stack(gts),
                "distance_map": torch.stack(dmaps),
            }
        )
    return batches


def make_pipeline():
    marker_net = SmallMarkerNet()
    final_net = DummyFinalNetwork()
    marker_step = MarkerStep(model=marker_net, differentiable=True, device="cpu")
    final_step = FrozenSegmentationStep(final_network=final_net, device="cpu")
    pipeline = TrainingPipeline([marker_step, final_step])
    return marker_net, final_net, pipeline


def test_marker_net_is_in_train_mode_during_training():
    """P3: com differentiable=True, a MarkerNet deve estar em train()."""
    marker_net, _, pipeline = make_pipeline()
    step = pipeline.steps[0]
    assert step.model.model.training is True


def test_marker_net_in_eval_mode_for_inference():
    """P3: em modo inferência (differentiable=False), a MarkerNet fica em eval()."""
    marker_net = SmallMarkerNet()
    step = MarkerStep(model=marker_net, differentiable=False, device="cpu")
    assert step.model.model.training is False


def test_gradient_flows_to_marker_net_through_frozen_final_network():
    """P1: o gradiente da loss deve chegar à MarkerNet com magnitude não-nula."""
    marker_net, final_net, pipeline = make_pipeline()
    composer = LossComposer([DiceTerm(), RMSETerm(), SizeTerm(weight=0.05)])
    trainer = Trainer(
        training_pipeline=pipeline,
        loss_composer=composer,
        optimizer=torch.optim.Adam(marker_net.parameters(), lr=1e-3),
        device="cpu",
    )

    loss, loss_log = trainer.train_step(make_batches()[0])

    marker_grads = [p.grad for p in marker_net.parameters() if p.grad is not None]
    final_grads = [p for p in final_net.parameters() if p.grad is not None]

    assert len(marker_grads) > 0, "Nenhum gradiente chegou à MarkerNet!"
    assert len(final_grads) == 0, "A rede final não deve acumular gradiente (congelada)!"
    total_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in marker_grads))
    assert total_norm.item() > 0.0, "Gradiente morto: grad-norm é zero (P1)!"
    assert set(loss_log) == {"dice", "rmse", "size"}


def test_dmap_term_supervises_markers_directly():
    """C3/C4: o DMapTerm deve reagir aos MARKERS, não à segmentação.

    Mantendo a segmentação (prediction) fixa e mudando apenas os markers, a
    perda do termo dmap deve mudar.
    """
    marker_net, _, pipeline = make_pipeline()
    composer = LossComposer([DMapTerm(weight=0.1)])

    batch = make_batches()[0]
    data = pipeline.run(batch, verbose=False)
    prediction = data["segmentation"]
    distance_maps = data["distance_map"]
    gt = data["ground_truth"]

    _, log1 = composer(prediction, distance_maps, gt, markers=data["markers"])
    _, log2 = composer(prediction, distance_maps, gt, markers=torch.zeros_like(data["markers"]))

    assert log1["dmap"].item() != log2["dmap"].item()
    # Com markers zerados, o termo dmap é zero (nada a penalizar).
    assert log2["dmap"].item() == 0.0


def test_full_training_loop_with_corrections():
    """Treina 2 épocas com grad clip + scheduler + GradNormCallback (P5/E1)."""
    marker_net, final_net, pipeline = make_pipeline()
    composer = LossComposer([DiceTerm(), RMSETerm(), SizeTerm(weight=0.05), DMapTerm(weight=0.1)])

    train_batches = make_batches(n_batches=3, batch_size=2)
    val_batches = make_batches(n_batches=2, batch_size=2, seed=1)

    optimizer = torch.optim.Adam(marker_net.parameters(), lr=1e-3)
    total_steps = 2 * len(train_batches)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    grad_norm_cb = GradNormCallback(log_every=1)

    trainer = Trainer(
        training_pipeline=pipeline,
        loss_composer=composer,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_clip=1.0,
        callbacks=[grad_norm_cb],
        device="cpu",
    )
    loop = TrainingLoop(
        trainer=trainer,
        train_loader=train_batches,
        val_loader=val_batches,
        num_epochs=2,
        validate_every=1,
        log_every=1,
    )

    history = loop.run()

    assert len(history["train_loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert all(torch.isfinite(torch.tensor(x)) for x in history["train_loss"])
    assert len(grad_norm_cb.history["total"]) == 2 * len(train_batches)
    assert all(n > 0.0 for n in grad_norm_cb.history["total"]), "Gradiente morto detectado (P1)!"
    assert "dmap" in history["train_terms"]
