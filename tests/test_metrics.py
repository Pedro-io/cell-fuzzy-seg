"""Testes das métricas binárias por imagem e da avaliação de um pipeline."""

import numpy as np
import pytest
import torch

from src.evaluation.metrics import binary_metrics, evaluate


def test_binary_metrics_hand_computed_case():
    # GT: bloco 2×2 (4 px). Predição acerta 3 deles e marca 1 px de fundo: tp=3, fp=1, fn=1.
    gt = np.zeros((4, 4), dtype=np.uint8)
    gt[0:2, 0:2] = 1
    pred = np.zeros((4, 4), dtype=np.float32)
    pred[0, 0] = pred[0, 1] = pred[1, 0] = 0.9
    pred[3, 3] = 0.7
    pred[1, 1] = 0.2  # abaixo do limiar: falso negativo

    m = binary_metrics(pred, gt)

    assert m["dice"] == pytest.approx(6 / 8)
    assert m["iou"] == pytest.approx(3 / 5)
    assert m["precision"] == pytest.approx(3 / 4)
    assert m["recall"] == pytest.approx(3 / 4)
    assert m["mass_ratio"] == pytest.approx(4 / 4)


def test_threshold_is_inclusive():
    gt = np.array([[1, 0]])
    pred = np.array([[0.5, 0.49]])

    m = binary_metrics(pred, gt, threshold=0.5)

    assert m["dice"] == 1.0


def test_ground_truth_uses_nonzero_as_foreground():
    # Rótulos de instância também valem como primeiro plano.
    gt = np.array([[0, 7, 3]])
    pred = np.array([[0.0, 1.0, 1.0]])

    assert binary_metrics(pred, gt)["dice"] == 1.0


@pytest.mark.parametrize(
    "pred, gt, expected",
    [
        # ambos vazios
        (np.zeros((2, 2)), np.zeros((2, 2)),
         {"dice": 1.0, "iou": 1.0, "precision": 1.0, "recall": 1.0, "mass_ratio": 0.0}),
        # nada previsto, GT com 2 px
        (np.zeros((2, 2)), np.array([[1, 1], [0, 0]]),
         {"dice": 0.0, "iou": 0.0, "precision": 1.0, "recall": 0.0, "mass_ratio": 0.0}),
        # 3 px previstos, GT vazio
        (np.array([[1.0, 1.0], [1.0, 0.0]]), np.zeros((2, 2)),
         {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 1.0, "mass_ratio": 3.0}),
    ],
)
def test_empty_cases(pred, gt, expected):
    assert binary_metrics(pred, gt) == pytest.approx(expected)


def test_accepts_tensors_and_arrays_alike():
    gt = np.array([[1, 0], [1, 1]])
    pred = np.array([[0.8, 0.6], [0.1, 0.9]], dtype=np.float32)

    assert binary_metrics(torch.from_numpy(pred), torch.from_numpy(gt)) == binary_metrics(pred, gt)


def test_rejects_different_shapes():
    with pytest.raises(ValueError, match="Formatos"):
        binary_metrics(np.zeros((2, 2)), np.zeros((3, 3)))


class RecordingStep:
    """Step de mentira: grava os modos pedidos e se havia gradiente ao rodar."""

    name = "RecordingStep"

    def __init__(self, segmentation):
        self.segmentation = segmentation
        self.modes = []
        self.grad_enabled = []

    def set_training(self, training):
        self.modes.append(training)

    def __call__(self, data):
        self.grad_enabled.append(torch.is_grad_enabled())
        n = data["ground_truth"].shape[0]
        data["markers"] = torch.ones(n, 1, 2, 2) * 0.8
        data["segmentation"] = self.segmentation[:n]
        return data


class FakePipeline:
    def __init__(self, steps):
        self.steps = steps

    def run(self, data, verbose=False):
        for step in self.steps:
            data = step(data)
        return data


def _batch(ids):
    n = len(ids)
    gt = torch.tensor([[1.0, 1.0], [0.0, 0.0]]).expand(n, 1, 2, 2).clone()
    cellpose = torch.tensor([[3, 0], [0, 0]], dtype=torch.int32).expand(n, 1, 2, 2).clone()
    return {
        "id": list(ids),
        "ground_truth": gt,
        "cellpose_segmentation": cellpose,
        "segmentation": torch.full((n, 1, 2, 2), -1.0),  # sentinela: não pode ser alterada
    }


def test_evaluate_rows_metrics_baseline_and_side_effects():
    prediction = torch.tensor([[0.9, 0.9], [0.9, 0.1]]).expand(2, 1, 2, 2).clone()
    step = RecordingStep(prediction)
    pipeline = FakePipeline([step])
    batches = [_batch(["a", "b"]), _batch(["c"])]

    rows = evaluate(pipeline, batches)

    assert [r["id"] for r in rows] == ["a", "b", "c"]
    # predição: tp=2, fp=1, fn=0 → Dice 4/5; Cellpose: 1 de 2 px → Dice 2/3
    assert rows[0]["dice"] == pytest.approx(4 / 5)
    assert rows[0]["cellpose_dice"] == pytest.approx(2 / 3)
    assert rows[0]["marker_fraction"] == 1.0
    assert step.modes == [False]                 # avaliação em modo eval
    assert step.grad_enabled == [False, False]  # sem gradiente
    assert torch.equal(batches[0]["segmentation"], torch.full((2, 1, 2, 2), -1.0))  # batch intacto


def test_evaluate_without_baseline():
    step = RecordingStep(torch.ones(1, 1, 2, 2))
    batch = _batch(["a"])
    del batch["cellpose_segmentation"]

    rows = evaluate(FakePipeline([step]), [batch])

    assert not any(k.startswith("cellpose_") for k in rows[0])
