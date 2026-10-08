"""Testes da divisão em folds, da validação cruzada e do treino final."""

import numpy as np
import pytest

from src.data.load.preprocessed_dataset import load_preprocessed
from src.losses.loss_composer import LossComposer
from src.losses.terms import DiceTerm
from src.pipeline.steps.inference.frozen_segmentation_step import FrozenSegmentationStep
from src.pipeline.steps.inference.marker_step import MarkerStep
from src.pipeline.training_pipeline import TrainingPipeline
from src.training.kfold import TrainConfig, final_epochs, make_folds, run_final, run_kfold
from tests.test_training_integration import DummyFinalNetwork, SmallMarkerNet

SIZE = 16


def _write_split(root, split, ids, seed):
    rng = np.random.default_rng(seed)
    for image_id in ids:
        gt = (rng.random((SIZE, SIZE)) > 0.6).astype(np.uint8)
        image = (rng.random((SIZE, SIZE, 3)) * 255).astype(np.uint8)
        arrays = {
            "image": image,
            "rgba": np.concatenate([image / 255.0, gt[..., None]], -1).astype(np.float32),
            "ground_truth": gt,
            "distance_map": (1.0 - gt).astype(np.float32),
            "segmentation": gt.astype(np.uint16),
            "cellpose_prob": gt.astype(np.float16),
        }
        for key, array in arrays.items():
            folder = root / split / key
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / f"{image_id}.npy", array)


@pytest.fixture()
def data(tmp_path):
    train_ids = [f"tr{i}" for i in range(6)]
    _write_split(tmp_path, "train", train_ids, seed=0)
    _write_split(tmp_path, "test", ["te0", "te1"], seed=1)
    return load_preprocessed("train", root=str(tmp_path)), load_preprocessed("test", root=str(tmp_path))


class Factory:
    """Monta um experimento novo a cada chamada e conta as chamadas."""

    def __init__(self):
        self.calls = 0

    def __call__(self):
        self.calls += 1
        net = SmallMarkerNet()
        pipeline = TrainingPipeline([
            MarkerStep(model=net, differentiable=True, target_size=SIZE, device="cpu"),
            FrozenSegmentationStep(final_network=DummyFinalNetwork(), device="cpu"),
        ])
        return net, pipeline, LossComposer([DiceTerm()])


CONFIG = TrainConfig(k=3, batch_size=2, max_epochs=3, patience=None, lr=1e-3, device="cpu", log_every=100)


def test_make_folds_is_a_disjoint_cover_and_reproducible():
    ids = [f"img{i}" for i in range(37)]

    folds = make_folds(ids, k=5, seed=42)

    assert sorted(i for f in folds for i in f) == sorted(ids)
    assert sorted(len(f) for f in folds) == [7, 7, 7, 8, 8]
    assert all(len(set(a) & set(b)) == 0 for n, a in enumerate(folds) for b in folds[n + 1:])
    assert make_folds(ids, 5, 42) == folds
    assert make_folds(ids, 5, 7) != folds


def test_make_folds_validates_arguments():
    with pytest.raises(ValueError, match="k deve"):
        make_folds(["a", "b"], k=3)
    with pytest.raises(ValueError, match="k deve"):
        make_folds(["a", "b"], k=1)
    with pytest.raises(ValueError, match="repetidos"):
        make_folds(["a", "a", "b"], k=2)


def test_run_kfold_trains_one_fresh_model_per_fold_without_leakage(data):
    train, _ = data
    factory = Factory()

    folds = run_kfold(factory, train, list(train), CONFIG)

    assert factory.calls == 3
    assert sorted(i for f in folds for i in f.val_ids) == sorted(train)
    for f in folds:
        assert not set(f.train_ids) & set(f.val_ids)          # validação fora do treino
        assert sorted(f.train_ids + f.val_ids) == sorted(train)
        assert 1 <= f.best_epoch <= CONFIG.max_epochs
        assert f.epochs_run == CONFIG.max_epochs
        assert [r["id"] for r in f.rows] == f.val_ids
        assert all(r["fold"] == f.fold and "cellpose_dice" in r for r in f.rows)
        assert len(f.val_dice_history) == CONFIG.max_epochs


def test_run_kfold_is_reproducible(data):
    train, _ = data

    first = run_kfold(Factory(), train, list(train), CONFIG)
    second = run_kfold(Factory(), train, list(train), CONFIG)

    assert [r["dice"] for f in first for r in f.rows] == [r["dice"] for f in second for r in f.rows]
    assert [f.train_loss_history for f in first] == [f.train_loss_history for f in second]


def test_final_epochs_is_the_median_of_best_epochs():
    class F:
        def __init__(self, e):
            self.best_epoch = e

    assert final_epochs([F(10), F(30), F(20)]) == 20
    assert final_epochs([F(10), F(11)]) == 10  # 10,5 arredonda para o par mais próximo


def test_run_final_trains_on_all_and_evaluates_the_test_once(data):
    train, test = data

    rows, history, pipeline = run_final(Factory(), train, test, CONFIG, num_epochs=2)

    assert [r["id"] for r in rows] == ["te0", "te1"]
    assert all("cellpose_dice" in r for r in rows)
    assert history["epochs_run"] == 2
    assert history["val_loss"] == []  # sem validação no treino final
    assert pipeline.steps[0].model.model.training is False  # devolvido em modo de avaliação
