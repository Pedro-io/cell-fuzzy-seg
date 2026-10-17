"""Testes da gravação e leitura dos modelos treinados."""

import json
import pickle

import pytest
import torch
import torch.nn as nn

from src.training.checkpoints import (
    FINAL_NAME,
    checkpoint_paths,
    ensure_new_models_dir,
    fold_name,
    load_checkpoint,
    load_pipeline,
    save_checkpoint,
)


class TinyStep:
    def __init__(self):
        self.modes = []

    def set_training(self, training):
        self.modes.append(training)


class TinyPipeline:
    def __init__(self):
        self.steps = [TinyStep()]


def test_save_and_load_round_trip_with_metadata(tmp_path):
    model = nn.Linear(3, 2)
    path = save_checkpoint(model, str(tmp_path), fold_name(1), {"fold": 1, "val_ids": ["a", "b"]})

    other = nn.Linear(3, 2)
    meta = load_checkpoint(other, str(tmp_path), fold_name(1))

    assert path.endswith("fold1.pt")
    assert meta == {"fold": 1, "val_ids": ["a", "b"]}
    for name, value in model.state_dict().items():
        assert torch.equal(other.state_dict()[name], value)
    assert json.load(open(checkpoint_paths(str(tmp_path), "fold1")[1]))["fold"] == 1


def test_load_missing_raises_and_missing_json_gives_empty_metadata(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(nn.Linear(3, 2), str(tmp_path), FINAL_NAME)
    save_checkpoint(nn.Linear(3, 2), str(tmp_path), FINAL_NAME)
    (tmp_path / "final.json").unlink()
    assert load_checkpoint(nn.Linear(3, 2), str(tmp_path), FINAL_NAME) == {}


class NotATensor:
    """Objeto arbitrário: só carrega com pickle completo, não com ``weights_only=True``."""


def test_load_refuses_files_with_arbitrary_objects(tmp_path):
    torch.save({"weight": NotATensor()}, tmp_path / "final.pt")

    with pytest.raises(pickle.UnpicklingError):
        load_checkpoint(nn.Linear(3, 2), str(tmp_path), FINAL_NAME)


def test_load_pipeline_builds_loads_and_sets_eval_mode(tmp_path):
    trained = nn.Linear(3, 2)
    save_checkpoint(trained, str(tmp_path), fold_name(0), {"fold": 0})
    built = {}

    def factory():
        built["model"], built["pipeline"] = nn.Linear(3, 2), TinyPipeline()
        return built["model"], built["pipeline"], None

    pipeline, meta = load_pipeline(factory, str(tmp_path), fold_name(0))

    assert pipeline is built["pipeline"] and meta == {"fold": 0}
    assert pipeline.steps[0].modes == [False]
    assert torch.equal(built["model"].weight, trained.weight)


def test_ensure_new_models_dir(tmp_path):
    ensure_new_models_dir(str(tmp_path / "nao_existe"), ["fold0"], overwrite=False)
    save_checkpoint(nn.Linear(1, 1), str(tmp_path), "fold0")
    with pytest.raises(FileExistsError):
        ensure_new_models_dir(str(tmp_path), ["fold0", "fold1"], overwrite=False)
    ensure_new_models_dir(str(tmp_path), ["fold0"], overwrite=True)
