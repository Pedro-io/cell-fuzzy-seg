"""Testes do CellposeStep: checagem do nome do modelo e chaves produzidas pelo ``forward``.

O Cellpose exige GPU e não roda nos testes. Um módulo ``cellpose`` falso reproduz só o que o step
usa, com os mesmos nomes do ``cellpose/models.py`` v4.1.1: ``MODEL_NAMES``, ``get_user_models()``
e ``CellposeModel`` (cujo ``eval`` devolve ``masks, [fluxo_rgb, dP, cellprob], styles``).
"""

import importlib
import sys
import types

import numpy as np
import pytest

# Saída do ``eval`` falso: máscara de instâncias e o logit de probabilidade (``flows[2]``).
FAKE_MASKS = np.array([[0, 1], [2, 0]], dtype=np.int32)
FAKE_LOGIT = np.array([[-4.0, 0.0], [2.0, 10.0]], dtype=np.float32)


class FakeCellposeModel:
    """Registra os argumentos em vez de carregar a rede."""

    def __init__(self, gpu=False, pretrained_model=None, diam_mean=None):
        self.pretrained_model = pretrained_model
        self.logit = FAKE_LOGIT

    def eval(self, image, **kwargs):
        flows = [np.zeros(image.shape[:2] + (3,), np.uint8), np.zeros((2,) + image.shape[:2]), self.logit]
        return FAKE_MASKS, flows, np.zeros(256)


@pytest.fixture()
def cellpose_step_module(monkeypatch):
    """Instala um ``cellpose`` falso (com GPU "disponível") e importa o step sobre ele."""
    fake_models = types.ModuleType("cellpose.models")
    fake_models.MODEL_NAMES = ["cpsam"]
    fake_models.get_user_models = lambda: ["meu_modelo"]
    fake_models.CellposeModel = FakeCellposeModel

    fake_core = types.ModuleType("cellpose.core")
    fake_core.use_gpu = lambda: True

    fake = types.ModuleType("cellpose")
    fake.models = fake_models
    fake.core = fake_core

    monkeypatch.setitem(sys.modules, "cellpose", fake)
    monkeypatch.setitem(sys.modules, "cellpose.models", fake_models)
    monkeypatch.setitem(sys.modules, "cellpose.core", fake_core)

    module = importlib.import_module("src.pipeline.steps.preprocessing.cellpose_step")
    module = importlib.reload(module)  # garante que ele use o falso deste teste
    yield module
    monkeypatch.undo()
    sys.modules.pop("src.pipeline.steps.preprocessing.cellpose_step", None)


def test_builtin_model_name_is_accepted(cellpose_step_module):
    step = cellpose_step_module.CellposeStep(pretrained_model="cpsam")

    assert step.model.pretrained_model == "cpsam"


def test_user_registered_model_is_accepted(cellpose_step_module):
    cellpose_step_module.CellposeStep._validate_model_name("meu_modelo")


def test_existing_model_file_is_accepted(cellpose_step_module, tmp_path):
    model_file = tmp_path / "modelo_proprio"
    model_file.write_bytes(b"pesos")

    cellpose_step_module.CellposeStep._validate_model_name(str(model_file))


def test_unknown_model_name_raises_before_loading(cellpose_step_module, monkeypatch):
    # "cpsam_v2" não existe no Cellpose 4: o modelo não pode nem ser carregado.
    def fail_if_built(*args, **kwargs):
        raise AssertionError("o CellposeModel não deveria ser construído")

    monkeypatch.setattr(cellpose_step_module.models, "CellposeModel", fail_if_built)

    with pytest.raises(ValueError, match="cpsam_v2"):
        cellpose_step_module.CellposeStep(pretrained_model="cpsam_v2")


def test_missing_model_file_raises(cellpose_step_module, tmp_path):
    with pytest.raises(ValueError, match="desconhecido"):
        cellpose_step_module.CellposeStep._validate_model_name(str(tmp_path / "nao_existe"))


def test_forward_adds_segmentation_and_probability_only(cellpose_step_module):
    step = cellpose_step_module.CellposeStep(pretrained_model="cpsam")
    data = {"image": np.zeros((2, 2, 3), dtype=np.uint8)}

    out = step(data)

    assert set(out) == {"image", "segmentation", "cellpose_prob"}
    np.testing.assert_array_equal(out["segmentation"], FAKE_MASKS)
    prob = out["cellpose_prob"]
    assert prob.dtype == np.float16
    # sigmoide do logit: 0 → 0,5 (o limiar cellprob_threshold=0 do Cellpose)
    expected = 1.0 / (1.0 + np.exp(-FAKE_LOGIT))
    np.testing.assert_allclose(prob.astype(np.float32), expected, atol=1e-3)
    assert prob[0, 1] == np.float16(0.5)


def test_forward_rejects_probability_with_wrong_shape(cellpose_step_module):
    step = cellpose_step_module.CellposeStep(pretrained_model="cpsam")
    step.model.logit = np.zeros((3, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="formato"):
        step({"image": np.zeros((2, 2, 3), dtype=np.uint8)})
