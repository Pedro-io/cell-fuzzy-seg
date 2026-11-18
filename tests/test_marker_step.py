import numpy as np
import pytest
import torch
import torch.nn as nn

from src.pipeline.steps.inference.marker_step import MarkerStep


class DummyMarkerNet:
    """MarkerNet dummy com a interface usada pelo MarkerStep (atributo ``model``)."""

    def __init__(self, in_channels: int = 4):
        self.model = nn.Sequential(nn.Conv2d(in_channels, 1, kernel_size=1))


def test_differentiable_forward_returns_tensor():
    step = MarkerStep(model=DummyMarkerNet(), differentiable=True)
    rgba = torch.rand(1, 4, 8, 8)  # (N, C, H, W) em [0, 1]

    data = step.forward({"rgba": rgba})

    assert isinstance(data["markers"], torch.Tensor)
    assert data["markers"].shape == (1, 1, 8, 8)
    assert data["markers"].min() >= 0.0 and data["markers"].max() <= 1.0


def test_differentiable_forward_preserves_gradient():
    """O gradiente deve fluir da saída até os parâmetros da MarkerNet."""
    net = DummyMarkerNet()
    step = MarkerStep(model=net, differentiable=True)
    rgba = torch.rand(1, 4, 8, 8, requires_grad=True)

    markers = step.forward({"rgba": rgba})["markers"]
    markers.sum().backward()

    assert rgba.grad is not None
    assert rgba.grad.abs().sum() > 0
    assert all(p.grad is not None for p in net.model.parameters())


def test_differentiable_forward_accepts_numpy_input():
    step = MarkerStep(model=DummyMarkerNet(), differentiable=True)
    rgba = (np.random.rand(8, 8, 4) * 255).astype(np.uint8)  # (H, W, 4) uint8

    data = step.forward({"rgba": rgba})

    assert data["markers"].shape == (1, 1, 8, 8)
    assert data["markers"].min() >= 0.0 and data["markers"].max() <= 1.0


def test_rejects_non_differentiable_mode():
    with pytest.raises(ValueError):
        MarkerStep(model=DummyMarkerNet(), differentiable=False)


def test_requires_model():
    with pytest.raises(ValueError):
        MarkerStep(model=None)


def test_batch_keeps_shape_and_skips_resize_at_target_size():
    """Um lote já no tamanho alvo passa sem redução e sai com um marcador por amostra."""
    step = MarkerStep(model=DummyMarkerNet(), target_size=8)
    rgba = torch.rand(3, 4, 8, 8)

    markers = step.forward({"rgba": rgba})["markers"]

    expected = torch.sigmoid(step.model.model(rgba))
    assert markers.shape == (3, 1, 8, 8)
    assert torch.allclose(markers, expected)


def test_resizes_to_target_and_back():
    step = MarkerStep(model=DummyMarkerNet(), target_size=8)
    rgba = torch.rand(2, 4, 20, 20)

    markers = step.forward({"rgba": rgba})["markers"]

    assert markers.shape == (2, 1, 20, 20)


def test_set_training_switches_mode():
    step = MarkerStep(model=DummyMarkerNet())
    assert step.model.model.training is True
    step.set_training(False)
    assert step.model.model.training is False
    step.set_training(True)
    assert step.model.model.training is True


def test_forward_requires_rgba_key():
    step = MarkerStep(model=DummyMarkerNet())

    try:
        step.forward({"image": np.zeros((8, 8, 3))})
    except KeyError:
        pass
    else:
        raise AssertionError("Expected KeyError when 'rgba' is missing")


def test_to_tensor_bchw_handles_channels_last_tensor():
    step = MarkerStep(model=DummyMarkerNet(), differentiable=True)
    rgba = torch.rand(2, 8, 8, 4)  # (N, H, W, C)

    t = step._to_tensor_bchw(rgba)

    assert t.shape == (2, 4, 8, 8)
