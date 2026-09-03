import numpy as np
import torch
import torch.nn as nn

from src.pipeline.steps.inference.marker_step import MarkerStep


class DummyMarkerNet:
    """MarkerNet dummy com interface compatível (atributo ``model`` + ``predict``)."""

    def __init__(self, in_channels: int = 4):
        self.model = nn.Sequential(nn.Conv2d(in_channels, 1, kernel_size=1))

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.model(x))


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


def test_inference_forward_returns_binary_numpy():
    step = MarkerStep(model=DummyMarkerNet(), differentiable=False)
    rgba = (np.random.rand(8, 8, 4) * 255).astype(np.uint8)

    data = step.forward({"rgba": rgba})

    assert isinstance(data["markers"], np.ndarray)
    assert data["markers"].shape == (8, 8)
    assert set(np.unique(data["markers"])).issubset({0.0, 1.0})


def test_inference_forward_accepts_tensor_input():
    step = MarkerStep(model=DummyMarkerNet(), differentiable=False)
    rgba = torch.rand(1, 4, 8, 8)

    data = step.forward({"rgba": rgba})

    assert isinstance(data["markers"], np.ndarray)
    assert data["markers"].shape == (8, 8)


def test_forward_requires_rgba_key():
    step = MarkerStep(model=DummyMarkerNet())

    try:
        step.forward({"image": np.zeros((8, 8, 3))})
    except KeyError:
        pass
    else:
        raise AssertionError("Expected KeyError when 'rgba' is missing")


def test_forward_without_model_falls_back_to_segmentation():
    step = MarkerStep(model=None)
    segmentation = np.random.rand(8, 8).astype(np.float32)
    segmentation[0, 0] = 1.0

    data = step.forward({"segmentation": segmentation})

    assert data["markers"].shape == (8, 8)
    assert np.array_equal(data["markers"], (segmentation > 0).astype(np.float32))


def test_to_tensor_bchw_handles_channels_last_tensor():
    step = MarkerStep(model=DummyMarkerNet(), differentiable=True)
    rgba = torch.rand(2, 8, 8, 4)  # (N, H, W, C)

    t = step._to_tensor_bchw(rgba)

    assert t.shape == (2, 4, 8, 8)
