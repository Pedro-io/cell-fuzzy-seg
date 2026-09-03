import numpy as np
import torch
import torch.nn as nn

from src.models.networks.base_networks import BaseNetwork
from src.models.networks.final_segmentation.base_final_segmentation import BaseFinalSegmentation
from src.pipeline.steps.inference.frozen_segmentation_step import FrozenSegmentationStep


class DummyFinalNetwork(BaseFinalSegmentation):
    """Rede final dummy que aplica uma convolução sobre os scribbles."""

    def __init__(self):
        super().__init__(config={})
        self.conv = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, data):
        scribbles = data["scribbles"]
        if not isinstance(scribbles, torch.Tensor):
            scribbles = torch.from_numpy(scribbles).float()
        if scribbles.ndim == 2:
            scribbles = scribbles[None, None]
        return self.conv(scribbles)

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)


def test_frozen_segmentation_step_adds_segmentation_key():
    step = FrozenSegmentationStep(final_network=DummyFinalNetwork())
    image = torch.zeros(1, 1, 8, 8)
    markers = torch.zeros(1, 1, 8, 8)

    data = step.forward({"image": image, "markers": markers})

    assert "segmentation" in data
    assert data["segmentation"].shape == (1, 1, 8, 8)


def test_frozen_segmentation_step_accepts_numpy_markers():
    step = FrozenSegmentationStep(final_network=DummyFinalNetwork())
    image = np.zeros((8, 8), dtype=np.float32)
    markers = np.zeros((8, 8), dtype=np.float32)

    data = step.forward({"image": image, "markers": markers})

    assert "segmentation" in data
    assert data["segmentation"].shape == (1, 1, 8, 8)


def test_frozen_segmentation_step_requires_image_key():
    step = FrozenSegmentationStep(final_network=DummyFinalNetwork())

    try:
        step.forward({"markers": torch.zeros(1, 1, 8, 8)})
    except KeyError:
        pass
    else:
        raise AssertionError("Expected KeyError when 'image' is missing")


def test_frozen_segmentation_step_requires_markers_key():
    step = FrozenSegmentationStep(final_network=DummyFinalNetwork())

    try:
        step.forward({"image": torch.zeros(1, 1, 8, 8)})
    except KeyError:
        pass
    else:
        raise AssertionError("Expected KeyError when 'markers' is missing")


def test_frozen_segmentation_step_requires_final_network():
    step = FrozenSegmentationStep(final_network=None)

    try:
        step.forward({"image": torch.zeros(1, 1, 8, 8), "markers": torch.zeros(1, 1, 8, 8)})
    except RuntimeError:
        pass
    else:
        raise AssertionError("Expected RuntimeError when no final_network is provided")


def test_base_final_segmentation_is_abstract():
    try:
        BaseFinalSegmentation(config={})
    except TypeError:
        pass
    else:
        raise AssertionError("Expected TypeError when instantiating abstract BaseFinalSegmentation")


def test_dummy_final_network_is_base_network():
    net = DummyFinalNetwork()
    assert isinstance(net, BaseNetwork)
