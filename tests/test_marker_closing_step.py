"""Testes do fechamento morfológico dos marcadores."""

import numpy as np
import pytest
import torch
from scipy import ndimage

from src.pipeline.steps.inference.frozen_segmentation_step import FrozenSegmentationStep
from src.pipeline.steps.inference.marker_closing_step import (
    MarkerClosingStep,
    disk_offsets,
    grey_closing,
    grey_dilation,
    grey_erosion,
)
from tests.test_frozen_segmentation_step import RecordingFinalNetwork


def _disk(radius):
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return x * x + y * y <= radius * radius


def test_disk_offsets_match_the_euclidean_disk():
    assert len(disk_offsets(0)) == 1
    assert len(disk_offsets(1)) == 5
    assert len(disk_offsets(2)) == int(_disk(2).sum()) == 13


def test_dilation_and_erosion_match_scipy_inside_the_image():
    torch.manual_seed(0)
    x = torch.rand(1, 1, 20, 24)
    a = x[0, 0].numpy()
    for r in (1, 2, 3):
        dil = ndimage.grey_dilation(a, footprint=_disk(r), mode="constant", cval=-np.inf)
        ero = ndimage.grey_erosion(a, footprint=_disk(r), mode="constant", cval=np.inf)
        np.testing.assert_allclose(grey_dilation(x, r)[0, 0].numpy(), dil, rtol=0, atol=0)
        np.testing.assert_allclose(grey_erosion(x, r)[0, 0].numpy(), ero, rtol=0, atol=0)


def test_closing_is_extensive_and_idempotent():
    torch.manual_seed(1)
    x = torch.rand(2, 1, 32, 32)
    for r in (1, 2, 3):
        c = grey_closing(x, r)
        assert (c >= x).all()
        assert torch.equal(grey_closing(c, r), c)


def test_closing_fills_a_hole_smaller_than_the_disk():
    x = torch.zeros(1, 1, 21, 21)
    x[..., 5:16, 5:16] = 0.9
    x[..., 10, 10] = 0.1  # buraco de 1 px
    c = grey_closing(x, 1)
    assert c[0, 0, 10, 10] == pytest.approx(0.9)
    assert torch.equal(c[..., 5:16, 5:16], torch.full((1, 1, 11, 11), 0.9))


def test_closing_keeps_a_convex_marker_and_the_background():
    x = torch.zeros(1, 1, 30, 30)
    x[..., 8:20, 8:20] = 1.0
    assert torch.equal(grey_closing(x, 2), x)


def test_closing_does_not_grow_a_marker_touching_the_border():
    x = torch.zeros(1, 1, 16, 16)
    x[..., :6, :6] = 1.0
    assert torch.equal(grey_closing(x, 2), x)


def test_closing_commutes_with_the_positive_threshold():
    torch.manual_seed(2)
    x = torch.rand(1, 1, 24, 24)
    tau = 0.5
    cut = lambda m: torch.relu(m - tau) / (1 - tau)  # noqa: E731
    torch.testing.assert_close(cut(grey_closing(x, 2)), grey_closing(cut(x), 2))


def test_radius_zero_is_the_identity():
    x = torch.rand(1, 1, 8, 8)
    assert grey_closing(x, 0) is x


def test_gradient_flows_through_the_closing():
    x = torch.rand(1, 1, 16, 16, requires_grad=True)
    grey_closing(x, 2).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_negative_radius_raises():
    with pytest.raises(ValueError):
        grey_closing(torch.zeros(1, 1, 4, 4), -1)
    with pytest.raises(ValueError):
        MarkerClosingStep(radius=-1)


def test_step_adds_the_closed_marker_and_keeps_the_raw_one():
    markers = torch.zeros(1, 1, 12, 12)
    markers[..., 3:9, 3:9] = 1.0
    markers[..., 6, 6] = 0.0
    data = MarkerClosingStep(radius=1)({"markers": markers})
    assert data["markers"] is markers and data["markers"][0, 0, 6, 6] == 0
    assert data["markers_closed"][0, 0, 6, 6] == 1


def test_step_requires_the_input_key_and_a_tensor():
    step = MarkerClosingStep(radius=1)
    with pytest.raises(KeyError):
        step({"image": torch.zeros(1, 1, 4, 4)})
    with pytest.raises(TypeError):
        step({"markers": np.zeros((4, 4), dtype=np.float32)})


def test_step_refuses_to_overwrite_its_input():
    with pytest.raises(ValueError):
        MarkerClosingStep(radius=1, output_key="markers")


def test_frozen_segmentation_step_reads_the_markers_key():
    net = RecordingFinalNetwork()
    closed = torch.ones(1, 1, 8, 8)
    step = FrozenSegmentationStep(final_network=net, markers_key="markers_closed")
    step.forward({"image": torch.zeros(1, 1, 8, 8), "markers": torch.zeros(1, 1, 8, 8), "markers_closed": closed})
    assert net.received["scribbles"] is closed
    with pytest.raises(KeyError):
        step.forward({"image": torch.zeros(1, 1, 8, 8), "markers": torch.zeros(1, 1, 8, 8)})
