import numpy as np

from src.pipeline.steps.rgba_step import RGBAStep


def test_rgba_step_returns_float32_rgba_in_zero_one_range():
    step = RGBAStep()
    image = np.arange(4, dtype=np.uint8).reshape(2, 2)
    segmentation = np.array([[0, 1], [0, 1]], dtype=np.uint8)

    rgba = step._build_rgba(image, segmentation)

    assert rgba.shape == (2, 2, 4)
    assert rgba.dtype == np.float32
    assert rgba[..., :3].min() >= 0.0
    assert rgba[..., :3].max() <= 1.0
    assert rgba[..., 3].min() >= 0.0
    assert rgba[..., 3].max() <= 1.0
    assert np.allclose(rgba[..., 3], np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32))
