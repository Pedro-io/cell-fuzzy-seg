import numpy as np
import pytest

from src.pipeline.steps.preprocessing.rgba_step import RGBAStep


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


def test_rgba_step_prob_alpha_uses_cellpose_probability():
    step = RGBAStep(alpha="prob")
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    prob = np.array([[0.1, 0.9], [0.5, 1.0]], dtype=np.float16)

    data = step({"image": image, "cellpose_prob": prob})

    assert data["rgba"].dtype == np.float32
    np.testing.assert_allclose(data["rgba"][..., 3], prob.astype(np.float32))


def test_rgba_step_mask_alpha_is_the_default_and_binary():
    step = RGBAStep()
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    segmentation = np.array([[0, 3], [7, 0]], dtype=np.int32)  # rótulos de instância

    data = step({"image": image, "segmentation": segmentation})

    assert step.alpha == "mask"
    np.testing.assert_array_equal(data["rgba"][..., 3], [[0.0, 1.0], [1.0, 0.0]])


def test_rgba_step_rejects_unknown_alpha():
    with pytest.raises(ValueError, match="alpha"):
        RGBAStep(alpha="flows")


def test_rgba_step_requires_the_key_of_its_alpha_source():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    segmentation = np.ones((2, 2), dtype=np.uint8)

    with pytest.raises(KeyError, match="cellpose_prob"):
        RGBAStep(alpha="prob")({"image": image, "segmentation": segmentation})


def test_rgba_step_rejects_probability_out_of_range_or_wrong_shape():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    step = RGBAStep(alpha="prob")

    with pytest.raises(ValueError, match="fora de"):
        step({"image": image, "cellpose_prob": np.full((2, 2), 2.0, dtype=np.float32)})
    with pytest.raises(ValueError, match="formato"):
        step({"image": image, "cellpose_prob": np.zeros((3, 3), dtype=np.float32)})
