import numpy as np
import pytest

from src.pipeline.steps.preprocessing.distance_map_step import (
    DistanceMapStep,
    compute_distance_map,
)


def test_compute_distance_map_returns_float32_in_zero_one_range():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:4, 1:4] = 1

    dmap = compute_distance_map(mask)

    assert dmap.shape == (5, 5)
    assert dmap.dtype == np.float32
    assert dmap.min() >= 0.0
    assert dmap.max() <= 1.0


def test_compute_distance_map_inverts_normalized_distance():
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:4, 1:4] = 1

    dmap = compute_distance_map(mask)

    # Interior da célula (maior distância à borda) mapeia para 0.
    assert dmap[2, 2] == 0.0
    # Fundo / fronteira (distância zero) mapeia para 1.
    assert dmap[0, 0] == 1.0
    # Ponto intermediário no limite do objeto.
    assert dmap[1, 1] == 0.5


def test_distance_map_step_adds_distance_map_key():
    step = DistanceMapStep()
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 1

    data = step({"ground_truth": mask})

    assert "distance_map" in data
    assert data["distance_map"].shape == (4, 4)
    assert data["distance_map"].dtype == np.float32


def test_distance_map_step_preserves_existing_keys():
    step = DistanceMapStep()
    mask = np.ones((2, 2), dtype=np.uint8)

    data = step({"image": np.zeros((2, 2)), "ground_truth": mask})

    assert "image" in data
    assert "distance_map" in data


def test_distance_map_step_raises_key_error_without_mask():
    step = DistanceMapStep()

    with pytest.raises(KeyError):
        step({"image": np.zeros((4, 4))})


def test_distance_map_step_custom_keys():
    step = DistanceMapStep(mask_key="segmentation", output_key="dmap")

    data = step({"segmentation": np.ones((3, 3), dtype=np.uint8)})

    assert "dmap" in data
    assert data["dmap"].shape == (3, 3)
