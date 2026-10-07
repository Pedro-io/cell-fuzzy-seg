import numpy as np
import pytest
from scipy.ndimage import distance_transform_edt

from src.pipeline.steps.preprocessing.distance_map_step import (
    DistanceMapStep,
    compute_instance_distance_map,
)


def disc(shape, center, radius, label, out=None):
    out = np.zeros(shape, dtype=np.int32) if out is None else out
    rr, cc = np.ogrid[: shape[0], : shape[1]]
    out[(rr - center[0]) ** 2 + (cc - center[1]) ** 2 <= radius**2] = label
    return out


def reference_full_image(labels):
    """EDT de cada núcleo calculada na imagem inteira (sem recorte)."""
    dmap = np.ones(labels.shape, dtype=np.float32)
    for label in range(1, labels.max() + 1):
        nucleus = labels == label
        if nucleus.any():
            distance = distance_transform_edt(nucleus)
            dmap[nucleus] = (1.0 - distance / distance.max())[nucleus]
    return dmap


def test_returns_float32_in_zero_one_range():
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[1:4, 1:4] = 1

    dmap = compute_instance_distance_map(labels)

    assert dmap.shape == (5, 5)
    assert dmap.dtype == np.float32
    assert dmap.min() >= 0.0
    assert dmap.max() <= 1.0


def test_single_nucleus_center_zero_background_one():
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[1:4, 1:4] = 1

    dmap = compute_instance_distance_map(labels)

    assert dmap[2, 2] == 0.0  # centro (EDT máxima = 2)
    assert dmap[0, 0] == 1.0  # fundo
    assert dmap[1, 1] == 0.5  # borda do núcleo (EDT = 1)


def test_every_nucleus_has_its_center_at_zero():
    # Um núcleo grande e um pequeno: os dois centros valem 0 (com o máximo da imagem, o pequeno
    # ficaria em 1 − 6/20 = 0,7).
    labels = disc((100, 100), (30, 30), 20, 1)
    disc((100, 100), (75, 75), 6, 2, out=labels)

    dmap = compute_instance_distance_map(labels)

    assert dmap[30, 30] == 0.0
    assert dmap[75, 75] == 0.0


def test_touching_nuclei_share_a_border():
    # Dois retângulos 5×5 colados (colunas 2–6 e 7–11): a coluna da fronteira é borda dos dois.
    labels = np.zeros((9, 14), dtype=np.int32)
    labels[2:7, 2:7] = 1
    labels[2:7, 7:12] = 2

    dmap = compute_instance_distance_map(labels)

    np.testing.assert_allclose(dmap[4, 6], dmap[4, 2])   # fronteira interna = borda externa
    np.testing.assert_allclose(dmap[4, 7], dmap[4, 11])
    assert dmap[4, 4] == 0.0 and dmap[4, 9] == 0.0      # um centro em cada núcleo


def test_crop_with_margin_matches_full_image_edt():
    # Núcleos encostados uns nos outros e na borda da imagem.
    labels = disc((60, 60), (10, 10), 9, 1)
    disc((60, 60), (10, 27), 8, 2, out=labels)
    disc((60, 60), (55, 30), 9, 3, out=labels)
    labels[30:40, 0:6] = 4

    np.testing.assert_array_equal(compute_instance_distance_map(labels), reference_full_image(labels))


def test_image_border_is_not_treated_as_background():
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[0:4, 0:4] = 1  # núcleo cortado pelo canto da imagem

    dmap = compute_instance_distance_map(labels)

    assert dmap[0, 0] == 0.0  # o ponto mais barato fica junto da borda


def test_rejects_non_integer_masks():
    with pytest.raises(ValueError, match="inteiros"):
        compute_instance_distance_map(np.zeros((4, 4), dtype=np.float32))


def test_distance_map_step_adds_distance_map_key():
    step = DistanceMapStep()
    labels = np.zeros((4, 4), dtype=np.int32)
    labels[1:3, 1:3] = 1

    data = step({"ground_truth_instances": labels})

    assert data["distance_map"].shape == (4, 4)
    assert data["distance_map"].dtype == np.float32


def test_distance_map_step_preserves_existing_keys():
    step = DistanceMapStep()

    data = step({"image": np.zeros((2, 2)), "ground_truth_instances": np.ones((2, 2), dtype=np.int32)})

    assert "image" in data
    assert "distance_map" in data


def test_distance_map_step_requires_instances():
    # Sem a máscara por instância, o step falha em vez de cair num mapa diferente.
    step = DistanceMapStep()

    with pytest.raises(KeyError, match="ground_truth_instances"):
        step({"ground_truth": np.ones((4, 4), dtype=np.uint8)})


def test_distance_map_step_custom_keys():
    step = DistanceMapStep(instances_key="segmentation", output_key="dmap")

    data = step({"segmentation": np.ones((3, 3), dtype=np.int32)})

    assert data["dmap"].shape == (3, 3)
