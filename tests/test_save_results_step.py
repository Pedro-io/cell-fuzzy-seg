import numpy as np
import pytest

from src.pipeline.steps.persistence.save_results_step import SaveResultsStep


def _sample(image_id="TCGA-test-01"):
    image = (np.random.rand(8, 8, 3) * 255).astype(np.uint8)
    segmentation = (np.random.rand(8, 8) > 0.7).astype(np.uint8)
    rgba = np.concatenate([image.astype(np.float32) / 255.0, segmentation[..., None].astype(np.float32)], axis=-1)
    ground_truth = segmentation.astype(np.float32)
    distance_map = 1.0 - ground_truth
    return {
        "id": image_id,
        "image": image,
        "segmentation": segmentation,
        "rgba": rgba,
        "ground_truth": ground_truth,
        "distance_map": distance_map,
    }


def test_save_results_step_persists_arrays_as_npy(tmp_path):
    step = SaveResultsStep(output_dir=str(tmp_path))
    sample = _sample()

    returned = step(sample)

    # O passo não altera o dicionário em memória.
    assert returned is sample
    for key in step.keys:
        path = tmp_path / key / f"{sample['id']}.npy"
        assert path.exists(), f"Arquivo esperado não encontrado: {path}"
        loaded = np.load(path)
        assert np.array_equal(loaded, sample[key]), f"Conteúdo de '{key}' divergente após reload"


def test_save_results_step_preserves_float_precision(tmp_path):
    """rgba/distance_map são float32 em [0, 1] — o .npy deve preservar exatamente."""
    step = SaveResultsStep(output_dir=str(tmp_path), keys=["rgba", "distance_map"])
    sample = _sample()

    step(sample)

    rgba_path = tmp_path / "rgba" / f"{sample['id']}.npy"
    dmap_path = tmp_path / "distance_map" / f"{sample['id']}.npy"

    assert np.load(rgba_path).dtype == np.float32
    assert np.load(dmap_path).dtype == np.float32
    assert np.array_equal(np.load(rgba_path), sample["rgba"])
    assert np.array_equal(np.load(dmap_path), sample["distance_map"])


def test_save_results_step_uses_default_keys(tmp_path):
    step = SaveResultsStep(output_dir=str(tmp_path))

    assert set(step.keys) == {"image", "segmentation", "rgba", "ground_truth", "distance_map"}


def test_save_results_step_custom_keys(tmp_path):
    step = SaveResultsStep(output_dir=str(tmp_path), keys=["ground_truth"])

    step(_sample())

    assert (tmp_path / "ground_truth" / "TCGA-test-01.npy").exists()
    assert not (tmp_path / "image" / "TCGA-test-01.npy").exists()


def test_save_results_step_raises_without_id(tmp_path):
    step = SaveResultsStep(output_dir=str(tmp_path))
    sample = _sample()
    del sample["id"]

    with pytest.raises(KeyError):
        step(sample)


def test_save_results_step_raises_on_missing_key(tmp_path):
    step = SaveResultsStep(output_dir=str(tmp_path), keys=["rgba", "flows"])
    sample = _sample()

    with pytest.raises(KeyError):
        step(sample)
