"""Testes da leitura dos .npy, do dataset com aumentação por época e do collate."""

import numpy as np
import pytest
import torch

from src.data.load.preprocessed_dataset import (
    PreprocessedDataset,
    collate_samples,
    load_preprocessed,
    make_loader,
)

H, W = 6, 8  # não quadrado: rotações de 90° trocam altura e largura


def _write_split(root, split, ids):
    """Grava amostras sintéticas em que todas as chaves derivam do mesmo padrão espacial."""
    for n, image_id in enumerate(ids):
        base = (np.arange(H * W).reshape(H, W) + 100 * n).astype(np.float32)  # valores únicos
        arrays = {
            "image": np.stack([base % 256] * 3, axis=-1).astype(np.uint8),
            "rgba": np.concatenate([np.stack([base] * 3, -1), (base[..., None] > 20)], -1).astype(np.float32),
            "ground_truth": (base > 20).astype(np.uint8),
            "distance_map": (base / base.max()).astype(np.float32),
            "segmentation": base.astype(np.uint16),
            "cellpose_prob": (base / base.max()).astype(np.float16),
        }
        for key, array in arrays.items():
            folder = root / split / key
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / f"{image_id}.npy", array)


@pytest.fixture()
def root(tmp_path):
    _write_split(tmp_path, "train", ["a", "b", "c"])
    return str(tmp_path)


def test_load_reads_all_ids_and_renames_cellpose_segmentation(root):
    samples = load_preprocessed("train", root=root)

    assert list(samples) == ["a", "b", "c"]
    assert set(samples["a"]) == {"image", "rgba", "ground_truth", "distance_map", "cellpose_segmentation"}
    assert samples["a"]["cellpose_segmentation"].dtype == np.uint16


def test_load_subset_of_ids(root):
    assert list(load_preprocessed("train", root=root, ids=["c", "a"])) == ["c", "a"]


def test_load_prob_alpha_replaces_fourth_channel(root):
    mask = load_preprocessed("train", root=root, ids=["a"])["a"]["rgba"]
    prob = load_preprocessed("train", root=root, ids=["a"], alpha="prob")["a"]["rgba"]
    expected = np.load(f"{root}/train/cellpose_prob/a.npy").astype(np.float32)

    np.testing.assert_array_equal(prob[..., 3], expected)
    np.testing.assert_array_equal(prob[..., :3], mask[..., :3])


def test_load_rejects_unknown_alpha_and_missing_files(root):
    with pytest.raises(ValueError, match="alpha"):
        load_preprocessed("train", root=root, alpha="flows")
    with pytest.raises(FileNotFoundError):
        load_preprocessed("train", root=root, ids=["nao_existe"])


def test_item_tensors_without_augmentation(root):
    samples = load_preprocessed("train", root=root)
    item = PreprocessedDataset(samples)[0]

    assert item["id"] == "a"
    assert item["image"].shape == (3, H, W) and item["image"].dtype == torch.float32
    assert item["rgba"].shape == (4, H, W)
    assert item["ground_truth"].shape == (1, H, W) and item["ground_truth"].dtype == torch.float32
    assert item["distance_map"].shape == (1, H, W)
    assert item["cellpose_segmentation"].shape == (1, H, W) and item["cellpose_segmentation"].dtype == torch.int32
    np.testing.assert_array_equal(item["cellpose_segmentation"][0].numpy(), samples["a"]["cellpose_segmentation"])


def test_dataset_rejects_unknown_ids(root):
    with pytest.raises(KeyError, match="x"):
        PreprocessedDataset(load_preprocessed("train", root=root), ids=["a", "x"])


def test_augmentation_is_the_same_for_every_key(root):
    samples = load_preprocessed("train", root=root)
    dataset = PreprocessedDataset(samples, augment=True, seed=0)

    for _ in range(12):
        item = dataset[1]
        seg = item["cellpose_segmentation"][0].numpy().astype(np.float32)  # o padrão de referência
        np.testing.assert_array_equal(item["rgba"][0].numpy(), seg)
        np.testing.assert_array_equal(item["image"][0].numpy(), seg % 256)
        np.testing.assert_array_equal(item["ground_truth"][0].numpy(), (seg > 20).astype(np.float32))
        np.testing.assert_allclose(item["distance_map"][0].numpy(), seg / seg.max())


def test_augmentation_changes_between_accesses_and_keeps_originals(root):
    samples = load_preprocessed("train", root=root)
    original = samples["a"]["cellpose_segmentation"].copy()
    dataset = PreprocessedDataset(samples, augment=True, seed=0)

    seen = {dataset[0]["cellpose_segmentation"].numpy().tobytes() for _ in range(30)}

    assert len(seen) > 1  # sorteios diferentes a cada acesso (= a cada época)
    np.testing.assert_array_equal(samples["a"]["cellpose_segmentation"], original)


def test_same_seed_repeats_the_augmentation_sequence(root):
    samples = load_preprocessed("train", root=root)
    d1, d2 = PreprocessedDataset(samples, augment=True, seed=7), PreprocessedDataset(samples, augment=True, seed=7)

    for _ in range(10):
        assert torch.equal(d1[0]["image"], d2[0]["image"])


def test_collate_and_loader(root):
    dataset = PreprocessedDataset(load_preprocessed("train", root=root))

    batch = collate_samples([dataset[0], dataset[2]])
    assert batch["id"] == ["a", "c"]
    assert batch["rgba"].shape == (2, 4, H, W)

    order = lambda seed: [i for b in make_loader(dataset, 2, shuffle=True, seed=seed) for i in b["id"]]  # noqa: E731
    assert order(3) == order(3)                     # mesma semente, mesma ordem
    assert sorted(order(3)) == ["a", "b", "c"]      # cada imagem uma vez por época
    loader = make_loader(dataset, 2, shuffle=False)
    assert [b["id"] for b in loader] == [["a", "b"], ["c"]]
