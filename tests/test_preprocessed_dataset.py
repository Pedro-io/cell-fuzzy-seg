"""Testes da leitura dos .npy, do dataset com aumentação por época e do collate."""

import numpy as np
import pytest
import torch

from src.data.load.preprocessed_dataset import (
    NEGATIVE_KEY,
    PreprocessedDataset,
    cellpose_negative,
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


def test_cellpose_negative_leaves_a_free_band_around_the_nuclei():
    seg = np.zeros((9, 9), np.int32)
    seg[4, 4] = 3

    touching = cellpose_negative(seg, radius=0)
    banded = cellpose_negative(seg, radius=2)

    assert touching.dtype == np.float32 and touching.sum() == 80 and touching[4, 4] == 0
    assert banded[4, 6] == 0 and banded[4, 7] == 1   # a 2 px: faixa livre; a 3 px: negativo
    assert banded[6, 6] == 1                         # diagonal: distância 2,83 > 2
    with pytest.raises(ValueError, match="radius"):
        cellpose_negative(seg, radius=-1)


def test_load_adds_the_negative_only_when_asked(root):
    assert NEGATIVE_KEY not in load_preprocessed("train", root=root, ids=["a"])["a"]
    sample = load_preprocessed("train", root=root, ids=["a"], negative_radius=0)["a"]

    np.testing.assert_array_equal(sample[NEGATIVE_KEY], (sample["cellpose_segmentation"] == 0).astype(np.float32))
    with pytest.raises(ValueError, match="negative_radius"):
        load_preprocessed("train", root=root, negative_radius=-1)


def test_negative_follows_the_augmentation(root):
    samples = load_preprocessed("train", root=root, negative_radius=0)
    dataset = PreprocessedDataset(samples, augment=True, seed=0)

    for _ in range(12):
        item = dataset[0]
        assert item[NEGATIVE_KEY].dtype == torch.float32
        np.testing.assert_array_equal(item[NEGATIVE_KEY][0].numpy(),
                                      (item["cellpose_segmentation"][0].numpy() == 0).astype(np.float32))
    plain = PreprocessedDataset(samples)
    assert collate_samples([plain[0], plain[1]])[NEGATIVE_KEY].shape == (2, 1, H, W)


def test_crops_have_the_requested_size_and_are_the_same_window_in_every_key(root):
    samples = load_preprocessed("train", root=root, negative_radius=0)
    dataset = PreprocessedDataset(samples, augment=True, seed=0, crop_size=4, crops_per_image=3)

    assert len(dataset) == 3 * 3
    assert [dataset.ids[i // 3] for i in range(len(dataset))] == ["a"] * 3 + ["b"] * 3 + ["c"] * 3
    for index in range(len(dataset)):
        item = dataset[index]
        seg = item["cellpose_segmentation"][0].numpy().astype(np.float32)
        assert seg.shape == (4, 4)
        np.testing.assert_array_equal(item["rgba"][0].numpy(), seg)
        np.testing.assert_array_equal(item["ground_truth"][0].numpy(), (seg > 20).astype(np.float32))
        np.testing.assert_array_equal(item[NEGATIVE_KEY][0].numpy(), (seg == 0).astype(np.float32))


def test_crops_change_between_accesses_and_repeat_with_the_same_seed(root):
    samples = load_preprocessed("train", root=root)
    a = PreprocessedDataset(samples, seed=3, crop_size=3, crops_per_image=2)
    b = PreprocessedDataset(samples, seed=3, crop_size=3, crops_per_image=2)

    first = [a[0]["image"].numpy().tobytes() for _ in range(20)]
    assert len(set(first)) > 1
    assert first == [b[0]["image"].numpy().tobytes() for _ in range(20)]


def test_crop_options_are_validated(root):
    samples = load_preprocessed("train", root=root)
    with pytest.raises(ValueError, match="crop_size"):
        PreprocessedDataset(samples, crop_size=0)
    with pytest.raises(ValueError, match="crops_per_image"):
        PreprocessedDataset(samples, crops_per_image=2)
    with pytest.raises(ValueError, match="maior que a imagem"):
        PreprocessedDataset(samples, crop_size=50)[0]
