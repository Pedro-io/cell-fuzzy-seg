"""Leitura dos dados pré-processados (``.npy``) e dataset de treino com aumentação por época.

Os arrays de um split são lidos do disco uma única vez por :func:`load_preprocessed` e podem ser
compartilhados por vários :class:`PreprocessedDataset` (por exemplo, os *folds* de uma validação
cruzada), cada um com o seu subconjunto de ids. A aumentação nunca altera os arrays carregados:
cada acesso devolve cópias transformadas.
"""

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

DEFAULT_ROOT = os.path.join("data_source", "MoNuSegPreprocessed")

# Chaves lidas do disco e o nome de cada uma na amostra. A segmentação do Cellpose entra como
# ``cellpose_segmentation`` porque a rede final grava a sua saída em ``segmentation``.
_DISK_KEYS = {
    "image": "image",
    "rgba": "rgba",
    "ground_truth": "ground_truth",
    "distance_map": "distance_map",
    "segmentation": "cellpose_segmentation",
}
ALPHA_SOURCES = ("mask", "prob")


def load_preprocessed(
    split: str,
    root: str = DEFAULT_ROOT,
    ids: Optional[Sequence[str]] = None,
    alpha: str = "mask",
) -> Dict[str, Dict[str, np.ndarray]]:
    """Lê do disco as amostras pré-processadas de um split.

    Args:
        split: Subpasta do split (``"train"`` ou ``"test"``).
        root: Pasta com os dados pré-processados (``<root>/<split>/<chave>/<id>.npy``).
        ids: Ids a carregar. ``None``: todos os ids presentes em ``<root>/<split>/image``.
        alpha: Origem do 4º canal do ``rgba``: ``"mask"`` mantém o que está salvo (a máscara binária
            do Cellpose); ``"prob"`` o troca pela probabilidade do Cellpose (``cellpose_prob``).

    Returns:
        Dicionário ``id -> amostra``, em que cada amostra tem ``image`` ``(H, W, 3)`` uint8, ``rgba``
        ``(H, W, 4)`` float32, ``ground_truth`` ``(H, W)``, ``distance_map`` ``(H, W)`` float32 e
        ``cellpose_segmentation`` ``(H, W)`` (instâncias do Cellpose).

    Raises:
        ValueError: Se ``alpha`` não for uma das opções.
        FileNotFoundError: Se algum arquivo esperado não existir.
    """
    if alpha not in ALPHA_SOURCES:
        raise ValueError(f"alpha deve ser um de {ALPHA_SOURCES}, recebido {alpha!r}.")

    base = os.path.join(root, split)
    if ids is None:
        ids = sorted(f[:-4] for f in os.listdir(os.path.join(base, "image")) if f.endswith(".npy"))

    samples: Dict[str, Dict[str, np.ndarray]] = {}
    for image_id in ids:
        sample = {
            name: np.load(os.path.join(base, disk_key, f"{image_id}.npy"))
            for disk_key, name in _DISK_KEYS.items()
        }
        if alpha == "prob":
            prob = np.load(os.path.join(base, "cellpose_prob", f"{image_id}.npy"))
            sample["rgba"] = sample["rgba"].copy()
            sample["rgba"][..., 3] = prob.astype(np.float32)
        samples[image_id] = sample
    return samples


def _augment(sample: Mapping[str, np.ndarray], rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Aplica a mesma rotação de 90° e os mesmos espelhamentos a todas as chaves da amostra."""
    k = int(rng.integers(0, 4))
    flip_lr = bool(rng.random() < 0.5)
    flip_ud = bool(rng.random() < 0.5)
    out = {}
    for key, array in sample.items():
        a = np.rot90(array, k, axes=(0, 1))
        if flip_lr:
            a = a[:, ::-1]
        if flip_ud:
            a = a[::-1]
        out[key] = np.ascontiguousarray(a)
    return out


class PreprocessedDataset(Dataset):
    """Dataset de amostras pré-processadas, com aumentação sorteada a cada acesso.

    Com ``augment=True``, cada acesso sorteia uma rotação de 90° (k ∈ {0, 1, 2, 3}) e espelhamentos
    horizontal e vertical (probabilidade 0,5 cada), aplicados juntos à imagem, ao RGBA, ao ground
    truth, ao mapa de distância e à segmentação do Cellpose. Como o ``DataLoader`` acessa cada amostra
    uma vez por época, cada época vê uma aumentação nova. O sorteio usa um gerador próprio com
    ``seed``: a mesma semente repete a mesma sequência. Use ``num_workers=0`` no ``DataLoader``
    para que a sequência seja reproduzível.

    Args:
        samples: Amostras carregadas por :func:`load_preprocessed` (``id -> amostra``).
        ids: Ids deste dataset, na ordem de acesso. ``None``: todos os de ``samples``.
        augment: Se ``True``, sorteia a aumentação a cada acesso.
        seed: Semente do gerador da aumentação.

    Raises:
        KeyError: Se algum id não estiver em ``samples``.
    """

    def __init__(
        self,
        samples: Mapping[str, Mapping[str, np.ndarray]],
        ids: Optional[Sequence[str]] = None,
        augment: bool = False,
        seed: Optional[int] = None,
    ) -> None:
        self.samples = samples
        self.ids = list(samples) if ids is None else list(ids)
        missing = [i for i in self.ids if i not in samples]
        if missing:
            raise KeyError(f"Ids ausentes das amostras carregadas: {missing}")
        self.augment = augment
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Devolve a amostra como tensores ``(C, H, W)``.

        Returns:
            Dicionário com ``id`` (str) e os tensores ``image`` ``(3, H, W)`` float32 em [0, 255],
            ``rgba`` ``(4, H, W)`` float32, ``ground_truth`` ``(1, H, W)`` float32, ``distance_map``
            ``(1, H, W)`` float32 e ``cellpose_segmentation`` ``(1, H, W)`` int32.
        """
        image_id = self.ids[index]
        sample = self.samples[image_id]
        if self.augment:
            sample = _augment(sample, self._rng)
        return {
            "id": image_id,
            "image": torch.from_numpy(np.ascontiguousarray(sample["image"].transpose(2, 0, 1))).float(),
            "rgba": torch.from_numpy(np.ascontiguousarray(sample["rgba"].transpose(2, 0, 1))).float(),
            "ground_truth": torch.from_numpy(sample["ground_truth"].astype(np.float32))[None],
            "distance_map": torch.from_numpy(sample["distance_map"].astype(np.float32))[None],
            "cellpose_segmentation": torch.from_numpy(sample["cellpose_segmentation"].astype(np.int32))[None],
        }


def collate_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Empilha uma lista de amostras num batch: tensores ``(B, C, H, W)`` e ``id`` como lista."""
    batch: Dict[str, Any] = {"id": [s["id"] for s in samples]}
    for key in samples[0]:
        if key != "id":
            batch[key] = torch.stack([s[key] for s in samples])
    return batch


def make_loader(
    dataset: PreprocessedDataset,
    batch_size: int,
    shuffle: bool,
    seed: Optional[int] = None,
) -> DataLoader:
    """Cria o ``DataLoader`` do projeto: ``collate_samples``, ``num_workers=0`` e ordem com semente.

    Args:
        dataset: Dataset das amostras.
        batch_size: Imagens por batch.
        shuffle: Se ``True``, embaralha a ordem a cada época.
        seed: Semente do embaralhamento (``None``: não determinística).

    Returns:
        ``DataLoader`` que pode ser iterado uma vez por época.
    """
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_samples,
        num_workers=0,
        generator=generator,
    )
