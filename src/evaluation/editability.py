"""Métricas de editabilidade do marcador: o quanto ele se aproxima de "um traço por núcleo".

O marcador é avaliado depois do corte em τ (o que a rede final recebe como scribble positivo). Um
marcador fácil de editar tem poucos componentes, poucos fragmentos e poucos buracos, e cada núcleo
recebe um único componente, sem componentes que atravessem núcleos vizinhos.
"""

from collections import Counter
from typing import Any, Dict

import numpy as np
from scipy import ndimage

DEFAULT_FRAGMENT_AREA = 20


def _to_numpy(array: Any) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.squeeze(np.asarray(array))


def marker_editability(
    marker: Any,
    ground_truth: Any,
    tau: float = 0.5,
    fragment_area: int = DEFAULT_FRAGMENT_AREA,
) -> Dict[str, float]:
    """Mede a forma do marcador cortado em ``tau`` contra as regiões do GT.

    Args:
        marker: Marcador ``(H, W)`` (numpy ou tensor; dimensões unitárias são removidas).
        ground_truth: GT binário ``(H, W)``. Núcleos que se tocam formam uma região só.
        tau: Corte do marcador; conta como marcado o pixel com valor ``> tau``.
        fragment_area: Componentes com menos px que isso contam como fragmentos.

    Returns:
        Dicionário com:
            - ``componentes``: componentes do marcador (8-vizinhança);
            - ``fragmentos``: componentes com menos de ``fragment_area`` px;
            - ``buracos``: regiões sem marcador cercadas por marcador (4-vizinhança, sem tocar a borda);
            - ``regioes_sem_marcador``, ``regioes_um_componente``, ``regioes_partidas``: frações das
              regiões do GT tocadas por nenhum, um ou dois ou mais componentes;
            - ``componentes_unindo``: componentes que tocam duas ou mais regiões do GT;
            - ``componentes_fora``: componentes que não tocam o GT.

    Raises:
        ValueError: Se ``marker`` e ``ground_truth`` não tiverem o mesmo formato 2D.
    """
    m = _to_numpy(marker) > tau
    gt = _to_numpy(ground_truth) > 0
    if m.ndim != 2 or m.shape != gt.shape:
        raise ValueError(f"marker {m.shape} e ground_truth {gt.shape} devem ser 2D e do mesmo formato.")

    labels, n = ndimage.label(m, structure=np.ones((3, 3)))
    areas = ndimage.sum(m, labels, range(1, n + 1)) if n else np.zeros(0)
    background, _ = ndimage.label(~m)
    on_border = set(np.unique(np.concatenate([background[0], background[-1], background[:, 0], background[:, -1]])))
    holes = len(set(np.unique(background)) - on_border - {0})

    regions, n_gt = ndimage.label(gt)
    inside = (labels > 0) & (regions > 0)
    pairs = np.unique(np.stack([labels[inside], regions[inside]]), axis=1) if inside.any() else np.zeros((2, 0), int)
    components_per_region = Counter(pairs[1].tolist())
    regions_per_component = Counter(pairs[0].tolist())
    total = max(n_gt, 1)
    return {
        "componentes": n,
        "fragmentos": int((areas < fragment_area).sum()),
        "buracos": holes,
        "regioes_sem_marcador": 1 - len(components_per_region) / total if n_gt else 0.0,
        "regioes_um_componente": sum(v == 1 for v in components_per_region.values()) / total,
        "regioes_partidas": sum(v >= 2 for v in components_per_region.values()) / total,
        "componentes_unindo": sum(v >= 2 for v in regions_per_component.values()),
        "componentes_fora": n - len(regions_per_component),
    }
