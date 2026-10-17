"""Figura das predições de um pipeline em algumas imagens: entradas, marcadores e segmentação.

Usada pelos notebooks de experimento (imagens da validação, cada uma com o modelo do *fold* em que
ficou de fora) e pelo notebook final (imagens do teste, com o modelo final).
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from src.data.load.preprocessed_dataset import PreprocessedDataset, collate_samples


def pick_cases(rows: Sequence[Mapping[str, Any]]) -> List[Tuple[str, Mapping[str, Any]]]:
    """Escolhe três imagens pela diferença de Dice contra o Cellpose: a pior, a mediana e a melhor.

    Args:
        rows: Métricas por imagem (com ``dice`` e ``cellpose_dice``), como as do ``evaluate``.

    Returns:
        ``[("pior × Cellpose", linha), ("mediana", linha), ("melhor × Cellpose", linha)]``.
    """
    order = sorted(rows, key=lambda r: r["dice"] - r["cellpose_dice"])
    return [("pior × Cellpose", order[0]), ("mediana", order[len(order) // 2]),
            ("melhor × Cellpose", order[-1])]


def _final_network(pipeline: Any) -> Optional[Any]:
    for step in getattr(pipeline, "steps", []):
        network = getattr(step, "final_network", None)
        if network is not None and hasattr(network, "scribble_channels"):
            return network
    return None


def _predict(pipeline: Any, samples: Mapping[str, Any], image_id: str,
             device: Optional[str]) -> Dict[str, Any]:
    batch = collate_samples([PreprocessedDataset(samples, ids=[image_id])[0]])
    if device is not None:
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    with torch.no_grad():
        return pipeline.run(batch, verbose=False)


def prediction_figure(
    cases: Sequence[Tuple[str, Mapping[str, Any], Any]],
    samples: Mapping[str, Mapping[str, np.ndarray]],
    crop: int = 400,
    threshold: float = 0.5,
    device: Optional[str] = None,
):
    """Monta a figura das predições: uma linha por imagem, sete painéis.

    Painéis: imagem; ground truth; máscara do Cellpose; marcador na escala 0–1; marcador com o
    contraste esticado entre os percentis 1 e 99 do recorte; canal positivo que chega à rede final,
    esticado do 0 ao máximo; segmentação binarizada. Os títulos trazem o Dice e os intervalos de
    valores dos painéis esticados. Mostra o recorte central ``crop × crop`` de cada imagem.

    Args:
        cases: ``(rótulo, linha de métricas, pipeline)`` por imagem; a linha precisa de ``id``,
            ``dice``, ``cellpose_dice`` e ``marker_fraction``. O pipeline já deve estar em modo de
            avaliação (ver :func:`src.training.checkpoints.load_pipeline`).
        samples: Amostras carregadas por ``load_preprocessed`` (as mesmas chaves do treino,
            inclusive o canal negativo, se o pipeline o usar).
        crop: Lado do recorte central, em pixels.
        threshold: Limiar de binarização da segmentação.
        device: Dispositivo para onde o batch é movido (``None``: não move).

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(cases), 7, figsize=(35, 5 * len(cases)), squeeze=False)
    for row_axes, (label, row, pipeline) in zip(axes, cases):
        image_id = row["id"]
        sample = samples[image_id]
        h, w = sample["ground_truth"].shape
        y0, x0 = max((h - crop) // 2, 0), max((w - crop) // 2, 0)
        window = (slice(y0, y0 + crop), slice(x0, x0 + crop))

        out = _predict(pipeline, samples, image_id, device)
        marker = out["markers"][0, 0].float().cpu().numpy()
        network = _final_network(pipeline)
        positive = (network.scribble_channels(out["markers"])[0, 0].float().cpu().numpy()
                    if network is not None else np.zeros_like(marker))
        segmentation = out["segmentation"][0, 0].float().cpu().numpy() >= threshold

        m_lo, m_hi = np.percentile(marker[window], [1, 99])
        p_hi = max(float(positive[window].max()), 1e-6)
        panels = [
            (f"{image_id[:16]}\n{label}", sample["image"], None),
            ("ground truth", sample["ground_truth"] > 0, "gray"),
            (f"Cellpose (Dice {row['cellpose_dice']:.3f})", sample["cellpose_segmentation"] > 0,
             "gray"),
            (f"marcador, 0–1 (≥ 0,5 em {100 * row['marker_fraction']:.1f}%)", marker, "magma"),
            (f"marcador, esticado ({m_lo:.3f}–{m_hi:.3f})",
             (marker - m_lo) / max(m_hi - m_lo, 1e-6), "magma"),
            (f"canal positivo, esticado (0–{p_hi:.3f})", positive / p_hi, "magma"),
            (f"segmentação (Dice {row['dice']:.3f})", segmentation, "gray"),
        ]
        for ax, (title, img, cmap) in zip(row_axes, panels):
            ax.imshow(img[window], cmap=cmap, **({} if cmap is None else {"vmin": 0, "vmax": 1}))
            ax.set_title(title)
            ax.axis("off")
    fig.tight_layout()
    return fig
