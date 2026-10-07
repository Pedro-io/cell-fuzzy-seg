"""Métricas de segmentação binária por imagem e avaliação de um pipeline de treino.

As métricas comparam uma predição contínua (ex.: a saída sigmoide da rede final) ou binária com o
ground truth binário, pixel a pixel, depois de binarizar a predição por um limiar. São calculadas
**por imagem**; médias e comparações entre métodos são feitas sobre essas linhas.
"""

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

METRIC_NAMES = ("dice", "iou", "precision", "recall", "mass_ratio")


def _to_numpy(array: Any) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def binary_metrics(prediction: Any, ground_truth: Any, threshold: float = 0.5) -> Dict[str, float]:
    """Calcula as métricas de segmentação binária de uma imagem.

    A predição é binarizada como ``prediction >= threshold``; o ground truth, como
    ``ground_truth > 0``.

    Casos sem nenhum pixel positivo (possíveis em recortes): Dice e IoU valem 1 quando predição e
    ground truth estão ambos vazios; precisão vale 1 quando não há pixel previsto (nenhum falso
    positivo); revocação vale 1 quando o ground truth está vazio (nenhum falso negativo).

    Args:
        prediction: Predição ``(H, W)`` (array NumPy ou tensor), contínua ou binária.
        ground_truth: Ground truth ``(H, W)``, com valores diferentes de zero no primeiro plano.
        threshold: Limiar de binarização da predição.

    Returns:
        Dicionário com ``dice``, ``iou``, ``precision``, ``recall`` e ``mass_ratio`` (massa prevista
        sobre a massa do ground truth, com a massa do ground truth limitada a no mínimo 1 pixel).

    Raises:
        ValueError: Se os formatos de ``prediction`` e ``ground_truth`` forem diferentes.
    """
    pred = _to_numpy(prediction) >= threshold
    gt = _to_numpy(ground_truth) > 0
    if pred.shape != gt.shape:
        raise ValueError(f"Formatos diferentes: predição {pred.shape}, ground truth {gt.shape}.")

    tp = float(np.logical_and(pred, gt).sum())
    fp = float(np.logical_and(pred, ~gt).sum())
    fn = float(np.logical_and(~pred, gt).sum())

    union = tp + fp + fn
    return {
        "dice": 2 * tp / (2 * tp + fp + fn) if union > 0 else 1.0,
        "iou": tp / union if union > 0 else 1.0,
        "precision": tp / (tp + fp) if tp + fp > 0 else 1.0,
        "recall": tp / (tp + fn) if tp + fn > 0 else 1.0,
        "mass_ratio": (tp + fp) / max(tp + fn, 1.0),
    }


def _set_training_mode(pipeline: Any, training: bool) -> None:
    for step in getattr(pipeline, "steps", []):
        set_training = getattr(step, "set_training", None)
        if set_training is not None:
            set_training(training)


def _to_device(batch: Dict[str, Any], device: Optional[str]) -> Dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) and device is not None else value
        for key, value in batch.items()
    }


@torch.no_grad()
def evaluate(
    pipeline: Any,
    batches: Iterable[Dict[str, Any]],
    threshold: float = 0.5,
    device: Optional[str] = None,
    prediction_key: str = "segmentation",
    ground_truth_key: str = "ground_truth",
    baseline_key: Optional[str] = "cellpose_segmentation",
    markers_key: str = "markers",
) -> List[Dict[str, Any]]:
    """Avalia um pipeline imagem por imagem, com a linha de base do Cellpose ao lado.

    Põe os passos do pipeline em modo de avaliação (``set_training(False)``) e roda sem gradiente.
    Cada batch é copiado antes de entrar no pipeline: os passos escrevem no dicionário que recebem
    (a rede final grava ``prediction_key``), e o batch original fica intacto.

    Args:
        pipeline: Pipeline com ``run(data, verbose)`` e, opcionalmente, ``steps`` com
            ``set_training``.
        batches: Iterável de batches ``dict`` com tensores ``(B, 1, H, W)`` e a lista ``"id"``.
        threshold: Limiar de binarização da predição.
        device: Dispositivo para onde os tensores do batch são movidos (``None``: não move).
        prediction_key: Chave da predição na saída do pipeline.
        ground_truth_key: Chave do ground truth no batch.
        baseline_key: Chave da máscara do Cellpose no batch; suas métricas entram com o prefixo
            ``"cellpose_"``. ``None`` ou chave ausente: sem linha de base.
        markers_key: Chave dos marcadores na saída; se presente, entra ``marker_fraction`` (fração de
            pixels com marcador ``>= threshold``).

    Returns:
        Uma linha (``dict``) por imagem, com ``"id"``, as métricas da predição e, quando houver, as da
        linha de base e ``marker_fraction``.
    """
    _set_training_mode(pipeline, False)
    rows: List[Dict[str, Any]] = []
    for batch in batches:
        output = pipeline.run(_to_device(batch, device), verbose=False)
        prediction = _to_numpy(output[prediction_key])
        ground_truth = _to_numpy(batch[ground_truth_key])
        baseline = _to_numpy(batch[baseline_key]) if baseline_key and baseline_key in batch else None
        markers = _to_numpy(output[markers_key]) if markers_key in output else None

        for b, image_id in enumerate(batch["id"]):
            row: Dict[str, Any] = {"id": image_id}
            row.update(binary_metrics(prediction[b, 0], ground_truth[b, 0], threshold))
            if baseline is not None:
                base = binary_metrics(baseline[b, 0] > 0, ground_truth[b, 0], threshold=0.5)
                row.update({f"cellpose_{name}": value for name, value in base.items()})
            if markers is not None:
                row["marker_fraction"] = float((markers[b, 0] >= threshold).mean())
            rows.append(row)
    return rows
