"""Resumo estatístico das métricas por imagem e registro reproduzível de um experimento.

:func:`summarize` resume as linhas produzidas por :func:`~src.evaluation.metrics.evaluate` (média,
desvio, intervalo de confiança por *bootstrap* e comparação pareada com o Cellpose). :func:`save_run`
grava, numa pasta própria do experimento, a configuração com a procedência (commit, versões, GPU),
as métricas por imagem dos *folds* e do teste, as curvas de treino e o resumo.
"""

import csv
import dataclasses
import json
import os
import platform
import subprocess
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from scipy.stats import wilcoxon

from .metrics import METRIC_NAMES


def _bootstrap_ci(values: np.ndarray, seed: int, n_resamples: int = 10000, level: float = 0.95) -> List[float]:
    """Intervalo de confiança da média por *bootstrap* percentil, reamostrando as imagens."""
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    alpha = (1.0 - level) / 2.0
    return [float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))]


def summarize(
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str] = METRIC_NAMES,
    baseline_prefix: str = "cellpose_",
    seed: int = 0,
) -> Dict[str, Any]:
    """Resume métricas por imagem e compara cada uma, pareada por imagem, com a linha de base.

    Args:
        rows: Uma linha por imagem (como as de ``evaluate``).
        metrics: Métricas a resumir; as que não estiverem em todas as linhas são ignoradas.
        baseline_prefix: Prefixo das colunas da linha de base.
        seed: Semente do *bootstrap*.

    Returns:
        ``{"n": ..., "<métrica>": {...}}``. Para cada métrica: ``mean``, ``std`` (amostral), ``ci95``
        (*bootstrap* da média) e, se a linha de base existir, ``baseline_mean``, ``baseline_std`` e a
        comparação pareada ``diff_mean``, ``diff_ci95``, ``n_better``, ``n_worse`` e ``wilcoxon_p``
        (``NaN`` quando todas as diferenças são zero ou há menos de 2 imagens).
    """
    summary: Dict[str, Any] = {"n": len(rows)}
    for name in [m for m in metrics if rows and all(m in r for r in rows)]:
        values = np.array([float(r[name]) for r in rows])
        entry: Dict[str, Any] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
            "ci95": _bootstrap_ci(values, seed),
        }
        base_key = baseline_prefix + name
        if rows and all(base_key in r for r in rows):
            base = np.array([float(r[base_key]) for r in rows])
            diff = values - base
            entry.update({
                "baseline_mean": float(base.mean()),
                "baseline_std": float(base.std(ddof=1)) if len(base) > 1 else float("nan"),
                "diff_mean": float(diff.mean()),
                "diff_ci95": _bootstrap_ci(diff, seed),
                "n_better": int((diff > 0).sum()),
                "n_worse": int((diff < 0).sum()),
                "wilcoxon_p": float(wilcoxon(values, base).pvalue) if len(diff) > 1 and np.any(diff != 0) else float("nan"),
            })
        summary[name] = entry
    return summary


def provenance(repo_dir: str = ".") -> Dict[str, Any]:
    """Procedência do run: commit, mudanças locais em ``src/``/``configs/``, versões e GPU."""
    def git(*args: str) -> Optional[str]:
        try:
            out = subprocess.run(["git", "-C", repo_dir, *args], capture_output=True, text=True, check=True)
            return out.stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    import torch

    status = git("status", "--porcelain", "--", "src", "configs")
    return {
        "commit": git("rev-parse", "HEAD"),
        "mudancas_locais_em_src": bool(status) if status is not None else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "criado_em": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _json_safe(value: Any) -> Any:
    """Troca ``NaN`` por ``None`` para o JSON ser válido fora do Python."""
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _write_csv(path: str, rows: Sequence[Mapping[str, Any]]) -> None:
    columns: List[str] = []
    for row in rows:
        columns += [c for c in row if c not in columns]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def save_run(
    name: str,
    config: Any,
    folds: Sequence[Any],
    test_rows: Optional[Sequence[Mapping[str, Any]]] = None,
    final_epochs: Optional[int] = None,
    description: Optional[Mapping[str, Any]] = None,
    output_dir: str = os.path.join("docs", "estudo", "resultados"),
    overwrite: bool = False,
) -> str:
    """Grava os resultados de um experimento em ``<output_dir>/<name>/``.

    Arquivos:

    - ``config.json``: a configuração do treino, a descrição do experimento (perdas, alpha, etc.), o
      número de épocas do treino final e a procedência (:func:`provenance`);
    - ``folds.csv``: métricas por imagem da validação de todos os *folds*;
    - ``folds_resumo.csv``: por *fold*, a melhor época, o melhor Dice e as épocas executadas;
    - ``curvas.csv``: por *fold* e época, a perda de treino e o Dice de validação;
    - ``teste.csv``: métricas por imagem do teste (se houver);
    - ``resumo.json``: :func:`summarize` dos *folds* e do teste.

    Args:
        name: Nome da pasta do experimento.
        config: Configuração do treino (dataclass, como ``TrainConfig``, ou ``dict``).
        folds: Resultados dos *folds* (``FoldResult``).
        test_rows: Métricas por imagem do teste (``None``: sem teste).
        final_epochs: Épocas do treino final.
        description: O que distingue o experimento (perdas e pesos, alpha, resolução...).
        output_dir: Pasta-mãe dos experimentos.
        overwrite: Se ``False``, recusa gravar numa pasta que já existe.

    Returns:
        Caminho da pasta gravada.

    Raises:
        FileExistsError: Se a pasta já existir e ``overwrite`` for ``False``.
    """
    run_dir = os.path.join(output_dir, name)
    if os.path.exists(run_dir) and not overwrite:
        raise FileExistsError(f"{run_dir} já existe; use outro nome ou overwrite=True.")
    os.makedirs(run_dir, exist_ok=True)

    config_dict = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else dict(config)
    meta = {
        "nome": name,
        "descricao": dict(description or {}),
        "config": config_dict,
        "final_epochs": final_epochs,
        "procedencia": provenance(),
    }
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(meta), f, indent=2, ensure_ascii=False, allow_nan=False)

    fold_rows = [row for fold in folds for row in fold.rows]
    _write_csv(os.path.join(run_dir, "folds.csv"), fold_rows)
    _write_csv(os.path.join(run_dir, "folds_resumo.csv"), [
        {"fold": f.fold, "best_epoch": f.best_epoch, "best_dice": f.best_dice, "epochs_run": f.epochs_run,
         "n_train": len(f.train_ids), "n_val": len(f.val_ids)}
        for f in folds
    ])
    curves = []
    for f in folds:
        val_dice = {h["epoch"]: h["dice"] for h in f.val_dice_history}
        for epoch, loss in enumerate(f.train_loss_history, start=1):
            curves.append({"fold": f.fold, "epoch": epoch, "train_loss": loss, "val_dice": val_dice.get(epoch, "")})
    _write_csv(os.path.join(run_dir, "curvas.csv"), curves)

    summary = {"folds": summarize(fold_rows) if fold_rows else None}
    if test_rows is not None:
        _write_csv(os.path.join(run_dir, "teste.csv"), test_rows)
        summary["teste"] = summarize(test_rows)
    with open(os.path.join(run_dir, "resumo.json"), "w", encoding="utf-8") as f:
        json.dump(_json_safe(summary), f, indent=2, ensure_ascii=False, allow_nan=False)
    return run_dir
