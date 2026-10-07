"""Testes do resumo estatístico e do registro de um experimento."""

import csv
import json

import numpy as np
import pytest
from scipy.stats import wilcoxon

from src.evaluation.report import save_run, summarize
from src.training.kfold import FoldResult, TrainConfig


def _rows(dices, baseline):
    return [{"id": f"i{n}", "dice": d, "cellpose_dice": b} for n, (d, b) in enumerate(zip(dices, baseline))]


def test_summarize_mean_std_and_paired_comparison():
    dices = [0.80, 0.70, 0.90, 0.60]
    base = [0.75, 0.72, 0.85, 0.50]

    s = summarize(_rows(dices, base), metrics=["dice"])["dice"]

    assert s["mean"] == pytest.approx(np.mean(dices))
    assert s["std"] == pytest.approx(np.std(dices, ddof=1))
    assert s["baseline_mean"] == pytest.approx(np.mean(base))
    assert s["diff_mean"] == pytest.approx(np.mean(np.subtract(dices, base)))
    assert (s["n_better"], s["n_worse"]) == (3, 1)
    assert s["wilcoxon_p"] == pytest.approx(wilcoxon(dices, base).pvalue)


def test_bootstrap_interval_contains_the_mean_and_is_reproducible():
    rows = _rows(list(np.linspace(0.5, 0.9, 20)), [0.7] * 20)

    a = summarize(rows, metrics=["dice"], seed=1)["dice"]
    b = summarize(rows, metrics=["dice"], seed=1)["dice"]

    assert a["ci95"][0] < a["mean"] < a["ci95"][1]
    assert a["ci95"] == b["ci95"]
    assert a["diff_ci95"][0] < a["diff_mean"] < a["diff_ci95"][1]


def test_identical_to_baseline_gives_nan_p_value_and_no_baseline_keys_without_it():
    s = summarize(_rows([0.8, 0.7], [0.8, 0.7]), metrics=["dice"])["dice"]
    assert np.isnan(s["wilcoxon_p"]) and s["n_better"] == s["n_worse"] == 0

    no_base = summarize([{"dice": 0.8}, {"dice": 0.6}], metrics=["dice"])["dice"]
    assert "diff_mean" not in no_base


def test_metrics_missing_from_some_rows_are_skipped():
    s = summarize([{"dice": 0.8, "iou": 0.6}, {"dice": 0.7}])

    assert "dice" in s and "iou" not in s


def _fold(fold, dices, base):
    rows = _rows(dices, base)
    for r in rows:
        r["fold"] = fold
    return FoldResult(
        fold=fold, train_ids=["t1", "t2"], val_ids=[r["id"] for r in rows], best_epoch=2, best_dice=float(np.mean(dices)),
        epochs_run=3, val_dice_history=[{"epoch": e, "dice": 0.1 * e} for e in (1, 2, 3)],
        train_loss_history=[1.0, 0.8, 0.7], rows=rows,
    )


def test_save_run_writes_all_files(tmp_path):
    folds = [_fold(0, [0.8, 0.7], [0.75, 0.8]), _fold(1, [0.6, 0.9], [0.7, 0.85])]
    test_rows = _rows([0.82, 0.78], [0.80, 0.79])

    run_dir = save_run("exp_base", TrainConfig(k=2, device="cpu"), folds, test_rows=test_rows, final_epochs=2,
                       description={"perdas": "Dice + TV 0,001"}, output_dir=str(tmp_path))

    meta = json.load(open(f"{run_dir}/config.json"))
    assert meta["config"]["k"] == 2 and meta["final_epochs"] == 2
    assert meta["descricao"] == {"perdas": "Dice + TV 0,001"}
    assert {"commit", "torch", "numpy", "criado_em"} <= set(meta["procedencia"])
    with open(f"{run_dir}/folds.csv") as f:
        rows = list(csv.DictReader(f))
    assert [r["fold"] for r in rows] == ["0", "0", "1", "1"] and float(rows[0]["dice"]) == 0.8
    with open(f"{run_dir}/curvas.csv") as f:
        assert len(list(csv.DictReader(f))) == 6  # 2 folds × 3 épocas
    with open(f"{run_dir}/folds_resumo.csv") as f:
        assert [r["best_epoch"] for r in csv.DictReader(f)] == ["2", "2"]
    resumo = json.load(open(f"{run_dir}/resumo.json"))
    assert resumo["folds"]["n"] == 4 and resumo["teste"]["n"] == 2
    assert resumo["teste"]["dice"]["mean"] == pytest.approx(0.80)


def test_save_run_refuses_to_overwrite_and_writes_strict_json(tmp_path):
    folds = [_fold(0, [0.8, 0.7], [0.8, 0.7])]  # diferenças nulas → p = NaN no resumo
    save_run("x", {"k": 1}, folds, output_dir=str(tmp_path))

    with pytest.raises(FileExistsError):
        save_run("x", {"k": 1}, folds, output_dir=str(tmp_path))
    save_run("x", {"k": 1}, folds, output_dir=str(tmp_path), overwrite=True)

    text = open(tmp_path / "x" / "resumo.json").read()
    assert "NaN" not in text
    assert json.loads(text)["folds"]["dice"]["wilcoxon_p"] is None
