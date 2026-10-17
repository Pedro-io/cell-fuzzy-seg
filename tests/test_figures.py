"""Testes da escolha de imagens e da figura das predições."""

import matplotlib

matplotlib.use("Agg")

from src.evaluation.figures import pick_cases, prediction_figure  # noqa: E402
from src.training.kfold import run_kfold  # noqa: E402
from tests.test_kfold import CONFIG, Factory, data  # noqa: E402,F401


def test_pick_cases_orders_by_difference_to_cellpose():
    rows = [{"id": i, "dice": d, "cellpose_dice": 0.5} for i, d in zip("abcde", [0.6, 0.2, 0.9, 0.4, 0.5])]

    cases = pick_cases(rows)

    assert [label for label, _ in cases] == ["pior × Cellpose", "mediana", "melhor × Cellpose"]
    assert [r["id"] for _, r in cases] == ["b", "e", "c"]


def test_prediction_figure_has_one_row_per_case(data):  # noqa: F811
    train, _ = data
    folds = run_kfold(Factory(), train, list(train), CONFIG)
    pipeline = Factory()()[1]
    cases = [(label, row, pipeline) for label, row in pick_cases([r for f in folds for r in f.rows])]

    fig = prediction_figure(cases, train, crop=8, device="cpu")

    assert len(fig.axes) == 3 * 7
    assert "Dice" in fig.axes[6].get_title()
