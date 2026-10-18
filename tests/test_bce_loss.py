"""Testes da entropia cruzada binária sobre probabilidades e do termo dela."""

import pytest
import torch
import torch.nn.functional as F

from src.losses import BCETerm, BinaryCrossEntropyLoss, LossComposer
from src.losses.terms import DiceTerm


def test_matches_the_torch_formula_on_probabilities():
    prob = torch.rand(2, 1, 6, 6) * 0.98 + 0.01
    gt = (torch.rand(2, 1, 6, 6) > 0.5).float()

    loss = BinaryCrossEntropyLoss(weight=2.0)(prob, gt)

    expected = -(gt * prob.log() + (1 - gt) * (1 - prob).log()).mean()
    assert torch.isclose(loss, 2.0 * expected)
    assert torch.isclose(loss, 2.0 * F.binary_cross_entropy(prob, gt))


def test_saturated_probabilities_stay_finite_and_have_gradient():
    prob = torch.tensor([[[[0.0, 1.0], [1.0, 0.0]]]], requires_grad=True)
    gt = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])

    loss = BinaryCrossEntropyLoss()(prob, gt)
    loss.backward()

    assert torch.isfinite(loss) and torch.isfinite(prob.grad).all()


def test_a_confident_false_positive_costs_more_than_an_uncertain_one():
    gt = torch.zeros(1, 1, 1, 2)
    confident = BinaryCrossEntropyLoss()(torch.tensor([[[[0.95, 0.0]]]]), gt)
    uncertain = BinaryCrossEntropyLoss()(torch.tensor([[[[0.55, 0.0]]]]), gt)

    assert confident > uncertain


def test_term_reads_prediction_and_gt_and_composes_with_dice():
    prob = torch.rand(1, 1, 4, 4)
    gt = (torch.rand(1, 1, 4, 4) > 0.5).float()
    term = BCETerm(weight=0.5)

    assert term.name == "bce"
    assert torch.isclose(term({"prediction": prob, "gt_masks": gt}), 0.5 * BinaryCrossEntropyLoss()(prob, gt))
    total, log = LossComposer([DiceTerm(), BCETerm()])(prob, torch.zeros_like(prob), gt)
    assert set(log) >= {"dice", "bce"}
    assert total == pytest.approx(float(log["dice"] + log["bce"]), rel=1e-5)
