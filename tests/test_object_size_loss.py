"""Testes do ObjectSizeLoss simétrico (P10).

A perda deve penalizar tanto superestimação quanto subestimação da massa
prevista, com ótimo em pred.sum() == gt.sum().
"""

import pytest
import torch

from src.losses.object_size_loss import ObjectSizeLoss


def test_loss_is_zero_when_sizes_match():
    loss = ObjectSizeLoss(weight=0.1)
    pred = torch.ones(1, 1, 4, 4)
    gt = torch.ones(1, 1, 4, 4)

    value = loss(pred, gt)

    assert value.item() == 0.0


def test_loss_penalizes_overprediction_symmetrically():
    loss = ObjectSizeLoss(weight=0.1)
    gt = torch.ones(1, 1, 4, 4)

    # 1.5x a massa do GT → ratio 1.5, desvio 0.5.
    pred_over = torch.full((1, 1, 4, 4), 1.5)
    # 0.5x a massa do GT → ratio 0.5, desvio 0.5.
    pred_under = torch.full((1, 1, 4, 4), 0.5)

    over = loss(pred_over, gt).item()
    under = loss(pred_under, gt).item()

    # Desvios iguais em módulo (0.5) → penalidades iguais, independentemente da direção.
    assert over > 0.0
    assert under > 0.0
    assert abs(over - under) < 1e-6
    assert abs(over - 0.05) < 1e-6  # weight * |ratio - 1| = 0.1 * 0.5


def test_gradient_pushes_toward_gt_size():
    loss = ObjectSizeLoss(weight=1.0)
    # Previsão menor que o GT: aumentar a previsão deve reduzir a perda.
    pred = torch.full((1, 1, 4, 4), 0.5, requires_grad=True)
    gt = torch.ones(1, 1, 4, 4)

    value = loss(pred, gt)
    value.backward()

    # d/dpred de weight*|pred.sum()/gt.sum() - 1| com pred.sum() < gt.sum() é -weight/gt.sum().
    assert pred.grad is not None
    assert pred.grad.sum().item() < 0.0


def test_empty_gt_penalizes_predicted_mass():
    loss = ObjectSizeLoss(weight=0.1)
    pred = torch.ones(1, 1, 4, 4)
    gt = torch.zeros(1, 1, 4, 4)

    value = loss(pred, gt)

    assert value.item() == pytest.approx(0.1 * 16.0)
