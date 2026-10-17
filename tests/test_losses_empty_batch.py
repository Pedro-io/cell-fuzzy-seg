"""Perdas normalizadas pela massa do GT num lote sem nenhum núcleo (possível com recortes)."""

import torch

from src.losses.distance_map_loss import DistanceMapLoss
from src.losses.total_variation_loss import TotalVariationLoss


def test_dmap_and_tv_stay_finite_without_any_nucleus_in_the_batch():
    pred = torch.rand(4, 1, 8, 8, requires_grad=True)
    empty = torch.zeros(4, 1, 8, 8)

    losses = [DistanceMapLoss(weight=1.0)(pred, torch.rand(4, 1, 8, 8), empty),
              TotalVariationLoss(weight=1.0)(pred, empty)]
    sum(losses).backward()

    assert all(torch.isfinite(loss) for loss in losses)
    assert torch.isfinite(pred.grad).all()


def test_normalization_is_unchanged_when_there_are_nuclei():
    pred, dmap = torch.rand(2, 1, 8, 8), torch.rand(2, 1, 8, 8)
    gt = torch.zeros(2, 1, 8, 8)
    gt[:, :, 2:6, 2:6] = 1.0

    expected = (pred * dmap).sum() / gt.sum()
    assert torch.isclose(DistanceMapLoss(weight=1.0)(pred, dmap, gt), expected)
