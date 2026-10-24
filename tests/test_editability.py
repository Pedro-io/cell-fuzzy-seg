"""Testes das métricas de editabilidade do marcador."""

import numpy as np
import pytest
import torch

from src.evaluation.editability import marker_editability


def _gt_two_nuclei():
    gt = np.zeros((40, 40), dtype=np.uint8)
    gt[5:15, 5:15] = 1
    gt[25:35, 25:35] = 1
    return gt


def test_one_marker_per_nucleus_is_the_ideal_case():
    gt = _gt_two_nuclei()
    marker = np.zeros((40, 40), dtype=np.float32)
    marker[7:13, 7:13] = 0.9
    marker[27:33, 27:33] = 0.9
    e = marker_editability(marker, gt)
    assert e["componentes"] == 2 and e["buracos"] == 0 and e["fragmentos"] == 0
    assert e["regioes_um_componente"] == 1.0 and e["regioes_partidas"] == 0.0 and e["regioes_sem_marcador"] == 0.0
    assert e["componentes_unindo"] == 0 and e["componentes_fora"] == 0


def test_split_marker_hole_fragment_and_outside_component():
    gt = _gt_two_nuclei()
    marker = np.zeros((40, 40), dtype=np.float32)
    marker[6:14, 6:14] = 0.9
    marker[9, 9] = 0.0                     # buraco dentro do primeiro núcleo
    marker[27:29, 27:29] = 0.9             # segundo núcleo partido em dois fragmentos
    marker[32:34, 32:34] = 0.9
    marker[20:22, 2:4] = 0.9               # componente fora do GT
    e = marker_editability(marker, gt, fragment_area=20)
    assert e["componentes"] == 4 and e["buracos"] == 1 and e["fragmentos"] == 3
    assert e["regioes_um_componente"] == 0.5 and e["regioes_partidas"] == 0.5
    assert e["componentes_fora"] == 1


def test_marker_bridging_two_regions_is_counted():
    gt = _gt_two_nuclei()
    marker = np.zeros((40, 40), dtype=np.float32)
    for i in range(8, 32):
        marker[i, i - 1:i + 2] = 0.9
    e = marker_editability(marker, gt)
    assert e["componentes"] == 1 and e["componentes_unindo"] == 1


def test_values_at_or_below_tau_are_not_marker():
    gt = _gt_two_nuclei()
    marker = np.full((40, 40), 0.5, dtype=np.float32)
    e = marker_editability(marker, gt, tau=0.5)
    assert e["componentes"] == 0 and e["regioes_sem_marcador"] == 1.0


def test_accepts_tensors_with_singleton_dimensions():
    gt = _gt_two_nuclei()
    marker = torch.zeros(1, 1, 40, 40)
    marker[..., 7:13, 7:13] = 0.9
    assert marker_editability(marker, torch.from_numpy(gt)[None])["componentes"] == 1


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        marker_editability(np.zeros((4, 4)), np.zeros((5, 5)))
