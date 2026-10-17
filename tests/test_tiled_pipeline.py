"""Testes da execução em blocos sobrepostos."""

import pytest
import torch

from src.pipeline.tiled_pipeline import TiledPipeline, hann_window, tile_starts


class PointwisePipeline:
    """Saída de cada pixel depende só daquele pixel: os blocos colados dão a imagem inteira."""

    def __init__(self):
        self.steps = ["passo"]
        self.calls = []

    def run(self, data, verbose=False):
        self.calls.append(tuple(data["image"].shape[-2:]))
        out = dict(data)
        out["segmentation"] = torch.sigmoid(data["image"][:, :1] / 50 - 2)
        out["markers"] = data["rgba"][:, 3:4] * 0.5
        return out


def _batch(h, w, n=2):
    return {"id": [f"i{k}" for k in range(n)], "image": torch.rand(n, 3, h, w) * 255,
            "rgba": torch.rand(n, 4, h, w), "ground_truth": (torch.rand(n, 1, h, w) > 0.5).float()}


def test_tile_starts_cover_the_axis_and_touch_the_border():
    assert tile_starts(1000, 256, 192) == [0, 192, 384, 576, 744]
    assert tile_starts(256, 256, 192) == [0]
    assert tile_starts(300, 256, 128) == [0, 44]


def test_hann_window_has_a_floor_and_peaks_in_the_center():
    w = hann_window(8, 0.05)
    assert w.shape == (8, 8) and w.min() == pytest.approx(0.05) and w[3:5, 3:5].min() == w.max()


def test_stitched_output_equals_the_whole_image_for_a_pointwise_pipeline():
    inner = PointwisePipeline()
    data = _batch(20, 30)

    out = TiledPipeline(inner, tile_size=8, overlap=0.25).run(data)
    whole = PointwisePipeline().run(_copy(data))

    assert out["segmentation"].shape == (2, 1, 20, 30)
    assert torch.allclose(out["segmentation"], whole["segmentation"], atol=1e-6)
    assert torch.allclose(out["markers"], whole["markers"], atol=1e-6)
    assert set(inner.calls) == {(8, 8)}                     # todo bloco tem o tamanho do treino
    assert len(inner.calls) == len(tile_starts(20, 8, 6)) * len(tile_starts(30, 8, 6))
    assert out["ground_truth"] is data["ground_truth"]      # a entrada continua inteira


def test_small_images_pass_straight_through_and_steps_are_exposed():
    inner = PointwisePipeline()
    tiled = TiledPipeline(inner, tile_size=8)

    tiled.run(_batch(8, 8))

    assert inner.calls == [(8, 8)] and tiled.steps == ["passo"]


def test_invalid_options_raise():
    with pytest.raises(ValueError, match="tile_size"):
        TiledPipeline(PointwisePipeline(), tile_size=0)
    with pytest.raises(ValueError, match="overlap"):
        TiledPipeline(PointwisePipeline(), tile_size=8, overlap=1.0)
    with pytest.raises(ValueError, match="menor que o bloco"):
        TiledPipeline(PointwisePipeline(), tile_size=8).run(_batch(6, 20))


def _copy(data):
    return {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
