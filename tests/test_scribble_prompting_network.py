"""Testes da rede final congelada ScribblePromptingNetwork.

O pacote ``scribbleprompt`` não está disponível no ambiente local, então estes
testes injetam um módulo fake (``scribbleprompt.models.unet``) com uma UNet
mínima para validar o contrato, a conversão de entradas, o congelamento dos
pesos e o fluxo de gradiente até os scribbles.
"""

import importlib.util
import sys
import types

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.networks.final_segmentation.scribble_prompting_network import (
    ScribblePromptingNetwork,
)


class FakeUNet(nn.Module):
    """UNet mínima com 5 canais de entrada (como a ScribblePrompt-UNet)."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(5, 1, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class FakeScribblePromptUNet:
    """Fake do wrapper ScribblePromptUNet do pacote."""

    weights = {"v1": "checkpoint_fake.pt"}

    def __init__(self, version="v1", device=None):
        self.version = version
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FakeUNet().to(self.device)
        self.input_size = (128, 128)

    def parameters(self):
        return self.model.parameters()

    def to(self, device):
        self.model = self.model.to(device)
        self.device = device
        return self


def fake_prepare_inputs(inputs):
    """Replica prepare_inputs do pacote: cat(img, box, scribbles, mask)."""
    img = inputs["img"]
    box = torch.zeros_like(img)
    scribbles = inputs.get("scribbles")
    if scribbles is None:
        scribbles = torch.zeros(img.shape[0], 2, *img.shape[-2:], device=img.device)
    mask = torch.zeros_like(img)
    return torch.cat([img, box, scribbles, mask], dim=1)


def fake_rescale_inputs(inputs, input_size=(128, 128)):
    """Replica rescale_inputs do pacote: redimensiona img e scribbles."""
    import torch.nn.functional as F

    if tuple(inputs["img"].shape[-2:]) != tuple(input_size):
        inputs["img"] = F.interpolate(inputs["img"], size=input_size, mode="bilinear")
    if inputs.get("scribbles") is not None and tuple(inputs["scribbles"].shape[-2:]) != tuple(input_size):
        inputs["scribbles"] = F.interpolate(inputs["scribbles"], size=input_size, mode="bilinear")
    return inputs


@pytest.fixture()
def fake_scribbleprompt(monkeypatch):
    """Instala um pacote scribbleprompt fake em sys.modules."""

    pkg = types.ModuleType("scribbleprompt")
    models = types.ModuleType("scribbleprompt.models")
    unet = types.ModuleType("scribbleprompt.models.unet")
    unet.ScribblePromptUNet = FakeScribblePromptUNet
    unet.prepare_inputs = fake_prepare_inputs
    unet.rescale_inputs = fake_rescale_inputs
    models.unet = unet
    pkg.models = models
    monkeypatch.setitem(sys.modules, "scribbleprompt", pkg)
    monkeypatch.setitem(sys.modules, "scribbleprompt.models", models)
    monkeypatch.setitem(sys.modules, "scribbleprompt.models.unet", unet)
    return pkg


def test_forward_returns_sigmoid_mask(fake_scribbleprompt):
    net = ScribblePromptingNetwork()
    image = torch.rand(1, 1, 32, 32)
    scribbles = torch.rand(1, 1, 32, 32)

    mask = net({"image": image, "scribbles": scribbles})

    assert mask.shape == (1, 1, 32, 32)
    assert mask.min() >= 0.0 and mask.max() <= 1.0


def test_forward_accepts_numpy_inputs(fake_scribbleprompt):
    net = ScribblePromptingNetwork()
    image = np.random.rand(32, 32).astype(np.float32)
    scribbles = np.random.rand(32, 32).astype(np.float32)

    mask = net({"image": image, "scribbles": scribbles})

    assert mask.shape == (1, 1, 32, 32)


def test_forward_requires_image_and_scribbles(fake_scribbleprompt):
    net = ScribblePromptingNetwork()

    with pytest.raises(KeyError):
        net({"image": torch.rand(1, 1, 32, 32)})
    with pytest.raises(KeyError):
        net({"scribbles": torch.rand(1, 1, 32, 32)})


def test_weights_are_frozen(fake_scribbleprompt):
    net = ScribblePromptingNetwork()

    assert all(not p.requires_grad for p in net.parameters())


def test_gradient_flows_to_scribbles(fake_scribbleprompt):
    """O gradiente deve fluir da loss até os scribbles (saída da MarkerNet)."""
    net = ScribblePromptingNetwork()
    image = torch.rand(1, 1, 32, 32)
    scribbles = torch.rand(1, 1, 32, 32, requires_grad=True)

    mask = net({"image": image, "scribbles": scribbles})
    loss = mask.sum()
    loss.backward()

    assert scribbles.grad is not None
    assert scribbles.grad.abs().sum() > 0


def test_predict_returns_mask_without_grad(fake_scribbleprompt):
    net = ScribblePromptingNetwork()
    image = torch.rand(1, 1, 32, 32)
    scribbles = torch.rand(1, 1, 32, 32, requires_grad=True)

    mask = net.predict({"image": image, "scribbles": scribbles})

    assert mask.shape == (1, 1, 32, 32)
    assert scribbles.grad is None


def test_resize_output_to_input_size(fake_scribbleprompt):
    net = ScribblePromptingNetwork(resize_output=True)
    image = torch.rand(1, 1, 64, 64)
    scribbles = torch.rand(1, 1, 64, 64)

    mask = net({"image": image, "scribbles": scribbles})

    assert mask.shape == (1, 1, 64, 64)


def test_no_resize_keeps_model_size(fake_scribbleprompt):
    net = ScribblePromptingNetwork(resize_output=False)
    image = torch.rand(1, 1, 32, 32)
    scribbles = torch.rand(1, 1, 32, 32)

    mask = net({"image": image, "scribbles": scribbles})

    assert mask.shape == (1, 1, 128, 128)


@pytest.mark.skipif(
    importlib.util.find_spec("scribbleprompt") is not None,
    reason="Pacote scribbleprompt real instalado no ambiente",
)
def test_import_error_without_scribbleprompt(monkeypatch):
    monkeypatch.delitem(sys.modules, "scribbleprompt", raising=False)
    monkeypatch.delitem(sys.modules, "scribbleprompt.models", raising=False)
    monkeypatch.delitem(sys.modules, "scribbleprompt.models.unet", raising=False)

    with pytest.raises(ImportError):
        ScribblePromptingNetwork()


def test_train_keeps_network_in_eval(fake_scribbleprompt):
    """O FrozenSegmentationStep chama .train(); a rede congelada deve ficar eval."""
    net = ScribblePromptingNetwork()

    net.train()

    assert net.training is False
    assert net.unet.training is False


def test_missing_checkpoint_raises_runtime_error(fake_scribbleprompt, monkeypatch):
    """Checkpoint ausente deve virar RuntimeError com dica de download."""
    class RaisingUNet(FakeScribblePromptUNet):
        def __init__(self, *args, **kwargs):
            raise AssertionError("Checkpoint file not found: ...")

    unet_mod = sys.modules["scribbleprompt.models.unet"]
    monkeypatch.setattr(unet_mod, "ScribblePromptUNet", RaisingUNet)

    with pytest.raises(RuntimeError, match="download_checkpoint"):
        ScribblePromptingNetwork()


def test_sharpened_scribbles_are_complementary_and_near_binary(fake_scribbleprompt):
    """C2: no modo padrão (sharpened), os 2 canais são complementares e quase binários."""
    net = ScribblePromptingNetwork(scribble_mode="sharpened", scribble_temperature=10.0)
    scribbles = torch.tensor([[[[0.43, 0.6, 0.79]]]], dtype=torch.float32)  # (1,1,1,3)

    s = net._prepare_scribbles(scribbles, torch.device("cpu"))

    assert s.shape == (1, 2, 1, 3)
    # pos + neg = 1 em todos os pixels (sigmoid(x) + sigmoid(-x) == 1).
    assert torch.allclose(s[:, 0] + s[:, 1], torch.ones_like(s[:, 0]), atol=1e-5)
    # Marcador alto (0.79) → canal positivo próximo de 1; marcador baixo (0.43) → positivo baixo.
    assert s[0, 0, 0, 2].item() > 0.9
    assert s[0, 0, 0, 0].item() < 0.4


def test_dense_soft_mode_keeps_legacy_behavior(fake_scribbleprompt):
    """O modo legado dense_soft mantém [s, 1-s]."""
    net = ScribblePromptingNetwork(scribble_mode="dense_soft")
    scribbles = torch.tensor([[[[0.6]]]], dtype=torch.float32)

    s = net._prepare_scribbles(scribbles, torch.device("cpu"))

    assert s.shape == (1, 2, 1, 1)
    assert torch.allclose(s[0, 0, 0, 0], torch.tensor(0.6))
    assert torch.allclose(s[0, 1, 0, 0], torch.tensor(0.4))


def test_gradient_flows_through_sharpened_scribbles(fake_scribbleprompt):
    """C2: o sharpening preserva o fluxo de gradiente até os marcadores."""
    net = ScribblePromptingNetwork(scribble_mode="sharpened", scribble_temperature=10.0)
    image = torch.rand(1, 1, 32, 32)
    scribbles = torch.rand(1, 1, 32, 32, requires_grad=True)

    mask = net({"image": image, "scribbles": scribbles})
    mask.sum().backward()

    assert scribbles.grad is not None
    assert scribbles.grad.abs().sum() > 0
