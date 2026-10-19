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
    mask = inputs.get("mask_input")
    if mask is None:
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


def test_positive_mode_zeroes_negative_and_cuts_below_threshold(fake_scribbleprompt):
    net = ScribblePromptingNetwork(scribble_mode="positive", positive_threshold=0.5)
    scribbles = torch.tensor([[[[0.0, 0.01, 0.5, 0.75, 0.9, 1.0]]]])

    s = net._prepare_scribbles(scribbles, torch.device("cpu"))

    assert torch.equal(s[0, 1], torch.zeros(1, 6))                 # sem canal negativo
    assert torch.equal(s[0, 0, 0, :3], torch.zeros(3))              # exatamente zero até τ
    assert torch.allclose(s[0, 0, 0, 3:], torch.tensor([0.5, 0.8, 1.0]))  # (s − τ)/(1 − τ)


def test_positive_mode_gradient_flows_only_above_threshold(fake_scribbleprompt):
    net = ScribblePromptingNetwork(scribble_mode="positive", positive_threshold=0.5)
    scribbles = torch.tensor([[[[0.2, 0.8]]]], requires_grad=True)

    net._prepare_scribbles(scribbles, torch.device("cpu"))[:, 0].sum().backward()

    assert torch.allclose(scribbles.grad, torch.tensor([[[[0.0, 2.0]]]]))  # 1/(1 − τ) acima de τ


def test_scribble_options_are_validated_at_construction(fake_scribbleprompt):
    with pytest.raises(ValueError, match="scribble_mode"):
        ScribblePromptingNetwork(scribble_mode="negative")
    with pytest.raises(ValueError, match="positive_threshold"):
        ScribblePromptingNetwork(scribble_mode="positive", positive_threshold=1.0)


def test_input_size_sets_the_network_resolution(fake_scribbleprompt):
    net = ScribblePromptingNetwork(input_size=(256, 256), resize_output=False)
    seen = []
    net.unet.register_forward_hook(lambda module, inputs, output: seen.append(tuple(inputs[0].shape[-2:])))

    mask = net({"image": torch.rand(1, 1, 64, 64), "scribbles": torch.rand(1, 1, 64, 64)})

    assert net.input_size == (256, 256)
    assert seen == [(256, 256)] and mask.shape[-2:] == (256, 256)
    assert ScribblePromptingNetwork().input_size == (128, 128)     # padrão do pacote


def test_input_size_must_be_divisible_by_16(fake_scribbleprompt):
    with pytest.raises(ValueError, match="divisíveis por 16"):
        ScribblePromptingNetwork(input_size=(250, 256))


def _unet_input(net, data):
    """Entrada (N, 5, h, w) que chega à UNet: [img, caixa, positivo, negativo, máscara]."""
    seen = {}
    handle = net.unet.register_forward_hook(lambda module, args, out: seen.setdefault("x", args[0].detach()))
    try:
        net(data)
    finally:
        handle.remove()
    return seen["x"]


def test_fixed_negative_replaces_the_mode_negative_channel(fake_scribbleprompt):
    net = ScribblePromptingNetwork(scribble_mode="positive", positive_threshold=0.5)
    scribbles = torch.full((1, 1, 128, 128), 0.75)
    negative = torch.zeros(1, 1, 128, 128)
    negative[..., :64] = 1.0

    x = _unet_input(net, {"image": torch.rand(1, 1, 128, 128), "scribbles": scribbles, "negative": negative})

    assert torch.allclose(x[:, 2], torch.full((1, 128, 128), 0.5))   # positivo: (0,75 − τ)/(1 − τ)
    assert torch.equal(x[:, 3:4], negative)                          # negativo: o canal fixo


def test_fixed_negative_keeps_the_gradient_to_the_scribbles(fake_scribbleprompt):
    net = ScribblePromptingNetwork(scribble_mode="positive", positive_threshold=0.5)
    scribbles = torch.full((1, 1, 32, 32), 0.8, requires_grad=True)

    net({"image": torch.rand(1, 1, 32, 32), "scribbles": scribbles, "negative": np.ones((32, 32), np.float32)}).sum().backward()

    assert scribbles.grad is not None and scribbles.grad.abs().sum() > 0


def test_fixed_negative_must_match_the_positive_shape(fake_scribbleprompt):
    net = ScribblePromptingNetwork(scribble_mode="positive")

    with pytest.raises(ValueError, match="negativo"):
        net({"image": torch.rand(1, 1, 32, 32), "scribbles": torch.rand(1, 1, 32, 32),
             "negative": torch.ones(1, 1, 16, 16)})


def test_scribble_channels_shows_what_the_mode_produces(fake_scribbleprompt):
    net = ScribblePromptingNetwork(scribble_mode="positive", positive_threshold=0.5)

    channels = net.scribble_channels(torch.tensor([[[[0.25, 0.75]]]]))

    assert channels.shape == (1, 2, 1, 2)
    assert torch.allclose(channels[0, 0, 0], torch.tensor([0.0, 0.5]))
    assert torch.equal(channels[0, 1], torch.zeros(1, 2))


def _fake_origin(monkeypatch, content, fail_after_write=False):
    """Substitui o download por uma escrita local de ``content`` e fixa o SHA-256 esperado de ``b"pesos"``."""
    import hashlib
    import urllib.request

    calls = []

    def fake_urlretrieve(url, dest):
        calls.append(url)
        with open(dest, "wb") as f:
            f.write(content)
        if fail_after_write:
            raise ConnectionError("conexão caiu")

    monkeypatch.setattr(urllib.request, "urlretrieve", fake_urlretrieve)
    monkeypatch.setitem(ScribblePromptingNetwork.CHECKPOINT_SHA256, "v1", hashlib.sha256(b"pesos").hexdigest())
    return calls


def test_download_checks_the_hash_and_reuses_a_verified_file(monkeypatch, tmp_path):
    calls = _fake_origin(monkeypatch, b"pesos")

    path = ScribblePromptingNetwork.download_checkpoint(str(tmp_path))
    again = ScribblePromptingNetwork.download_checkpoint(str(tmp_path))

    assert path == again and open(path, "rb").read() == b"pesos"
    assert len(calls) == 1                                     # a segunda chamada não baixa
    assert not list(tmp_path.glob("*.part"))


def test_download_rejects_a_different_file_and_leaves_nothing(monkeypatch, tmp_path):
    _fake_origin(monkeypatch, b"outro arquivo")

    with pytest.raises(RuntimeError, match="SHA-256"):
        ScribblePromptingNetwork.download_checkpoint(str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_interrupted_download_leaves_no_checkpoint(monkeypatch, tmp_path):
    _fake_origin(monkeypatch, b"pes", fail_after_write=True)

    with pytest.raises(ConnectionError):
        ScribblePromptingNetwork.download_checkpoint(str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_existing_corrupted_checkpoint_raises_instead_of_being_used(monkeypatch, tmp_path):
    calls = _fake_origin(monkeypatch, b"pesos")
    (tmp_path / "ScribblePrompt_unet_v1_nf192_res128.pt").write_bytes(b"pela metade")

    with pytest.raises(RuntimeError, match="Apague o arquivo"):
        ScribblePromptingNetwork.download_checkpoint(str(tmp_path))
    assert calls == []


def test_one_interaction_is_the_default_and_matches_a_single_pass(fake_scribbleprompt):
    data = {"image": torch.rand(1, 1, 128, 128), "scribbles": torch.rand(1, 1, 128, 128)}
    default = ScribblePromptingNetwork(scribble_mode="positive")
    single = ScribblePromptingNetwork(scribble_mode="positive", n_interactions=1)
    single.unet.load_state_dict(default.unet.state_dict())

    assert default.n_interactions == 1
    assert torch.equal(default(data), single(data))
    assert torch.equal(_unet_input(default, data)[:, 4], torch.zeros(1, 128, 128))   # mask_input vazio


def test_second_interaction_receives_the_first_logits_as_mask_input(fake_scribbleprompt):
    net = ScribblePromptingNetwork(scribble_mode="positive", n_interactions=2)
    data = {"image": torch.rand(1, 1, 128, 128), "scribbles": torch.rand(1, 1, 128, 128)}
    seen = []
    handle = net.unet.register_forward_hook(lambda m, args, out: seen.append((args[0].detach(), out.detach())))
    net(data)
    handle.remove()

    assert len(seen) == 2
    assert torch.equal(seen[0][0][:, 4], torch.zeros(1, 128, 128))
    assert torch.allclose(seen[1][0][:, 4:5], seen[0][1])                              # logits da 1ª passada


def test_gradient_flows_through_all_interactions(fake_scribbleprompt):
    net = ScribblePromptingNetwork(scribble_mode="positive", n_interactions=3)
    scribbles = torch.full((1, 1, 32, 32), 0.8, requires_grad=True)

    net({"image": torch.rand(1, 1, 32, 32), "scribbles": scribbles}).sum().backward()

    assert scribbles.grad is not None and scribbles.grad.abs().sum() > 0


def test_n_interactions_is_validated(fake_scribbleprompt):
    with pytest.raises(ValueError, match="n_interactions"):
        ScribblePromptingNetwork(n_interactions=0)
    with pytest.raises(ValueError, match="n_interactions"):
        ScribblePromptingNetwork(n_interactions=1.5)
