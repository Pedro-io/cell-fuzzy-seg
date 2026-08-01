"""Wrappers concretos de LossTerm para todas as funções de perda disponíveis."""

from typing import Dict

import torch

from .border_loss import BorderLoss
from .distance_map_loss import DistanceMapLoss
from .loss_term import LossTerm
from .not_too_thin_loss import NotTooThinLoss
from .object_size_loss import ObjectSizeLoss
from .rmse_loss import RMSELoss
from .soft_dice_loss import SoftDiceLoss
from .total_variation_loss import TotalVariationLoss


class SizeTerm(LossTerm):
    """Envolve :class:`ObjectSizeLoss`. Usa ``markers`` e ``gt_masks``."""

    def __init__(self, weight: float = 0.1) -> None:
        super().__init__()
        self._loss = ObjectSizeLoss(weight=weight)

    @property
    def name(self) -> str:
        return "size"

    def compute(self, ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._loss(ctx["markers"], ctx["gt_masks"])


class TVTerm(LossTerm):
    """Envolve :class:`TotalVariationLoss`. Usa ``markers`` e ``gt_masks``."""

    def __init__(self, weight: float = 0.1, power: int = 1) -> None:
        super().__init__()
        self._loss = TotalVariationLoss(weight=weight, power=power)

    @property
    def name(self) -> str:
        return "tv"

    def compute(self, ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._loss(ctx["markers"], ctx["gt_masks"])


class DMapTerm(LossTerm):
    """Envolve :class:`DistanceMapLoss`. Usa ``markers``, ``distance_maps`` e ``gt_masks``."""

    def __init__(self, weight: float = 0.1) -> None:
        super().__init__()
        self._loss = DistanceMapLoss(weight=weight)

    @property
    def name(self) -> str:
        return "dmap"

    def compute(self, ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._loss(ctx["markers"], ctx["distance_maps"], ctx["gt_masks"])

class BorderTerm(LossTerm):
    """Envolve :class:`BorderLoss`. Usa apenas ``markers``."""

    def __init__(self, weight: float = 1.0, border_size: int = 50) -> None:
        super().__init__()
        self._loss = BorderLoss(weight=weight, border_size=border_size)

    @property
    def name(self) -> str:
        return "border"

    def compute(self, ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._loss(ctx["markers"])


class DiceTerm(LossTerm):
    """Envolve :class:`SoftDiceLoss`. Usa ``markers`` e ``gt_masks``."""

    def __init__(self, epsilon: float = 1e-9) -> None:
        super().__init__()
        self._loss = SoftDiceLoss(epsilon=epsilon)

    @property
    def name(self) -> str:
        return "dice"

    def compute(self, ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._loss(ctx["markers"], ctx["gt_masks"])


class RMSETerm(LossTerm):
    """Envolve :class:`RMSELoss`. Usa ``markers`` e ``gt_masks``."""

    def __init__(self) -> None:
        super().__init__()
        self._loss = RMSELoss()

    @property
    def name(self) -> str:
        return "rmse"

    def compute(self, ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self._loss(ctx["markers"], ctx["gt_masks"])


class NotTooThinTerm(LossTerm):
    """Envolve :class:`NotTooThinLoss`. Usa apenas ``markers``.

    Aplica a perda a cada fatia ``(H, W)`` no batch ``(N, C, H, W)`` e retorna
    a média sobre todas as fatias.
    """

    def __init__(self, kernel: torch.Tensor, weight: float = 0.5) -> None:
        super().__init__()
        self._loss = NotTooThinLoss(kernel=kernel, weight=weight)

    @property
    def name(self) -> str:
        return "not_too_thin"

    def compute(self, ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        markers = ctx["markers"]
        n, c, h, w = markers.shape
        slices = [self._loss(markers[i, j]) for i in range(n) for j in range(c)]
        return sum(slices) / (n * c)


__all__ = [
    "SizeTerm",
    "TVTerm",
    "DMapTerm",
    "BorderTerm",
    "DiceTerm",
    "RMSETerm",
    "NotTooThinTerm",
]
