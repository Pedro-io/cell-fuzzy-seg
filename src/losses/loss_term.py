"""Abstract base class for composable loss terms."""

from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn


class LossTerm(nn.Module, ABC):
    """Interface for a single composable loss term.

    Each concrete term wraps one loss function and exposes a uniform
    ``compute(ctx)`` signature. The shared context dict carries all tensors
    that any term might need; each term pulls only what it requires.

    Expected context keys:
        - ``"markers"``: predicted marker tensor ``(N, C, H, W)``.
        - ``"distance_maps"``: distance map tensor ``(N, C, H, W)``.
        - ``"gt_masks"``: ground truth mask tensor ``(N, C, H, W)``.

    Subclasses must implement :attr:`name` and :meth:`compute`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique key used in the loss log returned by :class:`LossComposer`."""
        ...

    @abstractmethod
    def compute(self, ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Computes the loss from the shared context.

        Args:
            ctx: Shared tensor context. Required keys depend on the concrete
                term — see each subclass for details.

        Returns:
            Scalar loss tensor.
        """
        ...

    def forward(self, ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.compute(ctx)
