"""Strategy-based composition of LossTerm instances."""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from .loss_term import LossTerm


class LossComposer(nn.Module):
    """Composes an arbitrary list of :class:`LossTerm` instances into a single loss.

    Terms are registered as PyTorch submodules via ``nn.ModuleList``, so their
    parameters and buffers are properly tracked.  The active set is entirely
    determined at construction time — swap the list to run a different
    experiment without touching source code.

    Term names must be unique within a composer; duplicate names cause the
    earlier entry to be overwritten in the log.

    Example::

        composer = LossComposer([SizeTerm(0.1), TVTerm(0.05)])
        total, log = composer(markers, distance_maps, gt_masks)
        # log = {"size": tensor, "tv": tensor}
    """

    def __init__(self, terms: List[LossTerm]) -> None:
        """Initializes LossComposer.

        Args:
            terms: List of :class:`LossTerm` instances to compose.
        """
        super().__init__()
        self._terms = nn.ModuleList(terms)

    @property
    def terms(self) -> List[LossTerm]:
        return list(self._terms)

    def forward(
        self,
        markers: torch.Tensor,
        distance_maps: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Computes the total loss and a per-term log.

        Args:
            markers: Predicted marker tensor of shape ``(N, C, H, W)``.
            distance_maps: Distance map tensor of shape ``(N, C, H, W)``.
            gt_masks: Ground truth mask tensor of shape ``(N, C, H, W)``.

        Returns:
            A tuple ``(total_loss, loss_log)`` where ``loss_log`` maps each
            term's :attr:`~LossTerm.name` to its individual scalar tensor.
        """
        ctx: Dict[str, torch.Tensor] = {
            "markers": markers,
            "distance_maps": distance_maps,
            "gt_masks": gt_masks,
        }
        log: Dict[str, torch.Tensor] = {}
        active = []

        for term in self._terms:
            val = term(ctx)
            log[term.name] = val
            active.append(val)

        total = sum(active) if active else torch.tensor(0.0)
        return total, log
