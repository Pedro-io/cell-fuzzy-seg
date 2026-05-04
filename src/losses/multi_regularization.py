"""Multi-term regularization loss module."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .border_loss import BorderLoss
from .distance_map_loss import DistanceMapLoss
from .object_size_loss import ObjectSizeLoss
from .total_variation_loss import TotalVariationLoss
from .topology.attributes import attribute_max_altitudes
from .topology.topology_loss import TopologyLoss


class MultiRegularization(nn.Module):
    """Composite regularization combining multiple weighted penalty terms.

    Each term is active only when its corresponding weight is not ``None``.
    Supported terms: object size, total variation, distance map, and topology.

    Example::

        reg = MultiRegularization(size=0.1, tv=0.05, topo_weight=0.2)
        total_loss, log = reg(markers, distance_maps, gt_masks)
    """

    def __init__(
        self,
        size: Optional[float] = None,
        dmap: Optional[float] = None,
        tv: Optional[float] = None,
        tv_power: int = 2,
        topo_weight: Optional[float] = None,
        num_components: int = 3,
        cpus: int = 2,
    ) -> None:
        """Initializes MultiRegularization.

        Args:
            size: Weight for the object size regularization term. Disabled if
                ``None``.
            dmap: Weight for the distance map regularization term. Disabled if
                ``None``.
            tv: Weight for the total variation regularization term. Disabled if
                ``None``.
            tv_power: Exponent applied inside the total variation loss.
            topo_weight: Weight for the topology regularization term. Disabled
                if ``None``.
            num_components: Target number of prominent maxima per channel for
                the topology loss.
            cpus: Number of parallel workers used by the topology loss.
        """
        super().__init__()
        self._size_weight = size
        self._dmap_weight = dmap
        self._tv_weight = tv
        self._topo_weight = topo_weight

        if size is not None:
            self._size_loss = ObjectSizeLoss(weight=size)
        if tv is not None:
            self._tv_loss = TotalVariationLoss(weight=tv, power=tv_power)
        if dmap is not None:
            self._dmap_loss = DistanceMapLoss(weight=dmap)
        if topo_weight is not None:
            self._topo_loss = TopologyLoss(
                weight=topo_weight,
                num_target_maxima=num_components,
                attribute_function=attribute_max_altitudes,
                cpus=cpus,
            )

    def forward(
        self,
        markers: torch.Tensor,
        distance_maps: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Computes the total regularization loss and a per-term breakdown.

        Args:
            markers: Predicted marker tensor of shape ``(N, C, H, W)``.
            distance_maps: Distance map tensor of shape ``(N, C, H, W)``.
            gt_masks: Ground truth mask tensor of shape ``(N, C, H, W)``.

        Returns:
            A tuple ``(total_loss, loss_log)`` where ``loss_log`` maps each
            term name to its individual scalar tensor. Inactive terms are
            stored as ``torch.tensor(0.0)``.
        """
        loss_log: Dict[str, torch.Tensor] = {
            "size": torch.tensor(0.0),
            "tv": torch.tensor(0.0),
            "dmap": torch.tensor(0.0),
            "topo": torch.tensor(0.0),
        }
        active_losses = []

        if self._size_weight is not None:
            size_loss = self._size_loss(markers, gt_masks)
            active_losses.append(size_loss)
            loss_log["size"] = size_loss

        if self._tv_weight is not None:
            tv_loss = self._tv_loss(markers, gt_masks)
            active_losses.append(tv_loss)
            loss_log["tv"] = tv_loss

        if self._dmap_weight is not None:
            dmap_loss = self._dmap_loss(markers, distance_maps, gt_masks)
            active_losses.append(dmap_loss)
            loss_log["dmap"] = dmap_loss

        if self._topo_weight is not None:
            topo_loss = self._topo_loss(markers)
            active_losses.append(topo_loss)
            loss_log["topo"] = topo_loss

        total_loss = sum(active_losses) if active_losses else torch.tensor(0.0)
        return total_loss, loss_log
