"""Topology-aware loss module based on component tree dynamics."""

from itertools import repeat
from multiprocessing import get_context
from typing import Callable, Optional

import torch
import torch.nn as nn
import higra as hg

from .attributes import attribute_max_altitudes, attribute_saddle_nodes
from .component_tree import ComponentTree


def _loss_ranked_selection(
    ranked_measure: torch.Tensor,
    num_positives: int,
    margin: float,
    power: int = 2,
) -> torch.Tensor:
    """Pushes the top ``num_positives`` values above ``margin`` and suppresses the rest.

    Args:
        ranked_measure: 1-D tensor sorted by importance (most important first).
        num_positives: Number of elements to push above the margin.
        margin: Target lower bound for the top elements.
        power: Exponent used in the hinge penalty.

    Returns:
        Scalar loss tensor.
    """
    if len(ranked_measure) <= num_positives:
        return torch.sum(torch.relu(margin - ranked_measure**power))
    return (
        torch.sum(torch.relu(margin - ranked_measure[:num_positives] ** power))
        + torch.sum(ranked_measure[num_positives:] ** power)
    )


def _loss_dynamics_max(
    marker: torch.Tensor,
    attribute_function: Callable,
) -> torch.Tensor:
    """Computes sorted dynamics of maxima in a single 2-D marker map.

    Dynamics measure how prominent each maximum is relative to its saddle
    node. Used as input for :func:`_loss_ranked_selection`.

    Args:
        marker: 2-D or 3-D (1, H, W) marker tensor for a single image.
        attribute_function: Callable ``(tree, altitudes) -> attribute_array``
            used to determine the saddle node of each maximum.

    Returns:
        1-D tensor of dynamics sorted in descending order.
    """
    max_tree = ComponentTree("max")
    marker = marker.squeeze(0)
    graph = hg.get_8_adjacency_implicit_graph(marker.shape)
    tree, altitudes = max_tree(graph, marker)
    altitudes_np = altitudes.detach().numpy()

    extrema_mask = hg.attribute_extrema(tree, altitudes_np)
    extrema_indices = torch.arange(tree.num_vertices())[torch.from_numpy(extrema_mask)]
    extrema_altitudes = altitudes[extrema_indices]

    attribute = attribute_function(tree, altitudes_np)
    pass_nodes = torch.from_numpy(attribute_saddle_nodes(tree, altitudes_np, attribute)[0])
    pass_altitudes = altitudes[pass_nodes[extrema_indices]]

    dynamics = extrema_altitudes - pass_altitudes
    sorted_dynamics, _ = torch.sort(dynamics, descending=True)
    return sorted_dynamics


class TopologyLoss(nn.Module):
    """Topology-aware regularization loss based on max-tree dynamics.

    Encourages predictions to have exactly ``num_target_maxima`` prominent
    maxima by penalizing deviations from the target component count. The
    dynamics of each maximum (altitude minus its saddle node altitude) are
    used as the relevance measure.

    Parallelizes per-channel computation across multiple CPU workers.
    """

    def __init__(
        self,
        weight: float,
        num_target_maxima: int,
        margin: float = 1.0,
        attribute_function: Optional[Callable] = None,
        power: int = 2,
        cpus: int = 2,
    ) -> None:
        """Initializes TopologyLoss.

        Args:
            weight: Scalar multiplier applied to the total loss.
            num_target_maxima: Desired number of prominent maxima per channel.
            margin: Minimum dynamics value required for the top maxima.
            attribute_function: Callable ``(tree, altitudes) -> attribute``
                used to locate saddle nodes. Defaults to
                :func:`~.attributes.attribute_max_altitudes`.
            power: Exponent for the ranked-selection hinge penalty.
            cpus: Number of parallel worker processes.
        """
        super().__init__()
        self.weight = weight
        self.num_target_maxima = num_target_maxima
        self.margin = margin
        self.attribute_function = attribute_function or attribute_max_altitudes
        self.power = power
        self.cpus = cpus

    def forward(self, markers: torch.Tensor) -> torch.Tensor:
        """Computes the topology loss over a batch of marker maps.

        Args:
            markers: Predicted marker tensor of shape ``(N, C, H, W)``.

        Returns:
            Scalar weighted topology loss averaged over all ``N * C`` channels.
        """
        n, c, h, w = markers.size()
        flat_markers = markers.view(n * c, h, w)
        arguments = list(zip(flat_markers.detach(), repeat(self.attribute_function)))

        with get_context("spawn").Pool(self.cpus) as pool:
            sorted_dynamics_list = pool.starmap(_loss_dynamics_max, arguments)

        total = torch.tensor(0.0)
        for dynamics in sorted_dynamics_list:
            total = total + _loss_ranked_selection(
                dynamics, self.num_target_maxima, self.margin, power=self.power
            )

        return self.weight * total / len(sorted_dynamics_list)
