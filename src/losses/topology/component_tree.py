"""Component tree nn.Module wrapper."""

from typing import Tuple

import torch
from torch.nn import Module

from .component_tree_function import ComponentTreeFunction


class ComponentTree(Module):
    """Differentiable component tree module.

    Wraps :class:`ComponentTreeFunction` as an ``nn.Module`` so it can be
    composed with other layers and registered as a submodule.

    Supported tree types:

    * ``"max"`` — max-tree (component tree of maxima)
    * ``"min"`` — min-tree (component tree of minima)
    * ``"tos"`` — tree of shapes (2-D images only)
    """

    _VALID_TREE_TYPES = ("max", "min", "tos")

    def __init__(self, tree_type: str) -> None:
        """Initializes ComponentTree.

        Args:
            tree_type: One of ``"max"``, ``"min"``, or ``"tos"``.

        Raises:
            ValueError: If ``tree_type`` is not a supported value.
        """
        super().__init__()
        if tree_type not in self._VALID_TREE_TYPES:
            raise ValueError(
                f"Unknown tree type '{tree_type}'. "
                f"Valid options: {', '.join(self._VALID_TREE_TYPES)}."
            )
        self.tree_type = tree_type

    def forward(self, graph, vertex_weights: torch.Tensor) -> Tuple:
        """Builds the component tree and returns the tree structure and altitudes.

        Args:
            graph: Higra adjacency graph whose topology matches
                ``vertex_weights``.
            vertex_weights: 1-D float tensor of per-vertex intensities.

        Returns:
            A tuple ``(tree, altitudes)`` where ``tree`` is the Higra tree
            object and ``altitudes`` is a differentiable float tensor of node
            altitudes.
        """
        altitudes = ComponentTreeFunction.apply(graph, vertex_weights, self.tree_type)
        return altitudes.tree, altitudes
