"""Custom autograd Function for component tree construction."""

import numpy as np
import torch
import higra as hg
from torch.autograd import Function


class ComponentTreeFunction(Function):
    """Differentiable component tree construction via Higra.

    Supports max-tree, min-tree, and tree-of-shapes representations.
    Gradients are back-propagated from node altitudes to leaf vertex weights.
    """

    @staticmethod
    def forward(
        ctx,
        graph,
        vertex_weights: torch.Tensor,
        tree_type: str = "max",
        plateau_derivative: str = "single",
    ) -> torch.Tensor:
        """Builds a component tree and returns its node altitudes.

        The returned tensor carries a ``.tree`` attribute holding the Higra
        tree object, because ``torch.autograd.Function`` can only return
        tensors.

        Args:
            ctx: Autograd context for saving state used in the backward pass.
            graph: Higra adjacency graph matching the layout of
                ``vertex_weights``.
            vertex_weights: 1-D float tensor of per-vertex intensities.
            tree_type: Component tree variant. One of ``"max"``, ``"min"``, or
                ``"tos"``.
            plateau_derivative: Gradient distribution at plateaus. Use
                ``"single"`` to assign the gradient to one representative leaf
                per plateau or ``"full"`` to broadcast to all plateau leaves.

        Returns:
            Float tensor of node altitudes with shape
            ``(tree.num_vertices(),)``. The tensor carries a ``.tree``
            attribute.

        Raises:
            ValueError: If ``tree_type`` or ``plateau_derivative`` is unknown.
        """
        if tree_type == "max":
            tree, altitudes = hg.component_tree_max_tree(graph, vertex_weights.detach().numpy())
        elif tree_type == "min":
            tree, altitudes = hg.component_tree_min_tree(graph, vertex_weights.detach().numpy())
        elif tree_type == "tos":
            tree, altitudes = hg.component_tree_tree_of_shapes_image2d(vertex_weights)
        else:
            raise ValueError(f"Unknown tree type '{tree_type}'. Use 'max', 'min', or 'tos'.")

        if plateau_derivative == "full":
            use_full_plateau = True
        elif plateau_derivative == "single":
            use_full_plateau = False
        else:
            raise ValueError(
                f"Unknown plateau derivative '{plateau_derivative}'. Use 'single' or 'full'."
            )

        ctx.saved = (tree, graph, use_full_plateau)
        altitudes_tensor = torch.from_numpy(altitudes).clone().requires_grad_(True)
        altitudes_tensor.tree = tree
        return altitudes_tensor

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Propagates gradients from node altitudes back to vertex weights.

        Args:
            ctx: Autograd context populated during ``forward``.
            grad_output: Gradient tensor w.r.t. the output altitudes.

        Returns:
            A tuple of gradients aligned with the ``forward`` inputs. Only the
            ``vertex_weights`` gradient is non-None.
        """
        tree, graph, use_full_plateau = ctx.saved
        leaf_parents = tree.parents()[: tree.num_leaves()]

        if use_full_plateau:
            grad_in = grad_output[leaf_parents]
        else:
            _, indices = np.unique(leaf_parents, return_index=True)
            grad_in = torch.zeros(tree.num_leaves(), dtype=grad_output.dtype)
            grad_in[indices] = grad_output[leaf_parents[indices]]

        return None, hg.delinearize_vertex_weights(grad_in, graph), None, None
