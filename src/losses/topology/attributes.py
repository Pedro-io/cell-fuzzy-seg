"""Node attribute functions for component trees."""

import numpy as np
import higra as hg


def attribute_max_altitudes(tree, altitudes: np.ndarray) -> np.ndarray:
    """Computes the maximum descendant altitude for each tree node.

    The value of a node ``n`` equals the maximum altitude among all leaf
    descendants of ``n``.

    Args:
        tree: Higra component tree.
        altitudes: Array of node altitudes with shape
            ``(tree.num_vertices(),)``.

    Returns:
        Array of per-node max-descendant altitudes with the same shape as
        ``altitudes``.
    """
    return hg.accumulate_sequential(tree, altitudes[: tree.num_leaves()], hg.Accumulators.max)


def attribute_saddle_nodes(tree, altitudes: np.ndarray, attribute: np.ndarray):
    """Computes saddle and base nodes for each tree node under a given attribute.

    For a node ``n`` let ``an`` be the closest ancestor such that ``an`` has a
    child ``c`` with ``attr(c) > attr(ch(an → n))``. The saddle node is ``an``
    and the base node is ``ch(an → n)``.

    Args:
        tree: Higra component tree.
        altitudes: Array of node altitudes with shape
            ``(tree.num_vertices(),)``.
        attribute: Per-node attribute array with shape
            ``(tree.num_vertices(),)``.

    Returns:
        A tuple ``(pass_nodes, base_nodes)`` where each element is an integer
        array of shape ``(tree.num_vertices(),)`` indexing into the tree nodes.
    """
    max_child = hg.accumulate_parallel(tree, attribute, hg.Accumulators.max)
    main_branch = attribute == max_child[tree.parents()]
    main_branch[: tree.num_leaves()] = True

    node_indices = np.arange(tree.num_vertices())
    pass_nodes = hg.propagate_sequential(tree, node_indices[tree.parents()], main_branch)
    base_nodes = hg.propagate_sequential(tree, node_indices, main_branch)
    return pass_nodes, base_nodes
