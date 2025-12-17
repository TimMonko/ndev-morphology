"""
Directional tree analysis for skeleton structures.

This module provides tools for analyzing skeletons as directed trees
rooted at a soma/origin point. This enables:

- Branch ordering (primary, secondary, tertiary branches)
- Path analysis from soma to tips
- Axon/dendrite length measurements
- Directed graph traversal

Key Concepts
------------
- **Undirected skeleton**: skan represents skeletons as undirected graphs
  where edges are paths between junctions/endpoints.
- **Directed tree**: By specifying a root (soma), we can create a directed
  tree where edges flow from soma toward tips.
- **Branch order**: The "generation" of a branch. Primary branches connect
  directly to soma, secondary branch off primary, etc.
- **Centrifugal ordering**: Counting from soma outward (1, 2, 3...)
- **Strahler ordering**: Counting from tips inward (tips=1, merging increases)

This module uses networkx for graph operations, building on skan's
`skeleton_to_nx()` function.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
import pandas as pd
import skan
from skan.csr import skeleton_to_nx

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = [
    'DirectedTree',
    'BranchOrder',
    'create_directed_tree',
    'create_directed_trees_from_soma',
    'compute_branch_order',
    'compute_strahler_order',
    'find_longest_path',
    'get_paths_to_tips',
    'summarize_directed_tree',
]


class BranchOrder:
    """Branch order constants for semantic clarity."""

    PRIMARY = 1
    SECONDARY = 2
    TERTIARY = 3
    QUATERNARY = 4


@dataclass
class DirectedTree:
    """
    A skeleton represented as a directed tree from a root node.

    This class wraps a networkx DiGraph with additional metadata
    for morphological analysis.

    Attributes
    ----------
    graph : nx.DiGraph
        Directed graph where edges flow from root toward tips.
    root_node : int
        The node ID of the root (soma).
    root_coords : np.ndarray
        Coordinates of the root node.
    skeleton : skan.Skeleton
        The original skan Skeleton object.
    summary : pd.DataFrame
        Branch summary from skan.summarize().
    undirected_graph : nx.MultiGraph
        The original undirected graph from skeleton_to_nx().

    Properties
    ----------
    n_branches : int
        Number of branches (edges) in the tree.
    n_tips : int
        Number of terminal endpoints.
    n_junctions : int
        Number of branch points.
    tip_nodes : list[int]
        Node IDs of all tips (out-degree 0).
    junction_nodes : list[int]
        Node IDs of all junctions (out-degree > 1).
    """

    graph: nx.DiGraph
    root_node: int
    root_coords: np.ndarray
    skeleton: skan.Skeleton
    summary: pd.DataFrame
    undirected_graph: nx.MultiGraph
    _branch_orders: dict = field(default_factory=dict, repr=False)

    @property
    def n_branches(self) -> int:
        """Number of branches (edges) in the tree."""
        return self.graph.number_of_edges()

    @property
    def n_tips(self) -> int:
        """Number of terminal endpoints (out-degree 0, not root)."""
        return len(self.tip_nodes)

    @property
    def n_junctions(self) -> int:
        """Number of branch points (out-degree > 1)."""
        return len(self.junction_nodes)

    @property
    def tip_nodes(self) -> list[int]:
        """Node IDs of all tips (leaves of the tree)."""
        return [n for n in self.graph.nodes() if self.graph.out_degree(n) == 0]

    @property
    def junction_nodes(self) -> list[int]:
        """Node IDs of branch points (out-degree > 1)."""
        return [n for n in self.graph.nodes() if self.graph.out_degree(n) > 1]

    @property
    def primary_branches(self) -> list[tuple[int, int]]:
        """Edges that are primary branches (directly from root)."""
        return list(self.graph.out_edges(self.root_node))

    def get_branch_order(self, edge: tuple[int, int]) -> int:
        """
        Get the centrifugal branch order of an edge.

        Parameters
        ----------
        edge : tuple[int, int]
            Edge as (source, target) node IDs.

        Returns
        -------
        int
            Branch order (1=primary, 2=secondary, etc.)
        """
        if not self._branch_orders:
            self._branch_orders = compute_branch_order(self)
        return self._branch_orders.get(edge, 0)

    def path_to_node(self, target: int) -> list[int]:
        """
        Get the path from root to a target node.

        Parameters
        ----------
        target : int
            Target node ID.

        Returns
        -------
        list[int]
            List of node IDs from root to target.

        Raises
        ------
        nx.NodeNotFound
            If target is not in the directed tree.
        nx.NetworkXNoPath
            If no path exists (shouldn't happen in a tree).
        """
        if target not in self.graph:
            raise nx.NodeNotFound(
                f'Node {target} is not in the directed tree. '
                f'It may be in a disconnected skeleton component.'
            )
        return nx.shortest_path(self.graph, self.root_node, target)

    def distance_to_node(self, target: int) -> float:
        """
        Get the total path distance from root to a target node.

        Parameters
        ----------
        target : int
            Target node ID.

        Returns
        -------
        float
            Sum of branch distances along the path.

        Raises
        ------
        nx.NodeNotFound
            If target is not in the directed tree.
        """
        if target not in self.graph:
            raise nx.NodeNotFound(
                f'Node {target} is not in the directed tree.'
            )
        path = self.path_to_node(target)
        total = 0.0
        for i in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[i], path[i + 1])
            total += edge_data.get('branch_distance', 0.0)
        return total

    def subtree_from_node(self, node: int) -> nx.DiGraph:
        """
        Get the subtree rooted at a given node.

        Parameters
        ----------
        node : int
            Node ID to use as subtree root.

        Returns
        -------
        nx.DiGraph
            Subgraph containing node and all its descendants.
        """
        descendants = nx.descendants(self.graph, node)
        descendants.add(node)
        return self.graph.subgraph(descendants).copy()


def create_directed_tree(
    skeleton: skan.Skeleton,
    root_coords: ArrayLike,
    *,
    summary: pd.DataFrame | None = None,
    skeleton_id: int | None = None,
) -> DirectedTree:
    """
    Create a directed tree from a skeleton rooted at specified coordinates.

    Parameters
    ----------
    skeleton : skan.Skeleton
        The skeleton to analyze.
    root_coords : ArrayLike
        Coordinates of the root (soma) position. The closest skeleton
        node to these coordinates becomes the root.
    summary : pd.DataFrame, optional
        Pre-computed summary from skan.summarize(). If not provided,
        it will be computed.
    skeleton_id : int, optional
        If provided, only consider nodes from this skeleton component.
        This is useful when analyzing multi-cell images where each cell
        has a separate skeleton_id in the skan summary.

    Returns
    -------
    DirectedTree
        A directed tree representation of the skeleton.

    Examples
    --------
    >>> import skan
    >>> from ndev_morphology.tree import create_directed_tree
    >>> skeleton = skan.Skeleton(skeleton_image, spacing=(0.2, 0.2))
    >>> soma_centroid = [100, 150]  # y, x coordinates
    >>> tree = create_directed_tree(skeleton, soma_centroid)
    >>> print(f"Primary branches: {len(tree.primary_branches)}")
    >>> print(f"Tips: {tree.n_tips}")
    """
    root_coords = np.asarray(root_coords)

    # Get or compute summary
    if summary is None:
        summary = skan.summarize(skeleton, separator='_')

    # Convert to networkx
    undirected = skeleton_to_nx(skeleton, summary)

    # Find the root node
    if skeleton_id is not None:
        # Filter to nodes in the specified skeleton component
        root_node = _find_root_node_for_skeleton(
            skeleton, summary, root_coords, skeleton_id
        )
    else:
        # Find the closest node to root_coords (original behavior)
        root_node = _find_closest_node(skeleton.coordinates, root_coords)

    # Create directed tree via BFS from root
    directed = nx.bfs_tree(undirected, root_node)

    # Copy edge attributes from undirected graph
    _copy_edge_attributes(undirected, directed, summary)

    return DirectedTree(
        graph=directed,
        root_node=root_node,
        root_coords=root_coords,
        skeleton=skeleton,
        summary=summary,
        undirected_graph=undirected,
    )


def _find_root_node_for_skeleton(
    skeleton: skan.Skeleton,
    summary: pd.DataFrame,
    target_coords: np.ndarray,
    skeleton_id: int,
) -> int:
    """
    Find the root node for a specific skeleton component.

    Parameters
    ----------
    skeleton : skan.Skeleton
        The skeleton object.
    summary : pd.DataFrame
        Branch summary from skan.summarize().
    target_coords : np.ndarray
        Target coordinates (soma centroid).
    skeleton_id : int
        The skeleton component ID to search within.

    Returns
    -------
    int
        Node ID of the root node.

    Raises
    ------
    ValueError
        If no nodes found for the given skeleton_id.
    """
    # Get all node IDs that belong to this skeleton
    skel_rows = summary[summary['skeleton_id'] == skeleton_id]
    if len(skel_rows) == 0:
        raise ValueError(f'No branches found for skeleton_id={skeleton_id}')

    # Collect all unique node IDs from this skeleton
    node_ids = set()
    for _, row in skel_rows.iterrows():
        node_ids.add(int(row['node_id_src']))
        node_ids.add(int(row['node_id_dst']))

    node_ids = list(node_ids)

    # Get coordinates of these nodes
    node_coords = skeleton.coordinates[node_ids]

    # Find closest to target
    distances = np.linalg.norm(node_coords - target_coords, axis=1)
    closest_idx = np.argmin(distances)

    return node_ids[closest_idx]


def _find_closest_node(
    coordinates: np.ndarray,
    target: np.ndarray,
) -> int:
    """Find the node index closest to target coordinates."""
    # coordinates shape: (n_nodes, ndim)
    distances = np.linalg.norm(coordinates - target, axis=1)
    return int(np.argmin(distances))


def _copy_edge_attributes(
    undirected: nx.MultiGraph,
    directed: nx.DiGraph,
    summary: pd.DataFrame,
) -> None:
    """Copy edge attributes from undirected graph to directed graph."""
    # Build lookup from node pairs to summary rows
    edge_lookup = {}
    for _idx, row in summary.iterrows():
        src, dst = int(row['node_id_src']), int(row['node_id_dst'])
        edge_lookup[(src, dst)] = row
        edge_lookup[(dst, src)] = row  # Both directions

    for u, v in directed.edges():
        # Try to find matching row in summary
        if (u, v) in edge_lookup:
            row = edge_lookup[(u, v)]
            directed[u][v]['branch_distance'] = row['branch_distance']
            directed[u][v]['euclidean_distance'] = row['euclidean_distance']
            directed[u][v]['branch_type'] = row['branch_type']
            directed[u][v]['summary_index'] = _idx

        # Also copy path coordinates from undirected if available
        if undirected.has_edge(u, v):
            edge_data = undirected.get_edge_data(u, v)
            if edge_data:
                # MultiGraph returns dict of edge keys
                first_edge = list(edge_data.values())[0]
                if 'path' in first_edge:
                    directed[u][v]['path'] = first_edge['path']


def create_directed_trees_from_soma(
    skeleton_image: ArrayLike,
    soma_mask: ArrayLike,
    soma_centroid: ArrayLike,
    *,
    spacing: tuple[float, ...] = (1.0, 1.0),
    dilation_iterations: int = 2,
) -> list[DirectedTree]:
    """
    Create directed trees for all branches radiating from a soma.

    This function handles the case where multiple skeleton branches
    radiate outward from a soma region. It:

    1. Excludes the soma region from the skeleton (creating disconnected fragments)
    2. For each fragment, finds the node closest to the soma as its root
    3. Creates a directed tree for each fragment

    Parameters
    ----------
    skeleton_image : ArrayLike
        Binary or labeled skeleton image.
    soma_mask : ArrayLike
        Binary mask of the soma region. This region will be excluded
        from the skeleton before analysis.
    soma_centroid : ArrayLike
        Centroid of the soma (y, x) or (z, y, x). Used to find root nodes.
    spacing : tuple of float, optional
        Physical pixel spacing (default: (1.0, 1.0)).
    dilation_iterations : int, optional
        How much to dilate the soma mask before exclusion (default: 2).
        This ensures clean separation of branches.

    Returns
    -------
    list of DirectedTree
        One DirectedTree for each disconnected skeleton fragment radiating
        from the soma. Empty list if no skeleton fragments found.

    Examples
    --------
    >>> from ndev_morphology.tree import create_directed_trees_from_soma
    >>> # skeleton_image: binary skeleton of a neuron
    >>> # soma_mask: binary mask of the cell body
    >>> # soma_centroid: (y, x) center of the soma
    >>> trees = create_directed_trees_from_soma(
    ...     skeleton_image, soma_mask, soma_centroid,
    ...     spacing=(0.2, 0.2), dilation_iterations=3
    ... )
    >>> print(f"Found {len(trees)} branches radiating from soma")
    >>> for i, tree in enumerate(trees):
    ...     print(f"  Branch {i+1}: {tree.n_tips} tips, order {tree.n_junctions} junctions")

    Notes
    -----
    This function is designed for the common neuroscience case where:
    - A neuron has a central cell body (soma)
    - Multiple dendrites/axons radiate outward from the soma
    - Each radiating branch should be analyzed as a separate tree
    - All branches belong to the same neuron (same label_id)
    """
    from .skeleton import exclude_region_from_skeleton

    skeleton_image = np.asarray(skeleton_image)
    soma_mask = np.asarray(soma_mask).astype(bool)
    soma_centroid = np.asarray(soma_centroid)

    # Exclude soma region from skeleton
    skeleton_no_soma = exclude_region_from_skeleton(
        skeleton_image, soma_mask, dilation_iterations=dilation_iterations
    )

    # Check if anything remains
    if not np.any(skeleton_no_soma):
        return []

    # Create skan skeleton from the remaining fragments
    # Convert to binary first (in case it was labeled)
    skeleton_binary = (skeleton_no_soma > 0).astype(np.uint8)
    skel = skan.Skeleton(skeleton_binary, spacing=spacing)

    if skel.n_paths == 0:
        return []

    # Get summary and undirected graph
    summary = skan.summarize(skel, separator='_')
    undirected = skeleton_to_nx(skel, summary)

    # Find connected components - each is a separate tree
    components = list(nx.connected_components(undirected))

    trees = []
    for component_nodes in components:
        if len(component_nodes) < 2:
            # Skip single-node components
            continue

        # Find the node in this component closest to soma centroid
        component_nodes_list = list(component_nodes)
        node_coords = skel.coordinates[component_nodes_list]
        distances = np.linalg.norm(node_coords - soma_centroid, axis=1)
        root_idx = np.argmin(distances)
        root_node = component_nodes_list[root_idx]

        # Create directed tree for this component
        directed = nx.bfs_tree(undirected.subgraph(component_nodes), root_node)

        # Copy edge attributes
        _copy_edge_attributes(undirected, directed, summary)

        tree = DirectedTree(
            graph=directed,
            root_node=root_node,
            root_coords=soma_centroid,
            skeleton=skel,
            summary=summary,
            undirected_graph=undirected,
        )
        trees.append(tree)

    return trees


def compute_branch_order(tree: DirectedTree) -> dict[tuple[int, int], int]:
    """
    Compute centrifugal branch order for all edges.

    Branch order is computed as:
    - Primary (1): Edges directly from root
    - Secondary (2): Edges from nodes reached by primary branches
    - Tertiary (3): Edges from nodes reached by secondary branches
    - etc.

    Parameters
    ----------
    tree : DirectedTree
        The directed tree to analyze.

    Returns
    -------
    dict
        Mapping of (source, target) edge tuples to branch order (int).

    Examples
    --------
    >>> orders = compute_branch_order(tree)
    >>> primary = [e for e, o in orders.items() if o == 1]
    >>> secondary = [e for e, o in orders.items() if o == 2]
    """
    orders = {}

    # BFS from root, tracking depth
    for edge in nx.bfs_edges(tree.graph, tree.root_node):
        src, dst = edge
        if src == tree.root_node:
            orders[edge] = BranchOrder.PRIMARY
        else:
            # Find parent edge order
            parent_edges = [(u, v) for u, v in orders if v == src]
            if parent_edges:
                parent_order = orders[parent_edges[0]]
                # If src is a junction (has multiple children), increment order
                if tree.graph.out_degree(src) > 1:
                    orders[edge] = parent_order + 1
                else:
                    # Same order as parent (continuation)
                    orders[edge] = parent_order
            else:
                orders[edge] = BranchOrder.PRIMARY

    return orders


def find_longest_path(tree: DirectedTree) -> tuple[list[int], float]:
    """
    Find the longest path from root to any tip.

    This represents the "main axis" of the structure (e.g., axon length).

    Parameters
    ----------
    tree : DirectedTree
        The directed tree to analyze.

    Returns
    -------
    path : list[int]
        Node IDs along the longest path.
    length : float
        Total path length in physical units.

    Examples
    --------
    >>> path, length = find_longest_path(tree)
    >>> print(f"Axon length: {length:.2f} µm")
    """
    longest_path = []
    longest_length = 0.0

    for tip in tree.tip_nodes:
        path = tree.path_to_node(tip)
        length = tree.distance_to_node(tip)

        if length > longest_length:
            longest_length = length
            longest_path = path

    return longest_path, longest_length


def get_paths_to_tips(
    tree: DirectedTree,
) -> list[dict]:
    """
    Get all paths from root to tips with their properties.

    Parameters
    ----------
    tree : DirectedTree
        The directed tree to analyze.

    Returns
    -------
    list of dict
        Each dict contains:
        - 'tip_node': int, the tip node ID
        - 'path': list[int], node IDs from root to tip
        - 'length': float, total path length
        - 'n_branches': int, number of branches traversed
        - 'max_order': int, maximum branch order along path
    """
    paths = []

    for tip in tree.tip_nodes:
        path = tree.path_to_node(tip)
        length = tree.distance_to_node(tip)

        # Count branches and max order
        n_branches = len(path) - 1
        max_order = 0
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            order = tree.get_branch_order(edge)
            max_order = max(max_order, order)

        paths.append(
            {
                'tip_node': tip,
                'path': path,
                'length': length,
                'n_branches': n_branches,
                'max_order': max_order,
            }
        )

    # Sort by length descending
    paths.sort(key=lambda x: x['length'], reverse=True)

    return paths


def compute_strahler_order(tree: DirectedTree) -> dict[tuple[int, int], int]:
    """
    Compute Strahler order for all edges.

    Strahler ordering starts from tips:
    - Tips have order 1
    - When two branches of order n merge, result is order n+1
    - When branches of different orders merge, result is max order

    Parameters
    ----------
    tree : DirectedTree
        The directed tree to analyze.

    Returns
    -------
    dict
        Mapping of (source, target) edge tuples to Strahler order (int).

    Notes
    -----
    Strahler order is commonly used in hydrology and neuroscience to
    describe branching complexity. Higher orders indicate more complex
    upstream structure.
    """
    # We need to process in reverse topological order (tips first)
    orders = {}

    # Get reverse topological order
    try:
        topo_order = list(nx.topological_sort(tree.graph))
    except nx.NetworkXUnfeasible:
        # Graph has cycles - shouldn't happen for a tree
        return orders

    # Process from tips (end of topo order) to root
    for node in reversed(topo_order):
        out_edges = list(tree.graph.out_edges(node))

        if not out_edges:
            # Tip node - no outgoing edges to label
            continue

        # Get orders of child edges
        child_orders = []
        for _, child in out_edges:
            # Find edge from this node to child
            child_out = list(tree.graph.out_edges(child))
            if child_out:
                child_edge_orders = [orders.get(e, 1) for e in child_out]
                child_orders.append(
                    max(child_edge_orders) if child_edge_orders else 1
                )
            else:
                # Child is a tip
                child_orders.append(1)

        # Assign Strahler order to edges from this node
        for edge in out_edges:
            if len(child_orders) == 0:
                orders[edge] = 1
            elif len(child_orders) == 1:
                orders[edge] = child_orders[0]
            else:
                # Multiple children - Strahler rules
                max_child = max(child_orders)
                count_max = child_orders.count(max_child)
                if count_max >= 2:
                    orders[edge] = max_child + 1
                else:
                    orders[edge] = max_child

    return orders


def summarize_directed_tree(tree: DirectedTree) -> pd.DataFrame:
    """
    Create a summary DataFrame with directional branch information.

    This extends skan's summary with:
    - branch_order: Centrifugal order from soma
    - strahler_order: Strahler order from tips
    - distance_from_root: Path distance from soma to branch start
    - is_terminal: Whether branch ends at a tip
    - in_tree: Whether the branch is in the directed tree (connected to soma)

    Parameters
    ----------
    tree : DirectedTree
        The directed tree to summarize.

    Returns
    -------
    pd.DataFrame
        Extended summary with directional metrics.

    Notes
    -----
    Branches from disconnected skeleton components (not connected to the
    root/soma) will have NaN values for distance_from_root and order 0.
    """
    # Start with skan summary
    df = tree.summary.copy()

    # Compute orders
    centrifugal = compute_branch_order(tree)
    strahler = compute_strahler_order(tree)

    # Get set of nodes in tree for fast lookup
    tree_nodes = set(tree.graph.nodes())

    # Add new columns
    branch_orders = []
    strahler_orders = []
    distances_from_root = []
    is_terminal = []
    in_tree = []

    for _idx, row in df.iterrows():
        src = int(row['node_id_src'])
        dst = int(row['node_id_dst'])

        # Check if this branch is in the directed tree
        src_in_tree = src in tree_nodes
        dst_in_tree = dst in tree_nodes
        branch_in_tree = src_in_tree and dst_in_tree
        in_tree.append(branch_in_tree)

        # Try both edge directions
        edge = (src, dst) if (src, dst) in centrifugal else (dst, src)

        branch_orders.append(centrifugal.get(edge, 0))
        strahler_orders.append(strahler.get(edge, 0))

        # Distance from root to start of branch
        if branch_in_tree:
            try:
                dist = tree.distance_to_node(src)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                dist = np.nan
        else:
            dist = np.nan
        distances_from_root.append(dist)

        # Is this a terminal branch?
        is_terminal.append(dst in tree.tip_nodes or src in tree.tip_nodes)

    df['branch_order'] = branch_orders
    df['strahler_order'] = strahler_orders
    df['distance_from_root'] = distances_from_root
    df['is_terminal'] = is_terminal
    df['in_tree'] = in_tree

    return df
