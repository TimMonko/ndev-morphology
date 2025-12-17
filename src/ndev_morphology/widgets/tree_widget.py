"""
Widget for directional tree analysis.

Visualizes skeleton as a directed tree from soma, showing branch orders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from magicgui import magic_factory

if TYPE_CHECKING:
    import napari

__all__ = ['directed_tree_analysis']


# Colorblind-friendly palette for branch orders (Wong 2011)
BRANCH_ORDER_COLORS = {
    1: [0.0, 0.45, 0.70, 1.0],  # Blue - primary
    2: [0.90, 0.62, 0.0, 1.0],  # Orange - secondary
    3: [0.0, 0.62, 0.45, 1.0],  # Bluish green - tertiary
    4: [0.80, 0.47, 0.65, 1.0],  # Reddish purple - quaternary
    5: [0.94, 0.89, 0.26, 1.0],  # Yellow - quinary
    0: [0.5, 0.5, 0.5, 0.5],  # Gray - not in tree
}


@magic_factory(
    call_button='Analyze Tree',
    skeleton_layer={
        'label': 'Skeleton Layer',
        'tooltip': 'Binary or labeled skeleton image.',
    },
    soma_labels_layer={
        'label': 'Soma Labels',
        'tooltip': (
            'Labels layer containing soma/cell body regions. '
            'Each unique label is analyzed separately.'
        ),
    },
    soma_label_id={
        'label': 'Soma Label ID',
        'min': 1,
        'tooltip': 'Which soma label to analyze (1 = first label).',
    },
    dilation_iterations={
        'label': 'Soma Exclusion Dilation',
        'min': 0,
        'max': 20,
        'tooltip': (
            'Pixels to dilate soma mask before excluding from skeleton. '
            'Larger values create cleaner separation of branches.'
        ),
    },
    color_by={
        'label': 'Color by',
        'choices': ['branch_order', 'strahler_order', 'distance_from_root'],
        'tooltip': 'Property to color branches by.',
    },
    edge_width={'min': 0.5, 'max': 10.0, 'step': 0.5},
)
def directed_tree_analysis(
    skeleton_layer: napari.layers.Labels,
    soma_labels_layer: napari.layers.Labels,
    soma_label_id: int = 1,
    dilation_iterations: int = 3,
    color_by: str = 'branch_order',
    edge_width: float = 3.0,
) -> list[napari.types.LayerDataTuple]:
    """
    Analyze skeleton as directed trees radiating from a soma.

    This widget handles the common neuroscience case where multiple
    dendrites/axons radiate outward from a central cell body (soma).
    Each radiating branch is analyzed as a separate directed tree.

    The workflow is:
    1. Exclude the soma region from the skeleton (creating disconnected fragments)
    2. For each fragment, find the node closest to soma as its root
    3. Analyze each fragment as a directed tree
    4. Aggregate statistics for all trees belonging to the neuron

    Parameters
    ----------
    skeleton_layer : Labels
        Binary or labeled skeleton image.
    soma_labels_layer : Labels
        Labels layer containing soma/cell body regions.
    soma_label_id : int
        Which soma label to analyze (1 = first label).
    dilation_iterations : int
        How much to dilate soma mask before exclusion.
    color_by : str
        Property to use for coloring branches.
    edge_width : float
        Width of branch lines.

    Returns
    -------
    list of LayerDataTuple
        Shapes layer with colored branches, Points layer for soma centroid.
    """
    from scipy import ndimage as ndi

    from ..tree import (
        compute_branch_order,
        compute_strahler_order,
        create_directed_trees_from_soma,
        find_longest_path,
    )

    skeleton_arr = np.asarray(skeleton_layer.data)
    soma_labels_arr = np.asarray(soma_labels_layer.data)
    scale = skeleton_layer.scale
    spacing = tuple(scale[-2:])

    if not np.any(skeleton_arr):
        raise ValueError('Skeleton is empty')

    if not np.any(soma_labels_arr == soma_label_id):
        raise ValueError(
            f'Soma label {soma_label_id} not found in soma labels layer'
        )

    # Create binary mask for the selected soma
    soma_mask = soma_labels_arr == soma_label_id

    # Get soma centroid
    soma_centroid = ndi.center_of_mass(soma_mask)
    soma_centroid = np.array(soma_centroid)

    # Create directed trees for all branches radiating from this soma
    trees = create_directed_trees_from_soma(
        skeleton_image=skeleton_arr,
        soma_mask=soma_mask,
        soma_centroid=soma_centroid,
        spacing=spacing,
        dilation_iterations=dilation_iterations,
    )

    if not trees:
        raise ValueError(
            'No skeleton branches found radiating from soma. '
            'Try reducing dilation_iterations or check skeleton/soma overlap.'
        )

    # Build paths for visualization from all trees
    paths = []
    properties = {
        'branch_order': [],
        'strahler_order': [],
        'distance_from_root': [],
        'branch_distance': [],
        'is_terminal': [],
        'tree_id': [],  # Which tree (radiating branch) this path belongs to
    }

    total_branches = 0
    total_tips = 0
    total_junctions = 0
    max_path_length = 0.0
    order_distribution = {}

    for tree_idx, tree in enumerate(trees):
        # Compute orders for this tree
        branch_orders = compute_branch_order(tree)
        strahler_orders = compute_strahler_order(tree)

        # Find longest path in this tree
        _, path_length = find_longest_path(tree)
        max_path_length = max(max_path_length, path_length)

        total_tips += tree.n_tips
        total_junctions += tree.n_junctions

        # Compute distance from root for each node
        node_distances = {}
        for node in tree.graph.nodes():
            node_distances[node] = tree.distance_to_node(node)

        # Add paths from this tree
        for u, v in tree.graph.edges():
            edge_data = tree.graph.get_edge_data(u, v)
            branch_dist = edge_data.get('branch_distance', 0.0)

            # Get path coordinates for this edge
            # skan stores path coordinates - we need to extract them
            path_idx = edge_data.get('summary_index')
            if path_idx is not None and path_idx < tree.skeleton.n_paths:
                coords = tree.skeleton.path_coordinates(path_idx)
            else:
                # Fallback: create simple line between nodes
                coords = np.array(
                    [
                        tree.skeleton.coordinates[u],
                        tree.skeleton.coordinates[v],
                    ]
                )

            if len(coords) >= 2:
                paths.append(coords)
                order = branch_orders.get((u, v), 0)
                properties['branch_order'].append(order)
                properties['strahler_order'].append(
                    strahler_orders.get((u, v), 0)
                )
                properties['distance_from_root'].append(
                    node_distances.get(u, 0.0)
                )
                properties['branch_distance'].append(branch_dist)
                properties['is_terminal'].append(tree.graph.out_degree(v) == 0)
                properties['tree_id'].append(tree_idx)

                total_branches += 1
                order_distribution[order] = (
                    order_distribution.get(order, 0) + 1
                )

    layers = []

    # Branches layer
    if paths:
        if color_by == 'branch_order':
            # Use categorical colors for orders
            edge_colors = []
            for order in properties['branch_order']:
                color = BRANCH_ORDER_COLORS.get(
                    order, BRANCH_ORDER_COLORS.get(5)
                )
                if order > 5:
                    # Cycle through colors for higher orders
                    cycle_order = ((order - 1) % 5) + 1
                    color = BRANCH_ORDER_COLORS.get(cycle_order)
                edge_colors.append(color)

            layers.append(
                (
                    paths,
                    {
                        'name': f'Directed Trees (soma {soma_label_id})',
                        'shape_type': 'path',
                        'properties': properties,
                        'edge_color': edge_colors,
                        'edge_width': edge_width,
                        'scale': scale,
                    },
                    'shapes',
                )
            )
        else:
            # Use colormap for continuous properties
            layers.append(
                (
                    paths,
                    {
                        'name': f'Directed Trees (soma {soma_label_id})',
                        'shape_type': 'path',
                        'properties': properties,
                        'edge_color': color_by,
                        'edge_colormap': 'viridis',
                        'edge_width': edge_width,
                        'scale': scale,
                    },
                    'shapes',
                )
            )

    # Soma centroid marker
    soma_centroid_scaled = soma_centroid * np.array(
        scale[-len(soma_centroid) :]
    )
    layers.append(
        (
            np.array([soma_centroid_scaled]),
            {
                'name': f'Soma Centroid (label {soma_label_id})',
                'size': 15,
                'face_color': [0.84, 0.37, 0.0, 1.0],  # Vermillion
                'border_color': 'white',
                'border_width': 0.2,
                'symbol': 'star',
                'scale': (1.0,) * len(scale),  # Already scaled
            },
            'points',
        )
    )

    # Print summary
    print('=' * 60)
    print(f'Directed Tree Analysis - Soma {soma_label_id}')
    print('=' * 60)
    print(f'  Soma centroid: ({soma_centroid[0]:.1f}, {soma_centroid[1]:.1f})')
    print(f'  Dilation applied: {dilation_iterations} pixels')
    print(f'  Radiating branches (trees): {len(trees)}')
    print(f'  Total branches: {total_branches}')
    print(f'  Total tips: {total_tips}')
    print(f'  Total junctions: {total_junctions}')
    print(f'  Longest path length: {max_path_length:.2f}')
    print()
    print('Branch Order Distribution:')
    for order in sorted(order_distribution.keys()):
        if order > 0:
            count = order_distribution[order]
            print(f'  Order {order}: {count} branches')
    print()
    print('Colors: Blue=1°, Orange=2°, Green=3°, Purple=4°, Yellow=5°')
    print('=' * 60)

    return layers
