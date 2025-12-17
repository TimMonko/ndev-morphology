"""
Widget for directional tree analysis.

Visualizes skeleton as a directed tree from soma, showing branch orders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import skan
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


def _get_skeleton_id_for_soma(
    skeleton_arr: np.ndarray,
    soma_coord: np.ndarray,
    search_radius: int = 50,
) -> int | None:
    """
    Find the skeleton label ID that the soma connects to.

    Searches outward from the soma centroid to find the nearest skeleton
    pixel and returns its label value.

    Parameters
    ----------
    skeleton_arr : np.ndarray
        Labeled skeleton image (each cell has unique label).
    soma_coord : np.ndarray
        Soma centroid coordinates (y, x) or (z, y, x).
    search_radius : int
        Maximum distance to search for skeleton pixels.

    Returns
    -------
    int or None
        The skeleton label ID, or None if no skeleton found nearby.
    """
    # Create distance transform from soma point
    ndim = skeleton_arr.ndim
    soma_int = tuple(int(c) for c in soma_coord[-ndim:])

    # Check if soma is within bounds
    for i, c in enumerate(soma_int):
        if c < 0 or c >= skeleton_arr.shape[i]:
            return None

    # Search in expanding circles/spheres
    for radius in range(1, search_radius + 1):
        # Create a mask for points at approximately this radius
        slices = []
        for i, c in enumerate(soma_int):
            start = max(0, c - radius)
            stop = min(skeleton_arr.shape[i], c + radius + 1)
            slices.append(slice(start, stop))

        region = skeleton_arr[tuple(slices)]
        if np.any(region > 0):
            # Found skeleton pixels - return the most common label
            labels_found = region[region > 0]
            # Return the label closest to the soma
            # (for now, just return most common in the region)
            unique, counts = np.unique(labels_found, return_counts=True)
            return int(unique[np.argmax(counts)])

    return None


@magic_factory(
    call_button='Analyze Tree',
    skeleton_layer={
        'label': 'Skeleton Layer',
        'tooltip': 'Labeled skeleton from skeletonize_labels widget.',
    },
    soma_points={
        'label': 'Soma Points',
        'nullable': True,
        'tooltip': (
            'Points layer with soma centroids. Should have label_id property '
            'from Soma Detection widget.'
        ),
    },
    soma_index={
        'label': 'Soma Index',
        'min': 0,
        'tooltip': 'Which soma point to analyze (0 = first point).',
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
    soma_points: napari.layers.Points | None = None,
    soma_index: int = 0,
    color_by: str = 'branch_order',
    edge_width: float = 3.0,
) -> list[napari.types.LayerDataTuple]:
    """
    Analyze skeleton as a directed tree from soma.

    Creates a directed tree rooted at the soma position, then visualizes
    branches colored by their order (primary, secondary, etc.) or other
    directional properties.

    This widget matches the soma to the correct skeleton component using
    either the soma's label_id property or by finding the nearest skeleton.

    Parameters
    ----------
    skeleton_layer : Labels
        Labeled skeleton image (from skeletonize_labels widget).
        Each cell should have a unique label value.
    soma_points : Points, optional
        Points layer with soma centroids. Should have 'label_id' property
        from the Soma Detection widget.
    soma_index : int
        Which soma point to analyze (0-indexed).
    color_by : str
        Property to use for coloring.
    edge_width : float
        Width of branch lines.

    Returns
    -------
    list of LayerDataTuple
        Shapes layer with colored branches, Points layer for soma.
    """
    from ..tree import (
        create_directed_tree,
        find_longest_path,
        summarize_directed_tree,
    )

    skeleton_arr = np.asarray(skeleton_layer.data)
    scale = skeleton_layer.scale
    spacing = tuple(scale[-2:])

    if not np.any(skeleton_arr):
        raise ValueError('Skeleton is empty')

    # Get soma coordinates
    if soma_points is None:
        raise ValueError(
            'Soma points layer required. Use Soma Detection widget first.'
        )

    soma_data = np.asarray(soma_points.data)
    if len(soma_data) == 0:
        raise ValueError('Soma points layer is empty')

    if soma_index >= len(soma_data):
        raise ValueError(
            f'Soma index {soma_index} out of range (only {len(soma_data)} points)'
        )

    # Get the selected soma point
    soma_coords_world = soma_data[soma_index]

    # Convert to pixel coordinates if needed
    soma_scale = soma_points.scale
    if soma_scale is not None and not np.allclose(soma_scale, 1.0):
        # Points are in world coordinates, convert to pixel
        soma_coords = soma_coords_world / np.array(
            soma_scale[-len(soma_coords_world) :]
        )
    else:
        soma_coords = soma_coords_world.copy()

    # Try to get skeleton_id from soma properties
    skeleton_id = None
    if hasattr(soma_points, 'properties') and soma_points.properties:
        props = soma_points.properties
        if 'label_id' in props:
            label_ids = props['label_id']
            if (
                hasattr(label_ids, '__getitem__')
                and len(label_ids) > soma_index
            ):
                skeleton_id = int(label_ids[soma_index])
                print(f'Using label_id={skeleton_id} from soma properties')

    # If no label_id, try to find skeleton by proximity
    if skeleton_id is None:
        skeleton_id = _get_skeleton_id_for_soma(skeleton_arr, soma_coords)
        if skeleton_id is not None:
            print(f'Found skeleton_id={skeleton_id} near soma position')
        else:
            print(
                'Warning: Could not determine skeleton_id, using closest node'
            )

    # Create skan skeleton from the full labeled image
    # skan will treat each unique label as a separate skeleton
    skel = skan.Skeleton(skeleton_arr.astype(float), spacing=spacing)
    summary = skan.summarize(skel, separator='_')

    # Create directed tree for the specific skeleton
    tree = create_directed_tree(
        skel, soma_coords, summary=summary, skeleton_id=skeleton_id
    )

    # Get extended summary with directional info
    dir_summary = summarize_directed_tree(tree)

    # Find longest path (axon length)
    longest_path, axon_length = find_longest_path(tree)

    # Build paths for visualization - only for the selected skeleton
    paths = []
    properties = {
        'branch_order': [],
        'strahler_order': [],
        'distance_from_root': [],
        'branch_distance': [],
        'is_terminal': [],
        'in_tree': [],
        'skeleton_id': [],
    }

    for i in range(skel.n_paths):
        coords = skel.path_coordinates(i)
        if len(coords) >= 2:
            # Get properties from directed summary
            if i < len(dir_summary):
                row = dir_summary.iloc[i]
                row_skel_id = int(row.get('skeleton_id', -1))

                # Only include paths from the selected skeleton or mark others
                paths.append(coords)
                properties['skeleton_id'].append(row_skel_id)

                in_tree = bool(row.get('in_tree', False))
                properties['in_tree'].append(in_tree)

                if in_tree:
                    properties['branch_order'].append(
                        int(row.get('branch_order', 0))
                    )
                    properties['strahler_order'].append(
                        int(row.get('strahler_order', 0))
                    )
                    dist = row.get('distance_from_root', 0.0)
                    properties['distance_from_root'].append(
                        0.0 if np.isnan(dist) else float(dist)
                    )
                else:
                    properties['branch_order'].append(0)
                    properties['strahler_order'].append(0)
                    properties['distance_from_root'].append(0.0)

                properties['branch_distance'].append(
                    float(row.get('branch_distance', 0))
                )
                properties['is_terminal'].append(
                    bool(row.get('is_terminal', False))
                )
            else:
                paths.append(coords)
                properties['branch_order'].append(0)
                properties['strahler_order'].append(0)
                properties['distance_from_root'].append(0.0)
                properties['branch_distance'].append(0.0)
                properties['is_terminal'].append(False)
                properties['in_tree'].append(False)
                properties['skeleton_id'].append(-1)

    layers = []

    # Branches layer
    if paths:
        # Color by selected property
        if color_by == 'branch_order':
            # Use categorical colors for orders
            edge_colors = []
            for i, order in enumerate(properties['branch_order']):
                if not properties['in_tree'][i]:
                    # Branches not connected to soma are gray
                    edge_colors.append(BRANCH_ORDER_COLORS[0])
                else:
                    color = BRANCH_ORDER_COLORS.get(
                        order, BRANCH_ORDER_COLORS[0]
                    )
                    edge_colors.append(color)

            layers.append(
                (
                    paths,
                    {
                        'name': f'Directed Tree (cell {skeleton_id})',
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
            # For branches not in tree, set to 0
            layers.append(
                (
                    paths,
                    {
                        'name': f'Directed Tree (cell {skeleton_id})',
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

    # Soma marker (larger, distinct) - use actual root node coordinates
    # These are already in physical units from skan
    root_coords_physical = tree.skeleton.coordinates[tree.root_node]
    layers.append(
        (
            np.array([root_coords_physical]),
            {
                'name': f'Soma Root (cell {skeleton_id})',
                'size': 15,
                'face_color': [0.84, 0.37, 0.0, 1.0],  # Vermillion
                'border_color': 'white',
                'border_width': 0.2,
                'symbol': 'star',
                'scale': scale,
            },
            'points',
        )
    )

    # Print summary
    n_in_tree = sum(properties['in_tree'])
    n_total = len(properties['in_tree'])
    print('=' * 60)
    print(f'Directed Tree Analysis - Cell {skeleton_id}')
    print('=' * 60)
    print(f'  Root node: {tree.root_node}')
    print(f'  Root coords: {root_coords_physical}')
    print(f'  Branches in tree: {n_in_tree} / {n_total}')
    print(f'  Primary branches: {len(tree.primary_branches)}')
    print(f'  Tips (endpoints): {tree.n_tips}')
    print(f'  Junctions: {tree.n_junctions}')
    print(f'  Longest path length: {axon_length:.2f}')
    print()
    print('Branch Order Distribution:')
    for order in sorted(set(properties['branch_order'])):
        if order > 0:
            count = properties['branch_order'].count(order)
            print(f'  Order {order}: {count} branches')
    print()
    if n_in_tree < n_total:
        n_other = n_total - n_in_tree
        print(f'Note: {n_other} branches from other cells (gray)')
    print('Colors: Blue=1°, Orange=2°, Green=3°, Purple=4°, Yellow=5°')
    print('=' * 60)

    return layers
