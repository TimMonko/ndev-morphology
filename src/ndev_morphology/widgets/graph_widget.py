"""
Widget for visualizing skeleton graph structure.

Shows nodes (junctions/endpoints) and edges (branches) as separate layers.
"""

from __future__ import annotations

from typing import List

import numpy as np
import skan
from magicgui import magic_factory

__all__ = ['skeleton_graph']


@magic_factory(
    call_button='Show Graph Structure',
    skeleton_layer={'label': 'Skeleton Layer'},
    node_size={'min': 3, 'max': 30, 'tooltip': 'Size of node markers'},
    edge_width={'min': 0.5, 'max': 10.0, 'step': 0.5},
)
def skeleton_graph(
    skeleton_layer: napari.layers.Labels,
    node_size: float = 8.0,
    edge_width: float = 2.0,
) -> List[napari.types.LayerDataTuple]:
    """
    Visualize skeleton as graph: nodes (junctions/endpoints) and edges (branches).

    skan represents skeletons as graphs where:
    - **Nodes**: Junction points (degree >= 3) and endpoints (degree 1)
    - **Edges**: The traced pixel paths between nodes (not straight lines)
    - **skeleton_id**: Each disconnected component gets a unique ID

    Parameters
    ----------
    skeleton_layer : Labels
        Skeleton image.
    node_size : float
        Size of node markers.
    edge_width : float
        Width of edge lines.

    Returns
    -------
    list of LayerDataTuple
        Points layer for nodes, Shapes layer for edges.
    """
    skeleton_arr = np.asarray(skeleton_layer.data)
    scale = skeleton_layer.scale
    spacing = tuple(scale[-2:])

    if not np.any(skeleton_arr):
        raise ValueError('Skeleton is empty')

    # Create skan Skeleton
    skel = skan.Skeleton(skeleton_arr.astype(float), spacing=spacing)

    # Get branch summary for node info
    branches = skan.summarize(skel, separator='_')

    # Extract unique nodes (source and destination of each branch)
    # Nodes are stored as coordinates in the summary
    node_coords = set()
    node_types = {}  # (y, x) -> 'junction' or 'endpoint'

    for idx, row in branches.iterrows():
        # Source node
        src = (row['coord_src_0'], row['coord_src_1'])
        node_coords.add(src)

        # Destination node
        dst = (row['coord_dst_0'], row['coord_dst_1'])
        node_coords.add(dst)

        # Determine node types from branch_type
        # branch_type: 0=endpoint-endpoint, 1=junction-endpoint, 2=junction-junction
        bt = int(row['branch_type'])
        if bt == 0:  # endpoint-endpoint
            node_types[src] = 'endpoint'
            node_types[dst] = 'endpoint'
        elif bt == 1:  # junction-endpoint
            # One is junction, one is endpoint - can't tell which from summary alone
            # Mark as 'mixed' and we'll color by degree later
            if src not in node_types:
                node_types[src] = 'mixed'
            if dst not in node_types:
                node_types[dst] = 'mixed'
        elif bt == 2:  # junction-junction
            node_types[src] = 'junction'
            node_types[dst] = 'junction'

    # Convert to arrays
    nodes = np.array(list(node_coords))

    # Color nodes by type
    node_colors = []
    for coord in node_coords:
        ntype = node_types.get(coord, 'unknown')
        if ntype == 'endpoint':
            node_colors.append([1, 0, 0, 1])  # Red
        elif ntype == 'junction':
            node_colors.append([0, 1, 0, 1])  # Green
        else:
            node_colors.append([1, 1, 0, 1])  # Yellow for mixed/unknown

    # Get edges as paths
    paths = []
    edge_props = {'branch_distance': [], 'branch_type': [], 'skeleton_id': []}

    for i in range(skel.n_paths):
        coords = skel.path_coordinates(i)
        if len(coords) >= 2:
            paths.append(coords)
            edge_props['branch_distance'].append(
                branches.iloc[i]['branch_distance']
            )
            edge_props['branch_type'].append(branches.iloc[i]['branch_type'])
            edge_props['skeleton_id'].append(branches.iloc[i]['skeleton_id'])

    layers = []

    # Nodes layer
    if len(nodes) > 0:
        layers.append(
            (
                nodes,
                {
                    'name': 'Graph Nodes',
                    'size': node_size,
                    'face_color': node_colors,
                    'symbol': 'disc',
                    'scale': scale,
                },
                'points',
            )
        )

    # Edges layer
    if paths:
        layers.append(
            (
                paths,
                {
                    'name': 'Graph Edges',
                    'shape_type': 'path',
                    'properties': edge_props,
                    'edge_color': 'skeleton_id',
                    'edge_colormap': 'tab10',
                    'edge_width': edge_width,
                    'scale': scale,
                },
                'shapes',
            )
        )

    # Print summary
    print('=' * 50)
    print('Skeleton Graph Structure:')
    print(f'  Nodes: {len(nodes)}')
    print(f'  Edges (branches): {len(paths)}')
    print(f'  Connected components: {branches["skeleton_id"].nunique()}')
    print('  Node colors: Red=endpoint, Green=junction, Yellow=mixed')
    print('=' * 50)

    return layers
