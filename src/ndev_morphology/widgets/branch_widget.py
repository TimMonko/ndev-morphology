"""
Widget for branch analysis and visualization.

Provides interactive branch summarization with property-based coloring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import skan
from magicgui import magic_factory

from .._geometry import skeleton_to_paths
from ..branch import summarize_branches

if TYPE_CHECKING:
    import napari

__all__ = ['branch_analysis']


@magic_factory(
    call_button='Analyze Branches',
    skeleton_layer={'label': 'Skeleton Layer'},
    color_by={
        'choices': ['branch_distance', 'tortuosity', 'branch_type'],
        'tooltip': 'Property to color branches by',
    },
    edge_width={'min': 0.5, 'max': 10.0, 'step': 0.5},
)
def branch_analysis(
    skeleton_layer: napari.layers.Labels,
    color_by: str = 'branch_distance',
    edge_width: float = 2.0,
) -> napari.types.LayerDataTuple:
    """
    Analyze skeleton branches and visualize with property-based coloring.

    Prints branch summary statistics to console.

    Parameters
    ----------
    skeleton_layer : Labels
        Skeleton image layer.
    color_by : str
        Property to use for coloring.
    edge_width : float
        Width of branch lines.

    Returns
    -------
    list of LayerDataTuple
        Shapes layer with branches as paths.
    """
    skeleton_arr = np.asarray(skeleton_layer.data)
    scale = skeleton_layer.scale
    spacing = tuple(scale[-2:])

    if not np.any(skeleton_arr):
        raise ValueError('Skeleton is empty')

    # Create skeleton and summarize
    skeleton = skan.Skeleton(skeleton_arr.astype(float), spacing=spacing)
    branches = summarize_branches(skeleton)  # Already includes tortuosity

    # Print summary
    print('=' * 50)
    print(f'Branch Analysis: {len(branches)} branches')
    print(f'  Total length: {branches["branch_distance"].sum():.1f}')
    print(f'  Mean length: {branches["branch_distance"].mean():.2f}')
    if 'tortuosity' in branches.columns:
        print(f'  Mean tortuosity: {branches["tortuosity"].mean():.3f}')
    print('=' * 50)

    # Get paths for visualization
    paths, props = skeleton_to_paths(skeleton)

    if not paths:
        raise ValueError('No branches found')

    # For branch_type, use discrete colors
    if color_by == 'branch_type':
        type_colors = {
            0: [1, 0, 0, 1],  # endpoint-endpoint: red
            1: [0, 1, 0, 1],  # junction-endpoint: green
            2: [0, 0, 1, 1],  # junction-junction: blue
            3: [1, 1, 0, 1],  # isolated cycle: yellow
        }
        colors = [
            type_colors.get(int(t), [0.5, 0.5, 0.5, 1])
            for t in props['branch_type']
        ]
        return (
            paths,
            {
                'shape_type': 'path',
                'properties': props,
                'edge_color': colors,
                'edge_width': edge_width,
                'name': 'branches (by type)',
                'scale': scale,
            },
            'shapes',
        )

    return (
        paths,
        {
            'shape_type': 'path',
            'properties': props,
            'edge_color': color_by,
            'edge_colormap': 'viridis',
            'edge_width': edge_width,
            'name': f'branches (by {color_by})',
            'scale': scale,
        },
        'shapes',
    )
