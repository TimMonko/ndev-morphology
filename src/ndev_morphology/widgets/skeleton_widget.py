"""
Skeleton visualization widget for napari.

Converts skeleton images to Shapes layers with branch properties.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from magicgui import magic_factory

from .._geometry import skeleton_to_paths
from ._utils import get_or_create_model

if TYPE_CHECKING:
    import napari

__all__ = ['skeleton_to_shapes']


@magic_factory(
    call_button='Create Skeleton Shapes',
    skeleton_layer={'label': 'Skeleton Layer'},
    color_by={
        'label': 'Color by',
        'choices': ['branch_distance', 'euclidean_distance', 'branch_type'],
        'tooltip': 'Property to color branches by',
    },
    edge_width={
        'label': 'Line Width',
        'min': 0.5,
        'max': 10.0,
        'step': 0.5,
    },
)
def skeleton_to_shapes(
    skeleton_layer: napari.layers.Labels,
    color_by: str = 'branch_distance',
    edge_width: float = 2.0,
) -> napari.types.LayerDataTuple:
    """
    Convert skeleton to Shapes layer with colored branches.

    Parameters
    ----------
    skeleton_layer : Labels
        Skeleton image (from skeletonize_labels widget).
    color_by : str
        Property to use for coloring branches.
    edge_width : float
        Width of branch lines.

    Returns
    -------
    LayerDataTuple
        Shapes layer with branches as paths.

    Notes
    -----
    This widget uses a cached MorphologyModel object stored in
    layer.metadata. Subsequent runs reuse the same skan.Skeleton,
    improving performance for repeated analysis.
    """
    # Use cached model (or create if first time)
    model = get_or_create_model(skeleton_layer)

    # Get paths and properties from the cached skeleton
    paths, properties = skeleton_to_paths(model.skeleton)

    if not paths:
        raise ValueError('No branches found in skeleton')

    return (
        paths,
        {
            'shape_type': 'path',
            'properties': properties,
            'edge_color': color_by,
            'edge_colormap': 'viridis',
            'edge_width': edge_width,
            'name': f'{skeleton_layer.name}_shapes',
            'scale': skeleton_layer.scale,
        },
        'shapes',
    )
