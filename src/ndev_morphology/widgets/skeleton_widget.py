"""
Skeleton visualization widget for napari.

Converts skeleton images to Shapes layers with branch properties.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import skan
from magicgui import magic_factory

from .._geometry import skeleton_to_paths

if TYPE_CHECKING:
    import napari.types

__all__ = ['skeleton_to_shapes']


# Available properties for coloring (from skan.summarize output)
EDGE_COLOR_CHOICES = [
    'branch_distance',
    'euclidean_distance',
    'branch_type',
    'skeleton_id',
    'path_id',
]

COLORMAP_CHOICES = [
    'viridis',
    'plasma',
    'magma',
    'inferno',
    'turbo',
    'hot',
    'cool',
]


@magic_factory(
    call_button='Create Skeleton Shapes',
    skeleton_image={'label': 'Skeleton Image'},
    spacing_y={
        'label': 'Spacing Y',
        'min': 0.001,
        'max': 100.0,
        'step': 0.01,
        'tooltip': 'Physical pixel spacing in Y. Use 1.0 for pixel units.',
    },
    spacing_x={
        'label': 'Spacing X',
        'min': 0.001,
        'max': 100.0,
        'step': 0.01,
        'tooltip': 'Physical pixel spacing in X. Use 1.0 for pixel units.',
    },
    edge_color={
        'label': 'Color by',
        'choices': EDGE_COLOR_CHOICES,
        'tooltip': 'Property to use for edge coloring',
    },
    edge_colormap={
        'label': 'Colormap',
        'choices': COLORMAP_CHOICES,
    },
    edge_width={
        'label': 'Edge Width',
        'min': 0.5,
        'max': 10.0,
        'step': 0.5,
    },
)
def skeleton_to_shapes(
    skeleton_image: napari.types.LabelsData,
    spacing_y: float = 1.0,
    spacing_x: float = 1.0,
    edge_color: str = 'branch_distance',
    edge_colormap: str = 'viridis',
    edge_width: float = 2.0,
) -> napari.types.LayerDataTuple:
    """
    Create a Shapes layer from a skeleton image.

    Converts a binary or labeled skeleton image into a napari Shapes layer
    where each branch is represented as a path. Branches can be colored by
    various properties computed by skan.

    Parameters
    ----------
    skeleton_image : napari.types.LabelsData
        Binary or labeled skeleton image. Non-zero pixels are skeleton.
    spacing_y : float
        Physical pixel spacing in Y dimension.
    spacing_x : float
        Physical pixel spacing in X dimension.
    spacing : tuple of float
        Physical pixel spacing (Y, X). Used for computing branch distances
        in physical units.
    edge_color : str
        Property to use for coloring branches.
    edge_colormap : str
        Colormap for branch coloring.
    edge_width : float
        Width of skeleton path lines.

    Returns
    -------
    napari.types.LayerDataTuple
        Tuple of (data, kwargs, layer_type) for napari to create the layer.

    Examples
    --------
    In napari, this widget will appear in the Plugins menu. You can also
    call it programmatically:

    >>> from ndev_morphology.widgets import skeleton_to_shapes
    >>> layer_data = skeleton_to_shapes(skeleton_image, spacing_y=0.2, spacing_x=0.2)
    """
    # Validate input
    skeleton_arr = np.asarray(skeleton_image)
    if not np.any(skeleton_arr):
        raise ValueError('Skeleton image is empty (no non-zero pixels)')

    # Create skan Skeleton with spacing
    spacing = (spacing_y, spacing_x)
    skel = skan.Skeleton(skeleton_arr.astype(float), spacing=spacing)

    # Convert to paths and properties
    paths, properties = skeleton_to_paths(skel)

    # Return LayerDataTuple
    return (
        paths,
        {
            'shape_type': 'path',
            'properties': properties,
            'edge_color': edge_color,
            'edge_colormap': edge_colormap,
            'edge_width': edge_width,
            'name': f'skeleton ({edge_color})',
        },
        'shapes',
    )
