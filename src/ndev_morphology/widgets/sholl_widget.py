"""
Sholl analysis widget for napari.

Performs Sholl analysis on skeleton images and visualizes results
as concentric shell shapes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import skan
from magicgui import magic_factory

from .._geometry import sholl_shells_to_ellipses
from ..sholl import compute_sholl_profile

if TYPE_CHECKING:
    import napari

__all__ = ['sholl_analysis']


@magic_factory(
    call_button='Run Sholl Analysis',
    skeleton_layer={'label': 'Skeleton Layer'},
    center_point={'label': 'Center Point (optional)', 'nullable': True},
    center_y={'label': 'Center Y (pixels)', 'min': 0.0, 'max': 10000.0},
    center_x={'label': 'Center X (pixels)', 'min': 0.0, 'max': 10000.0},
    radius_step={
        'label': 'Radius Step (pixels)',
        'min': 1.0,
        'max': 50.0,
        'tooltip': 'Step between Sholl shells in pixels.',
    },
    max_radius={
        'label': 'Max Radius (pixels)',
        'min': 0.0,
        'tooltip': 'Maximum radius. Set to 0 for auto (half diagonal).',
    },
)
def sholl_analysis(
    skeleton_layer: napari.layers.Labels,
    center_point: napari.layers.Points | None = None,
    center_y: float = 100.0,
    center_x: float = 100.0,
    radius_step: float = 5.0,
    max_radius: float = 0.0,
) -> napari.types.LayerDataTuple:
    """
    Perform Sholl analysis and visualize with concentric shells.

    Parameters
    ----------
    skeleton_layer : Labels
        Skeleton image (binary or labeled).
    center_point : Points, optional
        Points layer with center. Uses first point if provided.
    center_y, center_x : float
        Manual center coordinates if no Points layer.
    radius_step : float
        Step between shells in pixels.
    max_radius : float
        Maximum radius. 0 = auto (half diagonal).

    Returns
    -------
    list of LayerDataTuple
        Shapes layer with Sholl shells.
    """
    skeleton_arr = np.asarray(skeleton_layer.data)
    scale = skeleton_layer.scale

    if not np.any(skeleton_arr):
        raise ValueError('Skeleton is empty')

    # Get center from Points layer or manual input
    if center_point is not None and len(center_point.data) > 0:
        center_px = center_point.data[0][:2]  # First point, YX
    else:
        center_px = np.array([center_y, center_x])

    # Convert to physical units using scale
    spacing = tuple(scale[-2:])  # YX spacing
    center_physical = center_px * np.array(spacing)

    # Auto max radius if not set
    if max_radius <= 0:
        diag = np.sqrt(
            skeleton_arr.shape[-2] ** 2 + skeleton_arr.shape[-1] ** 2
        )
        max_radius = diag / 2

    # Create radii array (in physical units)
    radii_physical = np.arange(
        radius_step * spacing[0],  # Start at one step
        max_radius * spacing[0],
        radius_step * spacing[0],
    )

    if len(radii_physical) == 0:
        raise ValueError('No valid radii generated')

    # Create skeleton and run Sholl
    skel = skan.Skeleton(skeleton_arr.astype(float), spacing=spacing)
    result = compute_sholl_profile(
        skel, center=center_physical, radii=radii_physical
    )

    # Print summary
    print('=' * 50)
    print('Sholl Analysis Results:')
    print(f'  Max crossings: {result.max_crossings}')
    print(f'  Critical radius: {result.critical_radius:.2f}')
    print(f'  Enclosing radius: {result.enclosing_radius:.2f}')
    print(f'  Total crossings: {result.total_crossings}')
    print('=' * 50)

    # Generate shells in pixel coordinates
    radii_px = result.radii / spacing[0]
    shells = sholl_shells_to_ellipses(center_px, radii_px, ndim=2)

    return (
        shells,
        {
            'shape_type': 'ellipse',
            'properties': {
                'radius': result.radii.tolist(),
                'crossings': result.counts.tolist(),
            },
            'edge_color': 'crossings',
            'edge_colormap': 'turbo',
            'edge_width': 1.5,
            'face_color': 'transparent',
            'name': 'Sholl shells',
            'scale': scale,
        },
        'shapes',
    )
