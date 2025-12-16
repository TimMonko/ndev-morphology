"""
Widget for per-cell morphology analysis.

Provides interactive single-cell analysis with visualization.
"""

from __future__ import annotations

from typing import List

import numpy as np
from magicgui import magic_factory

__all__ = ['cell_analysis']


@magic_factory(
    call_button='Analyze Cell',
    labels={'label': 'Cell Labels'},
    label_id={'label': 'Label ID', 'min': 1},
    soma_labels={
        'label': 'Soma Labels (optional)',
        'nullable': True,
    },
    run_sholl={'tooltip': 'Perform Sholl analysis from soma'},
    sholl_step={'label': 'Sholl Step (pixels)', 'min': 1.0, 'step': 1.0},
)
def cell_analysis(
    labels: napari.layers.Labels,
    label_id: int = 1,
    soma_labels: napari.layers.Labels | None = None,
    run_sholl: bool = True,
    sholl_step: float = 5.0,
) -> List[napari.types.LayerDataTuple]:
    """
    Analyze a single labeled cell's morphology.

    Returns visualization layers and prints summary to console.

    Parameters
    ----------
    labels : Labels
        Cell label image.
    label_id : int
        Which label ID to analyze.
    soma_labels : Labels, optional
        Separate nuclei labels for soma detection.
    run_sholl : bool
        Whether to perform Sholl analysis.
    sholl_step : float
        Step size for Sholl radii in pixels.

    Returns
    -------
    list of LayerDataTuple
        Layers for skeleton paths and soma point.
    """
    from .._geometry import sholl_shells_to_ellipses, skeleton_to_paths
    from ..analysis import analyze_single_cell

    labels_data = np.asarray(labels.data)
    scale = labels.scale
    spacing = tuple(scale[-2:])

    # Check if label exists
    if label_id not in np.unique(labels_data):
        raise ValueError(f'Label {label_id} not found in image')

    # Check if label has enough pixels
    cell_mask = labels_data == label_id
    if cell_mask.sum() < 10:
        raise ValueError(
            f'Label {label_id} is too small ({cell_mask.sum()} pixels)'
        )

    soma_data = None
    if soma_labels is not None:
        soma_data = np.asarray(soma_labels.data)

    # Run analysis (function handles empty skeleton errors)
    result = analyze_single_cell(
        labels_data,
        label_id=label_id,
        spacing=spacing,
        soma_labels=soma_data,
        run_sholl=run_sholl,
        sholl_step=sholl_step * spacing[0],  # Convert to physical
    )

    layers = []

    # 1. Skeleton as shapes
    paths, props = skeleton_to_paths(result.skeleton)
    if paths:
        layers.append(
            (
                paths,
                {
                    'name': f'Skeleton (cell {label_id})',
                    'shape_type': 'path',
                    'edge_color': 'branch_distance',
                    'edge_colormap': 'viridis',
                    'edge_width': 2.0,
                    'properties': props,
                    'scale': scale,
                },
                'shapes',
            )
        )

    # 2. Soma as point
    if result.soma_centroid is not None:
        layers.append(
            (
                np.array([result.soma_centroid]),
                {
                    'name': f'Soma (cell {label_id})',
                    'size': 15,
                    'face_color': 'magenta',
                    'symbol': 'disc',
                    'scale': scale,
                },
                'points',
            )
        )

    # 3. Sholl shells
    if result.sholl is not None and result.soma_centroid is not None:
        radii_px = result.sholl.radii / spacing[0]
        shells = sholl_shells_to_ellipses(result.soma_centroid, radii_px)

        layers.append(
            (
                shells,
                {
                    'name': f'Sholl (cell {label_id})',
                    'shape_type': 'ellipse',
                    'edge_color': 'turbo',
                    'edge_width': 1.5,
                    'face_color': 'transparent',
                    'scale': scale,
                },
                'shapes',
            )
        )

    # Print summary
    print(f'\n=== Cell {label_id} Analysis ===')
    for key, value in result.summary.items():
        if isinstance(value, float):
            print(f'  {key}: {value:.3f}')
        else:
            print(f'  {key}: {value}')
    print('=' * 35)

    return layers
