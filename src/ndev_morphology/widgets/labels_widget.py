"""
Labels processing widget for napari.

Provides label refinement operations like filtering by size,
removing edge-touching labels, and connecting nearby fragments.
"""

from __future__ import annotations

import numpy as np
from magicgui import magic_factory

from ..labels import (
    connect_breaks_between_labels,
    exclude_labels_on_edges,
    filter_labels_by_size,
)

__all__ = ['refine_labels_widget']


@magic_factory(
    call_button='Refine Labels',
    labels={'label': 'Labels Layer'},
    min_size={
        'label': 'Min Size (pixels)',
        'min': 0,
        'max': 1000000,
        'tooltip': 'Remove labels smaller than this. Set to 0 to disable.',
    },
    max_size={
        'label': 'Max Size (pixels)',
        'min': 0,
        'max': 10000000,
        'tooltip': 'Remove labels larger than this. Set to 0 to disable.',
    },
    exclude_edges={
        'label': 'Exclude Edge Labels',
        'tooltip': 'Remove labels that touch the image border.',
    },
    connect_distance={
        'label': 'Connect Distance (pixels)',
        'min': 0.0,
        'max': 50.0,
        'step': 0.5,
        'tooltip': 'Connect labels within this distance. Set to 0 to disable.',
    },
)
def refine_labels_widget(
    labels: napari.layers.Labels,
    min_size: int = 0,
    max_size: int = 0,
    exclude_edges: bool = False,
    connect_distance: float = 0.0,
) -> napari.types.LayerDataTuple:
    """
    Refine a labels image with filtering and connection operations.

    Apply a series of label refinement operations:
    1. Filter by size (min/max area in pixels)
    2. Exclude labels touching image edges
    3. Connect nearby label fragments

    Operations are applied in order, and each can be individually
    enabled/disabled.

    Parameters
    ----------
    labels : napari.types.LabelsData
        Input labeled image where each unique value represents a distinct object.
    min_size : int
        Minimum label size in pixels. Labels smaller than this are removed.
        Set to 0 to disable minimum size filtering.
    max_size : int
        Maximum label size in pixels. Labels larger than this are removed.
        Set to 0 to disable maximum size filtering.
    exclude_edges : bool
        If True, remove labels that touch the image border.
    connect_distance : float
        Connect labels within this distance (in pixels).
        Set to 0 to disable connecting.

    Returns
    -------
    napari.types.LayerDataTuple
        Refined labels layer.

    Notes
    -----
    This widget combines multiple label operations for convenience.
    Each operation can also be used independently via the API:

    - `filter_labels_by_size()`
    - `exclude_labels_on_edges()`
    - `connect_breaks_between_labels()`
    """
    result = np.asarray(labels.data).copy()

    # Apply operations in order
    # 1. Size filtering
    min_sz = min_size if min_size > 0 else None
    max_sz = max_size if max_size > 0 else None
    if min_sz is not None or max_sz is not None:
        result = filter_labels_by_size(
            result, min_size=min_sz, max_size=max_sz
        )

    # 2. Edge exclusion
    if exclude_edges:
        result = exclude_labels_on_edges(result)

    # 3. Connect nearby fragments
    if connect_distance > 0:
        result = connect_breaks_between_labels(
            result, connect_distance=connect_distance
        )

    # Build description of operations applied
    ops = []
    if min_sz is not None:
        ops.append(f'min>{min_sz}')
    if max_sz is not None:
        ops.append(f'max<{max_sz}')
    if exclude_edges:
        ops.append('no_edges')
    if connect_distance > 0:
        ops.append(f'connect={connect_distance:.1f}')

    suffix = f' ({", ".join(ops)})' if ops else ''

    return (
        result,
        {
            'name': f'refined_labels{suffix}',
            'scale': labels.scale,
        },
        'labels',
    )
