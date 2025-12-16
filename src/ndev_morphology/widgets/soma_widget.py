"""
Widget for soma detection and visualization.

Provides interactive soma centroid detection using different methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from magicgui import magic_factory

if TYPE_CHECKING:
    import napari

__all__ = ['soma_detection']


@magic_factory(
    call_button='Detect Soma',
    labels={'label': 'Labels (soma regions)'},
    id_labels={
        'label': 'ID Labels (optional)',
        'nullable': True,
        'tooltip': (
            'Optional labels layer to assign IDs from. '
            'Each detected soma will be assigned the ID of the overlapping '
            'region in this layer. Useful for matching soma to skeleton labels.'
        ),
    },
    method={
        'choices': ['largest_region', 'roundest_region'],
        'tooltip': (
            'Detection method:\n'
            '- largest_region: Centroid of largest connected component\n'
            '- roundest_region: Most circular region (lowest eccentricity)'
        ),
    },
    point_size={'min': 1, 'max': 50, 'tooltip': 'Size of soma marker points'},
)
def soma_detection(
    labels: napari.layers.Labels,
    id_labels: napari.layers.Labels | None = None,
    method: Literal['largest_region', 'roundest_region'] = 'largest_region',
    point_size: float = 10.0,
) -> napari.types.LayerDataTuple:
    """
    Detect soma centroids in labeled regions.

    Returns a Points layer with detected soma positions. Each point has a
    'label_id' property that can be used to match soma to skeleton labels.

    Parameters
    ----------
    labels : Labels
        Label image with soma regions (e.g., nuclei or cell bodies).
    id_labels : Labels, optional
        Optional labels layer to assign IDs from. If provided, each detected
        soma centroid is assigned the ID from this layer at that position.
        This is useful when you want soma IDs to match a separate cell mask
        or skeleton layer.
    method : str
        Detection method.
    point_size : float
        Size of soma marker points.

    Returns
    -------
    LayerDataTuple
        Points layer with soma centroids and label_id properties.
    """
    from ..soma import detect_soma_centroid

    labels_data = np.asarray(labels.data)
    scale = labels.scale

    # Get ID labels if provided
    id_labels_data = None
    if id_labels is not None:
        id_labels_data = np.asarray(id_labels.data)

    # Get unique labels
    label_ids = np.unique(labels_data)
    label_ids = label_ids[label_ids > 0]

    # Detect soma for each label
    centroids = []
    detected_labels = []

    for label_id in label_ids:
        try:
            centroid = detect_soma_centroid(
                labels_data,
                label_id=int(label_id),
                method=method,
            )
            centroids.append(centroid)

            # Assign ID from id_labels if provided, otherwise use original
            if id_labels_data is not None:
                # Get the ID at the centroid position
                centroid_int = tuple(int(c) for c in centroid)
                assigned_id = int(id_labels_data[centroid_int])
                detected_labels.append(assigned_id)
            else:
                detected_labels.append(int(label_id))
        except (ValueError, IndexError):
            continue

    if not centroids:
        return (
            np.empty((0, labels_data.ndim)),
            {
                'name': 'Soma Centroids',
                'size': point_size,
                'face_color': 'magenta',
            },
            'points',
        )

    points = np.array(centroids)

    return (
        points,
        {
            'name': 'Soma Centroids',
            'size': point_size,
            'face_color': 'magenta',
            'border_color': 'white',
            'symbol': 'disc',
            'properties': {'label_id': detected_labels},
            'scale': scale,
        },
        'points',
    )
