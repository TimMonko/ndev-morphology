"""
Widget for soma detection and visualization.

Provides interactive soma centroid detection using different methods.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from magicgui import magic_factory

__all__ = ['soma_detection']


@magic_factory(
    call_button='Detect Soma',
    labels={'label': 'Labels (cells or nuclei)'},
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
    method: Literal['largest_region', 'roundest_region'] = 'largest_region',
    point_size: float = 10.0,
) -> napari.types.LayerDataTuple:
    """
    Detect soma centroids in labeled regions.

    Returns a Points layer with detected soma positions.

    Parameters
    ----------
    labels : Labels
        Label image with cells or nuclei.
    method : str
        Detection method.
    point_size : float
        Size of soma marker points.

    Returns
    -------
    list of LayerDataTuple
        Points layer with soma centroids.
    """
    from ..soma import detect_soma_centroid

    labels_data = np.asarray(labels.data)
    scale = labels.scale

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
            detected_labels.append(int(label_id))
        except ValueError:
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
