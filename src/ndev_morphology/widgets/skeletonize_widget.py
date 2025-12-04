"""
Skeletonize labels widget for napari.

Provides label-aware skeletonization that preserves label identities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import napari.types
from magicgui import magic_factory

from ..skeleton import skeletonize_labels

if TYPE_CHECKING:
    pass

__all__ = ["skeletonize_labels_widget"]


@magic_factory(
    call_button="Skeletonize Labels",
    labels={"label": "Labels Layer"},
)
def skeletonize_labels_widget(
    labels: napari.types.LabelsData,
) -> napari.types.LayerDataTuple:
    """
    Skeletonize a labels image while preserving label identities.

    Unlike standard skeletonization which produces a binary image,
    this function preserves which original label each skeleton pixel
    belongs to. This is essential for per-object analysis like
    Sholl analysis on individual neurons.

    Parameters
    ----------
    labels : napari.types.LabelsData
        Labeled image where each unique value represents a distinct object.

    Returns
    -------
    napari.types.LayerDataTuple
        Labels layer with skeletonized objects, preserving original label values.

    Notes
    -----
    This is unique functionality not available in skan. Use skan's
    `labels_to_skeleton_shapes` widget if you want skeleton visualization
    as a Shapes layer.
    """
    skeleton = skeletonize_labels(labels)

    return (
        skeleton,
        {
            "name": "skeleton",
        },
        "labels",
    )
