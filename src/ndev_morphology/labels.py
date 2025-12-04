"""
Label preparation operations for morphology analysis.

Functions for filtering, cleaning, and preparing label images
before skeletonization. All functions accept numpy arrays and
use scikit-image/scipy for processing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from skimage import measure, morphology
from skimage.segmentation import clear_border

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = [
    "filter_labels_by_size",
    "exclude_labels_on_edges",
    "connect_breaks_between_labels",
]


def filter_labels_by_size(
    labels: ArrayLike,
    min_size: int | None = None,
    max_size: int | None = None,
) -> np.ndarray:
    """
    Filter labels by area (number of pixels).

    Parameters
    ----------
    labels : ArrayLike
        Label image.
    min_size : int, optional
        Minimum area in pixels. Labels smaller than this are removed.
    max_size : int, optional
        Maximum area in pixels. Labels larger than this are removed.

    Returns
    -------
    np.ndarray
        Filtered label image with same dtype as input.

    Examples
    --------
    >>> from ndev_morphology import filter_labels_by_size
    >>> # Keep only labels between 100 and 10000 pixels
    >>> filtered = filter_labels_by_size(labels, min_size=100, max_size=10000)
    """
    labels = np.asarray(labels)
    output = labels.copy()

    if min_size is None and max_size is None:
        return output

    props = measure.regionprops(labels.astype(np.int32))

    for prop in props:
        area = prop.area
        remove = False

        if min_size is not None and area < min_size:
            remove = True
        if max_size is not None and area > max_size:
            remove = True

        if remove:
            output[labels == prop.label] = 0

    return output


def exclude_labels_on_edges(labels: ArrayLike) -> np.ndarray:
    """
    Remove labels that touch the image border.

    Parameters
    ----------
    labels : ArrayLike
        Label image.

    Returns
    -------
    np.ndarray
        Label image with edge-touching labels removed.

    Examples
    --------
    >>> from ndev_morphology import exclude_labels_on_edges
    >>> filtered = exclude_labels_on_edges(labels)
    """
    labels = np.asarray(labels)
    return morphology.remove_small_objects(clear_border(labels), min_size=0)


def connect_breaks_between_labels(
    labels: ArrayLike,
    connect_distance: float,
) -> np.ndarray:
    """
    Connect breaks between labels within a specified distance.

    Dilates labels, merges touching ones, then masks back to original label
    extent. This connects nearby label fragments without changing overall
    morphology.

    Parameters
    ----------
    labels : ArrayLike
        Label image.
    connect_distance : float
        Maximum distance (in pixels) to connect labels. Labels are dilated
        by connect_distance/2 before merging.

    Returns
    -------
    np.ndarray
        Label image with connected labels. New merged labels may have
        different IDs than originals.

    Examples
    --------
    >>> from ndev_morphology import connect_breaks_between_labels
    >>> # Connect labels within 3 pixels of each other
    >>> connected = connect_breaks_between_labels(labels, connect_distance=3.0)
    """
    labels = np.asarray(labels)
    original_mask = labels > 0

    # Dilate labels using distance transform approach
    radius = int(np.ceil(connect_distance / 2))
    if radius < 1:
        return labels.copy()

    # Create dilated version using morphological dilation
    dilated = morphology.dilation(labels, footprint=morphology.disk(radius))

    # Relabel to merge touching components
    binary_dilated = dilated > 0
    merged = measure.label(binary_dilated, connectivity=2)

    # Mask back to original label extent
    result = (merged * original_mask).astype(labels.dtype)

    return result
