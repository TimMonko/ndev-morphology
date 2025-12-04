"""
Skeleton operations for morphology analysis.

Functions for creating and manipulating skeletons from label images.
All functions accept numpy arrays and use scikit-image/scipy for processing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage
from skimage import morphology

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = [
    "skeletonize_labels",
    "exclude_region_from_skeleton",
]


def skeletonize_labels(labels: ArrayLike) -> np.ndarray:
    """
    Create skeletons while maintaining label identities from a label image.

    Each labeled region is skeletonized independently and the skeleton pixels
    retain their original label values.

    Parameters
    ----------
    labels : ArrayLike
        Label image where each unique non-zero value represents a distinct object.

    Returns
    -------
    np.ndarray
        Skeletonized label image with same dtype as input labels.
        Background (0) pixels remain 0, skeleton pixels have original label values.

    Examples
    --------
    >>> import numpy as np
    >>> from ndev_morphology import skeletonize_labels
    >>> labels = np.array([[0, 1, 1, 1],
    ...                    [0, 1, 0, 1],
    ...                    [0, 1, 1, 1]])
    >>> skeleton = skeletonize_labels(labels)
    """
    labels = np.asarray(labels)
    binary = labels > 0
    skeleton = morphology.skeletonize(binary)
    return (labels * skeleton).astype(labels.dtype)


def exclude_region_from_skeleton(
    skeleton: ArrayLike,
    mask: ArrayLike,
    dilation_iterations: int = 0,
) -> np.ndarray:
    """
    Remove skeleton pixels that overlap with a mask region.

    Useful for excluding soma/nucleus regions from neuron skeletons.

    Parameters
    ----------
    skeleton : ArrayLike
        Skeleton image (binary or labeled).
    mask : ArrayLike
        Binary mask of region to exclude. Will be dilated if dilation_iterations > 0.
    dilation_iterations : int, optional
        Number of binary dilation iterations to apply to mask before exclusion.
        Default is 0 (no dilation).

    Returns
    -------
    np.ndarray
        Skeleton with masked region removed.

    Examples
    --------
    >>> from ndev_morphology import skeletonize_labels, exclude_region_from_skeleton
    >>> # Exclude nucleus region (with 3-pixel buffer) from neuron skeleton
    >>> skeleton_no_soma = exclude_region_from_skeleton(
    ...     skeleton, nucleus_mask, dilation_iterations=3
    ... )
    """
    skeleton = np.asarray(skeleton)
    mask = np.asarray(mask).astype(bool)

    if dilation_iterations > 0:
        mask = ndimage.binary_dilation(mask, iterations=dilation_iterations)

    return skeleton * ~mask
