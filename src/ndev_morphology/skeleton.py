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
    'skeletonize_labels',
    'separate_touching_skeleton_labels',
    'exclude_region_from_skeleton',
    'fill_skeleton_gaps',
]


def skeletonize_labels(labels: ArrayLike) -> np.ndarray:
    """
    Create skeletons while maintaining label identities from a label image.

    Each labeled region is skeletonized independently and the skeleton pixels
    retain their original label values; normally, scikit-image's morphology.skeletonize
    produces a binary skeleton image without label information.

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


def separate_touching_skeleton_labels(
    skeleton_labels: ArrayLike,
) -> np.ndarray:
    """
    Remove skeleton pixels where different labels touch.

    This is useful when skeletonizing adjacent labeled regions that result in
    touching skeletons. The function removes the minimal pixels needed to
    separate connected components from different original labels. Separates
    based on 8-connectivity in 2D or 26-connectivity in 3D.

    Parameters
    ----------
    skeleton_labels : ArrayLike
        Labeled skeleton image where each pixel value is the original label ID.
        Typically the output of `skeletonize_labels`.

    Returns
    -------
    np.ndarray
        Skeleton with touching pixels removed. Pixels at label boundaries
        are set to 0.

    Notes
    -----
    For most biological images, skeletons rarely touch because objects are
    naturally separated. This function handles the edge case where skeletons
    from adjacent labels happen to be neighbors (e.g., touching cells).

    This approach is much more efficient than eroding labels before
    skeletonizing, as it only removes the 1-2 pixels at actual touching
    boundaries rather than eroding all edges.

    Examples
    --------
    >>> from ndev_morphology import skeletonize_labels, separate_touching_skeleton_labels
    >>> # Create labels that touch
    >>> labels = np.zeros((10, 20), dtype=np.uint16)
    >>> labels[:, :10] = 1
    >>> labels[:, 10:] = 2
    >>> # Skeletonize - skeletons may touch at boundary
    >>> skeleton = skeletonize_labels(labels)
    >>> # Separate touching skeletons
    >>> separated = separate_touching_skeleton_labels(skeleton)
    """
    skeleton_labels = np.asarray(skeleton_labels)
    separated = skeleton_labels.copy()

    # Use full connectivity (8-connected in 2D, 26-connected in 3D) to detect
    # both orthogonal AND diagonal neighbors, matching skan's connectivity
    structure = ndimage.generate_binary_structure(skeleton_labels.ndim, 2)

    # Find pixels adjacent to a different nonzero label
    for label_id in np.unique(skeleton_labels):
        if label_id == 0:
            continue

        this_label = skeleton_labels == label_id
        other_labels = (skeleton_labels > 0) & (skeleton_labels != label_id)

        # Pixels of this label that touch other labels (including diagonals)
        touching = this_label & ndimage.binary_dilation(
            other_labels, structure=structure
        )
        # Replace touching pixels with 0 to separate
        separated[touching] = 0

    return separated.astype(skeleton_labels.dtype)


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


def fill_skeleton_gaps(
    skeleton: ArrayLike,
    dilation_size: int = 2,
) -> np.ndarray:
    """
    Fill small gaps in a skeleton by dilation and re-skeletonization.

    This function bridges discontinuities in skeletons caused by
    incomplete segmentation or thresholding artifacts.

    Parameters
    ----------
    skeleton : ArrayLike
        Binary skeleton image (or labeled - will be binarized).
    dilation_size : int
        Radius of disk structuring element for dilation.
        Larger values bridge bigger gaps but may create artifacts
        at junction points (small cycles).

    Returns
    -------
    np.ndarray
        Binary skeleton with gaps filled.

    Warnings
    --------
    This operation creates small cycles at junction points because
    dilation "fattens" the skeleton before re-skeletonizing.
    Consider using `prune_short_branches(..., prune_cycles=True)`
    after this function to remove unwanted cycles.

    Notes
    -----
    For labeled skeletons, the output will be binary. To restore
    labels, you can use the original label image to re-assign IDs
    to the filled skeleton.

    See Also
    --------
    connect_breaks_between_labels : Similar operation on labels (before skeletonization)

    Examples
    --------
    >>> from ndev_morphology import fill_skeleton_gaps, prune_short_branches
    >>> import skan
    >>> # Fill gaps then remove cycles
    >>> filled = fill_skeleton_gaps(skeleton, dilation_size=2)
    >>> skel = skan.Skeleton(filled)
    >>> pruned, _ = prune_short_branches(skel, prune_cycles=True)
    """
    from skimage.morphology import disk, skeletonize

    skeleton = np.asarray(skeleton)
    binary = skeleton.astype(bool)

    # Dilate to bridge gaps
    selem = disk(dilation_size)
    dilated = ndimage.binary_dilation(binary, structure=selem)

    # Re-skeletonize to get thin skeleton
    filled = skeletonize(dilated)

    return filled
