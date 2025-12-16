"""
Soma detection for neuronal morphology analysis.

Functions for identifying cell body (soma) location from label images,
intensity images, or skeleton structure. The soma centroid serves as
the origin for Sholl analysis and the root for directed tree analysis.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy import ndimage
from skimage import measure

if TYPE_CHECKING:
    import skan
    from numpy.typing import ArrayLike

__all__ = [
    'detect_soma_centroid',
    'find_soma_node',
    'get_label_centroid',
]


def detect_soma_centroid(
    labels: ArrayLike,
    *,
    method: Literal[
        'largest_region', 'roundest_region', 'intensity_peak'
    ] = 'largest_region',
    intensity_image: ArrayLike | None = None,
    label_id: int | None = None,
) -> np.ndarray:
    """
    Detect soma centroid from a label image.

    The soma (cell body) is typically the largest or most circular region
    in a neuronal segmentation. This function identifies the soma and
    returns its centroid coordinates.

    Parameters
    ----------
    labels : ArrayLike
        Label image where each connected region has a unique integer ID.
        For single-cell analysis, this should be a mask of one cell.
        For multi-cell, use `label_id` to specify which cell.
    method : {'largest_region', 'roundest_region', 'intensity_peak'}
        Method to identify the soma:
        - 'largest_region': Centroid of the largest connected component (default)
        - 'roundest_region': Centroid of most circular region (lowest eccentricity)
        - 'intensity_peak': Location of peak intensity (requires intensity_image)
    intensity_image : ArrayLike, optional
        Intensity image for 'intensity_peak' method. Typically DAPI/nuclear stain.
    label_id : int, optional
        Specific label ID to analyze. If None, uses all non-zero labels.

    Returns
    -------
    np.ndarray
        Centroid coordinates as (Y, X) for 2D or (Z, Y, X) for 3D.
        Coordinates are in pixel units.

    Raises
    ------
    ValueError
        If method is 'intensity_peak' but intensity_image is not provided.
        If no regions are found in the label image.

    Examples
    --------
    >>> from ndev_morphology import detect_soma_centroid
    >>> # Find soma as largest region
    >>> soma_center = detect_soma_centroid(cell_labels, method='largest_region')
    >>> # Use DAPI channel to find nucleus
    >>> soma_center = detect_soma_centroid(
    ...     cell_labels,
    ...     method='intensity_peak',
    ...     intensity_image=dapi_channel,
    ... )
    """
    labels = np.asarray(labels)

    # Filter to specific label if requested
    if label_id is not None:
        labels = (labels == label_id).astype(labels.dtype)

    if method == 'intensity_peak':
        if intensity_image is None:
            raise ValueError(
                "intensity_image is required for 'intensity_peak' method"
            )
        return _detect_by_intensity_peak(labels, intensity_image)

    elif method == 'roundest_region':
        return _detect_by_roundness(labels)

    else:  # 'largest_region' (default)
        return _detect_by_size(labels)


def _detect_by_size(labels: np.ndarray) -> np.ndarray:
    """Find centroid of the largest connected component."""
    props = measure.regionprops(labels.astype(np.int32))

    if not props:
        raise ValueError('No regions found in label image')

    # Find largest by area
    largest = max(props, key=lambda p: p.area)
    return np.array(largest.centroid)


def _detect_by_roundness(labels: np.ndarray) -> np.ndarray:
    """Find centroid of the most circular (lowest eccentricity) region."""
    props = measure.regionprops(labels.astype(np.int32))

    if not props:
        raise ValueError('No regions found in label image')

    # Find most circular (lowest eccentricity)
    # For 3D, use extent as proxy (higher extent = more compact)
    if labels.ndim == 3:
        roundest = max(props, key=lambda p: p.extent)
    else:
        roundest = min(props, key=lambda p: p.eccentricity)

    return np.array(roundest.centroid)


def _detect_by_intensity_peak(
    labels: np.ndarray,
    intensity_image: ArrayLike,
) -> np.ndarray:
    """Find location of peak intensity within labeled region."""
    intensity_image = np.asarray(intensity_image)
    mask = labels > 0

    # Mask intensity to labeled region
    masked_intensity = np.where(mask, intensity_image, 0)

    # Find peak location
    peak_idx = np.unravel_index(
        np.argmax(masked_intensity),
        masked_intensity.shape,
    )

    return np.array(peak_idx, dtype=float)


def get_label_centroid(
    labels: ArrayLike,
    label_id: int,
) -> np.ndarray:
    """
    Get centroid of a specific label.

    Simple utility to extract the centroid of a single labeled region.

    Parameters
    ----------
    labels : ArrayLike
        Label image.
    label_id : int
        ID of the label to get centroid for.

    Returns
    -------
    np.ndarray
        Centroid coordinates as (Y, X) or (Z, Y, X).

    Raises
    ------
    ValueError
        If label_id is not found in the label image.

    Examples
    --------
    >>> from ndev_morphology import get_label_centroid
    >>> centroid = get_label_centroid(labels, label_id=5)
    """
    labels = np.asarray(labels)
    mask = labels == label_id

    if not np.any(mask):
        raise ValueError(f'Label {label_id} not found in image')

    return np.array(ndimage.center_of_mass(mask))


def find_soma_node(
    skeleton: skan.Skeleton,
    soma_centroid: ArrayLike,
) -> int:
    """
    Find the skeleton node closest to the soma centroid.

    This node can serve as the root for directed tree analysis,
    where branches are oriented from soma toward endpoints.

    Parameters
    ----------
    skeleton : skan.Skeleton
        A skan.Skeleton object.
    soma_centroid : ArrayLike
        Soma centroid coordinates in the same units as skeleton coordinates.
        If skeleton was created with physical spacing, use physical units.

    Returns
    -------
    int
        Index of the skeleton node closest to the soma centroid.
        This can be used as the source node for networkx tree construction.

    Examples
    --------
    >>> import skan
    >>> from skan.csr import skeleton_to_nx
    >>> import networkx as nx
    >>> from ndev_morphology import detect_soma_centroid, find_soma_node
    >>>
    >>> # Create skeleton and find soma
    >>> skeleton = skan.Skeleton(skeleton_image, spacing=(0.2, 0.2))
    >>> soma = detect_soma_centroid(cell_labels)
    >>> soma_node = find_soma_node(skeleton, soma)
    >>>
    >>> # Create directed tree from soma
    >>> G = skeleton_to_nx(skeleton)
    >>> directed_tree = nx.bfs_tree(G, soma_node)
    """
    soma_centroid = np.asarray(soma_centroid)

    # Get all node coordinates from skeleton
    # skeleton.coordinates is shape (n_nodes, ndim)
    node_coords = skeleton.coordinates

    # Compute distances to soma
    distances = np.linalg.norm(node_coords - soma_centroid, axis=1)

    # Return index of closest node
    return int(np.argmin(distances))
