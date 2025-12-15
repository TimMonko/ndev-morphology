"""
Geometry utilities for napari visualization.

Internal module with functions to convert skeleton and Sholl data
to napari Shapes layer formats.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import skan

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = [
    'skeleton_to_paths',
    'sholl_shells_to_ellipses',
]


def skeleton_to_paths(
    skeleton: skan.Skeleton,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    """
    Convert skan Skeleton to paths and properties for napari Shapes layer.

    Parameters
    ----------
    skeleton : skan.Skeleton
        A skan.Skeleton object.

    Returns
    -------
    paths : list of np.ndarray
        List of path coordinate arrays, each shape (n_points, ndim).
        Coordinates are in pixel units (not physical units).
    properties : pd.DataFrame
        DataFrame with branch properties from skan.summarize(), plus path_id.

    Examples
    --------
    >>> import skan
    >>> import napari
    >>> skel = skan.Skeleton(skeleton_image, spacing=(0.2, 0.2))
    >>> paths, props = skeleton_to_paths(skel)
    >>> viewer = napari.Viewer()
    >>> viewer.add_shapes(
    ...     paths,
    ...     shape_type='path',
    ...     properties=props,
    ...     edge_color='branch_distance',
    ...     edge_colormap='viridis',
    ...     edge_width=2,
    ... )
    """
    # Get all path coordinates
    paths = [skeleton.path_coordinates(i) for i in range(skeleton.n_paths)]

    # Get branch properties from skan
    properties = skan.summarize(skeleton)

    # Add path_id for coloring
    properties['path_id'] = np.arange(len(properties))

    return paths, properties


def sholl_shells_to_ellipses(
    center: ArrayLike,
    radii: ArrayLike,
    *,
    ndim: int = 2,
) -> list[np.ndarray]:
    """
    Generate ellipse bounding boxes for napari Shapes layer.

    Creates circles (2D) as ellipses defined by their bounding boxes
    for Sholl visualization. Using ellipses is more efficient than
    polygon approximations and renders as true circles.

    Parameters
    ----------
    center : ArrayLike
        Center point coordinates (Y, X) for 2D or (Z, Y, X) for 3D.
        Should be in pixel units for napari display.
    radii : ArrayLike
        Array of shell radii in pixel units.
    ndim : int, optional
        Number of dimensions (2 or 3). Default is 2.
        For 3D, circles are drawn in the XY plane at the center Z.

    Returns
    -------
    list of np.ndarray
        List of ellipse bounding boxes, each shape (4, ndim).
        Each ellipse is defined by 4 corner points of its bounding box:
        [top-left, top-right, bottom-right, bottom-left].

    Notes
    -----
    napari's ellipse shape_type expects bounding box corners, not
    center + radii. This function converts center + radius to the
    4-corner format.

    Examples
    --------
    >>> shells = sholl_shells_to_ellipses(center=[100, 150], radii=[10, 20, 30])
    >>> viewer.add_shapes(
    ...     shells,
    ...     shape_type='ellipse',
    ...     edge_color='blue',
    ...     face_color='transparent',
    ... )
    """
    center = np.asarray(center)
    radii = np.asarray(radii)

    ellipses = []
    for radius in radii:
        if ndim == 2:
            # 2D ellipse bounding box: 4 corners (Y, X)
            # [top-left, top-right, bottom-right, bottom-left]
            y, x = center[0], center[1]
            bbox = np.array(
                [
                    [y - radius, x - radius],  # top-left
                    [y - radius, x + radius],  # top-right
                    [y + radius, x + radius],  # bottom-right
                    [y + radius, x - radius],  # bottom-left
                ]
            )
        else:
            # 3D ellipse in XY plane at center Z
            z, y, x = center[0], center[1], center[2]
            bbox = np.array(
                [
                    [z, y - radius, x - radius],
                    [z, y - radius, x + radius],
                    [z, y + radius, x + radius],
                    [z, y + radius, x - radius],
                ]
            )

        ellipses.append(bbox)

    return ellipses
