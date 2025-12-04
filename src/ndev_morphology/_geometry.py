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
    "skeleton_to_paths",
    "sholl_shells_to_shapes",
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
    properties["path_id"] = np.arange(len(properties))

    return paths, properties


def sholl_shells_to_shapes(
    center: ArrayLike,
    radii: ArrayLike,
    *,
    n_points: int = 64,
    ndim: int = 2,
) -> list[np.ndarray]:
    """
    Generate circle/sphere shell coordinates for napari Shapes layer.

    Creates concentric circles (2D) or polygonal approximations of
    spherical cross-sections (3D) for Sholl visualization.

    Parameters
    ----------
    center : ArrayLike
        Center point coordinates (Y, X) for 2D or (Z, Y, X) for 3D.
        Should be in pixel units for napari display.
    radii : ArrayLike
        Array of shell radii in pixel units.
    n_points : int, optional
        Number of points to approximate each circle. Default is 64.
    ndim : int, optional
        Number of dimensions (2 or 3). Default is 2.

    Returns
    -------
    list of np.ndarray
        List of shell coordinate arrays, each shape (n_points, ndim).
        For 3D, shells are circles in the XY plane at the center Z.

    Examples
    --------
    >>> shells = sholl_shells_to_shapes(center=[100, 150], radii=[10, 20, 30])
    >>> viewer.add_shapes(shells, shape_type='polygon', face_color='transparent')
    """
    center = np.asarray(center)
    radii = np.asarray(radii)

    # Generate angles for circle points
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    shells = []
    for radius in radii:
        if ndim == 2:
            # 2D circle: (Y, X) coordinates
            y = center[0] + radius * np.sin(angles)
            x = center[1] + radius * np.cos(angles)
            shell = np.column_stack([y, x])
        else:
            # 3D: circle in XY plane at center Z
            z = np.full(n_points, center[0])
            y = center[1] + radius * np.sin(angles)
            x = center[2] + radius * np.cos(angles)
            shell = np.column_stack([z, y, x])

        shells.append(shell)

    return shells
