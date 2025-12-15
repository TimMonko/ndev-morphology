"""
Sholl analysis for skeleton structures.

Wraps skan.sholl_analysis with convenient result dataclass and summary statistics.
Sholl analysis counts the number of skeleton crossings at concentric shells
from a center point (typically the soma/cell body of a neuron).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import skan

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = [
    'ShollResult',
    'compute_sholl_profile',
]


@dataclass
class ShollResult:
    """
    Result container for Sholl analysis.

    Attributes
    ----------
    center : np.ndarray
        Center point coordinates used for analysis.
    radii : np.ndarray
        Shell radii at which crossings were counted.
    counts : np.ndarray
        Number of skeleton crossings at each radius.
    max_crossings : int
        Maximum number of crossings (peak of Sholl profile).
    critical_radius : float
        Radius at which maximum crossings occur.
    enclosing_radius : float
        Largest radius with non-zero crossings.
    mean_crossings : float
        Mean number of crossings across all radii.
    total_crossings : int
        Sum of all crossings (related to total branch complexity).
    """

    center: np.ndarray
    radii: np.ndarray
    counts: np.ndarray
    max_crossings: int = field(init=False)
    critical_radius: float = field(init=False)
    enclosing_radius: float = field(init=False)
    mean_crossings: float = field(init=False)
    total_crossings: int = field(init=False)

    def __post_init__(self):
        """Compute derived statistics from counts."""
        self.max_crossings = (
            int(np.max(self.counts)) if len(self.counts) > 0 else 0
        )
        self.total_crossings = int(np.sum(self.counts))
        self.mean_crossings = (
            float(np.mean(self.counts)) if len(self.counts) > 0 else 0.0
        )

        if self.max_crossings > 0:
            max_idx = np.argmax(self.counts)
            self.critical_radius = float(self.radii[max_idx])
        else:
            self.critical_radius = 0.0

        # Find enclosing radius (last radius with non-zero crossings)
        nonzero_indices = np.nonzero(self.counts)[0]
        if len(nonzero_indices) > 0:
            self.enclosing_radius = float(self.radii[nonzero_indices[-1]])
        else:
            self.enclosing_radius = 0.0

    def to_dict(self) -> dict:
        """Convert result to dictionary for DataFrame creation."""
        return {
            'center': self.center.tolist(),
            'max_crossings': self.max_crossings,
            'critical_radius': self.critical_radius,
            'enclosing_radius': self.enclosing_radius,
            'mean_crossings': self.mean_crossings,
            'total_crossings': self.total_crossings,
        }


def compute_sholl_profile(
    skeleton: skan.Skeleton,
    center: ArrayLike,
    radii: ArrayLike | None = None,
    *,
    min_radius: float = 0.0,
    max_radius: float | None = None,
    step: float = 1.0,
) -> ShollResult:
    """
    Perform Sholl analysis on a skeleton.

    Counts the number of skeleton branch crossings at concentric circular (2D)
    or spherical (3D) shells centered on a given point.

    Parameters
    ----------
    skeleton : skan.Skeleton
        A skan.Skeleton object. Create one using `create_skeleton()`.
        If the skeleton was created with physical spacing, center and radii
        should be in the same physical units.
    center : ArrayLike
        Center point coordinates for Sholl analysis, typically the soma/nucleus
        centroid. Should be in the same units as the skeleton's coordinate system.
    radii : ArrayLike, optional
        Explicit array of radii at which to count crossings.
        If not provided, radii are generated from min_radius to max_radius with step.
    min_radius : float, optional
        Minimum shell radius. Default is 0.
    max_radius : float, optional
        Maximum shell radius. If None, estimated from skeleton extent.
    step : float, optional
        Step size between radii. Default is 1.0.

    Returns
    -------
    ShollResult
        Dataclass containing radii, counts, and summary statistics.

    Examples
    --------
    >>> from ndev_morphology import create_skeleton, compute_sholl_profile
    >>> skel = create_skeleton(skeleton_image, spacing=(0.2, 0.2))
    >>> # Center is in physical units (µm) since spacing was provided
    >>> center = np.array([10.0, 12.0])  # Y, X in µm
    >>> result = compute_sholl_profile(skel, center, min_radius=1, max_radius=50, step=1)
    >>> print(f"Max crossings: {result.max_crossings} at {result.critical_radius} µm")

    Notes
    -----
    The center coordinates should match the units of the skeleton. If the skeleton
    was created with physical spacing, center should be in physical units.
    If no spacing was provided (pixel units), center should be in pixels.
    """
    center = np.asarray(center)

    # Handle empty skeleton (no paths)
    if skeleton.n_paths == 0:
        # Empty skeleton - return result with zero counts
        if radii is None:
            if max_radius is None:
                max_radius = 100.0  # Default fallback
            radii = np.arange(min_radius, max_radius, step)
        radii = np.asarray(radii)
        return ShollResult(
            center=center,
            radii=radii,
            counts=np.zeros(len(radii), dtype=int),
        )

    # Generate radii if not provided
    if radii is None:
        if max_radius is None:
            # Estimate max radius from skeleton extent
            all_coords = np.vstack(
                [skeleton.path_coordinates(i) for i in range(skeleton.n_paths)]
            )
            # If spacing was used, coordinates are in physical units
            distances = np.linalg.norm(all_coords - center, axis=1)
            max_radius = float(np.max(distances)) * 1.1  # 10% margin

        radii = np.arange(min_radius, max_radius, step)

    radii = np.asarray(radii)

    # Perform Sholl analysis using skan
    center_result, radii_result, counts = skan.sholl_analysis(
        skeleton, center=center, shells=radii
    )

    return ShollResult(
        center=center_result,
        radii=radii_result,
        counts=counts,
    )
