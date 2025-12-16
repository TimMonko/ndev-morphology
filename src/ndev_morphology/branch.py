"""
Branch analysis for skeleton structures.

Functions for computing branch-level measurements using skan,
including branch classification, tortuosity, and intensity measurements.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import skan

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = [
    'BranchType',
    'summarize_branches',
    'compute_tortuosity',
    'filter_branches_by_type',
]


class BranchType(IntEnum):
    """
    Branch type enumeration matching skan's branch_type values.

    Attributes
    ----------
    ENDPOINT_TO_ENDPOINT : int
        Branch connecting two endpoints (isolated branch or terminal).
        Value: 0
    JUNCTION_TO_ENDPOINT : int
        Branch from a junction to an endpoint (terminal branch).
        Value: 1
    JUNCTION_TO_JUNCTION : int
        Branch connecting two junctions (internal branch).
        Value: 2
    ISOLATED_CYCLE : int
        Closed loop with no junctions.
        Value: 3
    """

    ENDPOINT_TO_ENDPOINT = 0
    JUNCTION_TO_ENDPOINT = 1
    JUNCTION_TO_JUNCTION = 2
    ISOLATED_CYCLE = 3

    @classmethod
    def label(cls, value: int) -> str:
        """Get human-readable label for branch type value."""
        labels = {
            0: 'endpoint-endpoint',
            1: 'junction-endpoint',
            2: 'junction-junction',
            3: 'isolated-cycle',
        }
        return labels.get(value, 'unknown')


def summarize_branches(
    skeleton: skan.Skeleton,
    *,
    intensity_image: ArrayLike | None = None,
    compute_tortuosity: bool = True,
) -> pd.DataFrame:
    """
    Summarize branch properties from a skeleton.

    Wraps `skan.summarize()` with additional computed metrics like tortuosity
    and human-readable branch type labels.

    Parameters
    ----------
    skeleton : skan.Skeleton
        A skan.Skeleton object created with physical spacing.
        For intensity measurements, create with source_image parameter:
        `skan.Skeleton(skeleton_image, source_image=intensity_image)`
    intensity_image : ArrayLike, optional
        Image to sample for intensity measurements along branches.
        If provided, adds mean_intensity/std_intensity columns.
        Note: For best results, pass intensity_image to skan.Skeleton
        constructor as source_image instead.
    compute_tortuosity : bool, optional
        Whether to compute tortuosity (path_length / euclidean_distance).
        Default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns from skan.summarize() plus:
        - 'branch_type_label': Human-readable branch type
        - 'tortuosity': Path length / euclidean distance (if computed)
        - 'mean_intensity', 'std_intensity': If intensity_image provided
          or if skeleton has source_image

    Examples
    --------
    >>> import skan
    >>> from ndev_morphology import summarize_branches
    >>> skeleton = skan.Skeleton(skeleton_image, spacing=(0.2, 0.2))
    >>> branches = summarize_branches(skeleton)
    >>> # Filter to terminal branches only
    >>> terminal = branches[branches['branch_type'] == 1]

    >>> # With intensity measurements
    >>> skeleton = skan.Skeleton(
    ...     skeleton_image, spacing=(0.2, 0.2), source_image=intensity_image
    ... )
    >>> branches = summarize_branches(skeleton)  # Includes mean_pixel_value
    """
    # Get base summary from skan
    # Use separator='_' for consistent column names across skan versions
    branches = skan.summarize(skeleton, separator='_')

    # Add human-readable branch type labels
    branches['branch_type_label'] = branches['branch_type'].apply(
        BranchType.label
    )

    # Compute tortuosity
    if compute_tortuosity:
        branches['tortuosity'] = _compute_tortuosity_series(branches)

    # Add intensity measurements if requested and not already present
    if (
        intensity_image is not None
        and 'mean_pixel_value' not in branches.columns
    ):
        intensity_image = np.asarray(intensity_image)
        mean_intensities, std_intensities = _compute_path_intensities(
            skeleton, intensity_image
        )
        branches['mean_intensity'] = mean_intensities
        branches['std_intensity'] = std_intensities

    return branches


def _compute_path_intensities(
    skeleton: skan.Skeleton,
    intensity_image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and std intensity along each path."""
    means = []
    stds = []

    for i in range(skeleton.n_paths):
        coords = skeleton.path_coordinates(i).astype(int)
        # Clip to image bounds
        for dim in range(coords.shape[1]):
            coords[:, dim] = np.clip(
                coords[:, dim], 0, intensity_image.shape[dim] - 1
            )

        # Sample intensity values
        if coords.shape[1] == 2:
            values = intensity_image[coords[:, 0], coords[:, 1]]
        else:  # 3D
            values = intensity_image[coords[:, 0], coords[:, 1], coords[:, 2]]

        means.append(np.mean(values))
        stds.append(np.std(values))

    return np.array(means), np.array(stds)


def _compute_tortuosity_series(branches: pd.DataFrame) -> pd.Series:
    """
    Compute tortuosity for each branch.

    Tortuosity = branch_distance / euclidean_distance
    A straight line has tortuosity = 1.0, curved branches > 1.0.
    """
    # Avoid division by zero for very short branches
    euclidean = branches['euclidean_distance'].replace(0, np.nan)
    tortuosity = branches['branch_distance'] / euclidean
    return tortuosity.fillna(1.0)


def compute_tortuosity(
    branch_distance: ArrayLike,
    euclidean_distance: ArrayLike,
) -> np.ndarray:
    """
    Compute tortuosity from path and straight-line distances.

    Tortuosity = path_length / euclidean_distance

    A perfectly straight branch has tortuosity = 1.0.
    More curved/tortuous branches have values > 1.0.

    Parameters
    ----------
    branch_distance : ArrayLike
        Path length along the branch (arc length).
    euclidean_distance : ArrayLike
        Straight-line distance between branch endpoints.

    Returns
    -------
    np.ndarray
        Tortuosity values. Returns 1.0 for branches with zero euclidean distance.

    Examples
    --------
    >>> from ndev_morphology import compute_tortuosity
    >>> tortuosity = compute_tortuosity(
    ...     branch_distance=[10.0, 15.0, 8.0],
    ...     euclidean_distance=[10.0, 10.0, 8.0],
    ... )
    >>> # [1.0, 1.5, 1.0]
    """
    branch_distance = np.asarray(branch_distance)
    euclidean_distance = np.asarray(euclidean_distance)

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        tortuosity = branch_distance / euclidean_distance
        tortuosity = np.where(
            euclidean_distance == 0,
            1.0,
            tortuosity,
        )

    return tortuosity


def filter_branches_by_type(
    branches: pd.DataFrame,
    branch_types: int | list[int] | BranchType | list[BranchType],
) -> pd.DataFrame:
    """
    Filter branch DataFrame to specific branch types.

    Parameters
    ----------
    branches : pd.DataFrame
        Branch summary DataFrame from `summarize_branches()` or `skan.summarize()`.
    branch_types : int, list of int, BranchType, or list of BranchType
        Branch type(s) to keep. Can use BranchType enum or raw int values.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only specified branch types.

    Examples
    --------
    >>> from ndev_morphology import summarize_branches, filter_branches_by_type, BranchType
    >>> branches = summarize_branches(skeleton)
    >>> # Get only terminal branches (junction to endpoint)
    >>> terminal = filter_branches_by_type(branches, BranchType.JUNCTION_TO_ENDPOINT)
    >>> # Get terminal and internal branches
    >>> main_branches = filter_branches_by_type(
    ...     branches,
    ...     [BranchType.JUNCTION_TO_ENDPOINT, BranchType.JUNCTION_TO_JUNCTION],
    ... )
    """
    # Normalize to list of ints
    if isinstance(branch_types, int | BranchType):
        branch_types = [int(branch_types)]
    else:
        branch_types = [int(bt) for bt in branch_types]

    return branches[branches['branch_type'].isin(branch_types)].copy()
