"""
Skeleton pruning operations.

Functions for removing unwanted branches from skeletons, such as
short spurious branches created during skeletonization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import skan

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = [
    'prune_short_branches',
    'remove_isolated_cycles',
    'BranchType',
]


class BranchType:
    """
    Branch type constants from skan.

    Branch types describe the topology of each branch based on its endpoints:
    - ENDPOINT_TO_ENDPOINT (0): An isolated branch with endpoints on both ends
    - JUNCTION_TO_ENDPOINT (1): One end is a junction, one is an endpoint (terminal branch)
    - JUNCTION_TO_JUNCTION (2): Both ends are junctions (connecting branch)
    - ISOLATED_CYCLE (3): A closed loop with no endpoints
    """

    ENDPOINT_TO_ENDPOINT = 0
    JUNCTION_TO_ENDPOINT = 1
    JUNCTION_TO_JUNCTION = 2
    ISOLATED_CYCLE = 3


def prune_short_branches(
    skeleton: skan.Skeleton,
    min_branch_distance: float = 10.0,
    *,
    prune_endpoint_to_endpoint: bool = True,
    prune_junction_to_endpoint: bool = True,
    prune_junction_to_junction: bool = False,
    prune_isolated_cycles: bool = True,
) -> tuple[skan.Skeleton, pd.DataFrame]:
    """
    Remove short branches from a skeleton.

    Uses skan's `prune_paths()` to remove branches shorter than the
    specified minimum distance. You can control which branch types
    are eligible for pruning.

    Parameters
    ----------
    skeleton : skan.Skeleton
        The skeleton to prune.
    min_branch_distance : float
        Minimum branch length to keep. Branches shorter than this
        (of the selected types) will be removed.
    prune_endpoint_to_endpoint : bool
        Whether to prune isolated branches (type 0). These are branches
        that don't connect to anything else - usually artifacts.
    prune_junction_to_endpoint : bool
        Whether to prune terminal branches (type 1). These are branches
        that end in an endpoint - the most common spurious branches.
    prune_junction_to_junction : bool
        Whether to prune connecting branches (type 2). Usually you want
        to keep these as they connect major parts of the skeleton.
    prune_isolated_cycles : bool
        Whether to prune closed loops (type 3). These are usually
        artifacts from holes in the original image.

    Returns
    -------
    pruned_skeleton : skan.Skeleton
        New skeleton with short branches removed.
    summary : pd.DataFrame
        Summary of remaining branches from skan.summarize().

    Notes
    -----
    The pruning process:
    1. Summarizes the skeleton to get branch properties
    2. Identifies branches that are both too short AND of a prunable type
    3. Calls `skeleton.prune_paths()` to remove those branches
    4. Re-summarizes the pruned skeleton

    After pruning, junction resolution may no longer be optimal. For
    best results, consider re-skeletonizing the result if accuracy is
    critical (though this is slower).

    Examples
    --------
    >>> import skan
    >>> from ndev_morphology import prune_short_branches
    >>> # Create skeleton from binary image
    >>> skeleton = skan.Skeleton(skeleton_image, spacing=(0.2, 0.2))
    >>> # Prune branches shorter than 5 microns
    >>> pruned, summary = prune_short_branches(skeleton, min_branch_distance=5.0)
    >>> print(f"Removed {skeleton.n_paths - pruned.n_paths} short branches")
    """
    # Get branch summary
    summary = skan.summarize(skeleton, separator='_')

    # Identify short branches
    too_short = summary['branch_distance'] < min_branch_distance

    # Identify prunable branch types
    types_to_prune = []
    if prune_endpoint_to_endpoint:
        types_to_prune.append(BranchType.ENDPOINT_TO_ENDPOINT)
    if prune_junction_to_endpoint:
        types_to_prune.append(BranchType.JUNCTION_TO_ENDPOINT)
    if prune_junction_to_junction:
        types_to_prune.append(BranchType.JUNCTION_TO_JUNCTION)
    if prune_isolated_cycles:
        types_to_prune.append(BranchType.ISOLATED_CYCLE)

    # Branches of prunable types
    prunable_type = summary['branch_type'].isin(types_to_prune)

    # Prune branches that are BOTH too short AND of a prunable type
    to_prune = too_short & prunable_type
    prune_indices = np.flatnonzero(to_prune)

    if len(prune_indices) == 0:
        # Nothing to prune
        return skeleton, summary

    # Use skan's prune_paths to remove branches
    pruned_skeleton = skeleton.prune_paths(prune_indices)

    # Re-summarize the pruned skeleton
    pruned_summary = skan.summarize(pruned_skeleton, separator='_')

    return pruned_skeleton, pruned_summary


def remove_isolated_cycles(
    skeleton: skan.Skeleton,
) -> tuple[skan.Skeleton, pd.DataFrame]:
    """
    Remove all isolated cycles (closed loops) from a skeleton.

    Isolated cycles are branches of type 3 in skan's classification.
    They often occur at junction points or from small holes in the
    original image.

    Parameters
    ----------
    skeleton : skan.Skeleton
        The skeleton to process.

    Returns
    -------
    pruned_skeleton : skan.Skeleton
        Skeleton with isolated cycles removed.
    summary : pd.DataFrame
        Summary of remaining branches.

    Examples
    --------
    >>> from ndev_morphology import remove_isolated_cycles
    >>> pruned, summary = remove_isolated_cycles(skeleton)
    """
    summary = skan.summarize(skeleton, separator='_')

    # Find all isolated cycles
    is_cycle = summary['branch_type'] == BranchType.ISOLATED_CYCLE
    cycle_indices = np.flatnonzero(is_cycle)

    if len(cycle_indices) == 0:
        return skeleton, summary

    pruned_skeleton = skeleton.prune_paths(cycle_indices)
    pruned_summary = skan.summarize(pruned_skeleton, separator='_')

    return pruned_skeleton, pruned_summary


def prune_skeleton_to_image(
    skeleton: skan.Skeleton,
    prune_indices: ArrayLike,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """
    Prune a skeleton and convert back to a binary image.

    This is useful when you want to re-skeletonize after pruning
    to get optimal junction resolution.

    Parameters
    ----------
    skeleton : skan.Skeleton
        The skeleton to prune.
    prune_indices : ArrayLike
        Indices of branches to remove.
    shape : tuple, optional
        Shape of output image. If None, uses skeleton.coordinates bounds.

    Returns
    -------
    np.ndarray
        Binary skeleton image with pruned branches removed.

    Notes
    -----
    To get a truly clean skeleton after pruning, you can:
    1. Use this function to get a binary image
    2. Optionally dilate slightly then re-skeletonize
    3. Create a new skan.Skeleton from the result

    Examples
    --------
    >>> from ndev_morphology.pruning import prune_skeleton_to_image
    >>> # Get branches to prune
    >>> summary = skan.summarize(skeleton)
    >>> too_short = summary['branch_distance'] < 5.0
    >>> # Create pruned image
    >>> pruned_image = prune_skeleton_to_image(skeleton, np.flatnonzero(too_short))
    """
    prune_indices = np.asarray(prune_indices)

    if len(prune_indices) == 0:
        # Return all paths as image
        return np.asarray(skeleton).astype(bool)

    pruned = skeleton.prune_paths(prune_indices)
    return np.asarray(pruned).astype(bool)
