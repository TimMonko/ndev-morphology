"""
Per-cell analysis pipelines for morphology analysis.

Functions for analyzing individual cells and batch processing
multi-cell images. Combines skeleton, branch, Sholl, and soma
analysis into unified workflows.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import skan

from .branch import BranchType, summarize_branches
from .sholl import ShollResult, compute_sholl_profile
from .skeleton import skeletonize_labels
from .soma import detect_soma_centroid, get_label_centroid

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

__all__ = [
    'CellAnalysisResult',
    'analyze_single_cell',
    'analyze_all_cells',
    'analyze_all_cells_generator',
    'aggregate_branch_stats',
]


@dataclass
class CellAnalysisResult:
    """
    Container for single-cell morphology analysis results.

    Attributes
    ----------
    label_id : int
        Original label ID from the input image.
    skeleton : skan.Skeleton
        The skan.Skeleton object for this cell.
    branches : pd.DataFrame
        Branch properties from summarize_branches().
    sholl : ShollResult | None
        Sholl analysis result, if soma was detected.
    soma_centroid : np.ndarray | None
        Soma centroid coordinates used for Sholl analysis.
    summary : dict
        Aggregate statistics for this cell.
    """

    label_id: int
    skeleton: skan.Skeleton
    branches: pd.DataFrame
    sholl: ShollResult | None
    soma_centroid: np.ndarray | None
    summary: dict = field(default_factory=dict)

    def __post_init__(self):
        """Compute summary statistics."""
        if not self.summary:
            self.summary = self._compute_summary()

    def _compute_summary(self) -> dict:
        """Compute aggregate statistics for the cell."""
        summary = {
            'label_id': self.label_id,
            'n_branches': len(self.branches),
            'total_length': self.branches['branch_distance'].sum(),
            'mean_branch_length': self.branches['branch_distance'].mean(),
        }

        # Count by branch type
        for bt in BranchType:
            count = (self.branches['branch_type'] == bt.value).sum()
            summary[f'n_{bt.name.lower()}'] = count

        # Tortuosity stats
        if 'tortuosity' in self.branches.columns:
            summary['mean_tortuosity'] = self.branches['tortuosity'].mean()
            summary['max_tortuosity'] = self.branches['tortuosity'].max()

        # Sholl stats
        if self.sholl is not None:
            summary.update(self.sholl.to_dict())

        return summary

    def to_dict(self) -> dict:
        """Convert summary to dict for DataFrame creation."""
        return self.summary


def analyze_single_cell(
    labels: ArrayLike,
    label_id: int | None = None,
    *,
    spacing: tuple[float, ...] = (1.0, 1.0),
    soma_labels: ArrayLike | None = None,
    intensity_image: ArrayLike | None = None,
    run_sholl: bool = True,
    sholl_step: float = 1.0,
) -> CellAnalysisResult:
    """
    Comprehensive morphology analysis of a single labeled cell.

    Performs skeletonization, branch analysis, and optionally Sholl analysis
    on a single cell identified by label_id.

    Parameters
    ----------
    labels : ArrayLike
        Label image. If label_id is None, treats all non-zero pixels as one cell.
    label_id : int, optional
        Specific label ID to analyze. If None, uses the entire non-zero region.
    spacing : tuple of float
        Physical pixel spacing (Y, X) or (Z, Y, X). Default is (1.0, 1.0).
    soma_labels : ArrayLike, optional
        Separate label image for soma detection (e.g., DAPI-based nuclei).
        If not provided, uses centroid of the cell mask.
    intensity_image : ArrayLike, optional
        Intensity image for measuring signal along branches.
    run_sholl : bool
        Whether to perform Sholl analysis. Default is True.
    sholl_step : float
        Step size for Sholl radii. Default is 1.0 (in physical units).

    Returns
    -------
    CellAnalysisResult
        Container with skeleton, branches, sholl, and summary statistics.

    Examples
    --------
    >>> from ndev_morphology import analyze_single_cell
    >>> result = analyze_single_cell(
    ...     cell_labels,
    ...     label_id=1,
    ...     spacing=(0.2, 0.2),
    ...     soma_labels=nuclei_labels,
    ... )
    >>> print(f"Total length: {result.summary['total_length']:.1f} µm")
    >>> print(f"Branch count: {result.summary['n_branches']}")
    """
    labels = np.asarray(labels)

    # Extract single cell mask
    if label_id is not None:
        cell_mask = (labels == label_id).astype(np.uint16)
        cell_mask[cell_mask > 0] = label_id
    else:
        cell_mask = (labels > 0).astype(np.uint16)
        label_id = 1
        cell_mask[cell_mask > 0] = label_id

    # Skeletonize
    skeleton_labels = skeletonize_labels(cell_mask)
    skeleton_binary = skeleton_labels > 0

    # Check if skeleton is empty
    if not np.any(skeleton_binary):
        raise ValueError(
            f'Label {label_id} has no skeleton pixels. '
            f'The cell may be too small or have no elongated structure.'
        )

    # Create skan.Skeleton
    skeleton = skan.Skeleton(skeleton_binary.astype(float), spacing=spacing)

    # Summarize branches
    branches = summarize_branches(
        skeleton,
        intensity_image=intensity_image,
    )
    branches['label_id'] = label_id

    # Detect soma centroid
    soma_centroid = None
    sholl_result = None

    if run_sholl:
        try:
            if soma_labels is not None:
                soma_centroid = detect_soma_centroid(
                    soma_labels,
                    label_id=label_id,
                    method='largest_region',
                )
            else:
                soma_centroid = get_label_centroid(cell_mask, label_id)

            # Convert to physical units if spacing provided
            soma_centroid_physical = soma_centroid * np.array(spacing)

            # Run Sholl analysis
            sholl_result = compute_sholl_profile(
                skeleton,
                center=soma_centroid_physical,
                step=sholl_step,
            )
        except ValueError:
            # No soma found or empty skeleton
            pass

    return CellAnalysisResult(
        label_id=label_id,
        skeleton=skeleton,
        branches=branches,
        sholl=sholl_result,
        soma_centroid=soma_centroid,
    )


def analyze_all_cells(
    labels: ArrayLike,
    *,
    spacing: tuple[float, ...] = (1.0, 1.0),
    soma_labels: ArrayLike | None = None,
    intensity_image: ArrayLike | None = None,
    run_sholl: bool = True,
    sholl_step: float = 1.0,
    min_skeleton_size: int = 10,
) -> tuple[pd.DataFrame, list[CellAnalysisResult]]:
    """
    Analyze all labeled cells in an image.

    Iterates over each unique label and performs comprehensive morphology
    analysis, returning both summary statistics and detailed results.

    Parameters
    ----------
    labels : ArrayLike
        Label image where each connected region has a unique integer ID.
    spacing : tuple of float
        Physical pixel spacing. Default is (1.0, 1.0).
    soma_labels : ArrayLike, optional
        Separate label image for soma detection.
    intensity_image : ArrayLike, optional
        Intensity image for measuring signal along branches.
    run_sholl : bool
        Whether to perform Sholl analysis. Default is True.
    sholl_step : float
        Step size for Sholl radii. Default is 1.0.
    min_skeleton_size : int
        Minimum skeleton size (in pixels) to include. Default is 10.

    Returns
    -------
    summary_df : pd.DataFrame
        DataFrame with one row per cell and columns for all summary metrics.
    results : list of CellAnalysisResult
        List of detailed results for each cell.

    Examples
    --------
    >>> from ndev_morphology import analyze_all_cells
    >>> summary, results = analyze_all_cells(
    ...     cell_labels,
    ...     spacing=(0.2, 0.2),
    ...     soma_labels=nuclei_labels,
    ... )
    >>> # Save summary to CSV
    >>> summary.to_csv('morphology_results.csv', index=False)
    >>> # Access individual cell skeleton
    >>> cell_1_skeleton = results[0].skeleton
    """
    labels = np.asarray(labels)

    # Get unique label IDs (excluding 0)
    label_ids = np.unique(labels)
    label_ids = label_ids[label_ids > 0]

    results = []
    summaries = []

    for label_id in label_ids:
        try:
            result = analyze_single_cell(
                labels,
                label_id=int(label_id),
                spacing=spacing,
                soma_labels=soma_labels,
                intensity_image=intensity_image,
                run_sholl=run_sholl,
                sholl_step=sholl_step,
            )

            # Skip cells with too-small skeletons
            if result.skeleton.n_paths < 1:
                continue

            skeleton_size = np.sum(result.skeleton.path_lengths())
            if skeleton_size < min_skeleton_size:
                continue

            results.append(result)
            summaries.append(result.to_dict())

        except Exception:
            # Skip cells that fail analysis
            continue

    summary_df = pd.DataFrame(summaries) if summaries else pd.DataFrame()

    return summary_df, results


def analyze_all_cells_generator(
    labels: ArrayLike,
    *,
    spacing: tuple[float, ...] = (1.0, 1.0),
    soma_labels: ArrayLike | None = None,
    intensity_image: ArrayLike | None = None,
    run_sholl: bool = True,
    sholl_step: float = 1.0,
    min_skeleton_size: int = 10,
    on_progress: Callable[[int, int, int], None] | None = None,
):
    """
    Generator version of analyze_all_cells for batch processing.

    Yields results one at a time, allowing for progress tracking
    and cancellation. Compatible with nbatch.BatchRunner.

    Parameters
    ----------
    labels : ArrayLike
        Label image where each connected region has a unique integer ID.
    spacing : tuple of float
        Physical pixel spacing. Default is (1.0, 1.0).
    soma_labels : ArrayLike, optional
        Separate label image for soma detection.
    intensity_image : ArrayLike, optional
        Intensity image for measuring signal along branches.
    run_sholl : bool
        Whether to perform Sholl analysis. Default is True.
    sholl_step : float
        Step size for Sholl radii. Default is 1.0.
    min_skeleton_size : int
        Minimum skeleton size (in pixels) to include. Default is 10.
    on_progress : callable, optional
        Callback(current, total, label_id) called after each cell.

    Yields
    ------
    CellAnalysisResult
        Result for each successfully analyzed cell.

    Examples
    --------
    >>> from ndev_morphology import analyze_all_cells_generator
    >>> results = []
    >>> for result in analyze_all_cells_generator(labels, spacing=(0.2, 0.2)):
    ...     results.append(result)
    ...     print(f"Analyzed cell {result.label_id}")
    """
    labels = np.asarray(labels)

    # Get unique label IDs (excluding 0)
    label_ids = np.unique(labels)
    label_ids = label_ids[label_ids > 0]
    total = len(label_ids)

    for idx, label_id in enumerate(label_ids):
        try:
            result = analyze_single_cell(
                labels,
                label_id=int(label_id),
                spacing=spacing,
                soma_labels=soma_labels,
                intensity_image=intensity_image,
                run_sholl=run_sholl,
                sholl_step=sholl_step,
            )

            # Skip cells with too-small skeletons
            if result.skeleton.n_paths < 1:
                continue

            skeleton_size = np.sum(result.skeleton.path_lengths())
            if skeleton_size < min_skeleton_size:
                continue

            if on_progress is not None:
                on_progress(idx + 1, total, int(label_id))

            yield result

        except Exception:
            # Skip cells that fail analysis
            if on_progress is not None:
                on_progress(idx + 1, total, int(label_id))
            continue


def aggregate_branch_stats(
    results: list[CellAnalysisResult],
) -> pd.DataFrame:
    """
    Combine branch DataFrames from multiple cells.

    Useful for population-level branch analysis.

    Parameters
    ----------
    results : list of CellAnalysisResult
        List of analysis results from analyze_all_cells or multiple
        analyze_single_cell calls.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with all branches from all cells.
        Includes 'label_id' column to identify source cell.

    Examples
    --------
    >>> summary, results = analyze_all_cells(labels)
    >>> all_branches = aggregate_branch_stats(results)
    >>> # Plot branch length distribution
    >>> all_branches['branch_distance'].hist()
    """
    if not results:
        return pd.DataFrame()

    all_branches = pd.concat(
        [r.branches for r in results],
        ignore_index=True,
    )

    return all_branches
