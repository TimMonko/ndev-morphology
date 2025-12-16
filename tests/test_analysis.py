"""Tests for analysis module."""

import numpy as np
import pandas as pd
import pytest

from ndev_morphology import (
    CellAnalysisResult,
    aggregate_branch_stats,
    analyze_all_cells,
    analyze_single_cell,
)


@pytest.fixture
def simple_cell_labels():
    """Create a simple cell-like label for testing."""
    labels = np.zeros((100, 100), dtype=np.uint16)

    # Cell body (soma-like)
    labels[45:55, 45:55] = 1

    # Dendrite-like extensions
    labels[40:46, 20:46] = 1  # Left branch
    labels[54:60, 54:80] = 1  # Right branch
    labels[20:46, 48:52] = 1  # Top branch
    labels[54:80, 48:52] = 1  # Bottom branch

    return labels


@pytest.fixture
def multi_cell_labels():
    """Create label image with multiple cells."""
    labels = np.zeros((200, 200), dtype=np.uint16)

    # Cell 1 - top left
    labels[20:30, 20:30] = 1
    labels[25:28, 30:60] = 1  # Branch

    # Cell 2 - top right
    labels[20:30, 120:130] = 2
    labels[25:28, 130:170] = 2  # Branch

    # Cell 3 - bottom center
    labels[150:160, 90:100] = 3
    labels[155:158, 50:90] = 3  # Left branch
    labels[155:158, 100:150] = 3  # Right branch

    return labels


class TestCellAnalysisResult:
    """Tests for CellAnalysisResult dataclass."""

    def test_summary_computed_on_init(self, simple_cell_labels):
        """Summary statistics are computed on initialization."""
        result = analyze_single_cell(simple_cell_labels, label_id=1)

        assert isinstance(result.summary, dict)
        assert 'label_id' in result.summary
        assert 'n_branches' in result.summary
        assert 'total_length' in result.summary

    def test_to_dict_returns_summary(self, simple_cell_labels):
        """to_dict returns the summary dictionary."""
        result = analyze_single_cell(simple_cell_labels, label_id=1)

        result_dict = result.to_dict()

        assert result_dict == result.summary


class TestAnalyzeSingleCell:
    """Tests for analyze_single_cell function."""

    def test_returns_cell_analysis_result(self, simple_cell_labels):
        """Returns a CellAnalysisResult object."""
        result = analyze_single_cell(simple_cell_labels, label_id=1)

        assert isinstance(result, CellAnalysisResult)

    def test_has_skeleton(self, simple_cell_labels):
        """Result contains skan.Skeleton object."""
        import skan

        result = analyze_single_cell(simple_cell_labels, label_id=1)

        assert isinstance(result.skeleton, skan.Skeleton)

    def test_has_branches_dataframe(self, simple_cell_labels):
        """Result contains branches DataFrame."""
        result = analyze_single_cell(simple_cell_labels, label_id=1)

        assert isinstance(result.branches, pd.DataFrame)
        assert len(result.branches) > 0

    def test_has_sholl_result(self, simple_cell_labels):
        """Result contains ShollResult when run_sholl=True."""
        from ndev_morphology import ShollResult

        result = analyze_single_cell(
            simple_cell_labels, label_id=1, run_sholl=True
        )

        assert isinstance(result.sholl, ShollResult)

    def test_no_sholl_when_disabled(self, simple_cell_labels):
        """Sholl is None when run_sholl=False."""
        result = analyze_single_cell(
            simple_cell_labels, label_id=1, run_sholl=False
        )

        assert result.sholl is None

    def test_spacing_used(self, simple_cell_labels):
        """Physical spacing is used in analysis."""
        result_pixel = analyze_single_cell(
            simple_cell_labels, label_id=1, spacing=(1.0, 1.0)
        )
        result_scaled = analyze_single_cell(
            simple_cell_labels, label_id=1, spacing=(0.5, 0.5)
        )

        # Total length should differ based on spacing
        assert (
            result_scaled.summary['total_length']
            < result_pixel.summary['total_length']
        )

    def test_no_label_id_uses_all_nonzero(self, simple_cell_labels):
        """When label_id is None, uses all non-zero pixels."""
        result = analyze_single_cell(simple_cell_labels, label_id=None)

        assert result.label_id == 1  # Gets set to 1 internally

    def test_branches_have_label_id_column(self, simple_cell_labels):
        """Branches DataFrame includes label_id column."""
        result = analyze_single_cell(simple_cell_labels, label_id=1)

        assert 'label_id' in result.branches.columns
        assert (result.branches['label_id'] == 1).all()


class TestAnalyzeAllCells:
    """Tests for analyze_all_cells function."""

    def test_returns_summary_and_results(self, multi_cell_labels):
        """Returns tuple of DataFrame and list of results."""
        summary, results = analyze_all_cells(multi_cell_labels)

        assert isinstance(summary, pd.DataFrame)
        assert isinstance(results, list)

    def test_one_row_per_cell(self, multi_cell_labels):
        """Summary has one row per analyzed cell."""
        summary, results = analyze_all_cells(multi_cell_labels)

        # Should have 3 cells
        assert len(summary) == 3
        assert len(results) == 3

    def test_results_match_summary(self, multi_cell_labels):
        """Results list matches summary DataFrame rows."""
        summary, results = analyze_all_cells(multi_cell_labels)

        assert len(summary) == len(results)
        for i, result in enumerate(results):
            assert result.label_id == summary.iloc[i]['label_id']

    def test_skips_small_skeletons(self, multi_cell_labels):
        """Skips cells with skeletons below min_skeleton_size."""
        # Add a tiny cell
        labels = multi_cell_labels.copy()
        labels[5:7, 5:7] = 4  # Very small, will have tiny skeleton

        summary, results = analyze_all_cells(labels, min_skeleton_size=20)

        # Should skip the tiny cell (label 4)
        label_ids = summary['label_id'].tolist()
        assert 4 not in label_ids

    def test_spacing_applied_to_all(self, multi_cell_labels):
        """Spacing is applied to all cells."""
        summary_pixel, _ = analyze_all_cells(
            multi_cell_labels, spacing=(1.0, 1.0)
        )
        summary_scaled, _ = analyze_all_cells(
            multi_cell_labels, spacing=(0.5, 0.5)
        )

        # Total lengths should differ
        assert (
            summary_scaled['total_length'].sum()
            < summary_pixel['total_length'].sum()
        )


class TestAggregateBranchStats:
    """Tests for aggregate_branch_stats function."""

    def test_combines_all_branches(self, multi_cell_labels):
        """Combines branches from all cells."""
        _, results = analyze_all_cells(multi_cell_labels)

        all_branches = aggregate_branch_stats(results)

        # Total branches should equal sum from all cells
        expected_count = sum(len(r.branches) for r in results)
        assert len(all_branches) == expected_count

    def test_preserves_label_id(self, multi_cell_labels):
        """Combined DataFrame preserves label_id column."""
        _, results = analyze_all_cells(multi_cell_labels)

        all_branches = aggregate_branch_stats(results)

        assert 'label_id' in all_branches.columns
        # Should have branches from multiple cells
        assert len(all_branches['label_id'].unique()) > 1

    def test_empty_results_returns_empty_df(self):
        """Returns empty DataFrame for empty results list."""
        result = aggregate_branch_stats([])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
