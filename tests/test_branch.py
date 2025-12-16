"""Tests for branch module."""

import numpy as np
import pandas as pd
import pytest
import skan

from ndev_morphology import (
    BranchType,
    compute_tortuosity,
    filter_branches_by_type,
    summarize_branches,
)


@pytest.fixture
def branching_skeleton():
    """Create a branching skeleton for testing."""
    skeleton = np.zeros((100, 100), dtype=bool)

    # Main horizontal trunk
    skeleton[50, 20:80] = True

    # Branch up-left
    for i in range(20):
        skeleton[50 - i, 30 + i] = True

    # Branch up-right
    for i in range(15):
        skeleton[50 - i, 60 - i] = True

    # Branch down
    skeleton[50:70, 50] = True

    return skan.Skeleton(skeleton.astype(float), spacing=(1.0, 1.0))


@pytest.fixture
def simple_intensity_image():
    """Create a simple intensity image for testing."""
    img = np.zeros((100, 100), dtype=np.float32)
    img[40:60, :] = 100.0  # High intensity band
    return img


class TestBranchType:
    """Tests for BranchType enum."""

    def test_branch_type_values(self):
        """Branch type values match skan conventions."""
        assert BranchType.ENDPOINT_TO_ENDPOINT == 0
        assert BranchType.JUNCTION_TO_ENDPOINT == 1
        assert BranchType.JUNCTION_TO_JUNCTION == 2
        assert BranchType.ISOLATED_CYCLE == 3

    def test_branch_type_label(self):
        """Branch type labels are human-readable."""
        assert BranchType.label(0) == 'endpoint-endpoint'
        assert BranchType.label(1) == 'junction-endpoint'
        assert BranchType.label(2) == 'junction-junction'
        assert BranchType.label(3) == 'isolated-cycle'
        assert BranchType.label(99) == 'unknown'


class TestSummarizeBranches:
    """Tests for summarize_branches function."""

    def test_returns_dataframe(self, branching_skeleton):
        """Returns a pandas DataFrame."""
        result = summarize_branches(branching_skeleton)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_has_expected_columns(self, branching_skeleton):
        """Result has expected columns from skan plus extras."""
        result = summarize_branches(branching_skeleton)

        # skan columns
        assert 'branch_distance' in result.columns
        assert 'euclidean_distance' in result.columns
        assert 'branch_type' in result.columns

        # Added columns
        assert 'branch_type_label' in result.columns
        assert 'tortuosity' in result.columns

    def test_tortuosity_computed(self, branching_skeleton):
        """Tortuosity is computed by default."""
        result = summarize_branches(branching_skeleton)

        assert 'tortuosity' in result.columns
        # Tortuosity should be >= 1.0 (path_length >= euclidean_distance)
        assert (result['tortuosity'] >= 1.0).all()

    def test_no_tortuosity_option(self, branching_skeleton):
        """Can disable tortuosity computation."""
        result = summarize_branches(
            branching_skeleton, compute_tortuosity=False
        )

        assert 'tortuosity' not in result.columns

    def test_branch_type_labels_added(self, branching_skeleton):
        """Human-readable branch type labels are added."""
        result = summarize_branches(branching_skeleton)

        assert 'branch_type_label' in result.columns
        # Check labels are valid
        valid_labels = {
            'endpoint-endpoint',
            'junction-endpoint',
            'junction-junction',
            'isolated-cycle',
        }
        assert all(
            label in valid_labels for label in result['branch_type_label']
        )

    def test_with_intensity_image(
        self, branching_skeleton, simple_intensity_image
    ):
        """Intensity measurements are added when image provided."""
        result = summarize_branches(
            branching_skeleton, intensity_image=simple_intensity_image
        )

        # Check for intensity columns (either our custom ones or skan's)
        has_mean = (
            'mean_intensity' in result.columns
            or 'mean_pixel_value' in result.columns
        )
        has_std = (
            'std_intensity' in result.columns
            or 'stdev_pixel_value' in result.columns
        )
        assert has_mean
        assert has_std


class TestComputeTortuosity:
    """Tests for compute_tortuosity function."""

    def test_straight_line_tortuosity_one(self):
        """Straight lines have tortuosity of 1.0."""
        branch_dist = np.array([10.0, 20.0, 5.0])
        euclidean_dist = np.array([10.0, 20.0, 5.0])

        result = compute_tortuosity(branch_dist, euclidean_dist)

        np.testing.assert_array_almost_equal(result, [1.0, 1.0, 1.0])

    def test_curved_branch_tortuosity_greater_than_one(self):
        """Curved branches have tortuosity > 1.0."""
        branch_dist = np.array([15.0, 25.0])
        euclidean_dist = np.array([10.0, 20.0])

        result = compute_tortuosity(branch_dist, euclidean_dist)

        assert result[0] == 1.5
        assert result[1] == 1.25

    def test_zero_euclidean_returns_one(self):
        """Zero euclidean distance returns 1.0 (not inf)."""
        branch_dist = np.array([5.0, 10.0])
        euclidean_dist = np.array([0.0, 10.0])

        result = compute_tortuosity(branch_dist, euclidean_dist)

        assert result[0] == 1.0
        assert result[1] == 1.0


class TestFilterBranchesByType:
    """Tests for filter_branches_by_type function."""

    @pytest.fixture
    def sample_branches(self):
        """Create sample branch DataFrame."""
        return pd.DataFrame(
            {
                'branch_type': [0, 1, 1, 2, 2, 2, 3],
                'branch_distance': [10, 20, 15, 25, 30, 22, 5],
            }
        )

    def test_filter_single_type_int(self, sample_branches):
        """Filter by single branch type as int."""
        result = filter_branches_by_type(sample_branches, 1)

        assert len(result) == 2
        assert (result['branch_type'] == 1).all()

    def test_filter_single_type_enum(self, sample_branches):
        """Filter by single branch type as enum."""
        result = filter_branches_by_type(
            sample_branches, BranchType.JUNCTION_TO_JUNCTION
        )

        assert len(result) == 3
        assert (result['branch_type'] == 2).all()

    def test_filter_multiple_types(self, sample_branches):
        """Filter by multiple branch types."""
        result = filter_branches_by_type(
            sample_branches,
            [BranchType.JUNCTION_TO_ENDPOINT, BranchType.JUNCTION_TO_JUNCTION],
        )

        assert len(result) == 5
        assert set(result['branch_type'].unique()) == {1, 2}

    def test_returns_copy(self, sample_branches):
        """Returns a copy, not a view."""
        result = filter_branches_by_type(sample_branches, 1)

        result['branch_distance'] = 999
        assert sample_branches['branch_distance'].iloc[1] != 999
