"""Tests for Sholl analysis functions."""

import numpy as np

from ndev_morphology.sholl import ShollResult, compute_sholl_profile


class TestShollResult:
    """Tests for ShollResult dataclass."""

    def test_dataclass_fields(self):
        """Test ShollResult has expected fields and computed properties."""
        # Only pass the core fields; computed properties are derived
        result = ShollResult(
            center=(16, 16),
            radii=np.array([1, 2, 3, 4, 5]),
            counts=np.array([0, 2, 4, 3, 1]),
        )

        assert result.center == (16, 16)
        assert len(result.radii) == 5
        assert len(result.counts) == 5
        # Computed properties
        assert result.max_crossings == 4  # max of counts
        assert result.critical_radius == 3  # radius at max crossings
        assert result.enclosing_radius == 5  # last radius with nonzero count
        assert result.mean_crossings == 2.0  # mean of counts
        assert result.total_crossings == 10  # sum of counts


class TestComputeShollProfile:
    """Tests for compute_sholl_profile function."""

    def test_basic_sholl_analysis(self, skeleton_with_center):
        """Test basic Sholl analysis."""
        skeleton, center = skeleton_with_center

        result = compute_sholl_profile(skeleton, center)

        assert isinstance(result, ShollResult)
        np.testing.assert_array_equal(result.center, center)
        assert len(result.radii) > 0
        assert len(result.counts) == len(result.radii)

    def test_radii_are_increasing(self, skeleton_with_center):
        """Test that radii are monotonically increasing."""
        skeleton, center = skeleton_with_center

        result = compute_sholl_profile(skeleton, center)

        if len(result.radii) > 1:
            diffs = np.diff(result.radii)
            assert np.all(diffs > 0)

    def test_counts_are_nonnegative(self, skeleton_with_center):
        """Test that counts are non-negative integers."""
        skeleton, center = skeleton_with_center

        result = compute_sholl_profile(skeleton, center)

        assert np.all(result.counts >= 0)

    def test_max_crossings_matches_counts(self, skeleton_with_center):
        """Test that max_crossings equals max of counts."""
        skeleton, center = skeleton_with_center

        result = compute_sholl_profile(skeleton, center)

        if len(result.counts) > 0:
            assert result.max_crossings == np.max(result.counts)

    def test_with_custom_radii(self, skeleton_with_center):
        """Test with custom radii specification."""
        skeleton, center = skeleton_with_center

        custom_radii = np.arange(1, 20, 2)  # 1, 3, 5, ..., 19
        result = compute_sholl_profile(skeleton, center, radii=custom_radii)

        # Should use the provided radii
        np.testing.assert_array_equal(result.radii, custom_radii)

    def test_empty_skeleton_raises(self):
        """Test that empty skeleton raises error from skan."""
        import pytest
        import skan

        skeleton_arr = np.zeros((32, 32), dtype=float)

        # skan.Skeleton with empty array raises sparse matrix error
        with pytest.raises(ValueError):
            skan.Skeleton(skeleton_arr)

    def test_total_crossings(self, skeleton_with_center):
        """Test that total_crossings is sum of counts."""
        skeleton, center = skeleton_with_center

        result = compute_sholl_profile(skeleton, center)

        assert result.total_crossings == np.sum(result.counts)

    def test_3d_sholl(self, simple_skeleton_3d):
        """Test Sholl analysis on 3D skeleton."""
        center = (8, 16, 16)

        result = compute_sholl_profile(simple_skeleton_3d, center)

        assert isinstance(result, ShollResult)
        assert len(result.center) == 3
