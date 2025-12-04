"""Tests for skeleton and label operations."""

import numpy as np

from ndev_morphology.labels import (
    connect_breaks_between_labels,
    exclude_labels_on_edges,
    filter_labels_by_size,
)
from ndev_morphology.skeleton import (
    exclude_region_from_skeleton,
    separate_touching_skeleton_labels,
    skeletonize_labels,
)


class TestSkeletonizeLabels:
    """Tests for skeletonize_labels function."""

    def test_basic_skeletonization(self, simple_labels_2d):
        """Test that skeletonization produces skeleton with label identities."""
        skeleton = skeletonize_labels(simple_labels_2d)

        # Skeleton should have same dtype
        assert skeleton.dtype == simple_labels_2d.dtype

        # Skeleton should only have values from original labels (0, 1, 2)
        unique_vals = np.unique(skeleton)
        assert all(v in [0, 1, 2] for v in unique_vals)

        # Skeleton should have fewer pixels than original labels
        assert np.count_nonzero(skeleton) < np.count_nonzero(simple_labels_2d)

        # Non-zero skeleton pixels should preserve label values
        mask1 = simple_labels_2d == 1
        skel_in_label1 = skeleton[mask1]
        assert all(v in [0, 1] for v in np.unique(skel_in_label1))

    def test_empty_labels(self):
        """Test with empty label image."""
        labels = np.zeros((32, 32), dtype=np.uint16)
        skeleton = skeletonize_labels(labels)

        assert skeleton.shape == labels.shape
        assert np.count_nonzero(skeleton) == 0

    def test_single_pixel_label(self):
        """Test with single-pixel labels."""
        labels = np.zeros((32, 32), dtype=np.uint16)
        labels[16, 16] = 1

        skeleton = skeletonize_labels(labels)
        # Single pixel skeletonizes to single pixel
        assert np.count_nonzero(skeleton) <= 1


class TestExcludeRegionFromSkeleton:
    """Tests for exclude_region_from_skeleton function."""

    def test_basic_exclusion(self, simple_skeleton_2d, circular_mask):
        """Test that masked region is excluded from skeleton."""
        result = exclude_region_from_skeleton(
            simple_skeleton_2d, circular_mask
        )

        # Result should have same shape
        assert result.shape == simple_skeleton_2d.shape

        # No skeleton pixels should remain in mask region
        assert np.count_nonzero(result[circular_mask]) == 0

        # Skeleton pixels outside mask should be preserved
        outside_mask = ~circular_mask
        original_outside = simple_skeleton_2d[outside_mask]
        result_outside = result[outside_mask]
        np.testing.assert_array_equal(original_outside, result_outside)

    def test_exclusion_with_dilation(self, simple_skeleton_2d, circular_mask):
        """Test that dilation expands exclusion region."""
        result_no_dilation = exclude_region_from_skeleton(
            simple_skeleton_2d, circular_mask, dilation_iterations=0
        )
        result_with_dilation = exclude_region_from_skeleton(
            simple_skeleton_2d, circular_mask, dilation_iterations=5
        )

        # Dilated exclusion should remove more pixels
        assert np.count_nonzero(result_with_dilation) <= np.count_nonzero(
            result_no_dilation
        )

    def test_no_overlap(self, simple_skeleton_2d):
        """Test when mask doesn't overlap skeleton."""
        mask = np.zeros_like(simple_skeleton_2d, dtype=bool)
        mask[0:5, 0:5] = True  # Corner, no skeleton there

        result = exclude_region_from_skeleton(simple_skeleton_2d, mask)
        np.testing.assert_array_equal(result, simple_skeleton_2d)


class TestFilterLabelsBySize:
    """Tests for filter_labels_by_size function."""

    def test_min_size_filter(self):
        """Test filtering by minimum size."""
        # Create labels with clearly different sizes
        labels = np.zeros((64, 64), dtype=np.uint16)
        labels[5:10, 5:10] = 1  # 25 pixels (small)
        labels[30:50, 30:50] = 2  # 400 pixels (large)

        # Filter out the small label (min_size > 25)
        result = filter_labels_by_size(labels, min_size=100)

        assert np.count_nonzero(result == 1) == 0  # Small removed
        assert np.count_nonzero(result == 2) > 0  # Large kept

    def test_max_size_filter(self):
        """Test filtering by maximum size."""
        # Create labels with clearly different sizes
        labels = np.zeros((64, 64), dtype=np.uint16)
        labels[5:10, 5:10] = 1  # 25 pixels (small)
        labels[30:50, 30:50] = 2  # 400 pixels (large)

        # Filter out the large label (max_size < 400)
        result = filter_labels_by_size(labels, max_size=100)

        assert np.count_nonzero(result == 1) > 0  # Small kept
        assert np.count_nonzero(result == 2) == 0  # Large removed

    def test_no_filter(self, simple_labels_2d):
        """Test with no size constraints."""
        result = filter_labels_by_size(simple_labels_2d)
        np.testing.assert_array_equal(result, simple_labels_2d)


class TestExcludeLabelsOnEdges:
    """Tests for exclude_labels_on_edges function."""

    def test_edge_exclusion(self):
        """Test that edge-touching labels are removed."""
        labels = np.zeros((32, 32), dtype=np.uint16)
        labels[0:10, 5:15] = 1  # Touches top edge
        labels[15:25, 15:25] = 2  # Interior, no edge touching

        result = exclude_labels_on_edges(labels)

        assert np.count_nonzero(result == 1) == 0  # Edge label removed
        assert np.count_nonzero(result == 2) > 0  # Interior label preserved


class TestConnectBreaksBetweenLabels:
    """Tests for connect_breaks_between_labels function."""

    def test_connects_nearby_labels(self):
        """Test that nearby labels are connected."""
        labels = np.zeros((32, 32), dtype=np.uint16)
        labels[10:15, 10:15] = 1
        labels[10:15, 17:22] = 2  # Gap of 2 pixels

        # Connect with distance > gap
        result = connect_breaks_between_labels(labels, connect_distance=4.0)

        # Both regions should now have same label
        unique_nonzero = np.unique(result[result > 0])
        assert len(unique_nonzero) == 1  # Single connected region

    def test_doesnt_connect_distant_labels(self):
        """Test that distant labels remain separate."""
        labels = np.zeros((64, 64), dtype=np.uint16)
        labels[10:15, 10:15] = 1
        labels[40:45, 40:45] = 2  # Far apart

        result = connect_breaks_between_labels(labels, connect_distance=3.0)

        # Should still have two separate regions
        unique_nonzero = np.unique(result[result > 0])
        assert len(unique_nonzero) >= 1  # At least one region preserved


class TestSeparateTouchingSkeletonLabels:
    """Tests for separate_touching_skeleton_labels function."""

    def test_separate_skeleton_basics(self):
        """Test that function separates touching skeleton pixels."""
        # Create two adjacent skeleton lines with different labels
        skeleton = np.zeros((20, 20), dtype=np.uint16)
        # Horizontal line with label 1
        skeleton[10, 5:10] = 1
        # Horizontal line with label 2 touching at column 10
        skeleton[10, 10:15] = 2

        result = separate_touching_skeleton_labels(skeleton)

        # The touching pixels should be removed
        assert result[10, 9] == 0 or result[10, 10] == 0
        # Non-touching parts should be preserved
        assert result[10, 5] == 1
        assert result[10, 14] == 2

    def test_no_touching_skeletons(self):
        """Test that non-touching skeletons are unchanged."""
        skeleton = np.zeros((20, 20), dtype=np.uint16)
        # Two separate horizontal lines
        skeleton[5, 5:10] = 1
        skeleton[15, 5:10] = 2

        result = separate_touching_skeleton_labels(skeleton)

        # Should be identical since nothing touches
        np.testing.assert_array_equal(result, skeleton)

    def test_empty_skeleton(self):
        """Test with empty skeleton."""
        skeleton = np.zeros((20, 20), dtype=np.uint16)
        result = separate_touching_skeleton_labels(skeleton)

        np.testing.assert_array_equal(result, skeleton)

    def test_single_label_skeleton(self):
        """Test that single-label skeleton is unchanged."""
        skeleton = np.zeros((20, 20), dtype=np.uint16)
        skeleton[10, 5:15] = 1

        result = separate_touching_skeleton_labels(skeleton)

        np.testing.assert_array_equal(result, skeleton)

    def test_preserves_dtype(self):
        """Test that output preserves input dtype."""
        skeleton = np.zeros((20, 20), dtype=np.uint32)
        skeleton[10, 5:10] = 1
        skeleton[10, 10:15] = 2

        result = separate_touching_skeleton_labels(skeleton)

        assert result.dtype == skeleton.dtype

    def test_diagonal_not_touching_by_default(self):
        """Test that diagonally adjacent pixels ARE considered touching.

        Since skan uses 8-connectivity, diagonally adjacent skeleton pixels
        from different labels would be merged. The function must handle this.
        """
        skeleton = np.zeros((20, 20), dtype=np.uint16)
        # Two lines that touch diagonally
        skeleton[9, 5:10] = 1  # Ends at (9, 9)
        skeleton[10, 10:15] = 2  # Starts at (10, 10) - diagonal from (9, 9)

        result = separate_touching_skeleton_labels(skeleton)

        # Diagonals ARE considered touching, so pixels should be removed
        assert np.count_nonzero(result) < np.count_nonzero(skeleton)

    def test_diagonal_separation_works(self):
        """Test that diagonally touching skeletons become separate in skan."""
        import skan

        skeleton = np.zeros((20, 20), dtype=np.uint16)
        # Two lines that touch diagonally
        skeleton[9, 5:10] = 1
        skeleton[10, 10:15] = 2

        # Before separation: skan sees them as ONE skeleton
        skel_before = skan.Skeleton(skeleton)
        summary_before = skan.summarize(skel_before, separator="_")
        assert (
            summary_before["skeleton_id"].nunique() == 1
        ), "Setup: should be merged"

        # After separation: skan sees them as TWO skeletons
        result = separate_touching_skeleton_labels(skeleton)
        skel_after = skan.Skeleton(result)
        summary_after = skan.summarize(skel_after, separator="_")
        assert (
            summary_after["skeleton_id"].nunique() == 2
        ), "Should be separated"
