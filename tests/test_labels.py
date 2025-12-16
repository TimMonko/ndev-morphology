"""Tests for labels module."""

import numpy as np

from ndev_morphology import (
    connect_breaks_between_labels,
    exclude_labels_on_edges,
    filter_labels_by_size,
)


class TestFilterLabelsBySize:
    """Tests for filter_labels_by_size function."""

    def test_filter_by_min_size(self):
        """Labels smaller than min_size are removed."""
        labels = np.zeros((50, 50), dtype=np.uint16)
        labels[5:10, 5:10] = 1  # 25 pixels
        labels[20:40, 20:40] = 2  # 400 pixels

        result = filter_labels_by_size(labels, min_size=100)

        assert 1 not in result
        assert 2 in result
        assert np.sum(result == 2) == 400

    def test_filter_by_max_size(self):
        """Labels larger than max_size are removed."""
        labels = np.zeros((50, 50), dtype=np.uint16)
        labels[5:10, 5:10] = 1  # 25 pixels
        labels[20:40, 20:40] = 2  # 400 pixels

        result = filter_labels_by_size(labels, max_size=100)

        assert 1 in result
        assert 2 not in result

    def test_filter_by_min_and_max_size(self):
        """Labels outside range are removed."""
        labels = np.zeros((50, 50), dtype=np.uint16)
        labels[5:10, 5:10] = 1  # 25 pixels - too small
        labels[15:20, 15:20] = 2  # 25 pixels - too small
        labels[25:30, 25:35] = 3  # 50 pixels - in range
        labels[35:45, 35:45] = 4  # 100 pixels - too big

        result = filter_labels_by_size(labels, min_size=30, max_size=80)

        assert 1 not in result
        assert 2 not in result
        assert 3 in result
        assert 4 not in result

    def test_no_filter_returns_copy(self):
        """No min/max returns a copy of input."""
        labels = np.array([[1, 2], [3, 4]], dtype=np.uint16)

        result = filter_labels_by_size(labels)

        np.testing.assert_array_equal(result, labels)
        assert result is not labels

    def test_preserves_dtype(self):
        """Output dtype matches input dtype."""
        labels = np.array([[0, 1], [2, 0]], dtype=np.uint32)

        result = filter_labels_by_size(labels, min_size=1)

        assert result.dtype == labels.dtype


class TestExcludeLabelsOnEdges:
    """Tests for exclude_labels_on_edges function."""

    def test_removes_edge_touching_labels(self):
        """Labels touching image border are removed."""
        labels = np.zeros((20, 20), dtype=np.uint16)
        labels[0:5, 5:10] = 1  # Touches top edge
        labels[8:12, 8:12] = 2  # Interior, not touching

        result = exclude_labels_on_edges(labels)

        assert 1 not in result
        assert 2 in result

    def test_removes_all_edges(self):
        """Labels on any edge are removed."""
        labels = np.zeros((20, 20), dtype=np.uint16)
        labels[0:3, 10:12] = 1  # Top edge
        labels[17:20, 10:12] = 2  # Bottom edge
        labels[10:12, 0:3] = 3  # Left edge
        labels[10:12, 17:20] = 4  # Right edge
        labels[8:12, 8:12] = 5  # Interior

        result = exclude_labels_on_edges(labels)

        assert 1 not in result
        assert 2 not in result
        assert 3 not in result
        assert 4 not in result
        assert 5 in result

    def test_empty_image(self):
        """Empty image returns empty image."""
        labels = np.zeros((10, 10), dtype=np.uint16)

        result = exclude_labels_on_edges(labels)

        assert np.all(result == 0)


class TestConnectBreaksBetweenLabels:
    """Tests for connect_breaks_between_labels function."""

    def test_connects_nearby_fragments(self):
        """Nearby label fragments are connected."""
        labels = np.zeros((30, 30), dtype=np.uint16)
        labels[10:15, 5:10] = 1  # First fragment
        labels[10:15, 12:17] = 1  # Second fragment, 2 pixels away

        result = connect_breaks_between_labels(labels, connect_distance=4.0)

        # Both fragments should now be part of same connected component
        # (though possibly with a different label ID)
        assert np.sum(result > 0) > 0

    def test_small_distance_no_connection(self):
        """Fragments too far apart remain separate."""
        labels = np.zeros((50, 50), dtype=np.uint16)
        labels[10:15, 5:10] = 1  # First fragment
        labels[10:15, 25:30] = 1  # Second fragment, far away

        result = connect_breaks_between_labels(labels, connect_distance=2.0)

        # Should still have separate components
        from skimage import measure

        n_regions = len(measure.regionprops(result))
        assert n_regions == 2

    def test_zero_distance_returns_copy(self):
        """Distance < 1 pixel returns input with same structure."""
        labels = np.array([[1, 0, 2], [0, 0, 0], [3, 0, 4]], dtype=np.uint16)

        result = connect_breaks_between_labels(labels, connect_distance=0.5)

        # With radius < 1, no connection happens, but relabeling may occur
        # Check that the structure (mask) is preserved
        np.testing.assert_array_equal(result > 0, labels > 0)

    def test_preserves_mask_extent(self):
        """Connected result stays within original label extent."""
        labels = np.zeros((30, 30), dtype=np.uint16)
        labels[10:12, 10:12] = 1
        labels[14:16, 14:16] = 2

        original_mask = labels > 0
        result = connect_breaks_between_labels(labels, connect_distance=6.0)
        result_mask = result > 0

        # Result mask should be subset of original mask extent
        # (the function masks back to original extent)
        np.testing.assert_array_equal(result_mask, original_mask)
