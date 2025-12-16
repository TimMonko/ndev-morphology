"""Tests for soma module."""

import numpy as np
import pytest
import skan

from ndev_morphology import (
    detect_soma_centroid,
    find_soma_node,
    get_label_centroid,
)


@pytest.fixture
def multi_region_labels():
    """Create label image with multiple regions of different sizes."""
    labels = np.zeros((100, 100), dtype=np.uint16)
    # Small region
    labels[10:15, 10:15] = 1  # 25 pixels
    # Large region (soma-like)
    labels[40:60, 40:60] = 2  # 400 pixels
    # Medium region
    labels[70:80, 70:85] = 3  # 150 pixels
    return labels


@pytest.fixture
def circular_and_elongated_labels():
    """Create label image with circular and elongated regions."""
    labels = np.zeros((100, 100), dtype=np.uint16)

    # Elongated region (high eccentricity)
    labels[40:45, 10:50] = 1  # 5x40 = 200 pixels

    # Circular region (low eccentricity)
    center = (50, 70)
    radius = 10
    y, x = np.ogrid[:100, :100]
    mask = (y - center[0]) ** 2 + (x - center[1]) ** 2 <= radius**2
    labels[mask] = 2  # ~314 pixels

    return labels


@pytest.fixture
def intensity_image_with_peak():
    """Create intensity image with a distinct peak inside labeled region."""
    img = np.zeros((100, 100), dtype=np.float32)
    # Low background
    img += 10.0
    # Bright peak at (50, 50) - inside the large region (40:60, 40:60)
    img[48:52, 48:52] = 200.0
    return img


@pytest.fixture
def simple_skeleton_for_soma():
    """Create skeleton radiating from a center point."""
    skeleton = np.zeros((100, 100), dtype=bool)
    center = (50, 50)

    # Create rays from center
    for angle in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        for r in range(5, 40):
            y = int(center[0] + r * np.sin(angle))
            x = int(center[1] + r * np.cos(angle))
            if 0 <= y < 100 and 0 <= x < 100:
                skeleton[y, x] = True

    return skan.Skeleton(skeleton.astype(float), spacing=(1.0, 1.0))


class TestDetectSomaCentroid:
    """Tests for detect_soma_centroid function."""

    def test_largest_region_method(self, multi_region_labels):
        """Finds centroid of largest region."""
        result = detect_soma_centroid(
            multi_region_labels, method='largest_region'
        )

        # Largest region is label 2 at (40:60, 40:60), centroid ~(50, 50)
        assert result[0] == pytest.approx(49.5, abs=1.0)
        assert result[1] == pytest.approx(49.5, abs=1.0)

    def test_roundest_region_method(self, circular_and_elongated_labels):
        """Finds centroid of most circular region."""
        result = detect_soma_centroid(
            circular_and_elongated_labels, method='roundest_region'
        )

        # Circular region is label 2 centered at (50, 70)
        assert result[0] == pytest.approx(50, abs=2.0)
        assert result[1] == pytest.approx(70, abs=2.0)

    def test_intensity_peak_method(
        self, multi_region_labels, intensity_image_with_peak
    ):
        """Finds location of peak intensity."""
        result = detect_soma_centroid(
            multi_region_labels,
            method='intensity_peak',
            intensity_image=intensity_image_with_peak,
        )

        # Peak is at approximately (50, 50) inside label 2 region
        assert result[0] == pytest.approx(50, abs=2.0)
        assert result[1] == pytest.approx(50, abs=2.0)

    def test_intensity_peak_requires_image(self, multi_region_labels):
        """Raises error when intensity_peak method used without image."""
        with pytest.raises(ValueError, match='intensity_image is required'):
            detect_soma_centroid(multi_region_labels, method='intensity_peak')

    def test_label_id_filter(self, multi_region_labels):
        """Can filter to specific label."""
        result = detect_soma_centroid(
            multi_region_labels, method='largest_region', label_id=1
        )

        # Label 1 is at (10:15, 10:15), centroid at (12, 12)
        assert result[0] == pytest.approx(12, abs=1.0)
        assert result[1] == pytest.approx(12, abs=1.0)

    def test_empty_labels_raises(self):
        """Raises error for empty label image."""
        labels = np.zeros((50, 50), dtype=np.uint16)

        with pytest.raises(ValueError, match='No regions found'):
            detect_soma_centroid(labels)


class TestGetLabelCentroid:
    """Tests for get_label_centroid function."""

    def test_returns_correct_centroid(self, multi_region_labels):
        """Returns centroid of specified label."""
        result = get_label_centroid(multi_region_labels, label_id=2)

        # Label 2 is at (40:60, 40:60), centroid at (49.5, 49.5)
        assert result[0] == pytest.approx(49.5, abs=0.1)
        assert result[1] == pytest.approx(49.5, abs=0.1)

    def test_missing_label_raises(self, multi_region_labels):
        """Raises error for non-existent label."""
        with pytest.raises(ValueError, match='Label 99 not found'):
            get_label_centroid(multi_region_labels, label_id=99)


class TestFindSomaNode:
    """Tests for find_soma_node function."""

    def test_finds_closest_node(self, simple_skeleton_for_soma):
        """Finds skeleton node closest to given centroid."""
        soma_centroid = np.array([50.0, 50.0])

        result = find_soma_node(simple_skeleton_for_soma, soma_centroid)

        assert isinstance(result, int)
        assert result >= 0
        assert result < len(simple_skeleton_for_soma.coordinates)

        # The closest node should be near the center
        node_coords = simple_skeleton_for_soma.coordinates[result]
        distance = np.linalg.norm(node_coords - soma_centroid)
        assert distance < 10.0  # Should be close to center

    def test_returns_int(self, simple_skeleton_for_soma):
        """Returns an integer node index."""
        soma_centroid = np.array([50.0, 50.0])

        result = find_soma_node(simple_skeleton_for_soma, soma_centroid)

        assert isinstance(result, int)
