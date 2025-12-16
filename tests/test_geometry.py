"""Tests for geometry module."""

import numpy as np
import pandas as pd
import pytest
import skan

from ndev_morphology import sholl_shells_to_ellipses, skeleton_to_paths


@pytest.fixture
def simple_skeleton():
    """Create a simple skeleton for testing."""
    skeleton = np.zeros((50, 50), dtype=bool)
    # Horizontal line
    skeleton[25, 10:40] = True
    # Vertical branch
    skeleton[15:35, 25] = True
    return skan.Skeleton(skeleton.astype(float), spacing=(1.0, 1.0))


class TestSkeletonToPaths:
    """Tests for skeleton_to_paths function."""

    def test_returns_paths_and_properties(self, simple_skeleton):
        """Returns list of paths and DataFrame of properties."""
        paths, properties = skeleton_to_paths(simple_skeleton)

        assert isinstance(paths, list)
        assert isinstance(properties, pd.DataFrame)

    def test_paths_are_coordinate_arrays(self, simple_skeleton):
        """Each path is a numpy array of coordinates."""
        paths, _ = skeleton_to_paths(simple_skeleton)

        assert len(paths) > 0
        for path in paths:
            assert isinstance(path, np.ndarray)
            assert path.ndim == 2
            assert path.shape[1] == 2  # (Y, X) coordinates

    def test_properties_has_path_id(self, simple_skeleton):
        """Properties DataFrame includes path_id column."""
        _, properties = skeleton_to_paths(simple_skeleton)

        assert 'path_id' in properties.columns
        # path_id should be sequential
        expected_ids = np.arange(len(properties))
        np.testing.assert_array_equal(
            properties['path_id'].values, expected_ids
        )

    def test_properties_has_branch_distance(self, simple_skeleton):
        """Properties DataFrame includes branch_distance column."""
        _, properties = skeleton_to_paths(simple_skeleton)

        assert 'branch_distance' in properties.columns
        # All distances should be positive
        assert (properties['branch_distance'] >= 0).all()

    def test_number_of_paths_matches_properties(self, simple_skeleton):
        """Number of paths matches number of rows in properties."""
        paths, properties = skeleton_to_paths(simple_skeleton)

        assert len(paths) == len(properties)


class TestShollShellsToEllipses:
    """Tests for sholl_shells_to_ellipses function."""

    def test_returns_list_of_arrays(self):
        """Returns list of numpy arrays."""
        center = [50, 50]
        radii = [10, 20, 30]

        result = sholl_shells_to_ellipses(center, radii)

        assert isinstance(result, list)
        assert len(result) == 3

    def test_ellipse_bounding_box_shape_2d(self):
        """Each ellipse has 4 corner points in 2D."""
        center = [50, 50]
        radii = [10]

        result = sholl_shells_to_ellipses(center, radii, ndim=2)

        assert len(result) == 1
        ellipse = result[0]
        assert ellipse.shape == (4, 2)  # 4 corners, 2D coords

    def test_ellipse_bounding_box_shape_3d(self):
        """Each ellipse has 4 corner points in 3D."""
        center = [25, 50, 50]  # Z, Y, X
        radii = [10]

        result = sholl_shells_to_ellipses(center, radii, ndim=3)

        assert len(result) == 1
        ellipse = result[0]
        assert ellipse.shape == (4, 3)  # 4 corners, 3D coords

    def test_ellipse_corners_2d(self):
        """2D ellipse corners are computed correctly."""
        center = [100, 150]  # Y, X
        radii = [10]

        result = sholl_shells_to_ellipses(center, radii, ndim=2)

        ellipse = result[0]
        # Expected corners for radius 10:
        # top-left: (90, 140), top-right: (90, 160)
        # bottom-right: (110, 160), bottom-left: (110, 140)
        expected = np.array(
            [
                [90, 140],  # top-left
                [90, 160],  # top-right
                [110, 160],  # bottom-right
                [110, 140],  # bottom-left
            ]
        )
        np.testing.assert_array_equal(ellipse, expected)

    def test_multiple_radii(self):
        """Multiple radii produce multiple ellipses."""
        center = [50, 50]
        radii = [10, 20, 30, 40, 50]

        result = sholl_shells_to_ellipses(center, radii)

        assert len(result) == 5

    def test_ellipse_size_increases_with_radius(self):
        """Larger radii produce larger ellipse bounding boxes."""
        center = [50, 50]
        radii = [10, 20]

        result = sholl_shells_to_ellipses(center, radii)

        # Check that the second ellipse is larger
        # By comparing the span from top-left to bottom-right
        ellipse_small = result[0]
        ellipse_large = result[1]

        span_small = ellipse_small[2, 0] - ellipse_small[0, 0]  # Y span
        span_large = ellipse_large[2, 0] - ellipse_large[0, 0]

        assert span_large > span_small
