"""Shared test fixtures for ndev-morphology tests."""

import numpy as np
import pytest
import skan


@pytest.fixture
def simple_labels_2d():
    """Simple 2D label image with two rectangular regions."""
    labels = np.zeros((64, 64), dtype=np.uint16)
    labels[10:20, 10:30] = 1  # Horizontal rectangle
    labels[30:50, 25:35] = 2  # Vertical rectangle
    return labels


@pytest.fixture
def simple_skeleton_2d():
    """Simple 2D binary skeleton (cross shape)."""
    skeleton = np.zeros((64, 64), dtype=bool)
    # Vertical line
    skeleton[10:50, 32] = True
    # Horizontal line
    skeleton[30, 10:55] = True
    return skeleton


@pytest.fixture
def branching_skeleton_2d():
    """2D skeleton with multiple branches for testing."""
    skeleton = np.zeros((100, 100), dtype=bool)

    # Main trunk
    skeleton[50, 20:80] = True

    # Branch 1: up-left
    for i in range(20):
        skeleton[50 - i, 30 + i] = True

    # Branch 2: up-right
    for i in range(15):
        skeleton[50 - i, 60 - i] = True

    # Branch 3: down
    skeleton[50:70, 50] = True

    return skeleton


@pytest.fixture
def branching_skan_skeleton(branching_skeleton_2d):
    """skan.Skeleton from branching_skeleton_2d."""
    return skan.Skeleton(branching_skeleton_2d.astype(float))


@pytest.fixture
def circular_mask():
    """Circular binary mask centered in image."""
    mask = np.zeros((64, 64), dtype=bool)
    center = (32, 32)
    radius = 10
    y, x = np.ogrid[:64, :64]
    mask[(y - center[0]) ** 2 + (x - center[1]) ** 2 <= radius**2] = True
    return mask


@pytest.fixture
def skeleton_with_center():
    """skan.Skeleton radiating from center point for Sholl testing."""
    skeleton = np.zeros((100, 100), dtype=bool)
    center = (50, 50)

    # Create rays radiating from center
    for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        for r in range(5, 40):
            y = int(center[0] + r * np.sin(angle))
            x = int(center[1] + r * np.cos(angle))
            if 0 <= y < 100 and 0 <= x < 100:
                skeleton[y, x] = True

    skel = skan.Skeleton(skeleton.astype(float))
    return skel, np.array(center)


@pytest.fixture
def simple_labels_3d():
    """Simple 3D label image."""
    labels = np.zeros((32, 64, 64), dtype=np.uint16)
    labels[10:20, 20:40, 20:40] = 1
    labels[15:25, 30:50, 35:55] = 2
    return labels


@pytest.fixture
def simple_skeleton_3d():
    """skan.Skeleton for 3D skeleton testing."""
    skeleton = np.zeros((32, 64, 64), dtype=bool)
    # Line along Z axis
    skeleton[:, 32, 32] = True
    # Line along Y axis
    skeleton[16, :, 32] = True
    # Line along X axis
    skeleton[16, 32, :] = True
    return skan.Skeleton(skeleton.astype(float))
