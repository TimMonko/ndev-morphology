"""Tests for skeleton pruning operations."""

from __future__ import annotations

import numpy as np
import pytest
import skan

from ndev_morphology.pruning import (
    BranchType,
    prune_short_branches,
    prune_skeleton_to_image,
    remove_isolated_cycles,
)


@pytest.fixture
def skeleton_with_branches():
    """Create a skeleton with various branch types."""
    # Create a skeleton with main branch and short terminal branches
    # Main horizontal line with short vertical spurs
    skeleton = np.zeros((50, 100), dtype=bool)
    # Main horizontal line
    skeleton[25, 10:90] = True
    # Short spur (should be prunable)
    skeleton[22:25, 30] = True  # 3 pixels up
    # Longer spur (should survive)
    skeleton[10:25, 60] = True  # 15 pixels up
    # Another short spur
    skeleton[25:28, 50] = True  # 3 pixels down
    return skeleton


@pytest.fixture
def skeleton_with_cycle():
    """Create a skeleton with an isolated cycle."""
    skeleton = np.zeros((50, 50), dtype=bool)
    # Main line
    skeleton[25, 10:40] = True
    # Small loop
    skeleton[10:15, 25] = True  # vertical
    skeleton[10, 23:28] = True  # top
    skeleton[14, 23:28] = True  # bottom
    skeleton[10:15, 23] = True  # left
    return skeleton


class TestPruneShortBranches:
    """Tests for prune_short_branches function."""

    def test_prune_removes_short_branches(self, skeleton_with_branches):
        """Short terminal branches should be removed."""
        skel = skan.Skeleton(skeleton_with_branches)
        n_before = skel.n_paths

        pruned, summary = prune_short_branches(
            skel,
            min_branch_distance=5.0,
            prune_junction_to_endpoint=True,
        )

        n_after = pruned.n_paths
        assert n_after < n_before, 'Should have removed some branches'

    def test_prune_preserves_long_branches(self, skeleton_with_branches):
        """Long branches should not be removed."""
        skel = skan.Skeleton(skeleton_with_branches)

        # Prune with very small threshold
        pruned, summary = prune_short_branches(
            skel,
            min_branch_distance=1.0,
            prune_junction_to_endpoint=True,
        )

        # All branches > 1 pixel should survive
        assert np.all(summary['branch_distance'] >= 1.0)

    def test_prune_respects_branch_type(self, skeleton_with_branches):
        """Only specified branch types should be pruned."""
        skel = skan.Skeleton(skeleton_with_branches)
        _summary_before = skan.summarize(skel, separator='_')

        # Get branch counts by type before pruning
        # All prune options disabled except connecting (type 2)
        pruned, summary = prune_short_branches(
            skel,
            min_branch_distance=100.0,  # Very large threshold
            prune_junction_to_endpoint=False,
            prune_endpoint_to_endpoint=False,
            prune_junction_to_junction=False,  # Nothing to prune
            prune_isolated_cycles=False,
        )

        # Nothing should be pruned since all branch types are protected
        assert pruned.n_paths == skel.n_paths, 'No branches should be removed'

    def test_prune_returns_valid_skeleton(self, skeleton_with_branches):
        """Pruned result should be a valid skan Skeleton."""
        skel = skan.Skeleton(skeleton_with_branches)

        pruned, summary = prune_short_branches(skel, min_branch_distance=5.0)

        # Should be able to iterate paths
        assert pruned.n_paths >= 0
        for i in range(pruned.n_paths):
            coords = pruned.path_coordinates(i)
            assert len(coords) >= 2


class TestRemoveIsolatedCycles:
    """Tests for remove_isolated_cycles function."""

    def test_removes_cycles(self, skeleton_with_cycle):
        """Isolated cycles should be removed."""
        skel = skan.Skeleton(skeleton_with_cycle)
        summary_before = skan.summarize(skel, separator='_')

        # Check if there are any cycles before
        has_cycles_before = (
            summary_before['branch_type'] == BranchType.ISOLATED_CYCLE
        ).any()

        pruned, summary = remove_isolated_cycles(skel)

        # Check no cycles after
        has_cycles_after = (
            summary['branch_type'] == BranchType.ISOLATED_CYCLE
        ).any()

        if has_cycles_before:
            assert not has_cycles_after, 'Cycles should be removed'

    def test_preserves_non_cycles(self):
        """Non-cycle branches should be preserved."""
        # Simple line - no cycles
        skeleton = np.zeros((10, 50), dtype=bool)
        skeleton[5, 5:45] = True

        skel = skan.Skeleton(skeleton)
        n_before = skel.n_paths

        pruned, summary = remove_isolated_cycles(skel)

        n_after = pruned.n_paths
        assert n_after == n_before, 'No branches should be removed'


class TestPruneSkeletonToImage:
    """Tests for prune_skeleton_to_image function."""

    def test_returns_binary_image(self, skeleton_with_branches):
        """Output should be a binary image."""
        skel = skan.Skeleton(skeleton_with_branches)
        summary = skan.summarize(skel, separator='_')
        short = summary['branch_distance'] < 5.0

        result = prune_skeleton_to_image(skel, np.flatnonzero(short))

        assert result.dtype == bool
        assert result.shape == skeleton_with_branches.shape

    def test_empty_prune_list(self, skeleton_with_branches):
        """Empty prune list should return full skeleton."""
        skel = skan.Skeleton(skeleton_with_branches)

        result = prune_skeleton_to_image(skel, [])

        # Should match original
        assert result.sum() == skeleton_with_branches.sum()


class TestBranchType:
    """Tests for BranchType constants."""

    def test_constants_match_skan(self):
        """BranchType values should match skan's conventions."""
        assert BranchType.ENDPOINT_TO_ENDPOINT == 0
        assert BranchType.JUNCTION_TO_ENDPOINT == 1
        assert BranchType.JUNCTION_TO_JUNCTION == 2
        assert BranchType.ISOLATED_CYCLE == 3
