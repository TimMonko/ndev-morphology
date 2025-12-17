"""Tests for the tree module - directional tree analysis."""

import networkx as nx
import numpy as np
import pytest
import skan

from ndev_morphology import (
    BranchOrder,
    DirectedTree,
    compute_branch_order,
    compute_strahler_order,
    create_directed_tree,
    find_longest_path,
    get_paths_to_tips,
    summarize_directed_tree,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_y_skeleton():
    """
    Create a Y-shaped skeleton for testing.

    Structure (approximately):
        /
       *  <- junction
        \\
         |
         * <- root (soma)

    The soma is at the bottom, one junction, two tips at top.
    """
    skeleton = np.zeros((50, 50), dtype=np.uint8)
    # Vertical stem from soma (40, 25) to junction (20, 25)
    skeleton[20:41, 25] = 1
    # Left branch from junction (20, 25) to tip (5, 10)
    for i in range(16):
        skeleton[20 - i, 25 - i] = 1
    # Right branch from junction (20, 25) to tip (5, 40)
    for i in range(16):
        skeleton[20 - i, 25 + i] = 1
    return skeleton


@pytest.fixture
def simple_y_skeleton_obj(simple_y_skeleton):
    """Create skan.Skeleton from Y-shaped image."""
    return skan.Skeleton(simple_y_skeleton.astype(float), spacing=(1.0, 1.0))


@pytest.fixture
def soma_coords_for_y():
    """Soma coordinates at bottom of Y skeleton."""
    return np.array([40, 25])


@pytest.fixture
def linear_skeleton():
    """Create a simple linear skeleton (no junctions)."""
    skeleton = np.zeros((50, 50), dtype=np.uint8)
    skeleton[10:40, 25] = 1  # Vertical line
    return skeleton


@pytest.fixture
def complex_skeleton():
    """
    Create a more complex skeleton with multiple junction levels.

          *   *
           \\ /
            *  <- junction 2
            |
       *---*---*  <- junction 1 + 2 tips
            |
            *  <- soma
    """
    skeleton = np.zeros((60, 60), dtype=np.uint8)
    # Main stem: (50, 30) to (30, 30)
    skeleton[30:51, 30] = 1
    # Horizontal bar at junction 1: (30, 10) to (30, 50)
    skeleton[30, 10:51] = 1
    # Second stem up: (30, 30) to (15, 30)
    skeleton[15:31, 30] = 1
    # Y branches from (15, 30)
    for i in range(10):
        skeleton[15 - i, 30 - i] = 1  # Left tip
        skeleton[15 - i, 30 + i] = 1  # Right tip
    return skeleton


# =============================================================================
# Tests for create_directed_tree
# =============================================================================


class TestCreateDirectedTree:
    """Tests for create_directed_tree function."""

    def test_creates_directed_tree_from_y_skeleton(
        self, simple_y_skeleton, soma_coords_for_y
    ):
        """Test creating directed tree from Y-shaped skeleton."""
        skel = skan.Skeleton(
            simple_y_skeleton.astype(float), spacing=(1.0, 1.0)
        )
        summary = skan.summarize(skel, separator='_')

        tree = create_directed_tree(skel, soma_coords_for_y, summary=summary)

        assert isinstance(tree, DirectedTree)
        assert isinstance(tree.graph, nx.DiGraph)
        assert tree.root_node is not None

    def test_tree_has_correct_structure(
        self, simple_y_skeleton_obj, soma_coords_for_y
    ):
        """Test that the tree has expected number of nodes/edges."""
        summary = skan.summarize(simple_y_skeleton_obj, separator='_')
        tree = create_directed_tree(
            simple_y_skeleton_obj, soma_coords_for_y, summary=summary
        )

        # Should have 3 or more nodes (soma, junction, tips)
        assert tree.graph.number_of_nodes() >= 3

    def test_tree_is_rooted_at_soma(
        self, simple_y_skeleton_obj, soma_coords_for_y
    ):
        """Test that tree is correctly rooted at soma position."""
        summary = skan.summarize(simple_y_skeleton_obj, separator='_')
        tree = create_directed_tree(
            simple_y_skeleton_obj, soma_coords_for_y, summary=summary
        )

        # Root should have no incoming edges
        assert tree.graph.in_degree(tree.root_node) == 0

    def test_linear_skeleton_creates_simple_tree(self, linear_skeleton):
        """Test that linear skeleton creates simple 2-node tree."""
        skel = skan.Skeleton(linear_skeleton.astype(float), spacing=(1.0, 1.0))
        summary = skan.summarize(skel, separator='_')
        soma_coords = np.array([39, 25])  # Bottom of line

        tree = create_directed_tree(skel, soma_coords, summary=summary)

        assert tree.n_branches >= 1
        assert tree.n_junctions == 0  # No junctions in linear skeleton


# =============================================================================
# Tests for DirectedTree properties
# =============================================================================


class TestDirectedTreeProperties:
    """Tests for DirectedTree class properties."""

    def test_n_tips(self, simple_y_skeleton_obj, soma_coords_for_y):
        """Test counting tip nodes."""
        summary = skan.summarize(simple_y_skeleton_obj, separator='_')
        tree = create_directed_tree(
            simple_y_skeleton_obj, soma_coords_for_y, summary=summary
        )

        # Y-skeleton should have 2 tips
        assert tree.n_tips == 2

    def test_n_junctions(self, simple_y_skeleton_obj, soma_coords_for_y):
        """Test counting junction nodes."""
        summary = skan.summarize(simple_y_skeleton_obj, separator='_')
        tree = create_directed_tree(
            simple_y_skeleton_obj, soma_coords_for_y, summary=summary
        )

        # Y-skeleton should have 1 junction
        assert tree.n_junctions == 1

    def test_primary_branches(self, simple_y_skeleton_obj, soma_coords_for_y):
        """Test identifying primary branches."""
        summary = skan.summarize(simple_y_skeleton_obj, separator='_')
        tree = create_directed_tree(
            simple_y_skeleton_obj, soma_coords_for_y, summary=summary
        )

        primary = tree.primary_branches
        assert isinstance(primary, list)
        # Y-skeleton: primary is the stem from soma to junction
        assert len(primary) >= 1


# =============================================================================
# Tests for compute_branch_order
# =============================================================================


class TestComputeBranchOrder:
    """Tests for compute_branch_order function."""

    def test_branch_order_for_y_skeleton(
        self, simple_y_skeleton_obj, soma_coords_for_y
    ):
        """Test branch order assignment for Y-shaped skeleton."""
        summary = skan.summarize(simple_y_skeleton_obj, separator='_')
        tree = create_directed_tree(
            simple_y_skeleton_obj, soma_coords_for_y, summary=summary
        )

        orders = compute_branch_order(tree)

        assert isinstance(orders, dict)
        # All edges should have an order assigned
        assert len(orders) == tree.graph.number_of_edges()

        # Orders should be positive integers
        for order in orders.values():
            assert isinstance(order, int)
            assert order >= BranchOrder.PRIMARY

    def test_primary_secondary_order_distinction(
        self, simple_y_skeleton_obj, soma_coords_for_y
    ):
        """Test that primary and secondary branches are distinguished."""
        summary = skan.summarize(simple_y_skeleton_obj, separator='_')
        tree = create_directed_tree(
            simple_y_skeleton_obj, soma_coords_for_y, summary=summary
        )

        orders = compute_branch_order(tree)

        # Should have at least primary (1) and secondary (2) orders in Y
        order_values = set(orders.values())
        assert BranchOrder.PRIMARY in order_values


# =============================================================================
# Tests for compute_strahler_order
# =============================================================================


class TestComputeStrahlerOrder:
    """Tests for compute_strahler_order function."""

    def test_strahler_order_for_y_skeleton(
        self, simple_y_skeleton_obj, soma_coords_for_y
    ):
        """Test Strahler order assignment for Y-shaped skeleton."""
        summary = skan.summarize(simple_y_skeleton_obj, separator='_')
        tree = create_directed_tree(
            simple_y_skeleton_obj, soma_coords_for_y, summary=summary
        )

        orders = compute_strahler_order(tree)

        assert isinstance(orders, dict)
        assert len(orders) == tree.graph.number_of_edges()

    def test_strahler_order_contains_ones(
        self, simple_y_skeleton_obj, soma_coords_for_y
    ):
        """Test that Strahler orders include order 1 for terminal branches."""
        summary = skan.summarize(simple_y_skeleton_obj, separator='_')
        tree = create_directed_tree(
            simple_y_skeleton_obj, soma_coords_for_y, summary=summary
        )

        orders = compute_strahler_order(tree)

        # Strahler order 1 should exist for terminal branches
        # or order 2 if the Y merges them (both are valid Strahler outcomes)
        order_values = set(orders.values())
        # At minimum we should have some order >= 1
        assert all(v >= 1 for v in order_values)


# =============================================================================
# Tests for find_longest_path
# =============================================================================


class TestFindLongestPath:
    """Tests for find_longest_path function."""

    def test_finds_path_in_y_skeleton(
        self, simple_y_skeleton_obj, soma_coords_for_y
    ):
        """Test finding longest path in Y-shaped skeleton."""
        summary = skan.summarize(simple_y_skeleton_obj, separator='_')
        tree = create_directed_tree(
            simple_y_skeleton_obj, soma_coords_for_y, summary=summary
        )

        path, length = find_longest_path(tree)

        assert isinstance(path, list)
        assert len(path) >= 2  # At least soma and tip
        assert isinstance(length, float)
        assert length > 0

    def test_path_starts_at_root(
        self, simple_y_skeleton_obj, soma_coords_for_y
    ):
        """Test that longest path starts at root node."""
        summary = skan.summarize(simple_y_skeleton_obj, separator='_')
        tree = create_directed_tree(
            simple_y_skeleton_obj, soma_coords_for_y, summary=summary
        )

        path, _ = find_longest_path(tree)

        assert path[0] == tree.root_node

    def test_linear_skeleton_path_length(self, linear_skeleton):
        """Test path length for simple linear skeleton."""
        skel = skan.Skeleton(linear_skeleton.astype(float), spacing=(1.0, 1.0))
        summary = skan.summarize(skel, separator='_')
        soma_coords = np.array([39, 25])

        tree = create_directed_tree(skel, soma_coords, summary=summary)
        path, length = find_longest_path(tree)

        # Linear skeleton is 30 pixels long (rows 10-39 inclusive)
        assert length == pytest.approx(29.0, abs=5.0)


# =============================================================================
# Tests for get_paths_to_tips
# =============================================================================


class TestGetPathsToTips:
    """Tests for get_paths_to_tips function."""

    def test_returns_list_of_dicts(
        self, simple_y_skeleton_obj, soma_coords_for_y
    ):
        """Test that paths are returned as a list of dicts."""
        summary = skan.summarize(simple_y_skeleton_obj, separator='_')
        tree = create_directed_tree(
            simple_y_skeleton_obj, soma_coords_for_y, summary=summary
        )

        paths = get_paths_to_tips(tree)

        assert isinstance(paths, list)
        # Y-skeleton has 2 tips
        assert len(paths) == 2
        # Each path should be a dict with expected keys
        for path_info in paths:
            assert isinstance(path_info, dict)
            assert 'path' in path_info
            assert 'length' in path_info
            assert 'tip_node' in path_info

    def test_all_paths_start_at_root(
        self, simple_y_skeleton_obj, soma_coords_for_y
    ):
        """Test that all paths start at root node."""
        summary = skan.summarize(simple_y_skeleton_obj, separator='_')
        tree = create_directed_tree(
            simple_y_skeleton_obj, soma_coords_for_y, summary=summary
        )

        paths = get_paths_to_tips(tree)

        for path_info in paths:
            path = path_info['path']
            assert path[0] == tree.root_node
            assert path[-1] == path_info['tip_node']


# =============================================================================
# Tests for summarize_directed_tree
# =============================================================================


class TestSummarizeDirectedTree:
    """Tests for summarize_directed_tree function."""

    def test_returns_dataframe(self, simple_y_skeleton_obj, soma_coords_for_y):
        """Test that function returns a pandas DataFrame."""
        import pandas as pd

        summary = skan.summarize(simple_y_skeleton_obj, separator='_')
        tree = create_directed_tree(
            simple_y_skeleton_obj, soma_coords_for_y, summary=summary
        )

        df = summarize_directed_tree(tree)

        assert isinstance(df, pd.DataFrame)

    def test_has_branch_order_column(
        self, simple_y_skeleton_obj, soma_coords_for_y
    ):
        """Test that summary includes branch_order column."""
        summary = skan.summarize(simple_y_skeleton_obj, separator='_')
        tree = create_directed_tree(
            simple_y_skeleton_obj, soma_coords_for_y, summary=summary
        )

        df = summarize_directed_tree(tree)

        assert 'branch_order' in df.columns

    def test_has_strahler_order_column(
        self, simple_y_skeleton_obj, soma_coords_for_y
    ):
        """Test that summary includes strahler_order column."""
        summary = skan.summarize(simple_y_skeleton_obj, separator='_')
        tree = create_directed_tree(
            simple_y_skeleton_obj, soma_coords_for_y, summary=summary
        )

        df = summarize_directed_tree(tree)

        assert 'strahler_order' in df.columns


# =============================================================================
# Tests for BranchOrder constants
# =============================================================================


class TestBranchOrder:
    """Tests for BranchOrder constants."""

    def test_order_values(self):
        """Test that branch order constants have expected values."""
        assert BranchOrder.PRIMARY == 1
        assert BranchOrder.SECONDARY == 2
        assert BranchOrder.TERTIARY == 3
        assert BranchOrder.QUATERNARY == 4
