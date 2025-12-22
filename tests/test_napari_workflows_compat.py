"""
Tests for napari-workflows compatibility.

These tests verify that ndev-morphology functions work with napari-workflows'
Workflow class for building composable image processing pipelines.

napari-workflows compatibility requirements:
1. Functions should accept numpy arrays (or types that convert to numpy)
2. Type annotations help workflows track data flow
3. Functions should be pure (no side effects on inputs)
4. Return types should be numpy arrays or LayerDataTuples
"""

import numpy as np
import pytest

# Skip all tests if napari-workflows is not installed
pytest.importorskip('napari_workflows')


class TestWorkflowBasics:
    """Test that core functions work with napari_workflows.Workflow."""

    @pytest.fixture
    def simple_labels(self):
        """Create a simple labels image with two objects."""
        labels = np.zeros((50, 50), dtype=np.uint16)
        # Object 1: horizontal bar
        labels[20:22, 10:40] = 1
        # Object 2: vertical bar
        labels[10:40, 35:37] = 2
        return labels

    @pytest.fixture
    def simple_skeleton(self, simple_labels):
        """Create a skeleton from the simple labels."""
        from ndev_morphology import skeletonize_labels

        return skeletonize_labels(simple_labels)

    def test_workflow_with_skeletonize_labels(self, simple_labels):
        """Test that skeletonize_labels works in a Workflow."""
        from napari_workflows import Workflow

        from ndev_morphology import skeletonize_labels

        w = Workflow()
        w.set('input', simple_labels)
        w.set('skeleton', skeletonize_labels, 'input')

        result = w.get('skeleton')

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == simple_labels.shape
        # Skeleton should have fewer non-zero pixels than original
        assert np.count_nonzero(result) < np.count_nonzero(simple_labels)
        # Label values should be preserved
        assert set(np.unique(result)) <= set(np.unique(simple_labels))

    def test_workflow_with_filter_labels(self, simple_labels):
        """Test that filter_labels_by_size works in a Workflow."""
        from napari_workflows import Workflow

        from ndev_morphology import filter_labels_by_size

        w = Workflow()
        w.set('input', simple_labels)
        # Filter to keep only labels with at least 50 pixels
        w.set('filtered', filter_labels_by_size, 'input', min_size=50)

        result = w.get('filtered')

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape == simple_labels.shape

    def test_workflow_chain(self, simple_labels):
        """Test a multi-step workflow: filter -> skeletonize."""
        from napari_workflows import Workflow

        from ndev_morphology import filter_labels_by_size, skeletonize_labels

        w = Workflow()
        w.set('input', simple_labels)
        w.set('filtered', filter_labels_by_size, 'input', min_size=10)
        w.set('skeleton', skeletonize_labels, 'filtered')

        # Get the final result - all dependencies should auto-execute
        result = w.get('skeleton')

        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_workflow_with_separate_touching(self, simple_labels):
        """Test the separate_touching_skeleton_labels function."""
        from napari_workflows import Workflow

        from ndev_morphology import (
            separate_touching_skeleton_labels,
            skeletonize_labels,
        )

        w = Workflow()
        w.set('input', simple_labels)
        w.set('skeleton', skeletonize_labels, 'input')
        w.set('separated', separate_touching_skeleton_labels, 'skeleton')

        result = w.get('separated')

        assert result is not None
        assert isinstance(result, np.ndarray)


class TestWorkflowWithSkan:
    """Test workflow compatibility with skan-based functions."""

    @pytest.fixture
    def cross_skeleton(self):
        """Create a cross-shaped skeleton for testing."""
        skeleton = np.zeros((50, 50), dtype=np.uint8)
        skeleton[25, 10:40] = 1  # Horizontal line
        skeleton[10:40, 25] = 1  # Vertical line
        return skeleton

    def test_summarize_branches_workflow(self, cross_skeleton):
        """Test that summarize_branches works in a workflow context.

        Note: summarize_branches takes a skan.Skeleton, not a numpy array.
        This is acceptable for programmatic use but may need a wrapper
        for workflow compatibility.
        """
        import skan

        from ndev_morphology import summarize_branches

        # Create skan Skeleton directly (this is the expected API)
        skeleton_obj = skan.Skeleton(cross_skeleton > 0)
        branches = summarize_branches(skeleton=skeleton_obj)

        assert branches is not None
        assert len(branches) > 0
        assert 'branch_distance' in branches.columns
        assert 'tortuosity' in branches.columns


class TestWorkflowRoots:
    """Test that workflow can track inputs and outputs correctly."""

    def test_workflow_structure(self):
        """Test that workflow correctly identifies roots and leafs."""
        from napari_workflows import Workflow

        from ndev_morphology import filter_labels_by_size, skeletonize_labels

        labels = np.zeros((30, 30), dtype=np.uint16)
        labels[10:20, 10:20] = 1

        w = Workflow()
        w.set('input', labels)
        w.set('filtered', filter_labels_by_size, 'input', min_size=5)
        w.set('skeleton', skeletonize_labels, 'filtered')

        # Check workflow structure
        roots = w.roots()
        leafs = w.leafs()

        assert 'input' in roots
        assert 'skeleton' in leafs
        assert len(w.followers_of('input')) == 1
        assert len(w.sources_of('skeleton')) == 1


class TestWorkflowSaveLoad:
    """Test that workflows with ndev-morphology can be saved and loaded."""

    def test_save_load_workflow(self, tmp_path):
        """Test saving and loading a workflow."""
        from napari_workflows import Workflow
        from napari_workflows._io_yaml_v1 import load_workflow, save_workflow

        from ndev_morphology import skeletonize_labels

        # Create a workflow
        w = Workflow()
        w.set('skeleton', skeletonize_labels, 'input')

        # Save it
        filepath = tmp_path / 'test_workflow.yaml'
        save_workflow(str(filepath), w)

        # Load it back
        w2 = load_workflow(str(filepath))

        # Check that the task was preserved
        assert 'skeleton' in w2._tasks
        task = w2.get_task('skeleton')
        assert task[0] == skeletonize_labels

    def test_loaded_workflow_executable(self, tmp_path):
        """Test that a loaded workflow can be executed."""
        from napari_workflows import Workflow
        from napari_workflows._io_yaml_v1 import load_workflow, save_workflow

        from ndev_morphology import skeletonize_labels

        # Create and save workflow
        w = Workflow()
        w.set('skeleton', skeletonize_labels, 'input')

        filepath = tmp_path / 'test_workflow.yaml'
        save_workflow(str(filepath), w)

        # Load and execute
        w2 = load_workflow(str(filepath))

        # Set input and get result
        labels = np.zeros((20, 20), dtype=np.uint16)
        labels[5:15, 8:12] = 1
        w2.set('input', labels)

        result = w2.get('skeleton')

        assert result is not None
        assert isinstance(result, np.ndarray)


class TestWorkflowWithLabelsModule:
    """Test labels module functions with napari-workflows."""

    @pytest.fixture
    def multi_object_labels(self):
        """Create labels with multiple objects of different sizes."""
        labels = np.zeros((100, 100), dtype=np.uint16)
        # Small object (100 pixels)
        labels[10:20, 10:20] = 1
        # Medium object (400 pixels)
        labels[30:50, 30:50] = 2
        # Large object (900 pixels)
        labels[60:90, 60:90] = 3
        return labels

    def test_filter_labels_by_size_workflow(self, multi_object_labels):
        """Test filter_labels_by_size in a workflow."""
        from napari_workflows import Workflow

        from ndev_morphology import filter_labels_by_size

        w = Workflow()
        w.set('input', multi_object_labels)
        w.set(
            'filtered',
            filter_labels_by_size,
            'input',
            min_size=200,
            max_size=500,
        )

        result = w.get('filtered')

        # Only the medium object (400 pixels) should remain
        unique_labels = np.unique(result)
        assert 2 in unique_labels  # Medium object kept
        assert 1 not in unique_labels  # Small removed
        assert 3 not in unique_labels  # Large removed

    def test_exclude_labels_on_edges_workflow(self):
        """Test exclude_labels_on_edges in a workflow."""
        from napari_workflows import Workflow

        from ndev_morphology import exclude_labels_on_edges

        labels = np.zeros((50, 50), dtype=np.uint16)
        labels[0:10, 20:30] = 1  # Touches top edge
        labels[20:30, 20:30] = 2  # Interior, should be kept

        w = Workflow()
        w.set('input', labels)
        w.set('interior_only', exclude_labels_on_edges, 'input')

        result = w.get('interior_only')

        assert 1 not in result  # Edge-touching removed
        assert 2 in result  # Interior kept

    def test_connect_breaks_between_labels_workflow(self):
        """Test connect_breaks_between_labels in a workflow."""
        from napari_workflows import Workflow

        from ndev_morphology import connect_breaks_between_labels

        labels = np.zeros((50, 50), dtype=np.uint16)
        labels[20:25, 10:15] = 1
        labels[20:25, 17:22] = 1  # Small gap of 2 pixels

        w = Workflow()
        w.set('input', labels)
        w.set(
            'connected',
            connect_breaks_between_labels,
            'input',
            connect_distance=5,
        )

        result = w.get('connected')

        assert result is not None
        assert isinstance(result, np.ndarray)


class TestWorkflowWithSkeletonModule:
    """Test skeleton module functions with napari-workflows."""

    @pytest.fixture
    def simple_labels(self):
        """Create simple bar-shaped labels."""
        labels = np.zeros((50, 50), dtype=np.uint16)
        labels[22:28, 10:40] = 1
        return labels

    def test_exclude_region_from_skeleton_workflow(self, simple_labels):
        """Test exclude_region_from_skeleton in a workflow."""
        from napari_workflows import Workflow

        from ndev_morphology import (
            exclude_region_from_skeleton,
            skeletonize_labels,
        )

        # Create a mask for the center region
        mask = np.zeros((50, 50), dtype=bool)
        mask[20:30, 20:30] = True

        w = Workflow()
        w.set('labels', simple_labels)
        w.set('skeleton', skeletonize_labels, 'labels')
        w.set('excluded', exclude_region_from_skeleton, 'skeleton', mask)

        result = w.get('excluded')

        assert result is not None
        # The center should be excluded
        assert np.all(result[20:30, 20:30] == 0)

    def test_fill_skeleton_gaps_workflow(self, simple_labels):
        """Test fill_skeleton_gaps in a workflow."""
        from napari_workflows import Workflow

        from ndev_morphology import fill_skeleton_gaps, skeletonize_labels

        w = Workflow()
        w.set('labels', simple_labels)
        w.set('skeleton', skeletonize_labels, 'labels')
        w.set('filled', fill_skeleton_gaps, 'skeleton', dilation_size=3)

        result = w.get('filled')

        assert result is not None
        assert isinstance(result, np.ndarray)


class TestComplexWorkflowPipeline:
    """Test complex multi-step workflows typical in morphology analysis."""

    def test_full_preprocessing_pipeline(self):
        """Test a realistic preprocessing pipeline."""
        from napari_workflows import Workflow

        from ndev_morphology import (
            exclude_labels_on_edges,
            filter_labels_by_size,
            separate_touching_skeleton_labels,
            skeletonize_labels,
        )

        # Create test data with edge-touching and small objects
        labels = np.zeros((100, 100), dtype=np.uint16)
        labels[0:20, 40:60] = 1  # Edge-touching (should be removed)
        labels[30:35, 30:35] = 2  # Small (should be removed by size filter)
        labels[50:80, 50:80] = 3  # Good object (should be kept)

        w = Workflow()
        w.set('raw_labels', labels)
        w.set('no_edges', exclude_labels_on_edges, 'raw_labels')
        w.set('size_filtered', filter_labels_by_size, 'no_edges', min_size=100)
        w.set('skeleton', skeletonize_labels, 'size_filtered')
        w.set('final', separate_touching_skeleton_labels, 'skeleton')

        result = w.get('final')

        assert result is not None
        # Only label 3 should remain (as skeleton)
        unique = np.unique(result)
        assert 1 not in unique
        assert 2 not in unique

    def test_workflow_code_generation(self):
        """Test that workflow can generate Python code."""
        from napari_workflows import Workflow

        from ndev_morphology import filter_labels_by_size, skeletonize_labels

        w = Workflow()
        w.set('filtered', filter_labels_by_size, 'input', min_size=50)
        w.set('skeleton', skeletonize_labels, 'filtered')

        # Workflow should have valid structure
        assert len(w._tasks) == 2
        assert 'filtered' in w._tasks
        assert 'skeleton' in w._tasks

        # Check task dependencies
        skeleton_task = w.get_task('skeleton')
        assert skeleton_task[0] == skeletonize_labels
        assert 'filtered' in skeleton_task  # Input reference
