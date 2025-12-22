"""
Tests for the MorphologyModel data model.
"""

import numpy as np
import pandas as pd
import pytest
import skan

from ndev_morphology.model import (
    METADATA_KEY,
    MorphologyModel,
    attach_model_to_layer,
    get_model_from_layer,
)


@pytest.fixture
def simple_skeleton():
    """Create a simple Y-shaped skeleton."""
    skeleton = np.zeros((50, 50), dtype=np.uint8)
    # Main trunk
    skeleton[25, 10:30] = 1
    # Two branches
    skeleton[15:26, 30] = 1  # Upper branch
    skeleton[25:36, 30] = 1  # Lower branch
    return skeleton


@pytest.fixture
def simple_model(simple_skeleton):
    """Create MorphologyModel from simple skeleton."""
    return MorphologyModel.from_array(
        simple_skeleton,
        spacing=(1.0, 1.0),
        source_name='test_skeleton',
    )


class TestMorphologyModelCreation:
    """Tests for creating MorphologyModel objects."""

    def test_from_array(self, simple_skeleton):
        """Can create from numpy array."""
        model = MorphologyModel.from_array(simple_skeleton, spacing=(0.2, 0.2))
        assert isinstance(model.skeleton, skan.Skeleton)
        assert model.spacing == (0.2, 0.2)

    def test_from_array_with_name(self, simple_skeleton):
        """Source name is stored."""
        model = MorphologyModel.from_array(
            simple_skeleton, source_name='my_skeleton'
        )
        assert model.source_layer_name == 'my_skeleton'

    def test_direct_construction(self, simple_skeleton):
        """Can construct directly with skan.Skeleton."""
        skeleton = skan.Skeleton(
            simple_skeleton.astype(float), spacing=(1.0, 1.0)
        )
        model = MorphologyModel(
            skeleton=skeleton,
            spacing=(1.0, 1.0),
            source_layer_name='direct',
        )
        assert model.n_branches > 0


class TestMorphologyModelProperties:
    """Tests for computed properties."""

    def test_summary_is_dataframe(self, simple_model):
        """Summary property returns a DataFrame."""
        assert isinstance(simple_model.summary, pd.DataFrame)

    def test_summary_is_cached(self, simple_model):
        """Summary is computed once and cached."""
        summary1 = simple_model.summary
        summary2 = simple_model.summary
        assert summary1 is summary2  # Same object

    def test_n_branches(self, simple_model):
        """n_branches matches summary length."""
        assert simple_model.n_branches == len(simple_model.summary)
        assert simple_model.n_branches > 0

    def test_total_length(self, simple_model):
        """total_length is sum of branch distances."""
        expected = simple_model.summary['branch_distance'].sum()
        assert simple_model.total_length == pytest.approx(expected)

    def test_repr(self, simple_model):
        """repr includes key information."""
        rep = repr(simple_model)
        assert 'MorphologyModel' in rep
        assert 'n_branches=' in rep
        assert 'total_length=' in rep


class TestBranchComputation:
    """Tests for branch analysis methods."""

    def test_compute_branches(self, simple_model):
        """compute_branches returns extended DataFrame."""
        branches = simple_model.compute_branches()
        assert isinstance(branches, pd.DataFrame)
        assert 'branch_distance' in branches.columns

    def test_compute_branches_cached(self, simple_model):
        """compute_branches result is cached."""
        branches1 = simple_model.compute_branches()
        branches2 = simple_model.compute_branches()
        assert branches1 is branches2


class TestShollCaching:
    """Tests for Sholl result caching."""

    def test_sholl_result_initially_none(self, simple_model):
        """No Sholl result before computation."""
        result = simple_model.get_sholl_result('10.0,10.0')
        assert result is None

    def test_add_and_get_sholl_result(self, simple_model):
        """Can add and retrieve Sholl results."""
        from ndev_morphology.sholl import ShollResult

        result = ShollResult(
            center=np.array([10.0, 10.0]),
            radii=np.array([1.0, 2.0, 3.0]),
            counts=np.array([0, 1, 2]),
        )
        simple_model.add_sholl_result('10.0,10.0', result)

        retrieved = simple_model.get_sholl_result('10.0,10.0')
        assert retrieved is result

    def test_sholl_results_property(self, simple_model):
        """sholl_results returns copy of cache."""
        from ndev_morphology.sholl import ShollResult

        result = ShollResult(
            center=np.array([10.0, 10.0]),
            radii=np.array([1.0]),
            counts=np.array([1]),
        )
        simple_model.add_sholl_result('key1', result)

        results = simple_model.sholl_results
        assert 'key1' in results
        # Modifying returned dict doesn't affect cache
        results['key2'] = result
        assert 'key2' not in simple_model.sholl_results


class TestDirectedTreeCaching:
    """Tests for directed tree caching."""

    def test_directed_trees_initially_empty(self, simple_model):
        """No trees before computation."""
        assert simple_model.directed_trees == []

    def test_add_directed_trees(self, simple_model):
        """Can add directed trees."""
        # Mock tree (just using a dict for testing)
        mock_tree = {'root': 0, 'branches': 3}
        simple_model.add_directed_trees([mock_tree])
        assert len(simple_model.directed_trees) == 1

    def test_clear_directed_trees(self, simple_model):
        """Can clear cached trees."""
        mock_tree = {'root': 0}
        simple_model.add_directed_trees([mock_tree])
        simple_model.clear_directed_trees()
        assert simple_model.directed_trees == []


class TestCacheClearing:
    """Tests for cache management."""

    def test_clear_cache(self, simple_model):
        """clear_cache resets all cached data."""
        # Populate caches
        _ = simple_model.summary
        _ = simple_model.compute_branches()
        simple_model.add_directed_trees([{}])

        simple_model.clear_cache()

        # All caches should be reset
        assert simple_model._summary is None
        assert simple_model._branch_summary is None
        assert len(simple_model._directed_trees) == 0
        assert len(simple_model._sholl_results) == 0


class TestLayerIntegration:
    """Tests for napari layer integration."""

    def test_get_model_from_layer_without_cache(self):
        """Returns None when no model cached."""

        class MockLayer:
            metadata = {}

        result = get_model_from_layer(MockLayer())
        assert result is None

    def test_attach_model_to_layer(self, simple_model):
        """Can attach model to layer metadata."""

        class MockLayer:
            metadata = {}

        layer = MockLayer()
        attach_model_to_layer(layer, simple_model)
        assert METADATA_KEY in layer.metadata
        assert layer.metadata[METADATA_KEY] is simple_model

    def test_get_model_after_attach(self, simple_model):
        """Can retrieve attached model."""

        class MockLayer:
            metadata = {}

        layer = MockLayer()
        attach_model_to_layer(layer, simple_model)
        retrieved = get_model_from_layer(layer)
        assert retrieved is simple_model


class TestFromLayerMethod:
    """Tests for MorphologyModel.from_layer()."""

    @pytest.fixture
    def mock_labels_layer(self, simple_skeleton):
        """Create a mock napari Labels layer."""

        class MockLabelsLayer:
            def __init__(self, data, scale, name):
                self.data = data
                self.scale = scale
                self.name = name
                self.metadata = {}

        return MockLabelsLayer(
            data=simple_skeleton,
            scale=(1.0, 1.0),
            name='test_labels',
        )

    def test_from_layer_creates_model(self, mock_labels_layer):
        """from_layer creates new model for uncached layer."""
        model = MorphologyModel.from_layer(mock_labels_layer)
        assert isinstance(model, MorphologyModel)
        assert model.source_layer_name == 'test_labels'

    def test_from_layer_caches_on_layer(self, mock_labels_layer):
        """from_layer caches model in layer.metadata."""
        model = MorphologyModel.from_layer(mock_labels_layer)
        assert METADATA_KEY in mock_labels_layer.metadata
        assert mock_labels_layer.metadata[METADATA_KEY] is model

    def test_from_layer_returns_cached(self, mock_labels_layer):
        """from_layer returns cached model on subsequent calls."""
        model1 = MorphologyModel.from_layer(mock_labels_layer)
        model2 = MorphologyModel.from_layer(mock_labels_layer)
        assert model1 is model2

    def test_from_layer_force_recompute(self, mock_labels_layer):
        """force_recompute creates new model."""
        model1 = MorphologyModel.from_layer(mock_labels_layer)
        model2 = MorphologyModel.from_layer(
            mock_labels_layer, force_recompute=True
        )
        assert model1 is not model2

    def test_from_layer_extracts_spacing(self, simple_skeleton):
        """from_layer extracts spacing from layer scale."""

        class MockLabelsLayer:
            def __init__(self):
                self.data = simple_skeleton
                self.scale = (0.5, 0.5)  # Non-default spacing
                self.name = 'test'
                self.metadata = {}

        layer = MockLabelsLayer()
        model = MorphologyModel.from_layer(layer)
        assert model.spacing == (0.5, 0.5)
