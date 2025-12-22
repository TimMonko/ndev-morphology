"""
Central data model for morphology analysis and visualization.

This module provides `MorphologyModel`, a unified data model that:
- Wraps skan.Skeleton with napari-specific context
- Caches computed results (summary, Sholl, branches, trees)
- Attaches to napari layers via layer.metadata
- Provides a clean API for widgets

Usage
-----
>>> from ndev_morphology.model import MorphologyModel
>>> # From a napari layer
>>> model = MorphologyModel.from_layer(skeleton_layer)
>>> # Access cached skeleton and summary
>>> branches = model.summary
>>> # Compute branch analysis (cached)
>>> branch_df = model.compute_branches()

The model follows the pattern used by napari-swc-editor for storing
structured data in layer.metadata.

Design Philosophy
-----------------
MorphologyModel is a **convenience wrapper**, not a processing state machine.

The processing pipeline should be explicit and replayable:

    segmentation = load_labels("neuron.tif")
    skeleton = skeletonize_labels(segmentation)
    skeleton_pruned = prune_short_branches(skeleton, min_length=5)

    # Don't like the pruning? Start fresh:
    skeleton_alt = prune_short_branches(skeleton, min_length=10)

    # The model wraps the FINAL result for visualization/export:
    model = MorphologyModel.from_array(skeleton_pruned, spacing=(0.2, 0.2))

The model does NOT:
- Track undo/redo history
- Manage mutable processing pipelines
- Replace explicit function calls
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import skan

from .branch import summarize_branches
from .sholl import ShollResult

if TYPE_CHECKING:
    import napari
    from numpy.typing import ArrayLike

__all__ = [
    'MorphologyModel',
    'get_model_from_layer',
    'attach_model_to_layer',
    # Backwards compatibility aliases
    'MorphologyAnalysis',
    'get_analysis_from_layer',
    'attach_analysis_to_layer',
]

# Key used to store MorphologyModel in layer.metadata
METADATA_KEY = 'morphology_model'


@dataclass
class MorphologyModel:
    """
    Unified data model for skeleton morphology analysis and visualization.

    Wraps skan.Skeleton with computed analysis results and napari context.
    Designed to be stored in layer.metadata['morphology_model'].

    Parameters
    ----------
    skeleton : skan.Skeleton
        The skan Skeleton object.
    spacing : tuple of float
        Physical pixel spacing used to create the skeleton.
    source_layer_name : str, optional
        Name of the napari layer this was created from.

    Attributes
    ----------
    skeleton : skan.Skeleton
        The underlying skan Skeleton object.
    spacing : tuple of float
        Physical pixel spacing (e.g., (0.2, 0.2) for 200nm pixels).
    source_layer_name : str
        Name of the source napari layer.
    summary : pd.DataFrame
        Branch summary from skan.summarize() (lazily computed).

    Examples
    --------
    >>> import skan
    >>> from ndev_morphology.model import MorphologyModel
    >>> skeleton = skan.Skeleton(binary_image, spacing=(0.2, 0.2))
    >>> model = MorphologyModel(skeleton, spacing=(0.2, 0.2))
    >>> print(f"Branches: {len(model.summary)}")
    >>> branches = model.compute_branches()
    """

    skeleton: skan.Skeleton
    spacing: tuple[float, ...]
    source_layer_name: str = ''

    # Cached results (computed lazily)
    _summary: pd.DataFrame | None = field(default=None, repr=False)
    _branch_summary: pd.DataFrame | None = field(default=None, repr=False)
    _sholl_results: dict[str, ShollResult] = field(
        default_factory=dict, repr=False
    )
    _directed_trees: list[Any] = field(
        default_factory=list, repr=False
    )  # Any to avoid circular import

    @classmethod
    def from_array(
        cls,
        skeleton_array: ArrayLike,
        *,
        spacing: tuple[float, ...] = (1.0, 1.0),
        source_name: str = '',
    ) -> MorphologyModel:
        """
        Create MorphologyModel from a numpy array.

        Parameters
        ----------
        skeleton_array : ArrayLike
            Binary or labeled skeleton image.
        spacing : tuple of float, optional
            Physical pixel spacing (default: (1.0, 1.0)).
        source_name : str, optional
            Name to identify the source (default: '').

        Returns
        -------
        MorphologyModel
            New model object.

        Examples
        --------
        >>> model = MorphologyModel.from_array(
        ...     skeleton_image, spacing=(0.2, 0.2), source_name='neuron_1'
        ... )
        """
        arr = np.asarray(skeleton_array)
        skeleton = skan.Skeleton(arr.astype(float), spacing=spacing)
        return cls(
            skeleton=skeleton,
            spacing=spacing,
            source_layer_name=source_name,
        )

    @classmethod
    def from_layer(
        cls,
        layer: napari.layers.Labels,
        *,
        force_recompute: bool = False,
    ) -> MorphologyModel:
        """
        Create or retrieve MorphologyModel from a napari Labels layer.

        If the layer already has a cached model in its metadata,
        that cached version is returned (unless force_recompute=True).

        Parameters
        ----------
        layer : napari.layers.Labels
            The napari Labels layer containing the skeleton.
        force_recompute : bool, optional
            If True, create a new model even if one is cached.
            Default is False.

        Returns
        -------
        MorphologyModel
            The model object (new or cached).

        Examples
        --------
        >>> # First call creates and caches
        >>> model = MorphologyModel.from_layer(skeleton_layer)
        >>> # Second call returns cached version
        >>> same_model = MorphologyModel.from_layer(skeleton_layer)
        >>> assert model is same_model
        >>> # Force recomputation
        >>> new_model = MorphologyModel.from_layer(
        ...     skeleton_layer, force_recompute=True
        ... )
        """
        # Check cache first
        if not force_recompute and METADATA_KEY in layer.metadata:
            cached = layer.metadata[METADATA_KEY]
            if isinstance(cached, MorphologyModel):
                return cached

        # Create new model
        arr = np.asarray(layer.data)
        spacing = tuple(layer.scale[-2:])
        skeleton = skan.Skeleton(arr.astype(float), spacing=spacing)

        model = cls(
            skeleton=skeleton,
            spacing=spacing,
            source_layer_name=layer.name,
        )

        # Cache on layer
        layer.metadata[METADATA_KEY] = model
        return model

    @property
    def summary(self) -> pd.DataFrame:
        """
        Lazily compute and cache skan summary.

        Returns
        -------
        pd.DataFrame
            Branch summary with columns like 'branch_distance',
            'euclidean_distance', 'branch_type', 'node_id_src', etc.
        """
        if self._summary is None:
            self._summary = skan.summarize(self.skeleton, separator='_')
        return self._summary

    @property
    def n_branches(self) -> int:
        """Number of branches in the skeleton."""
        return len(self.summary)

    @property
    def total_length(self) -> float:
        """Total branch length in physical units."""
        return float(self.summary['branch_distance'].sum())

    @property
    def n_tips(self) -> int:
        """Number of endpoint branches (branch_type 0 or 1)."""
        return int((self.summary['branch_type'] <= 1).sum())

    @property
    def n_junctions(self) -> int:
        """Number of junction-junction branches (branch_type 2)."""
        return int((self.summary['branch_type'] == 2).sum())

    def compute_branches(
        self, intensity_image: ArrayLike | None = None
    ) -> pd.DataFrame:
        """
        Compute or retrieve branch summary with additional metrics.

        This extends skan.summarize() with computed metrics like
        tortuosity.

        Parameters
        ----------
        intensity_image : ArrayLike, optional
            Intensity image for measuring mean intensity along branches.

        Returns
        -------
        pd.DataFrame
            Extended branch summary with tortuosity, etc.

        Notes
        -----
        The result is cached. If called multiple times, the first
        computed result is returned (unless the skeleton is recreated).
        """
        if self._branch_summary is None:
            self._branch_summary = summarize_branches(
                self.skeleton, intensity_image=intensity_image
            )
        return self._branch_summary

    def get_sholl_result(self, center_key: str) -> ShollResult | None:
        """
        Get cached Sholl result for a center point.

        Parameters
        ----------
        center_key : str
            Key identifying the center (e.g., "100.0,150.0").

        Returns
        -------
        ShollResult or None
            Cached result or None if not computed.
        """
        return self._sholl_results.get(center_key)

    def add_sholl_result(self, center_key: str, result: ShollResult) -> None:
        """
        Cache a Sholl result.

        Parameters
        ----------
        center_key : str
            Key identifying the center (e.g., "100.0,150.0").
        result : ShollResult
            The computed Sholl result to cache.
        """
        self._sholl_results[center_key] = result

    @property
    def sholl_results(self) -> dict[str, ShollResult]:
        """All cached Sholl results."""
        return self._sholl_results.copy()

    def add_directed_trees(self, trees: list) -> None:
        """
        Store directed tree analysis results.

        Parameters
        ----------
        trees : list of DirectedTree
            Trees to cache.
        """
        self._directed_trees.extend(trees)

    def clear_directed_trees(self) -> None:
        """Clear cached directed trees."""
        self._directed_trees.clear()

    @property
    def directed_trees(self) -> list:
        """Get cached directed trees (list of DirectedTree)."""
        return self._directed_trees

    def clear_cache(self) -> None:
        """Clear all cached computation results."""
        self._summary = None
        self._branch_summary = None
        self._sholl_results.clear()
        self._directed_trees.clear()

    def __repr__(self) -> str:
        """Return a string representation."""
        return (
            f'MorphologyModel('
            f'n_branches={self.n_branches}, '
            f'total_length={self.total_length:.1f}, '
            f"source='{self.source_layer_name}')"
        )


def get_model_from_layer(
    layer: napari.layers.Labels,
) -> MorphologyModel | None:
    """
    Get MorphologyModel from a layer's metadata if it exists.

    Parameters
    ----------
    layer : napari.layers.Labels
        The napari layer to check.

    Returns
    -------
    MorphologyModel or None
        The cached model, or None if not present.
    """
    return layer.metadata.get(METADATA_KEY)


def attach_model_to_layer(
    layer: napari.layers.Labels,
    model: MorphologyModel,
) -> None:
    """
    Attach a MorphologyModel to a layer's metadata.

    Parameters
    ----------
    layer : napari.layers.Labels
        The napari layer.
    model : MorphologyModel
        The model to attach.
    """
    layer.metadata[METADATA_KEY] = model


# Backwards compatibility aliases
MorphologyAnalysis = MorphologyModel
get_analysis_from_layer = get_model_from_layer
attach_analysis_to_layer = attach_model_to_layer
