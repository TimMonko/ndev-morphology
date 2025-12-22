"""
Shared utilities for morphology widgets.

This module provides common helper functions used across widgets
to reduce boilerplate and ensure consistent behavior.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..model import MorphologyModel

if TYPE_CHECKING:
    import napari

__all__ = [
    'get_or_create_model',
    'get_spacing_from_layer',
    'validate_skeleton_layer',
]


def get_or_create_model(
    layer: napari.layers.Labels,
    *,
    force_recompute: bool = False,
) -> MorphologyModel:
    """
    Get or create MorphologyModel for a skeleton layer.

    This is the recommended way to get a model in widgets.
    It handles caching automatically.

    Parameters
    ----------
    layer : napari.layers.Labels
        Skeleton layer.
    force_recompute : bool, optional
        Force creation of a new model. Default False.

    Returns
    -------
    MorphologyModel
        The model object.

    Raises
    ------
    ValueError
        If skeleton is empty.

    Examples
    --------
    >>> model = get_or_create_model(skeleton_layer)
    >>> branches = model.summary
    """
    # Validate first
    validate_skeleton_layer(layer)

    return MorphologyModel.from_layer(layer, force_recompute=force_recompute)


def get_spacing_from_layer(
    layer: napari.layers.Labels | napari.layers.Image,
) -> tuple[float, ...]:
    """
    Extract physical spacing from layer scale.

    Parameters
    ----------
    layer : napari.layers.Labels or napari.layers.Image
        The napari layer.

    Returns
    -------
    tuple of float
        Spacing as (y, x) or (z, y, x) depending on dimensionality.
    """
    return tuple(layer.scale[-2:])


def validate_skeleton_layer(layer: napari.layers.Labels) -> None:
    """
    Validate that a layer contains a non-empty skeleton.

    Parameters
    ----------
    layer : napari.layers.Labels
        The skeleton layer to validate.

    Raises
    ------
    ValueError
        If skeleton is empty.
    TypeError
        If layer is not a Labels layer.
    """
    if not hasattr(layer, 'data'):
        raise TypeError(f'Expected napari layer, got {type(layer)}')

    arr = np.asarray(layer.data)
    if not np.any(arr):
        raise ValueError('Skeleton is empty')


def ensure_binary_skeleton(arr: np.ndarray) -> np.ndarray:
    """
    Convert labeled skeleton to binary.

    Parameters
    ----------
    arr : np.ndarray
        Binary or labeled skeleton.

    Returns
    -------
    np.ndarray
        Binary skeleton (dtype uint8).
    """
    return (arr > 0).astype(np.uint8)


def get_layer_name_suffix(
    layer: napari.layers.Layer,
    suffix: str,
) -> str:
    """
    Generate a derived layer name with suffix.

    Parameters
    ----------
    layer : napari.layers.Layer
        Source layer.
    suffix : str
        Suffix to add.

    Returns
    -------
    str
        Name like "skeleton_shapes" or "skeleton_sholl".
    """
    return f'{layer.name}_{suffix}'
