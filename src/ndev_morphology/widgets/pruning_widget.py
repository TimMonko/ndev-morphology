"""
Widget for skeleton pruning operations.

Provides interactive skeleton branch pruning with type-based filtering.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import skan
from magicgui import magic_factory

from ..pruning import prune_short_branches

if TYPE_CHECKING:
    import napari

__all__ = ['prune_skeleton_widget']


@magic_factory(
    call_button='Prune Skeleton',
    skeleton_layer={'label': 'Skeleton Layer'},
    min_branch_distance={
        'label': 'Min Branch Length',
        'min': 0.0,
        'max': 1000.0,
        'step': 1.0,
        'tooltip': 'Branches shorter than this will be pruned (in pixels or physical units).',
    },
    prune_terminal={
        'label': 'Prune Terminal Branches (type 1)',
        'tooltip': (
            'Prune junction-to-endpoint branches. '
            'These are the most common spurious branches from skeletonization.'
        ),
    },
    prune_isolated={
        'label': 'Prune Isolated Branches (type 0)',
        'tooltip': (
            'Prune endpoint-to-endpoint branches. '
            'These are completely disconnected branch fragments.'
        ),
    },
    prune_cycles={
        'label': 'Prune Isolated Cycles (type 3)',
        'tooltip': 'Remove closed loops. Often artifacts from holes in the image.',
    },
    prune_connecting={
        'label': 'Prune Connecting Branches (type 2)',
        'tooltip': (
            'Prune junction-to-junction branches. '
            'Usually you want to KEEP these as they connect major skeleton parts!'
        ),
    },
)
def prune_skeleton_widget(
    skeleton_layer: napari.layers.Labels,
    min_branch_distance: float = 10.0,
    prune_terminal: bool = True,
    prune_isolated: bool = True,
    prune_cycles: bool = True,
    prune_connecting: bool = False,
) -> napari.types.LayerDataTuple:
    """
    Prune short branches from a skeleton.

    Removes branches shorter than the specified minimum distance.
    You can control which branch types are eligible for pruning.

    Branch Types
    ------------
    - Type 0 (endpoint-to-endpoint): Isolated branch fragments
    - Type 1 (junction-to-endpoint): Terminal branches (most common artifacts)
    - Type 2 (junction-to-junction): Connecting branches (usually keep these!)
    - Type 3 (isolated cycle): Closed loops (usually artifacts)

    Parameters
    ----------
    skeleton_layer : Labels
        Skeleton image (from skeletonize_labels widget).
    min_branch_distance : float
        Minimum branch length to keep.
    prune_terminal : bool
        Prune junction-to-endpoint branches.
    prune_isolated : bool
        Prune endpoint-to-endpoint branches.
    prune_cycles : bool
        Prune isolated cycles.
    prune_connecting : bool
        Prune junction-to-junction branches (careful!).

    Returns
    -------
    LayerDataTuple
        Labels layer with pruned skeleton.
    """
    skeleton_arr = np.asarray(skeleton_layer.data)
    scale = skeleton_layer.scale
    spacing = tuple(scale[-2:])

    if not np.any(skeleton_arr):
        raise ValueError('Skeleton is empty')

    # Create skan Skeleton - use float to get proper measurements
    skel = skan.Skeleton(skeleton_arr.astype(float), spacing=spacing)

    # Count branches before pruning
    n_before = skel.n_paths

    # Prune
    pruned_skel, summary = prune_short_branches(
        skel,
        min_branch_distance=min_branch_distance,
        prune_endpoint_to_endpoint=prune_isolated,
        prune_junction_to_endpoint=prune_terminal,
        prune_junction_to_junction=prune_connecting,
        prune_isolated_cycles=prune_cycles,
    )

    n_after = pruned_skel.n_paths
    n_removed = n_before - n_after

    # Convert back to image
    pruned_image = np.asarray(pruned_skel)

    # For labeled skeletons, try to preserve label IDs
    # This is approximate - we mark pruned skeleton pixels with original labels
    if skeleton_arr.max() > 1:
        # Original was labeled - try to preserve labels
        # Create output with original label IDs where pruned skeleton exists
        pruned_binary = pruned_image > 0
        pruned_labeled = skeleton_arr * pruned_binary
        result = pruned_labeled.astype(skeleton_arr.dtype)
    else:
        # Binary skeleton - return pruned labels from skan
        result = pruned_image.astype(skeleton_arr.dtype)

    print(f'Pruned {n_removed} branches (from {n_before} to {n_after})')
    print(
        f'Remaining branch types: {summary["branch_type"].value_counts().to_dict()}'
    )

    return (
        result,
        {
            'name': f'{skeleton_layer.name}_pruned',
            'scale': scale,
            'metadata': {
                'n_branches_before': n_before,
                'n_branches_after': n_after,
                'n_removed': n_removed,
                'min_branch_distance': min_branch_distance,
            },
        },
        'labels',
    )
