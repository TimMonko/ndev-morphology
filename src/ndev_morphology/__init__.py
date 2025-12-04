"""
ndev-morphology: Morphology analysis of skeleton-like objects.

A library for skeleton analysis, Sholl analysis, and branch
quantification. Designed for bioimage analysis of neuronal
and other branching structures.

Core Functions
--------------
- Skeleton operations: skeletonize_labels, exclude_region_from_skeleton, separate_touching_skeleton_labels
- Geometry utilities: skeleton_to_paths, sholl_shells_to_ellipses
- Label operations: filter_labels_by_size, exclude_labels_on_edges, connect_breaks_between_labels
- Sholl analysis: compute_sholl_profile, ShollResult

For interactive napari visualization, see the widgets subpackage.

Examples
--------
>>> import skan
>>> from ndev_morphology import skeletonize_labels, compute_sholl_profile
>>> from ndev_morphology.widgets import skeleton_to_shapes, sholl_analysis
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

# Skeleton operations
# Geometry utilities (for programmatic use)
from ._geometry import sholl_shells_to_ellipses, skeleton_to_paths

# Label operations
from .labels import (
    connect_breaks_between_labels,
    exclude_labels_on_edges,
    filter_labels_by_size,
)

# Sholl analysis
from .sholl import ShollResult, compute_sholl_profile
from .skeleton import (
    exclude_region_from_skeleton,
    separate_touching_skeleton_labels,
    skeletonize_labels,
)

__all__ = [
    # Skeleton
    "skeletonize_labels",
    "exclude_region_from_skeleton",
    "separate_touching_skeleton_labels",
    # Labels
    "filter_labels_by_size",
    "exclude_labels_on_edges",
    "connect_breaks_between_labels",
    # Sholl
    "ShollResult",
    "compute_sholl_profile",
    # Geometry
    "skeleton_to_paths",
    "sholl_shells_to_ellipses",
]
