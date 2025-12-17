"""
ndev-morphology: Morphology analysis of skeleton-like objects.

A library for skeleton analysis, Sholl analysis, and branch
quantification. Designed for bioimage analysis of neuronal
and other branching structures.

Core Modules
------------
- skeleton: skeletonize_labels, exclude_region_from_skeleton, separate_touching_skeleton_labels
- labels: filter_labels_by_size, exclude_labels_on_edges, connect_breaks_between_labels
- sholl: compute_sholl_profile, ShollResult
- branch: summarize_branches, BranchType, compute_tortuosity, filter_branches_by_type
- soma: detect_soma_centroid, find_soma_node, get_label_centroid
- analysis: analyze_single_cell, analyze_all_cells, CellAnalysisResult

For interactive napari visualization, see the widgets subpackage.

Examples
--------
>>> import skan
>>> from ndev_morphology import skeletonize_labels, compute_sholl_profile
>>> from ndev_morphology import summarize_branches, analyze_single_cell
>>> from ndev_morphology.widgets import skeleton_to_shapes, sholl_analysis
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = 'unknown'

# Geometry utilities (for programmatic use)
from ._geometry import sholl_shells_to_ellipses, skeleton_to_paths

# Analysis pipeline
from .analysis import (
    CellAnalysisResult,
    aggregate_branch_stats,
    analyze_all_cells,
    analyze_all_cells_generator,
    analyze_single_cell,
)

# Branch analysis
from .branch import (
    BranchType,
    compute_tortuosity,
    filter_branches_by_type,
    summarize_branches,
)

# Label operations
from .labels import (
    connect_breaks_between_labels,
    exclude_labels_on_edges,
    filter_labels_by_size,
)

# Pruning operations
from .pruning import (
    prune_short_branches,
    prune_skeleton_to_image,
    remove_isolated_cycles,
)

# Sholl analysis
from .sholl import ShollResult, compute_sholl_profile

# Skeleton operations
from .skeleton import (
    exclude_region_from_skeleton,
    fill_skeleton_gaps,
    separate_touching_skeleton_labels,
    skeletonize_labels,
)

# Soma detection
from .soma import detect_soma_centroid, find_soma_node, get_label_centroid

# Directed tree analysis
from .tree import (
    BranchOrder,
    DirectedTree,
    compute_branch_order,
    compute_strahler_order,
    create_directed_tree,
    create_directed_trees_from_soma,
    find_longest_path,
    get_paths_to_tips,
    summarize_directed_tree,
)

__all__ = [
    # Skeleton
    'skeletonize_labels',
    'exclude_region_from_skeleton',
    'separate_touching_skeleton_labels',
    'fill_skeleton_gaps',
    # Labels
    'filter_labels_by_size',
    'exclude_labels_on_edges',
    'connect_breaks_between_labels',
    # Pruning
    'prune_short_branches',
    'remove_isolated_cycles',
    'prune_skeleton_to_image',
    # Sholl
    'ShollResult',
    'compute_sholl_profile',
    # Branch
    'BranchType',
    'summarize_branches',
    'compute_tortuosity',
    'filter_branches_by_type',
    # Soma
    'detect_soma_centroid',
    'find_soma_node',
    'get_label_centroid',
    # Tree
    'DirectedTree',
    'BranchOrder',
    'create_directed_tree',
    'create_directed_trees_from_soma',
    'compute_branch_order',
    'compute_strahler_order',
    'find_longest_path',
    'get_paths_to_tips',
    'summarize_directed_tree',
    # Analysis
    'CellAnalysisResult',
    'analyze_single_cell',
    'analyze_all_cells',
    'analyze_all_cells_generator',
    'aggregate_branch_stats',
    # Geometry
    'skeleton_to_paths',
    'sholl_shells_to_ellipses',
]
