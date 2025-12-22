# AGENTS.md - ndev-morphology

**Single-cell morphological analysis for napari**

This document provides guidance for AI assistants and developers working on the `ndev-morphology` package. For workspace-level guidance, see the root `AGENTS.md`.

---

## Quick Start for Development

```powershell
# Navigate to ndev-morphology directory
cd C:\Users\timmo\ndev-kit\ndev-morphology

# Create and activate virtual environment (using uv)
uv venv
.venv\Scripts\activate

# Install in editable mode with dev dependencies
uv pip install -e . --group dev

# Run tests to verify setup
pytest -v
```

**IMPORTANT**: Always activate the virtual environment before running tests or development commands.

---

## Development Philosophy

### No Backwards Compatibility Required

**This code is not yet public.** We can make whatever breaking changes are needed at the moment. Do not hesitate to refactor, rename, or restructure when it improves the architecture. There is no need to maintain deprecated APIs or provide migration paths.

### napari-workflows & napari-assistant Integration

**Core Design Principle**: All ndev-morphology functions should be compatible with [napari-workflows](https://github.com/haesleinhuepf/napari-workflows) and discoverable in [napari-assistant](https://github.com/haesleinhuepf/napari-assistant).

This is **more important than building our own workflow management system**. By designing functions that work with napari-workflows, we get:

1. **Undo/Redo** — napari-workflows provides `UndoRedoController` that tracks workflow states
2. **Workflow saving/loading** — YAML-based workflow persistence
3. **Code generation** — Auto-generate Python scripts and Jupyter notebooks from workflows
4. **Composability** — Our functions become building blocks in larger workflows with other tools (pyclesperanto, devbio-napari, etc.)
5. **Interoperability** — Works with napari-workflow-optimizer, napari-workflow-inspector, napari-script-editor

**How napari-workflows works**:
```python
from napari_workflows import Workflow

# Define a workflow using function references and layer names
w = Workflow()
w.set("skeleton", skeletonize_labels, "labels_input")
w.set("pruned", prune_short_branches, "skeleton", min_length=5)
w.set("branches", summarize_branches, "pruned")

# Execute any step (dependencies auto-resolve)
result = w.get("branches")
```

**Verified Compatible Functions** (tested in `tests/test_napari_workflows_compat.py`):
- `skeletonize_labels`, `separate_touching_skeleton_labels`, `exclude_region_from_skeleton`, `fill_skeleton_gaps`
- `filter_labels_by_size`, `exclude_labels_on_edges`, `connect_breaks_between_labels`

These functions accept numpy arrays and return numpy arrays, making them fully compatible with napari-workflows' `Workflow.set()` pattern. Workflows can be saved/loaded as YAML files.

**How napari-assistant discovers functions**:
Functions are discovered via the `napari.yaml` manifest. Use proper `display_name` prefixes:

```yaml
widgets:
  - command: ndev-morphology.skeletonize_labels_widget
    display_name: Segmentation post-processing > Skeletonize Labels (ndev)
```

**Valid prefixes for napari-assistant**:
- `Filtering / noise removal >`
- `Filtering / background removal >`
- `Filtering >`
- `Image math >`
- `Transform >`
- `Projection >`
- `Segmentation / binarization >`
- `Segmentation / labeling >`
- `Segmentation post-processing >`
- `Measurement >`
- `Label neighbor filters >`
- `Label filters >`
- `Visualization >`

---

## Package Purpose

`ndev-morphology` provides single-cell morphological analysis tools, primarily for neuronal structures (dendrites, axons). The package bridges **skan** (skeleton analysis) and **napari** (visualization) with a focus on:

1. **Label preparation** — Filtering, cleaning, and preparing segmentation masks
2. **Skeleton creation** — Label-aware skeletonization preserving cell identity
3. **Branch analysis** — Quantifying branch properties using skan
4. **Sholl analysis** — Counting skeleton crossings at concentric shells
5. **Directed tree analysis** — Understanding branch hierarchy from soma to tips (via networkx)
6. **napari visualization** — Interactive widgets for analysis and exploration
7. **SWC interoperability** — Import/export standard neuron morphology format

### Target Users

- **Neuroscientists** analyzing dendritic/axonal morphology in vitro
- **Image analysts** quantifying branching structures in biology
- **Developers** needing programmatic access to morphology metrics

---

## Architecture Overview

### Core Principle: Separation of Concerns

Following the ndev-kit pattern, **business logic is separate from UI**:

```
src/ndev_morphology/
├── model.py           # MorphologyModel - unified data wrapper
├── labels.py          # Label preprocessing (pure Python)
├── skeleton.py        # Skeleton creation/modification (pure Python)
├── sholl.py           # Sholl analysis with ShollResult dataclass
├── branch.py          # Branch-level measurements and classification
├── soma.py            # Soma detection and centroid extraction
├── tree.py            # Directed tree analysis (networkx)
├── analysis.py        # Per-cell analysis pipelines, batch processing
├── swc.py             # SWC format import/export (future)
├── _geometry.py       # Visualization helpers (skeleton_to_paths, etc.)
├── widgets/           # napari/magicgui widgets (UI layer)
│   ├── _utils.py      # Shared widget utilities
│   ├── labels_widget.py
│   ├── skeletonize_widget.py
│   ├── skeleton_widget.py
│   ├── sholl_widget.py
│   ├── tree_widget.py
│   └── ...
└── napari.yaml        # Plugin manifest
```

**Rule**: Any function in `widgets/` should be a thin wrapper around core functionality.

---

## MorphologyModel Design

### Purpose

`MorphologyModel` is a **convenience wrapper** for working with skan.Skeleton objects:
- Bundles `skan.Skeleton` with metadata (spacing, source name)
- Provides lazy-computed properties (`summary`, `n_branches`, `total_length`)
- Can attach to napari layers for visualization/export (e.g., SWC)

### When to Use

| Use Case | Pattern |
|----------|--------|
| Creating from array | `MorphologyModel.from_array(skeleton_image, spacing=(0.2, 0.2))` |
| Creating from layer | `MorphologyModel.from_layer(labels_layer)` |
| Accessing branch data | `model.summary`, `model.compute_branches()` |
| SWC export (future) | `model.to_swc()` |

### Layer Metadata Pattern

```python
# Attach model to output layer for later retrieval
shapes_layer.metadata['morphology_model'] = model

# Retrieve for export or inspection
model = layer.metadata.get('morphology_model')
```

**Note**: Workflow state and undo/redo should be handled by **napari-workflows**, not MorphologyModel. The model is for data representation, not pipeline management.

---

## SWC Format Specification

SWC is the standard format for neuron morphology, used by neuromorpho.org.

### Format

ASCII text, 7 columns per line:

| Column | Name | Description |
|--------|------|-------------|
| 1 | Sample ID | Integer, typically starting from 1 |
| 2 | Type | 0=undefined, 1=soma, 2=axon, 3=dendrite, 4=apical dendrite |
| 3 | X | X coordinate in micrometers |
| 4 | Y | Y coordinate in micrometers |
| 5 | Z | Z coordinate in micrometers |
| 6 | Radius | Half the thickness in micrometers |
| 7 | Parent | Parent sample ID (-1 for root) |

### Example

```
# SWC file for simple neuron
1 1 0.0 0.0 0.0 5.0 -1    # Soma at origin
2 3 10.0 0.0 0.0 1.0 1    # Dendrite from soma
3 3 20.0 5.0 0.0 0.8 2    # Branch continues
4 3 20.0 -5.0 0.0 0.8 2   # Fork
```

### Mapping to ndev-morphology

| SWC Concept | ndev-morphology Equivalent |
|-------------|---------------------------|
| Sample ID | Node index in directed tree |
| Type | Need classification (soma labels, branch type heuristics) |
| X, Y, Z | `skeleton.coordinates[node]` |
| Radius | Could derive from skeleton width or set constant |
| Parent | Edge source in DirectedTree.graph |

### Integration Strategy

Prefer **napari-swc-reader** as optional dependency for I/O, with our own conversion functions:

```python
# Future: ndev_morphology/swc.py
def tree_to_swc(tree: DirectedTree, soma_radius: float = 5.0) -> pd.DataFrame:
    """Convert DirectedTree to SWC-format DataFrame."""
    ...

def swc_to_tree(swc_df: pd.DataFrame, spacing: tuple) -> DirectedTree:
    """Convert SWC DataFrame to DirectedTree."""
    ...
```

---

## skan vs networkx Design

| Library | Purpose | When to Use |
|---------|---------|-------------|
| **skan** | Efficient skeleton representation (sparse CSR matrices), branch summarization, Sholl analysis | Default for skeleton analysis, branch measurements |
| **networkx** | Directed graph algorithms, tree traversal, branch ordering | When you need directed trees (soma→tips), branch order computation |

**Key insight**: skan already provides `skeleton_to_nx()` for conversion to networkx. Don't reinvent this.

**Workflow**:
1. Create skeleton with `skan.Skeleton(image, spacing=...)`
2. Use `skan.summarize(skeleton)` for branch properties
3. Convert to networkx only when you need directed tree operations:
   ```python
   from skan.csr import skeleton_to_nx
   G = skeleton_to_nx(skeleton)
   # Add direction from soma node
   ```

---

## Module Specifications

### `model.py` — MorphologyModel

**Purpose**: Unified data wrapper for skeleton analysis and visualization.

**Classes**:
- `MorphologyModel` — Wraps skan.Skeleton with metadata and lazy-computed properties

**Functions**:
- `get_model_from_layer(layer)` — Retrieve model from layer metadata
- `attach_model_to_layer(layer, model)` — Store model in layer metadata

**Usage Pattern**:
```python
from ndev_morphology import MorphologyModel

# From numpy array (explicit creation)
model = MorphologyModel.from_array(skeleton_image, spacing=(0.2, 0.2))

# From napari layer (widget convenience)
model = MorphologyModel.from_layer(skeleton_layer)

# Access properties
print(model.n_branches, model.total_length)
branches_df = model.compute_branches()
```

**Status**: ✅ Implemented and tested

### `labels.py` — Label Preprocessing

**Purpose**: Prepare segmentation masks for skeletonization.

**Functions**:
- `filter_labels_by_size(labels, min_size, max_size)` — Remove labels outside area range
- `exclude_labels_on_edges(labels)` — Remove border-touching labels
- `connect_breaks_between_labels(labels, connect_distance)` — Merge nearby fragments

**Status**: ✅ Implemented and tested

### `skeleton.py` — Skeleton Operations

**Purpose**: Create and modify skeletons while preserving label identity.

**Functions**:
- `skeletonize_labels(labels)` — Skeletonize preserving label IDs (unique vs scikit-image)
- `separate_touching_skeleton_labels(skeleton_labels)` — Remove touching points between labels
- `exclude_region_from_skeleton(skeleton, mask, dilation_iterations)` — Exclude soma region

**Status**: ✅ Implemented and tested

### `sholl.py` — Sholl Analysis

**Purpose**: Quantify branching complexity via shell crossing counts.

**Classes**:
- `ShollResult` — Dataclass with radii, counts, and computed statistics:
  - `max_crossings`, `critical_radius`, `enclosing_radius`, `mean_crossings`, `total_crossings`

**Functions**:
- `compute_sholl_profile(skeleton, center, radii, ...)` — Wrapper around `skan.sholl_analysis`

**Status**: ✅ Implemented and tested

### `branch.py` — Branch Analysis

**Purpose**: Branch-level measurements and classification using skan.

**Classes**:
- `BranchType` — Enum for branch types (ENDPOINT_TO_ENDPOINT, JUNCTION_TO_ENDPOINT, etc.)

**Functions**:
- `summarize_branches(skeleton, *, intensity_image=None)` — Extended skan.summarize with tortuosity
- `compute_tortuosity(branches_df)` — Path length / euclidean distance for each branch
- `filter_branches_by_type(branches_df, types)` — Filter DataFrame by branch types

**Status**: ✅ Implemented and tested

### `soma.py` — Soma Detection

**Purpose**: Identify cell body for directed tree analysis.

**Functions**:
- `detect_soma_centroid(labels, *, method, intensity_image)` — Find soma center via multiple methods
- `find_soma_node(skeleton, soma_centroid)` — Find nearest skeleton node to soma
- `get_label_centroid(labels, label_id)` — Get centroid of a specific label

**Status**: ✅ Implemented and tested

### `tree.py` — Directed Tree Analysis

**Purpose**: Analyze skeletons as directed trees rooted at soma.

**Classes**:
- `DirectedTree` — Wraps networkx DiGraph with morphology-specific methods
- `BranchOrder` — Constants for semantic branch order values

**Functions**:
- `create_directed_tree(skeleton, root_coords)` — Create single directed tree
- `create_directed_trees_from_soma(skeleton_image, soma_mask, soma_centroid)` — Create multiple trees from radiating branches
- `compute_branch_order(tree)` — Centrifugal ordering (1=primary, 2=secondary, etc.)
- `compute_strahler_order(tree)` — Strahler ordering (tips=1, merging increases)
- `find_longest_path(tree)` — Find longest path from root to tip
- `summarize_directed_tree(tree)` — DataFrame with per-branch order information

**Status**: ✅ Implemented and tested

### `analysis.py` — Per-Cell Analysis

**Purpose**: Batch processing pipeline for multi-cell images.

**Classes**:
- `CellAnalysisResult` — Dataclass containing skeleton, branches, Sholl, summary

**Functions**:
- `analyze_single_cell(skeleton_labels, label_id, ...)` — Comprehensive analysis of one cell
- `analyze_all_cells(skeleton_labels, ...)` — Analyze all labeled cells, returns DataFrame
- `analyze_all_cells_generator(...)` — Memory-efficient generator version
- `aggregate_branch_stats(results)` — Combine branch DataFrames from multiple cells

**Status**: ✅ Implemented and tested

### `pruning.py` — Skeleton Pruning

**Purpose**: Remove unwanted branches from skeletons.

**Functions**:
- `prune_short_branches(skeleton, min_length)` — Remove branches below length threshold
- `remove_isolated_cycles(skeleton)` — Remove isolated cycle branches
- `prune_skeleton_to_image(skeleton, prune_indices)` — Convert pruned skeleton back to image

**Status**: ✅ Implemented and tested

### `_geometry.py` — Visualization Utilities

**Purpose**: Convert analysis results to napari layer formats.

**Functions**:
- `skeleton_to_paths(skeleton)` — Convert skan.Skeleton to paths + properties for Shapes
- `sholl_shells_to_ellipses(center, radii)` — Generate ellipse bounding boxes for Sholl visualization

**Status**: ✅ Implemented, needs tests

---

## Testing Strategy

### Test Organization

```
tests/
├── conftest.py           # Shared fixtures
├── test_labels.py        # Label preprocessing tests
├── test_skeleton.py      # Skeleton operations tests ✅
├── test_sholl.py         # Sholl analysis tests ✅
├── test_branch.py        # Branch analysis tests
├── test_soma.py          # Soma detection tests
├── test_analysis.py      # Per-cell pipeline tests
├── test_geometry.py      # Visualization utility tests
└── resources/            # Test data files (if needed)
```

### Testing Priorities

1. **Core functions first** — Pure Python, no napari dependency
2. **Unit tests > Integration tests** — Fast iteration
3. **Defer widget tests** — Manual testing with sample data is faster during development

### Testing magicgui Functions Without napari

The `@magic_factory` decorated functions can be tested without napari:

```python
def test_skeleton_to_shapes_core():
    """Test skeleton_to_shapes logic without napari viewer."""
    from ndev_morphology.widgets.skeleton_widget import skeleton_to_shapes

    # Create test skeleton
    skeleton = np.zeros((50, 50), dtype=np.uint8)
    skeleton[25, 10:40] = 1

    # Call the function directly (returns LayerDataTuple)
    result = skeleton_to_shapes(
        skeleton_image=skeleton,
        spacing_y=1.0,
        spacing_x=1.0,
        edge_color='branch_distance',
        edge_colormap='viridis',
        edge_width=2.0,
    )

    # Validate result structure
    data, kwargs, layer_type = result
    assert layer_type == 'shapes'
    assert 'properties' in kwargs
```

### ndevio Sample Data Integration

Use `ndev-sampledata` for realistic test fixtures:

```python
# In conftest.py
import pytest

@pytest.fixture
def neuron_labels_data():
    """Load neuron labels from ndevio sample data."""
    from ndevio.sampledata import neuron_labels

    layer_data_tuples = neuron_labels()
    # Extract the numpy array from LayerDataTuple
    data, kwargs, layer_type = layer_data_tuples[0]
    return data
```

---

## Widget Development

### Pattern: Thin Wrapper Widgets

```python
# In widgets/some_widget.py
from magicgui import magic_factory
from ..core_module import core_function

@magic_factory(
    call_button='Run Analysis',
    param={'tooltip': 'Description of parameter'},
)
def some_widget(
    image: 'napari.types.LabelsData',
    param: float = 1.0,
) -> 'napari.types.LayerDataTuple':
    """Widget docstring shown in napari."""
    # Call core function
    result = core_function(image, param)

    # Convert to napari layer format
    return (result_data, {'name': 'Result'}, 'labels')
```

### Existing Widgets

| Widget | Purpose | Core Function |
|--------|---------|---------------|
| `refine_labels_widget` | Filter/clean labels | `filter_labels_by_size`, `exclude_labels_on_edges`, `connect_breaks_between_labels` |
| `skeletonize_labels_widget` | Create label-aware skeleton | `skeletonize_labels`, `separate_touching_skeleton_labels` |
| `skeleton_to_shapes` | Visualize skeleton as Shapes | `skeleton_to_paths` |
| `sholl_analysis` | Interactive Sholl | `compute_sholl_profile`, `sholl_shells_to_ellipses` |
| `branch_analysis` | Branch measurements | `summarize_branches` |
| `directed_tree_analysis` | Directed tree from soma | `create_directed_trees_from_soma`, `compute_branch_order` |
| `prune_skeleton` | Remove short branches | `prune_short_branches` |

---

## Implementation Status

### Complete ✅
- [x] `model.py` — MorphologyModel data wrapper
- [x] `labels.py` — Label preprocessing
- [x] `skeleton.py` — Skeleton creation and modification
- [x] `sholl.py` — Sholl analysis
- [x] `branch.py` — Branch measurements and classification
- [x] `soma.py` — Soma detection
- [x] `tree.py` — Directed tree analysis with branch ordering
- [x] `analysis.py` — Per-cell batch processing
- [x] `pruning.py` — Skeleton pruning
- [x] `_geometry.py` — Visualization utilities
- [x] Core widgets
- [x] 153 tests passing

### In Progress 🔄
- [ ] napari-assistant compatibility — ensure all widgets have proper `display_name` prefixes
- [ ] napari-workflows compatibility — verify all core functions work with `Workflow.set()`

### Future ⏳
- [ ] `swc.py` — SWC import/export
- [ ] Shaft ordering (Neurolucida-style main axon identification)
- [ ] Axon/dendrite classification (intensity-based)
- [ ] 3D support improvements
- [ ] Integration with napari-swc-reader
- [ ] Upstream improvements to napari-workflows if features are missing

---

## Common Patterns

### Creating a skan.Skeleton

```python
import skan

# From binary skeleton image
skeleton = skan.Skeleton(skeleton_image, spacing=(0.2, 0.2))

# From labels (use ndev-morphology first)
from ndev_morphology import skeletonize_labels
skeleton_labels = skeletonize_labels(labels)
# Then create skan.Skeleton per label or from combined binary
```

### Getting Branch Properties

```python
import skan

skeleton = skan.Skeleton(skeleton_image, spacing=(0.2, 0.2))
branches = skan.summarize(skeleton)

# branches is a DataFrame with columns:
# - skeleton_id, branch_type, branch_distance, euclidean_distance
# - coord_src_0, coord_src_1, coord_dst_0, coord_dst_1 (endpoints)
```

### Converting to networkx

```python
from skan.csr import skeleton_to_nx

G = skeleton_to_nx(skeleton)  # Returns networkx.Graph

# For directed tree from soma:
import networkx as nx
soma_node = find_closest_node_to_soma(skeleton, soma_centroid)
DG = nx.bfs_tree(G, soma_node)  # Directed graph from soma
```

---

## Dependencies

### Core Dependencies
```toml
dependencies = [
    "numpy",
    "pandas",
    "scipy",
    "skan",
    "scikit-image>=0.18.0",
]
```

### Optional Dependencies
```toml
[project.optional-dependencies]
napari = ["napari", "magicgui"]
gpu = ["pyclesperanto"]
```

### Dev Dependencies
```toml
[dependency-groups]
dev = [
    "pytest",
    "pytest-cov",
    "napari[pyqt6]",
    "ndev-sampledata",  # For realistic test fixtures
]
```

---

## Anti-Patterns to Avoid

### ❌ Don't: Embed analysis logic in widgets

```python
# BAD
@magic_factory()
def analyze_widget(labels):
    # 50 lines of analysis code here
    skeleton = skeletonize(labels)
    branches = compute_branches(skeleton)
    ...
```

### ✅ Do: Call core functions from widgets

```python
# GOOD
@magic_factory()
def analyze_widget(labels):
    from ..analysis import analyze_single_cell
    result = analyze_single_cell(labels)
    return (result['data'], result['kwargs'], 'shapes')
```

---

### ❌ Don't: Reinvent skan functionality

```python
# BAD - skan already does this
def my_branch_summary(skeleton_image):
    # Custom branch counting...
```

### ✅ Do: Wrap skan with convenience

```python
# GOOD - add value on top of skan
def summarize_branches(skeleton, intensity_image=None):
    branches = skan.summarize(skeleton, extra_properties=[intensity_mean])
    branches['tortuosity'] = branches['branch_distance'] / branches['euclidean_distance']
    return branches
```

---

## Resources

### skan Documentation
- [skan API Reference](https://skeleton-analysis.org/stable/api/skan.csr.html)
- [skan.Skeleton class](https://skeleton-analysis.org/stable/api/skan.csr.html#skan.csr.Skeleton)
- [skan.summarize](https://skeleton-analysis.org/stable/api/skan.csr.html#skan.csr.summarize)
- [skan.sholl_analysis](https://skeleton-analysis.org/stable/api/skan.csr.html#skan.csr.sholl_analysis)
- [skeleton_to_nx](https://skeleton-analysis.org/stable/api/skan.csr.html#skan.csr.skeleton_to_nx)

### SWC Format
- [SWC Specification](http://www.neuronland.org/NLMorphologyConverter/MorphologyFormats/SWC/Spec.html)
- [NeuroMorpho.org](http://www.neuromorpho.org/) — Standard SWC archive

### Related Projects
- [napari-workflows](https://github.com/haesleinhuepf/napari-workflows) — **Core dependency** for workflow management
- [napari-assistant](https://github.com/haesleinhuepf/napari-assistant) — **Target UI** for workflow building
- [devbio-napari](https://www.napari-hub.org/plugins/devbio-napari) — Bundle of compatible napari plugins
- [MorphoPy](https://github.com/berenslab/MorphoPy) — Morphology analysis inspiration
- [napari-skan](https://github.com/jni/skan) — skan's napari integration (reference)
- [napari-swc-reader](https://github.com/kephale/napari-swc-reader) — SWC I/O for napari
- [napari-swc-editor](https://github.com/LaboratoryOpticsBiosciences/napari-swc-editor) — SWC editing in napari

### ndev-kit Ecosystem
- [ndevio](../ndevio/) — Image I/O and sample data
- [ndev-settings](../ndev-settings/) — Settings framework
- Root [AGENTS.md](../AGENTS.md) — Workspace-level guidance

---

## Changelog

- **2025-12-17**: napari-workflows integration decision
  - Added "No Backwards Compatibility Required" statement
  - Added napari-workflows and napari-assistant as core design principles
  - Simplified MorphologyModel section (workflow management delegated to napari-workflows)
  - Updated implementation priorities to focus on napari-assistant compatibility

- **2025-12-17**: Major architecture update
  - Added MorphologyModel data wrapper with pipeline philosophy
  - Added SWC format specification and integration strategy
  - Added tree.py module documentation
  - Updated implementation status (all core modules now complete)
  - Added 153 passing tests

- **2025-12-15**: Initial AGENTS.md created
  - Documented package architecture and module specifications
  - Defined skan vs networkx design decisions
  - Outlined implementation phases
  - Established testing strategy

---

## Contact

**Package Maintainer**: Tim Monko (timmonko@gmail.com)

**Issues**: GitHub Issues (when public)

---

*This is living documentation. Update as the package evolves.*
