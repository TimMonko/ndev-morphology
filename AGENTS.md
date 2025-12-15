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

## Package Purpose

`ndev-morphology` provides single-cell morphological analysis tools, primarily for neuronal structures (dendrites, axons). The package bridges **skan** (skeleton analysis) and **napari** (visualization) with a focus on:

1. **Label preparation** — Filtering, cleaning, and preparing segmentation masks
2. **Skeleton creation** — Label-aware skeletonization preserving cell identity
3. **Branch analysis** — Quantifying branch properties using skan
4. **Sholl analysis** — Counting skeleton crossings at concentric shells
5. **Directed tree analysis** — Understanding branch hierarchy from soma to tips (via networkx)
6. **napari visualization** — Interactive widgets for analysis and exploration

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
├── labels.py          # Label preprocessing (pure Python)
├── skeleton.py        # Skeleton creation/modification (pure Python)
├── sholl.py           # Sholl analysis with ShollResult dataclass
├── branch.py          # Branch-level measurements and classification
├── soma.py            # Soma detection and centroid extraction
├── analysis.py        # Per-cell analysis pipelines, batch processing
├── _geometry.py       # Visualization helpers (skeleton_to_paths, etc.)
├── widgets/           # napari/magicgui widgets (UI layer)
│   ├── labels_widget.py
│   ├── skeletonize_widget.py
│   ├── skeleton_widget.py
│   ├── sholl_widget.py
│   └── ...
└── napari.yaml        # Plugin manifest
```

**Rule**: Any function in `widgets/` should be a thin wrapper around core functionality.

### skan vs networkx Design

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

### `labels.py` — Label Preprocessing

**Purpose**: Prepare segmentation masks for skeletonization.

**Functions**:
- `filter_labels_by_size(labels, min_size, max_size)` — Remove labels outside area range
- `exclude_labels_on_edges(labels)` — Remove border-touching labels
- `connect_breaks_between_labels(labels, connect_distance)` — Merge nearby fragments

**Status**: ✅ Implemented, needs dedicated tests

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

### `branch.py` — Branch Analysis (TO IMPLEMENT)

**Purpose**: Branch-level measurements and classification using skan.

**Planned Functions**:
```python
def summarize_branches(skeleton: skan.Skeleton, intensity_image: ArrayLike | None = None) -> pd.DataFrame:
    """
    Summarize branch properties from a skeleton.

    Wraps skan.summarize() with optional intensity measurements.
    Adds computed metrics like tortuosity.
    """

def classify_branch_type(branch_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Classify branches by type.

    skan branch_type: 0=endpoint-endpoint, 1=junction-endpoint, 2=junction-junction
    Adds human-readable labels and filters.
    """

def compute_tortuosity(branch_summary: pd.DataFrame) -> pd.Series:
    """
    Compute tortuosity (path_length / euclidean_distance) for each branch.
    """
```

**Status**: ⏳ Not implemented

### `soma.py` — Soma Detection (TO IMPLEMENT)

**Purpose**: Identify cell body for directed tree analysis.

**Planned Functions**:
```python
def detect_soma_centroid(
    labels: ArrayLike,
    *,
    method: Literal['largest_region', 'roundest_region', 'intensity_peak'] = 'largest_region',
    intensity_image: ArrayLike | None = None,
) -> np.ndarray:
    """
    Detect soma centroid from label image.

    Methods:
    - 'largest_region': Centroid of largest connected component
    - 'roundest_region': Most circular region (lowest eccentricity)
    - 'intensity_peak': Peak of DAPI/nucleus channel
    """

def find_soma_node(skeleton: skan.Skeleton, soma_centroid: ArrayLike) -> int:
    """
    Find the skeleton node closest to the soma centroid.

    This node becomes the root for directed tree analysis.
    """
```

**Status**: ⏳ Not implemented

### `analysis.py` — Per-Cell Analysis (TO IMPLEMENT)

**Purpose**: Batch processing pipeline for multi-cell images.

**Planned Functions**:
```python
def analyze_single_cell(
    skeleton_image: ArrayLike,
    label_id: int,
    soma_centroid: ArrayLike,
    *,
    spacing: tuple[float, ...] = (1.0, 1.0),
    sholl_step: float = 1.0,
) -> dict:
    """
    Comprehensive analysis of a single labeled cell.

    Returns dict with:
    - 'skeleton': skan.Skeleton object
    - 'branches': pd.DataFrame of branch properties
    - 'sholl': ShollResult
    - 'summary': dict of aggregate metrics
    """

def analyze_all_cells(
    skeleton_labels: ArrayLike,
    soma_labels: ArrayLike | None = None,
    *,
    spacing: tuple[float, ...] = (1.0, 1.0),
) -> pd.DataFrame:
    """
    Analyze all labeled cells in an image.

    Returns DataFrame with one row per cell, columns for all metrics.
    """
```

**Status**: ⏳ Not implemented

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

---

## Implementation Phases

### Phase 1: Foundation (Current)
- [x] `labels.py` — Label preprocessing
- [x] `skeleton.py` — Skeleton creation
- [x] `sholl.py` — Sholl analysis
- [x] `_geometry.py` — Visualization utilities
- [x] Basic widgets
- [ ] `test_labels.py` — Add dedicated tests
- [ ] `test_geometry.py` — Add tests

### Phase 2: Branch Analysis
- [ ] `branch.py` — Branch measurements and classification
- [ ] Integrate skan.summarize with intensity measurements
- [ ] Tortuosity and other derived metrics
- [ ] `test_branch.py`

### Phase 3: Soma and Directed Trees
- [ ] `soma.py` — Soma detection
- [ ] Convert skeleton to networkx directed tree
- [ ] Branch order computation
- [ ] `test_soma.py`

### Phase 4: Per-Cell Pipeline
- [ ] `analysis.py` — Batch processing
- [ ] Per-cell DataFrame output
- [ ] Export utilities (CSV, etc.)
- [ ] `test_analysis.py`

### Phase 5: Advanced Features
- [ ] Axon/dendrite classification (intensity-based)
- [ ] 3D support improvements
- [ ] GPU acceleration (pyclesperanto)
- [ ] Widget tests

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

### Related Projects
- [MorphoPy](https://github.com/berenslab/MorphoPy) — Morphology analysis inspiration
- [napari-skan](https://github.com/jni/skan) — skan's napari integration (reference)

### ndev-kit Ecosystem
- [ndevio](../ndevio/) — Image I/O and sample data
- [ndev-settings](../ndev-settings/) — Settings framework
- Root [AGENTS.md](../AGENTS.md) — Workspace-level guidance

---

## Changelog

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
