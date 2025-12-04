"""
Sholl analysis widget for napari.

Performs Sholl analysis on skeleton images and visualizes results
as concentric shell shapes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import napari.types
import numpy as np
import skan
from magicgui import magic_factory

from .._geometry import sholl_shells_to_ellipses
from ..sholl import compute_sholl_profile

if TYPE_CHECKING:
    pass

__all__ = ["sholl_analysis"]


COLORMAP_CHOICES = [
    "viridis",
    "plasma",
    "magma",
    "inferno",
    "turbo",
    "hot",
    "cool",
]


@magic_factory(
    call_button="Run Sholl Analysis",
    skeleton_image={"label": "Skeleton Image"},
    center_y={"label": "Center Y (pixels)", "min": 0.0, "max": 10000.0},
    center_x={"label": "Center X (pixels)", "min": 0.0, "max": 10000.0},
    spacing_y={
        "label": "Spacing Y",
        "min": 0.001,
        "max": 100.0,
        "step": 0.01,
        "tooltip": "Physical pixel spacing in Y for distance calculations.",
    },
    spacing_x={
        "label": "Spacing X",
        "min": 0.001,
        "max": 100.0,
        "step": 0.01,
        "tooltip": "Physical pixel spacing in X for distance calculations.",
    },
    radius_start={
        "label": "Start Radius",
        "min": 0.1,
        "tooltip": "Starting radius in physical units (µm).",
    },
    radius_end={
        "label": "End Radius",
        "min": 1.0,
        "tooltip": "Ending radius in physical units (µm).",
    },
    radius_step={
        "label": "Radius Step",
        "min": 0.1,
        "max": 50.0,
        "tooltip": "Step between radii in physical units (µm).",
    },
    shell_opacity={
        "label": "Shell Opacity",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
    },
    edge_colormap={
        "label": "Colormap",
        "choices": COLORMAP_CHOICES,
        "tooltip": "Colormap for shell coloring by crossing count.",
    },
)
def sholl_analysis(
    skeleton_image: napari.types.LabelsData,
    center_y: float = 100.0,
    center_x: float = 100.0,
    spacing_y: float = 1.0,
    spacing_x: float = 1.0,
    radius_start: float = 5.0,
    radius_end: float = 100.0,
    radius_step: float = 5.0,
    shell_opacity: float = 0.2,
    edge_colormap: str = "viridis",
) -> napari.types.LayerDataTuple:
    """
    Perform Sholl analysis and create a Shapes layer with concentric shells.

    Sholl analysis counts skeleton crossings at concentric circles/shells
    from a center point, providing a measure of branching complexity.

    Parameters
    ----------
    skeleton_image : napari.types.LabelsData
        Binary or labeled skeleton image.
    center_y : float
        Y coordinate of center point in pixels.
    center_x : float
        X coordinate of center point in pixels.
    spacing : tuple of float
        Physical pixel spacing (Y, X) for distance calculations.
    radius_start : float
        Starting radius in physical units.
    radius_end : float
        Ending radius in physical units.
    radius_step : float
        Step between radii in physical units.
    shell_opacity : float
        Opacity of shell faces (0-1).
    edge_colormap : str
        Colormap for coloring shells by crossing count.

    Returns
    -------
    napari.types.LayerDataTuple
        Shapes layer with Sholl shells colored by crossing count.

    Notes
    -----
    The Sholl result statistics (max_crossings, critical_radius, etc.)
    are stored in the layer metadata and printed to console.
    """
    # Validate input
    skeleton_arr = np.asarray(skeleton_image)
    if not np.any(skeleton_arr):
        raise ValueError("Skeleton image is empty (no non-zero pixels)")

    # Create spacing tuple
    spacing = (spacing_y, spacing_x)

    # Create skan Skeleton with spacing
    skel = skan.Skeleton(skeleton_arr.astype(float), spacing=spacing)

    # Center in physical units
    center_physical = np.array([center_y, center_x]) * np.array(spacing)

    # Create radii array
    radii = np.arange(radius_start, radius_end, radius_step)
    if len(radii) == 0:
        raise ValueError(
            f"No radii generated with start={radius_start}, "
            f"end={radius_end}, step={radius_step}"
        )

    # Run Sholl analysis
    result = compute_sholl_profile(skel, center=center_physical, radii=radii)

    # Print summary to console
    print("=" * 50)
    print("Sholl Analysis Results:")
    print(f"  Max crossings: {result.max_crossings}")
    print(f"  Critical radius: {result.critical_radius:.2f}")
    print(f"  Enclosing radius: {result.enclosing_radius:.2f}")
    print(f"  Mean crossings: {result.mean_crossings:.2f}")
    print(f"  Total crossings: {result.total_crossings}")
    print("=" * 50)

    # Convert radii back to pixels for display
    radii_px = result.radii / spacing_y  # Use Y spacing for radius
    center_px = np.array([center_y, center_x])

    # Generate shell shapes as ellipses (circles)
    shells = sholl_shells_to_ellipses(center_px, radii_px, ndim=2)

    # Properties for coloring
    properties = {
        "radius": result.radii,
        "crossings": result.counts,
    }

    # Metadata with full Sholl result
    metadata = {
        "sholl_result": result.to_dict(),
        "center_pixels": (center_y, center_x),
        "spacing": spacing,
    }

    return (
        shells,
        {
            "shape_type": "ellipse",
            "properties": properties,
            "edge_color": "crossings",
            "edge_colormap": edge_colormap,
            "edge_width": 1.5,
            "face_color": "transparent",
            "opacity": shell_opacity,
            "name": "Sholl shells",
            "metadata": metadata,
        },
        "shapes",
    )
