"""
Widget for batch cell morphology analysis.

Provides batch processing with progress tracking and cancellation
using nbatch.BatchRunner.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from magicgui.widgets import (
    CheckBox,
    ComboBox,
    Container,
    FileEdit,
    FloatSpinBox,
    Label,
    ProgressBar,
    PushButton,
    SpinBox,
    Table,
)

if TYPE_CHECKING:
    from napari import Viewer

__all__ = ['BatchAnalysisWidget']


class BatchAnalysisWidget(Container):
    """
    Widget for batch morphology analysis with progress tracking.

    Uses analyze_all_cells_generator for item-by-item progress
    and nbatch.BatchRunner for threaded execution.
    """

    def __init__(self, viewer: Viewer):
        super().__init__()
        self.viewer = viewer
        self._results = []
        self._summary_df = None
        self._is_running = False

        # Build UI
        self._build_ui()

    def _build_ui(self):
        """Construct widget UI elements."""
        # Layer selection
        self._labels_combo = ComboBox(
            label='Cell Labels',
            choices=self._get_labels_layers,
        )
        self._soma_combo = ComboBox(
            label='Soma Labels (optional)',
            choices=self._get_labels_layers_optional,
            nullable=True,
        )

        # Spacing
        self._spacing_y = FloatSpinBox(
            label='Spacing Y (µm)',
            value=1.0,
            min=0.001,
            step=0.1,
        )
        self._spacing_x = FloatSpinBox(
            label='Spacing X (µm)',
            value=1.0,
            min=0.001,
            step=0.1,
        )

        # Analysis options
        self._run_sholl = CheckBox(label='Run Sholl Analysis', value=True)
        self._sholl_step = FloatSpinBox(
            label='Sholl Step (µm)',
            value=1.0,
            min=0.1,
            step=0.5,
        )
        self._min_size = SpinBox(
            label='Min Skeleton Size (px)',
            value=10,
            min=1,
        )

        # Output
        self._output_path = FileEdit(
            label='Save Results To',
            mode='w',
            filter='*.csv',
        )

        # Status and progress
        self._status_label = Label(value='Ready')
        self._progress = ProgressBar(min=0, max=100, value=0)

        # Buttons
        self._run_button = PushButton(text='Run Batch Analysis')
        self._cancel_button = PushButton(text='Cancel', enabled=False)

        # Results table (shows summary)
        self._results_table = Table(value={'Column': []})

        # Connect signals
        self._run_button.clicked.connect(self._on_run)
        self._cancel_button.clicked.connect(self._on_cancel)

        # Add all widgets
        self.extend(
            [
                self._labels_combo,
                self._soma_combo,
                self._spacing_y,
                self._spacing_x,
                self._run_sholl,
                self._sholl_step,
                self._min_size,
                self._output_path,
                self._status_label,
                self._progress,
                self._run_button,
                self._cancel_button,
                self._results_table,
            ]
        )

    def _get_labels_layers(self, widget=None):
        """Get list of Labels layers in viewer."""
        import napari

        return [
            layer.name
            for layer in self.viewer.layers
            if isinstance(layer, napari.layers.Labels)
        ]

    def _get_labels_layers_optional(self, widget=None):
        """Get list of Labels layers plus None option."""
        layers = self._get_labels_layers()
        return [None] + layers

    def _on_run(self):
        """Start batch analysis."""
        import pandas as pd

        from ..analysis import analyze_all_cells_generator

        # Get selected layers
        labels_name = self._labels_combo.value
        if not labels_name:
            self._status_label.value = 'Error: Select a labels layer'
            return

        labels_layer = self.viewer.layers[labels_name]
        labels = np.asarray(labels_layer.data)

        soma_labels = None
        soma_name = self._soma_combo.value
        if soma_name:
            soma_labels = np.asarray(self.viewer.layers[soma_name].data)

        # Get parameters
        spacing = (self._spacing_y.value, self._spacing_x.value)

        # Count cells
        label_ids = np.unique(labels)
        label_ids = label_ids[label_ids > 0]
        total_cells = len(label_ids)

        if total_cells == 0:
            self._status_label.value = 'Error: No cells found in labels'
            return

        # Update UI state
        self._is_running = True
        self._run_button.enabled = False
        self._cancel_button.enabled = True
        self._progress.max = total_cells
        self._progress.value = 0
        self._results = []

        self._status_label.value = f'Analyzing 0/{total_cells} cells...'

        # Run analysis (in main thread for now, nbatch threading can be added)
        try:
            for result in analyze_all_cells_generator(
                labels,
                spacing=spacing,
                soma_labels=soma_labels,
                run_sholl=self._run_sholl.value,
                sholl_step=self._sholl_step.value,
                min_skeleton_size=self._min_size.value,
                on_progress=self._on_progress_update,
            ):
                if not self._is_running:
                    # Cancelled
                    break
                self._results.append(result)

            # Create summary DataFrame
            if self._results:
                summaries = [r.to_dict() for r in self._results]
                self._summary_df = pd.DataFrame(summaries)

                # Update table display
                self._results_table.value = self._summary_df.head(50).to_dict(
                    'list'
                )

                # Save if path provided
                output_path = self._output_path.value
                if output_path:
                    self._summary_df.to_csv(output_path, index=False)
                    self._status_label.value = (
                        f'Done! Analyzed {len(self._results)} cells. '
                        f'Saved to {output_path}'
                    )
                else:
                    self._status_label.value = (
                        f'Done! Analyzed {len(self._results)} cells.'
                    )
            else:
                self._status_label.value = 'No cells passed filtering criteria'

        except Exception as e:
            self._status_label.value = f'Error: {e}'

        finally:
            self._is_running = False
            self._run_button.enabled = True
            self._cancel_button.enabled = False

    def _on_progress_update(self, current: int, total: int, label_id: int):
        """Update progress bar and status."""
        self._progress.value = current
        self._status_label.value = (
            f'Analyzing {current}/{total} cells (cell {label_id})...'
        )
        # Allow Qt to process events
        from qtpy.QtWidgets import QApplication

        QApplication.processEvents()

    def _on_cancel(self):
        """Cancel running analysis."""
        self._is_running = False
        self._status_label.value = 'Cancelled'
        self._cancel_button.enabled = False
