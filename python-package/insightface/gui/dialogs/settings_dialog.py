"""Application settings dialog."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.config import AppConfig, save_config
from ..core.navigation import NAVIGATION_MODES, mode_from_value
from ..core.paths import ensure_workspace


class SettingsDialog(QDialog):
    settingsSaved = Signal()

    def __init__(self, context, parent=None):
        super().__init__(parent)
        self.context = context
        self.setWindowTitle("Settings")
        self.resize(900, 650)
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        self._build_general_tab()
        self._build_paths_tab()
        self._build_recognition_tab()
        self._build_privacy_tab()
        self._build_exports_tab()
        self._build_appearance_tab()
        self._build_advanced_tab()
        self.tabs.setCurrentIndex(max(0, min(self.context.config.ui_settings_dialog_last_tab, self.tabs.count() - 1)))

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Apply | QDialogButtonBox.Cancel | QDialogButtonBox.Reset)
        buttons.button(QDialogButtonBox.Save).clicked.connect(self.save_and_close)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply)
        buttons.button(QDialogButtonBox.Cancel).clicked.connect(self.reject)
        buttons.button(QDialogButtonBox.Reset).clicked.connect(self.reset_fields)
        layout.addWidget(buttons)

    def _tab(self) -> tuple[QWidget, QGridLayout]:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        grid = QGridLayout(body)
        grid.setColumnStretch(1, 1)
        grid.setVerticalSpacing(10)
        scroll.setWidget(body)
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.addWidget(scroll)
        return wrapper, grid

    def _add_row(self, grid: QGridLayout, row: int, label: str, widget, button: QPushButton | None = None) -> None:
        grid.addWidget(QLabel(label), row, 0)
        grid.addWidget(widget, row, 1)
        if button is not None:
            grid.addWidget(button, row, 2)

    def _path_row(self, grid: QGridLayout, row: int, label: str, line_edit: QLineEdit, folder: bool = True) -> None:
        line_edit.setMinimumWidth(520)
        button = QPushButton("Browse")
        button.clicked.connect(lambda checked=False, target=line_edit, is_folder=folder: self._browse_path(target, is_folder))
        self._add_row(grid, row, label, line_edit, button)

    def _build_general_tab(self) -> None:
        cfg = self.context.config
        tab, grid = self._tab()
        self.default_mode = QComboBox()
        for mode in NAVIGATION_MODES.values():
            self.default_mode.addItem(mode.title, mode.id.value)
        self.default_mode.setCurrentIndex(max(0, self.default_mode.findData(mode_from_value(cfg.ui_default_mode).value)))
        self.safe_mode = QCheckBox("Start next launch without automatic model loading")
        self.safe_mode.setChecked(cfg.safe_mode)
        self._add_row(grid, 0, "Default mode", self.default_mode)
        self._add_row(grid, 1, "Safe mode", self.safe_mode)
        self.tabs.addTab(tab, "General")

    def _build_paths_tab(self) -> None:
        cfg = self.context.config
        tab, grid = self._tab()
        self.workspace = QLineEdit(cfg.workspace_path)
        self.database = QLineEdit(cfg.database_path)
        self.crops = QLineEdit(cfg.crop_dir)
        self.reports = QLineEdit(cfg.report_dir)
        self.exports = QLineEdit(cfg.export_dir)
        self.cache = QLineEdit(cfg.cache_dir)
        for row, (label, edit, folder) in enumerate(
            [
                ("Workspace path", self.workspace, True),
                ("Database path", self.database, False),
                ("Crop output directory", self.crops, True),
                ("Report output directory", self.reports, True),
                ("Export output directory", self.exports, True),
                ("Cache directory", self.cache, True),
            ]
        ):
            self._path_row(grid, row, label, edit, folder)
        open_button = QPushButton("Open Workspace Folder")
        open_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.workspace.text().strip())))
        grid.addWidget(open_button, 6, 1)
        self.tabs.addTab(tab, "Paths & Storage")

    def _build_recognition_tab(self) -> None:
        cfg = self.context.config
        tab, grid = self._tab()
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0, 1)
        self.threshold.setSingleStep(0.01)
        self.threshold.setValue(cfg.recognition_threshold)
        self.top_k = QSpinBox()
        self.top_k.setRange(1, 100)
        self.top_k.setValue(cfg.default_top_k)
        self.min_det = QDoubleSpinBox()
        self.min_det.setRange(0, 1)
        self.min_det.setSingleStep(0.01)
        self.min_det.setValue(cfg.min_detection_score)
        self.min_face = QSpinBox()
        self.min_face.setRange(1, 4096)
        self.min_face.setValue(cfg.min_face_size)
        self.video_interval = QSpinBox()
        self.video_interval.setRange(1, 300)
        self.video_interval.setValue(cfg.video_frame_interval)
        self.camera_skip = QSpinBox()
        self.camera_skip.setRange(1, 60)
        self.camera_skip.setValue(cfg.camera_frame_skip)
        for row, (label, widget) in enumerate(
            [
                ("Default recognition threshold", self.threshold),
                ("Default Top-K", self.top_k),
                ("Minimum detection score", self.min_det),
                ("Minimum face size", self.min_face),
                ("Video frame interval", self.video_interval),
                ("Camera frame skip", self.camera_skip),
            ]
        ):
            self._add_row(grid, row, label, widget)
        self.tabs.addTab(tab, "Recognition Defaults")

    def _build_privacy_tab(self) -> None:
        cfg = self.context.config
        tab, grid = self._tab()
        self.save_crops = QCheckBox("Save face crops")
        self.save_crops.setChecked(cfg.save_crops)
        self.save_logs = QCheckBox("Save recognition logs")
        self.save_logs.setChecked(cfg.save_recognition_logs)
        self.anonymize = QCheckBox("Anonymize report paths")
        self.anonymize.setChecked(cfg.anonymize_report_paths)
        self._add_row(grid, 0, "Crops", self.save_crops)
        self._add_row(grid, 1, "Recognition logs", self.save_logs)
        self._add_row(grid, 2, "Reports", self.anonymize)
        open_logs = QPushButton("Open Log Folder")
        open_logs.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.context.config.log_dir)))
        grid.addWidget(open_logs, 3, 1)
        self.tabs.addTab(tab, "Privacy & Logs")

    def _build_exports_tab(self) -> None:
        tab, grid = self._tab()
        self.default_export = QComboBox()
        self.default_export.addItems(["CSV", "JSON", "Markdown", "HTML"])
        report_dir = QLabel(self.context.config.report_dir)
        report_dir.setTextInteractionFlags(Qt.TextSelectableByMouse)
        anonymize_note = QLabel("Configured in Privacy & Logs.")
        self._add_row(grid, 0, "Default export format", self.default_export)
        self._add_row(grid, 1, "Report output directory", report_dir)
        self._add_row(grid, 2, "Anonymize absolute paths in reports", anonymize_note)
        self.tabs.addTab(tab, "Exports")

    def _build_appearance_tab(self) -> None:
        cfg = self.context.config
        tab, grid = self._tab()
        self.theme = QComboBox()
        self.theme.addItems(["system", "light", "dark"])
        self.theme.setCurrentText(cfg.ui_theme)
        self.sidebar_compact = QCheckBox("Use compact sidebar")
        self.sidebar_compact.setChecked(cfg.ui_sidebar_compact)
        self.show_chips = QCheckBox("Show status chips in top bar")
        self.show_chips.setChecked(cfg.ui_show_status_chips)
        self._add_row(grid, 0, "UI theme", self.theme)
        self._add_row(grid, 1, "Sidebar compact mode", self.sidebar_compact)
        self._add_row(grid, 2, "Status chips", self.show_chips)
        self.tabs.addTab(tab, "Appearance")

    def _build_advanced_tab(self) -> None:
        tab, grid = self._tab()
        export_button = QPushButton("Export Settings")
        export_button.clicked.connect(self.export_settings)
        import_button = QPushButton("Import Settings")
        import_button.clicked.connect(self.import_settings)
        reset_button = QPushButton("Reset Settings")
        reset_button.clicked.connect(self.reset_fields)
        config_path = QLabel(str(Path(self.context.config.workspace_path) / "config.json"))
        config_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self._add_row(grid, 0, "Config file", config_path)
        grid.addWidget(export_button, 1, 1)
        grid.addWidget(import_button, 2, 1)
        grid.addWidget(reset_button, 3, 1)
        self.tabs.addTab(tab, "Advanced")

    def _browse_path(self, line_edit: QLineEdit, folder: bool) -> None:
        if folder:
            path = QFileDialog.getExistingDirectory(self, "Select folder", line_edit.text().strip() or str(Path.home()))
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select file", line_edit.text().strip() or str(Path.home()), "All Files (*)")
        if path:
            line_edit.setText(path)

    def apply(self) -> None:
        cfg = self.context.config
        cfg.ui_default_mode = self.default_mode.currentData()
        cfg.safe_mode = self.safe_mode.isChecked()
        cfg.workspace_path = self.workspace.text().strip()
        paths = ensure_workspace(cfg.workspace_path)
        cfg.database_path = self.database.text().strip() or str(paths["database"])
        cfg.crop_dir = self.crops.text().strip() or str(paths["crops"])
        cfg.report_dir = self.reports.text().strip() or str(paths["reports"])
        cfg.export_dir = self.exports.text().strip() or str(paths["exports"])
        cfg.cache_dir = self.cache.text().strip() or str(paths["cache"])
        cfg.recognition_threshold = float(self.threshold.value())
        cfg.default_top_k = int(self.top_k.value())
        cfg.min_detection_score = float(self.min_det.value())
        cfg.min_face_size = int(self.min_face.value())
        cfg.video_frame_interval = int(self.video_interval.value())
        cfg.camera_frame_skip = int(self.camera_skip.value())
        cfg.save_crops = self.save_crops.isChecked()
        cfg.save_recognition_logs = self.save_logs.isChecked()
        cfg.anonymize_report_paths = self.anonymize.isChecked()
        cfg.ui_theme = self.theme.currentText()
        cfg.ui_sidebar_compact = self.sidebar_compact.isChecked()
        cfg.ui_show_status_chips = self.show_chips.isChecked()
        cfg.ui_settings_dialog_last_tab = self.tabs.currentIndex()
        save_config(cfg)
        self.settingsSaved.emit()

    def save_and_close(self) -> None:
        self.apply()
        self.accept()

    def reset_fields(self) -> None:
        self.context.config = AppConfig()
        save_config(self.context.config)
        self.settingsSaved.emit()
        self.accept()

    def export_settings(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Export settings", str(Path(self.context.config.export_dir) / "insightface_gui_settings.json"), "JSON (*.json)")
        if path:
            Path(path).write_text(json.dumps(self.context.config.to_dict(), indent=2), encoding="utf-8")

    def import_settings(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import settings", str(Path.home()), "JSON (*.json)")
        if path:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            self.context.config = AppConfig.from_dict(data)
            save_config(self.context.config)
            self.settingsSaved.emit()
            self.accept()
