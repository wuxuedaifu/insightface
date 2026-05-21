"""Application settings dialog."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QComboBox, QDialog, QDialogButtonBox, QFormLayout, QLabel, QVBoxLayout

from ..core.config import save_config
from ..core.theme import THEME_OPTIONS, normalize_theme, theme_description


class SettingsDialog(QDialog):
    settingsSaved = Signal()

    def __init__(self, context, parent=None):
        super().__init__(parent)
        self.context = context
        self.setWindowTitle("Settings")
        self.resize(560, 220)

        layout = QVBoxLayout(self)
        title = QLabel("Application Settings")
        title.setStyleSheet("font-size:18px; font-weight:700;")
        note = QLabel("Only appearance settings are configurable here. Workspace paths are fixed after first launch.")
        note.setWordWrap(True)
        note.setProperty("role", "muted")
        layout.addWidget(title)
        layout.addWidget(note)

        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.theme = QComboBox()
        for option in THEME_OPTIONS:
            self.theme.addItem(option.label, option.value)
        current = normalize_theme(self.context.config.ui_theme)
        current_index = self.theme.findData(current)
        self.theme.setCurrentIndex(max(0, current_index))
        self.theme.currentIndexChanged.connect(self._update_theme_description)
        form.addRow("UI theme", self.theme)
        layout.addLayout(form, 1)
        self.theme_help = QLabel("")
        self.theme_help.setWordWrap(True)
        self.theme_help.setProperty("role", "muted")
        layout.addWidget(self.theme_help)
        self._update_theme_description()

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Save).clicked.connect(self.save_and_close)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(self.apply)
        buttons.button(QDialogButtonBox.Cancel).clicked.connect(self.reject)
        layout.addWidget(buttons)

    def apply(self) -> None:
        self.context.config.ui_theme = self.theme.currentData()
        save_config(self.context.config)
        self.settingsSaved.emit()

    def save_and_close(self) -> None:
        self.apply()
        self.accept()

    def _update_theme_description(self) -> None:
        self.theme_help.setText(theme_description(self.theme.currentData()))
