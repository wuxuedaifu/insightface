"""Model runtime and download manager dialog."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QDialog, QLabel, QTabWidget, QVBoxLayout

from ..pages.model_download_page import ModelDownloadPage
from ..pages.model_settings_page import ModelSettingsPage


class ModelManagerDialog(QDialog):
    modelChanged = Signal()

    def __init__(self, context, parent=None):
        super().__init__(parent)
        self.context = context
        self.setWindowTitle("Models")
        self.resize(980, 680)
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.runtime_page = ModelSettingsPage(context)
        self.downloads_page = ModelDownloadPage(context)
        self.tabs.addTab(self.runtime_page, "Runtime")
        self.tabs.addTab(self.downloads_page, "Downloads")
        layout.addWidget(self.tabs, 1)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    def open_page(self, label: str) -> None:
        if label == "Model Downloads":
            self.tabs.setCurrentWidget(self.downloads_page)
        elif label == "Model Settings":
            self.tabs.setCurrentWidget(self.runtime_page)

    def set_status(self, message: str) -> None:
        self.status_label.setText(message)
        self.modelChanged.emit()
        if self.parent() is not None and hasattr(self.parent(), "set_status"):
            self.parent().set_status(message)

    def refresh_statusbar(self) -> None:
        self.modelChanged.emit()
        if self.parent() is not None and hasattr(self.parent(), "refresh_statusbar"):
            self.parent().refresh_statusbar()
