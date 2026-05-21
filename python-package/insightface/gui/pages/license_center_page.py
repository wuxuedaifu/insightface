"""License Center page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QLabel, QPushButton, QTableWidget, QTableWidgetItem, QTextEdit

from ..core import constants
from ..core.exporters import export_markdown
from ..core.licensing import allowed_usage_summary, find_license_text, license_summary_text
from ..core.utils import timestamp_for_filename
from ..widgets.table_utils import configure_table_columns, refresh_table_columns
from .base import BasePage


class LicenseCenterPage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(context, "License Center", "Review local package, model license, allowed usage, and commercial paths.", parent)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.content.addWidget(self.summary)
        self.content.addWidget(self.notice(constants.LICENSE_NOTICE))
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Usage", "Status"])
        configure_table_columns(self.table, [260, 420])
        self.content.addWidget(self.table)
        self.license_text = QTextEdit()
        self.license_text.setReadOnly(True)
        self.license_text.setMinimumHeight(140)
        self.content.addWidget(QLabel("Code License"))
        self.content.addWidget(self.license_text)
        self.content.addWidget(self.notice(constants.RESPONSIBLE_USE_NOTICE))
        self.content.addWidget(
            self.row(
                self.button("Request Commercial Model License", self.contact),
                self.button("Request Private Model Evaluation", self.contact),
                self.button("Request SDK / API", self.contact),
                self.button("Request Face Swap Commercial License", self.contact),
                self.button("Export License Summary", self.export_summary),
            )
        )
        self.refresh()

    def refresh(self) -> None:
        cfg = self.context.config
        self.summary.setText(
            "\n".join(
                [
                    f"insightface version: 1.0",
                    f"GUI version: 1.0",
                    f"current model name: {cfg.model_name}",
                    f"provider: {cfg.provider}",
                    f"workspace path: {cfg.workspace_path}",
                    f"license status: {cfg.license_status}",
                ]
            )
        )
        summary = allowed_usage_summary()
        self.table.setRowCount(len(summary))
        for row, (usage, status) in enumerate(summary.items()):
            self.table.setItem(row, 0, QTableWidgetItem(usage))
            self.table.setItem(row, 1, QTableWidgetItem(status))
        refresh_table_columns(self.table)
        self.license_text.setPlainText(find_license_text(Path(__file__).parents[4]))

    def contact(self) -> None:
        QDesktopServices.openUrl(
            QUrl(
                "mailto:contact@insightface.ai?subject=InsightFace%20Commercial%20Licensing%20Request"
            )
        )
        self.set_status("Opened mail client for InsightFace commercial licensing.")

    def export_summary(self) -> None:
        cfg = self.context.config
        path = Path(cfg.export_dir) / f"license_summary_{timestamp_for_filename()}.md"
        export_markdown(path, license_summary_text(cfg.license_status, cfg.model_name, cfg.provider, cfg.workspace_path))
        self.set_status(f"License summary exported to {path}")
