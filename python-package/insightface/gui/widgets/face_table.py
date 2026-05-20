"""Table helpers for face/search results."""

from __future__ import annotations

from typing import Iterable, Mapping

from PySide6.QtWidgets import QTableWidget, QTableWidgetItem


class FaceTable(QTableWidget):
    def set_rows(self, rows: Iterable[Mapping[str, object]], columns: list[str]) -> None:
        rows = list(rows)
        self.setColumnCount(len(columns))
        self.setHorizontalHeaderLabels(columns)
        self.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            for col_index, column in enumerate(columns):
                self.setItem(row_index, col_index, QTableWidgetItem(str(row.get(column, ""))))
        self.resizeColumnsToContents()
