"""Compact person/cluster card."""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout


class PersonCard(QFrame):
    def __init__(self, title: str, subtitle: str = "", parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.title = QLabel(title)
        self.title.setStyleSheet("font-weight: 600;")
        self.subtitle = QLabel(subtitle)
        self.action = QPushButton("Open")
        layout = QVBoxLayout(self)
        layout.addWidget(self.title)
        layout.addWidget(self.subtitle)
        layout.addWidget(self.action)
