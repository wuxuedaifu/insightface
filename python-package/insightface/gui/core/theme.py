"""Qt stylesheet helpers for the desktop GUI."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication


def effective_theme(theme: str | None) -> str:
    value = (theme or "system").strip().lower()
    if value in {"light", "dark"}:
        return value
    app = QApplication.instance()
    if app is not None:
        try:
            if app.styleHints().colorScheme() == Qt.ColorScheme.Dark:
                return "dark"
        except Exception:
            pass
    return "light"


def application_stylesheet(theme: str | None) -> str:
    palette = {
        "light": {
            "bg": "#f6f7f9",
            "surface": "#ffffff",
            "surface_alt": "#f9fafb",
            "text": "#111827",
            "muted": "#4b5563",
            "border": "#d1d5db",
            "accent": "#2563eb",
            "accent_soft": "#eff6ff",
            "selection": "#dbeafe",
            "button": "#ffffff",
            "button_hover": "#f3f4f6",
            "field": "#ffffff",
            "disabled": "#9ca3af",
        },
        "dark": {
            "bg": "#111827",
            "surface": "#1f2937",
            "surface_alt": "#243244",
            "text": "#f9fafb",
            "muted": "#cbd5e1",
            "border": "#374151",
            "accent": "#60a5fa",
            "accent_soft": "#1e3a5f",
            "selection": "#1d4ed8",
            "button": "#273449",
            "button_hover": "#334155",
            "field": "#172033",
            "disabled": "#64748b",
        },
    }[effective_theme(theme)]

    return f"""
    QMainWindow, QDialog {{
        background: {palette["bg"]};
        color: {palette["text"]};
    }}
    QWidget {{
        color: {palette["text"]};
        selection-background-color: {palette["selection"]};
    }}
    QWidget#topAppBar {{
        background: {palette["surface"]};
        border-bottom: 1px solid {palette["border"]};
    }}
    QWidget#modeSidebar {{
        background: {palette["surface_alt"]};
        border-right: 1px solid {palette["border"]};
    }}
    QLabel[role="muted"] {{
        color: {palette["muted"]};
    }}
    QLabel[role="statusChip"] {{
        padding: 4px 8px;
        border: 1px solid {palette["border"]};
        border-radius: 6px;
        background: {palette["surface_alt"]};
        color: {palette["text"]};
    }}
    QPushButton {{
        background: {palette["button"]};
        color: {palette["text"]};
        border: 1px solid {palette["border"]};
        border-radius: 6px;
        padding: 6px 10px;
    }}
    QPushButton:hover {{
        background: {palette["button_hover"]};
    }}
    QPushButton:pressed {{
        background: {palette["accent_soft"]};
        border-color: {palette["accent"]};
    }}
    QPushButton:disabled {{
        color: {palette["disabled"]};
    }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {{
        background: {palette["field"]};
        color: {palette["text"]};
        border: 1px solid {palette["border"]};
        border-radius: 6px;
        padding: 5px 7px;
    }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {palette["accent"]};
    }}
    QListWidget {{
        background: transparent;
        border: 0;
        outline: 0;
    }}
    QListWidget::item {{
        border-radius: 6px;
        padding: 8px 10px;
        margin: 2px 0;
    }}
    QListWidget::item:selected {{
        background: {palette["accent_soft"]};
        color: {palette["text"]};
    }}
    QListWidget::item:hover {{
        background: {palette["button_hover"]};
    }}
    QTableWidget, QTableView {{
        background: {palette["surface"]};
        alternate-background-color: {palette["surface_alt"]};
        color: {palette["text"]};
        gridline-color: {palette["border"]};
        border: 1px solid {palette["border"]};
        border-radius: 6px;
    }}
    QHeaderView::section {{
        background: {palette["surface_alt"]};
        color: {palette["text"]};
        border: 0;
        border-bottom: 1px solid {palette["border"]};
        padding: 6px;
    }}
    QMenuBar, QMenu {{
        background: {palette["surface"]};
        color: {palette["text"]};
    }}
    QMenuBar::item:selected, QMenu::item:selected {{
        background: {palette["accent_soft"]};
    }}
    QStatusBar {{
        background: {palette["surface"]};
        color: {palette["muted"]};
        border-top: 1px solid {palette["border"]};
    }}
    QToolTip {{
        background: {palette["surface"]};
        color: {palette["text"]};
        border: 1px solid {palette["border"]};
    }}
    """
