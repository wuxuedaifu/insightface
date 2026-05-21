"""Qt stylesheet helpers for the desktop GUI."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class ThemeOption:
    value: str
    label: str
    description: str


THEME_OPTIONS = [
    ThemeOption("system", "System", "Follow the operating system color scheme."),
    ThemeOption("precision_light", "Precision Light", "Clean, bright, and readable for long review sessions."),
    ThemeOption("studio_dark", "Studio Dark", "Default dark studio look for local AI evaluation workflows."),
    ThemeOption("graphite_pro", "Graphite Pro", "Neutral enterprise console with strong contrast and restrained accents."),
    ThemeOption("azure_lab", "Azure Lab", "Cool laboratory workspace with blue and cyan interaction states."),
    ThemeOption("emerald_focus", "Emerald Focus", "Privacy-first workspace with deep green and cyan accents."),
]

_THEME_ALIASES = {
    "light": "precision_light",
    "dark": "studio_dark",
}

_LIGHT_THEME = "precision_light"
_DARK_THEME = "studio_dark"


def normalize_theme(theme: str | None) -> str:
    value = (theme or "system").strip().lower()
    value = _THEME_ALIASES.get(value, value)
    valid = {option.value for option in THEME_OPTIONS}
    return value if value in valid else "system"


def theme_label(theme: str | None) -> str:
    value = normalize_theme(theme)
    for option in THEME_OPTIONS:
        if option.value == value:
            return option.label
    return "System"


def theme_description(theme: str | None) -> str:
    value = normalize_theme(theme)
    for option in THEME_OPTIONS:
        if option.value == value:
            return option.description
    return THEME_OPTIONS[0].description


def effective_theme(theme: str | None) -> str:
    value = normalize_theme(theme)
    if value != "system":
        return value
    app = QApplication.instance()
    if app is not None:
        try:
            if app.styleHints().colorScheme() == Qt.ColorScheme.Dark:
                return _DARK_THEME
        except Exception:
            pass
    return _LIGHT_THEME


def _palettes() -> dict[str, dict[str, str]]:
    return {
        "precision_light": {
            "font": '"Segoe UI", "Inter", "Noto Sans", "Helvetica Neue", Arial, sans-serif',
            "bg": "#f4f7fa",
            "surface": "#ffffff",
            "surface_alt": "#f8fafc",
            "panel": "#eef3f8",
            "panel_hover": "#e8f0fb",
            "text": "#17202c",
            "muted": "#5e6b7a",
            "subtle": "#7b8797",
            "border": "#d7e0ea",
            "border_strong": "#aebdcb",
            "accent": "#2563eb",
            "accent_2": "#0891b2",
            "accent_soft": "#dbeafe",
            "accent_faint": "#eff6ff",
            "field": "#ffffff",
            "button": "#ffffff",
            "button_hover": "#eef4fb",
            "button_pressed": "#dbeafe",
            "selection": "#bfdbfe",
            "success": "#16a34a",
            "success_soft": "#dcfce7",
            "warning": "#d97706",
            "danger": "#dc2626",
            "danger_hover": "#b91c1c",
            "upload_bg": "#f8fafc",
            "upload_hover": "#eef7ff",
            "upload_drag": "#e6fffa",
            "upload_file": "#ffffff",
            "notice_bg": "#edf4ff",
            "notice_border": "#bfdbfe",
            "table_alt": "#f8fafc",
            "header": "#eef3f8",
            "chip": "#f8fafc",
        },
        "studio_dark": {
            "font": '"Segoe UI", "Inter", "Noto Sans", "Helvetica Neue", Arial, sans-serif',
            "bg": "#0b0f14",
            "surface": "#111821",
            "surface_alt": "#16202b",
            "panel": "#1b2633",
            "panel_hover": "#223246",
            "text": "#eaf0f7",
            "muted": "#91a1b5",
            "subtle": "#728196",
            "border": "#283545",
            "border_strong": "#425268",
            "accent": "#4f8cff",
            "accent_2": "#34d3e6",
            "accent_soft": "#173354",
            "accent_faint": "#101f35",
            "field": "#101720",
            "button": "#16202b",
            "button_hover": "#203047",
            "button_pressed": "#173354",
            "selection": "#224a78",
            "success": "#22c55e",
            "success_soft": "#123423",
            "warning": "#f59e0b",
            "danger": "#ef4444",
            "danger_hover": "#b91c1c",
            "upload_bg": "#101720",
            "upload_hover": "#13253a",
            "upload_drag": "#0e332f",
            "upload_file": "#111821",
            "notice_bg": "#10233c",
            "notice_border": "#244e7d",
            "table_alt": "#141e29",
            "header": "#182331",
            "chip": "#151f2b",
        },
        "graphite_pro": {
            "font": '"Segoe UI", "Inter", "Noto Sans", "Helvetica Neue", Arial, sans-serif',
            "bg": "#101214",
            "surface": "#171a1f",
            "surface_alt": "#1f232a",
            "panel": "#252a32",
            "panel_hover": "#2d3440",
            "text": "#f2f4f7",
            "muted": "#a7b0bd",
            "subtle": "#7f8a99",
            "border": "#343b46",
            "border_strong": "#515b6a",
            "accent": "#7aa2f7",
            "accent_2": "#8bd5ca",
            "accent_soft": "#22304a",
            "accent_faint": "#192131",
            "field": "#11151a",
            "button": "#20252d",
            "button_hover": "#2b3340",
            "button_pressed": "#253757",
            "selection": "#31476e",
            "success": "#7dcfff",
            "success_soft": "#14313e",
            "warning": "#e0af68",
            "danger": "#f7768e",
            "danger_hover": "#d94f67",
            "upload_bg": "#15191f",
            "upload_hover": "#1c2632",
            "upload_drag": "#173337",
            "upload_file": "#181c22",
            "notice_bg": "#192338",
            "notice_border": "#31476e",
            "table_alt": "#1b2027",
            "header": "#222832",
            "chip": "#1f252d",
        },
        "azure_lab": {
            "font": '"Segoe UI", "Inter", "Noto Sans", "Helvetica Neue", Arial, sans-serif',
            "bg": "#eef6fb",
            "surface": "#ffffff",
            "surface_alt": "#f2f8fc",
            "panel": "#e5f1f9",
            "panel_hover": "#d9edf8",
            "text": "#102033",
            "muted": "#5b6f82",
            "subtle": "#7c8da0",
            "border": "#c9ddea",
            "border_strong": "#93b7ca",
            "accent": "#1769d1",
            "accent_2": "#00a7c8",
            "accent_soft": "#d7effa",
            "accent_faint": "#eefaff",
            "field": "#ffffff",
            "button": "#ffffff",
            "button_hover": "#e7f4fb",
            "button_pressed": "#d7effa",
            "selection": "#bee3f8",
            "success": "#0f9f6e",
            "success_soft": "#dcfce7",
            "warning": "#c47a00",
            "danger": "#d13855",
            "danger_hover": "#ae2841",
            "upload_bg": "#f6fbfe",
            "upload_hover": "#e8f7ff",
            "upload_drag": "#ddfbff",
            "upload_file": "#ffffff",
            "notice_bg": "#e4f4ff",
            "notice_border": "#9ad7ef",
            "table_alt": "#f4f9fc",
            "header": "#e5f1f9",
            "chip": "#f3f9fc",
        },
        "emerald_focus": {
            "font": '"Segoe UI", "Inter", "Noto Sans", "Helvetica Neue", Arial, sans-serif',
            "bg": "#07120f",
            "surface": "#0d1c18",
            "surface_alt": "#122820",
            "panel": "#18342b",
            "panel_hover": "#1f4036",
            "text": "#e8f7f1",
            "muted": "#9bb9ae",
            "subtle": "#76978b",
            "border": "#294a40",
            "border_strong": "#4a7467",
            "accent": "#2dd4bf",
            "accent_2": "#7dd3fc",
            "accent_soft": "#103d38",
            "accent_faint": "#0b2927",
            "field": "#091713",
            "button": "#13261f",
            "button_hover": "#1a342b",
            "button_pressed": "#12433d",
            "selection": "#14534c",
            "success": "#34d399",
            "success_soft": "#0f3329",
            "warning": "#fbbf24",
            "danger": "#fb7185",
            "danger_hover": "#e11d48",
            "upload_bg": "#0a1714",
            "upload_hover": "#102820",
            "upload_drag": "#0e3a35",
            "upload_file": "#0d1c18",
            "notice_bg": "#0d2f2b",
            "notice_border": "#1f6d62",
            "table_alt": "#10221d",
            "header": "#142b24",
            "chip": "#122820",
        },
    }


def application_stylesheet(theme: str | None) -> str:
    palette = _palettes()[effective_theme(theme)]

    return f"""
    * {{
        font-family: {palette["font"]};
        font-size: 13px;
    }}
    QMainWindow, QDialog {{
        background: {palette["bg"]};
        color: {palette["text"]};
    }}
    QWidget {{
        color: {palette["text"]};
        selection-background-color: {palette["selection"]};
        selection-color: {palette["text"]};
    }}
    QWidget#topAppBar {{
        background: {palette["surface"]};
        border-bottom: 1px solid {palette["border"]};
    }}
    QWidget#modeSidebar {{
        background: {palette["surface_alt"]};
        border-right: 1px solid {palette["border"]};
    }}
    QLabel[role="muted"], QLabel[role="status"], QLabel[role="secondary"] {{
        color: {palette["muted"]};
    }}
    QLabel[role="statusChip"] {{
        padding: 4px 9px;
        border: 1px solid {palette["border"]};
        border-radius: 6px;
        background: {palette["chip"]};
        color: {palette["text"]};
        font-weight: 600;
    }}
    QLabel#noticeLabel {{
        background: {palette["notice_bg"]};
        border: 1px solid {palette["notice_border"]};
        border-radius: 6px;
        color: {palette["text"]};
        padding: 9px 10px;
    }}
    QPushButton {{
        background: {palette["button"]};
        color: {palette["text"]};
        border: 1px solid {palette["border"]};
        border-radius: 6px;
        padding: 7px 12px;
        font-weight: 600;
    }}
    QPushButton:hover {{
        background: {palette["button_hover"]};
        border-color: {palette["border_strong"]};
    }}
    QPushButton:pressed {{
        background: {palette["button_pressed"]};
        border-color: {palette["accent"]};
    }}
    QPushButton:disabled {{
        color: {palette["subtle"]};
        background: {palette["surface_alt"]};
        border-color: {palette["border"]};
    }}
    QPushButton#primaryButton {{
        background: {palette["accent"]};
        border-color: {palette["accent"]};
        color: #ffffff;
    }}
    QPushButton#primaryButton:hover {{
        background: {palette["accent_2"]};
        border-color: {palette["accent_2"]};
    }}
    QPushButton#removeUpload {{
        background: {palette["button_pressed"]};
        color: {palette["text"]};
        border: 1px solid {palette["border_strong"]};
        border-radius: 11px;
        font-weight: 700;
        padding: 0;
    }}
    QPushButton#removeUpload:hover {{
        background: {palette["danger"]};
        border-color: {palette["danger"]};
        color: #ffffff;
    }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {{
        background: {palette["field"]};
        color: {palette["text"]};
        border: 1px solid {palette["border"]};
        border-radius: 6px;
        padding: 6px 8px;
    }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
    QTextEdit:focus, QPlainTextEdit:focus {{
        border-color: {palette["accent"]};
        background: {palette["surface"]};
    }}
    QComboBox::drop-down {{
        border: 0;
        width: 24px;
    }}
    QAbstractItemView {{
        background: {palette["surface"]};
        color: {palette["text"]};
        border: 1px solid {palette["border"]};
        selection-background-color: {palette["selection"]};
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
        alternate-background-color: {palette["table_alt"]};
        color: {palette["text"]};
        gridline-color: {palette["border"]};
        border: 1px solid {palette["border"]};
        border-radius: 6px;
        selection-background-color: {palette["selection"]};
    }}
    QTableWidget::item, QTableView::item {{
        padding: 4px;
    }}
    QTableWidget::item:selected, QTableView::item:selected {{
        background: {palette["selection"]};
        color: {palette["text"]};
    }}
    QHeaderView::section {{
        background: {palette["header"]};
        color: {palette["text"]};
        border: 0;
        border-bottom: 1px solid {palette["border"]};
        padding: 7px;
        font-weight: 650;
    }}
    QFrame, QWidget#dashboardCard {{
        border-color: {palette["border"]};
    }}
    QWidget#dashboardCard {{
        border: 1px solid {palette["border"]};
        border-radius: 6px;
        background: {palette["surface"]};
    }}
    QWidget#dashboardCard:hover {{
        border-color: {palette["accent"]};
        background: {palette["surface_alt"]};
    }}
    QFrame#uploadPreview, QFrame#dropInput, QFrame#galleryUpload,
    QFrame#imageOrFolderImport, QListWidget#albumDirectoryList {{
        border: 1px dashed {palette["border_strong"]};
        border-radius: 6px;
        background: {palette["upload_bg"]};
        padding: 8px;
    }}
    QFrame#uploadPreview[hoverActive="true"], QFrame#dropInput[hoverActive="true"],
    QFrame#galleryUpload[hoverActive="true"], QFrame#imageOrFolderImport[hoverActive="true"],
    QListWidget#albumDirectoryList[hoverActive="true"] {{
        border-color: {palette["accent"]};
        background: {palette["upload_hover"]};
    }}
    QFrame#uploadPreview[dragActive="true"], QFrame#dropInput[dragActive="true"],
    QFrame#galleryUpload[dragActive="true"], QFrame#imageOrFolderImport[dragActive="true"],
    QListWidget#albumDirectoryList[dragActive="true"] {{
        border-color: {palette["success"]};
        background: {palette["upload_drag"]};
    }}
    QFrame#uploadPreview[hasFile="true"], QFrame#dropInput[hasFiles="true"],
    QFrame#galleryUpload[hasFiles="true"] {{
        border: 1px solid {palette["border"]};
        background: {palette["upload_file"]};
    }}
    QFrame#uploadPreview QLabel, QFrame#dropInput QLabel, QFrame#galleryUpload QLabel,
    QFrame#imageOrFolderImport QLabel {{
        background: transparent;
    }}
    QFrame#uploadPreview QGraphicsView, QFrame#galleryUpload QListWidget {{
        background: transparent;
        border: 0;
    }}
    QLabel#uploadPrompt, QLabel#dropPrompt {{
        color: {palette["muted"]};
        font-size: 15px;
        font-weight: 650;
        padding: 18px;
    }}
    QLabel#pathLabel {{
        color: {palette["muted"]};
        padding: 0 8px 8px 8px;
    }}
    QMenuBar, QMenu {{
        background: {palette["surface"]};
        color: {palette["text"]};
        border-color: {palette["border"]};
    }}
    QMenuBar::item:selected, QMenu::item:selected {{
        background: {palette["accent_soft"]};
    }}
    QTabWidget::pane {{
        border: 1px solid {palette["border"]};
        border-radius: 6px;
        background: {palette["surface"]};
    }}
    QTabBar::tab {{
        background: {palette["surface_alt"]};
        border: 1px solid {palette["border"]};
        border-bottom: 0;
        padding: 7px 12px;
        margin-right: 2px;
        border-top-left-radius: 6px;
        border-top-right-radius: 6px;
    }}
    QTabBar::tab:selected {{
        background: {palette["surface"]};
        color: {palette["accent"]};
    }}
    QProgressBar {{
        background: {palette["surface_alt"]};
        border: 1px solid {palette["border"]};
        border-radius: 6px;
        color: {palette["text"]};
        text-align: center;
    }}
    QProgressBar::chunk {{
        background: {palette["accent"]};
        border-radius: 5px;
    }}
    QSlider::groove:horizontal {{
        height: 6px;
        background: {palette["surface_alt"]};
        border: 1px solid {palette["border"]};
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {palette["accent"]};
        border: 1px solid {palette["accent"]};
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }}
    QSplitter::handle {{
        background: {palette["bg"]};
    }}
    QScrollArea {{
        background: transparent;
        border: 0;
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
        border-radius: 4px;
        padding: 4px;
    }}
    """
