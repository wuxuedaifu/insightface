"""Main window, mode navigation, and task orchestration."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QAction, QDesktopServices, QFontMetrics, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from .core.config import save_config
from .core.constants import LOCAL_PROCESSING_NOTICE, RESPONSIBLE_USE_NOTICE, WINDOW_TITLE
from .core.face_engine import is_cuda_provider_available
from .core.navigation import (
    AppMode,
    MODE_LABELS,
    NAVIGATION_MODES,
    last_page_attr,
    mode_from_value,
)
from .core.theme import application_stylesheet
from .dialogs.license_dialog import LicenseDialog
from .dialogs.model_manager_dialog import ModelManagerDialog
from .dialogs.settings_dialog import SettingsDialog
from .page_registry import PageRegistry
from .widgets.progress_dialog import StudioProgressDialog


class WorkerSignals(QObject):
    result = Signal(object)
    error = Signal(str)
    progress = Signal(int, int, str)
    finished = Signal()


class Worker(QRunnable):
    def __init__(self, fn: Callable):
        super().__init__()
        self.fn = fn
        self.signals = WorkerSignals()
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True

    @Slot()
    def run(self) -> None:
        try:
            signature = inspect.signature(self.fn)
            if "progress" in signature.parameters or "is_cancelled" in signature.parameters:
                result = self.fn(
                    progress=lambda current, total, message="": self.signals.progress.emit(int(current), int(total), str(message)),
                    is_cancelled=lambda: self.cancelled,
                )
            else:
                result = self.fn()
            self.signals.result.emit(result)
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            self.signals.finished.emit()


class FirstLaunchWizard(QDialog):
    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.setWindowTitle("First Launch Wizard")
        self.config = config
        layout = QVBoxLayout(self)
        intro = QLabel(
            "\n".join(
                [
                    "Welcome to InsightFace Evaluation Studio.",
                    LOCAL_PROCESSING_NOTICE,
                    RESPONSIBLE_USE_NOTICE,
                ]
            )
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QFormLayout()
        self.workspace = QLineEdit(config.workspace_path)
        browse = QLabel("<a href='#'>Browse</a>")
        browse.linkActivated.connect(self.browse_workspace)
        workspace_row = QWidget()
        workspace_layout = QVBoxLayout(workspace_row)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.addWidget(self.workspace)
        workspace_layout.addWidget(browse)
        self.mode = QComboBox()
        self.mode.addItems(["Personal / Research", "Enterprise Evaluation"])
        self.model = QComboBox()
        self.model.addItems(["buffalo_l", "buffalo_s", "antelopev2", "custom model directory"])
        self.provider = QComboBox()
        self.provider.addItems(["Auto", "CPU", "CUDA"])
        self._update_provider_availability()
        self.license_notice = QLabel(
            "This application runs locally. Code and model licenses may differ. "
            "Research models may be limited to non-commercial use. Commercial deployment "
            "requires appropriate model license. This tool does not provide legal advice."
        )
        self.license_notice.setWordWrap(True)
        form.addRow("Workspace", workspace_row)
        form.addRow("Mode", self.mode)
        form.addRow("Model package", self.model)
        form.addRow("Provider", self.provider)
        form.addRow("License notice", self.license_notice)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def browse_workspace(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select workspace", str(Path.home()))
        if folder:
            self.workspace.setText(folder)

    def accept(self) -> None:
        self.config.workspace_path = self.workspace.text().strip()
        self.config.mode = self.mode.currentText()
        self.config.ui_last_mode = (
            AppMode.ENTERPRISE_EVALUATION.value
            if self.mode.currentText() == "Enterprise Evaluation"
            else AppMode.FACE_VERIFICATION.value
        )
        selected_model = self.model.currentText()
        self.config.model_name = selected_model if selected_model != "custom model directory" else self.config.model_name
        provider = self.provider.currentText()
        self.config.provider = "Auto" if provider == "CUDA" and not is_cuda_provider_available() else provider
        self.config.apply_workspace_defaults()
        save_config(self.config)
        super().accept()

    def _update_provider_availability(self) -> None:
        cuda_available = is_cuda_provider_available()
        cuda_index = self.provider.findText("CUDA")
        if cuda_index >= 0:
            item = self.provider.model().item(cuda_index)
            if item is not None:
                item.setEnabled(cuda_available)
                item.setToolTip(
                    "CUDAExecutionProvider is available."
                    if cuda_available
                    else "CUDAExecutionProvider is not available. Install a matching onnxruntime-gpu, CUDA runtime, and GPU driver first."
                )
        self.provider.setToolTip("Auto uses CUDA when CUDAExecutionProvider is available, otherwise CPU.")


class MainWindow(QMainWindow):
    LEGACY_PAGE_MAP = {
        "Dashboard": "face_dashboard",
        "Face Recognition": "verification",
        "Verification": "verification",
        "Verification Dashboard": "face_dashboard",
        "Album Dashboard": "album_dashboard",
        "Swap Dashboard": "swap_dashboard",
        "Evaluation Dashboard": "enterprise_dashboard",
        "1:1 Compare": "compare",
        "1:N Face Search": "face_search",
        "Multi-face Photo Recognition": "multiface_photo",
        "Batch Folder Processing": "batch_processing",
        "Camera Recognition": "camera_recognition",
        "Video Person Search": "video_search",
        "People Library": "people_library",
        "Album People Clustering": "album_people_clustering",
        "Enterprise Evaluation": "enterprise_evaluation",
        "Reports": "reports",
        "Image Face Swap": "image_face_swap",
    }

    GLOBAL_PAGES = {"Settings", "Model Settings", "Model Downloads", "License Center"}

    def __init__(self, context, parent=None):
        super().__init__(parent)
        self.context = context
        self.apply_theme()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(int(context.config.ui_window_width), int(context.config.ui_window_height))
        self.thread_pool = QThreadPool.globalInstance()
        self.active_workers: set[Worker] = set()
        self.page_registry = PageRegistry(context)
        self.pages = self.page_registry.pages
        self.current_mode = mode_from_value(context.config.ui_last_mode)
        self.current_page_key = ""

        self.stack = QStackedWidget()
        self.sidebar = QWidget()
        self.sidebar.setObjectName("modeSidebar")
        self.sidebar.setFixedWidth(int(context.config.ui_sidebar_width or 240))
        self.sidebar_title = QLabel()
        self.sidebar_title.setStyleSheet("font-size:18px; font-weight:700;")
        self.sidebar_description = QLabel()
        self.sidebar_description.setWordWrap(True)
        self.sidebar_description.setProperty("role", "muted")
        self.sidebar_list = QListWidget()
        self.sidebar_list.setAlternatingRowColors(False)
        self.sidebar_list.itemClicked.connect(self._sidebar_clicked)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_top_bar())
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(14, 14, 10, 14)
        sidebar_layout.addWidget(self.sidebar_title)
        sidebar_layout.addWidget(self.sidebar_description)
        sidebar_layout.addWidget(self.sidebar_list, 1)
        body_layout.addWidget(self.sidebar)
        body_layout.addWidget(self.stack, 1)
        root.addWidget(body, 1)
        self.setCentralWidget(central)

        self._build_menu()
        self._build_statusbar()
        self.change_mode(self.current_mode, restore_last=True, save=False)
        self.refresh_statusbar()
        if not self.context.config_exists:
            QTimer.singleShot(100, self.show_first_launch)
        if self.context.config.auto_load_model and not self.context.runtime_safe_mode:
            QTimer.singleShot(250, self.auto_load_model)

    def _build_top_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("topAppBar")
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(14, 8, 14, 8)
        title = QLabel("InsightFace Evaluation Studio")
        title.setStyleSheet("font-size:18px; font-weight:700; border:0;")
        version = QLabel("v1.0")
        version.setProperty("role", "muted")
        self.mode_combo = QComboBox()
        for mode in NAVIGATION_MODES.values():
            self.mode_combo.addItem(mode.title, mode.id.value)
        self.mode_combo.currentIndexChanged.connect(self._mode_combo_changed)
        self.model_chip = QLabel()
        self.provider_chip = QLabel()
        self.license_chip = QLabel()
        for chip in (self.model_chip, self.provider_chip, self.license_chip):
            chip.setProperty("role", "statusChip")
        layout.addWidget(title)
        layout.addWidget(version)
        layout.addSpacing(18)
        layout.addWidget(QLabel("Mode"))
        layout.addWidget(self.mode_combo)
        layout.addStretch(1)
        layout.addWidget(self.model_chip)
        layout.addWidget(self.provider_chip)
        layout.addWidget(self.license_chip)
        layout.addWidget(self._top_button("Models", self.open_model_manager))
        layout.addWidget(self._top_button("License", self.open_license_dialog))
        layout.addWidget(self._top_button("Settings", self.open_settings_dialog))
        return bar

    def _top_button(self, text: str, callback: Callable) -> QPushButton:
        button = QPushButton(text)
        button.clicked.connect(callback)
        return button

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        open_workspace = file_menu.addAction("Open Workspace Folder")
        open_workspace.triggered.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.context.config.workspace_path)))
        quit_action = file_menu.addAction("Exit")
        quit_action.setShortcut(QKeySequence("Ctrl+Q"))
        quit_action.triggered.connect(self.close)

        mode_menu = self.menuBar().addMenu("Mode")
        for mode in NAVIGATION_MODES:
            action = mode_menu.addAction(NAVIGATION_MODES[mode].title)
            action.triggered.connect(lambda checked=False, target=mode: self.change_mode(target))

        tools_menu = self.menuBar().addMenu("Tools")
        models_action = QAction("Models", self)
        models_action.setShortcut(QKeySequence("Ctrl+M"))
        models_action.triggered.connect(self.open_model_manager)
        settings_action = QAction("Settings", self)
        settings_action.setShortcut(QKeySequence("Ctrl+,"))
        settings_action.triggered.connect(self.open_settings_dialog)
        license_action = QAction("License Center", self)
        license_action.setShortcut(QKeySequence("Ctrl+L"))
        license_action.triggered.connect(self.open_license_dialog)
        tools_menu.addAction(models_action)
        tools_menu.addAction(settings_action)
        tools_menu.addAction(license_action)

        help_menu = self.menuBar().addMenu("Help")
        about_action = help_menu.addAction("About")
        about_action.triggered.connect(lambda: QMessageBox.information(self, "About", "InsightFace Evaluation Studio v1.0"))
        responsible_action = help_menu.addAction("Responsible Use Notice")
        responsible_action.triggered.connect(lambda: QMessageBox.information(self, "Responsible Use Notice", RESPONSIBLE_USE_NOTICE))

    def _build_statusbar(self) -> None:
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self.status_labels = {
            "model": QLabel(),
            "provider": QLabel(),
            "database": QLabel(),
            "license": QLabel(),
            "running": QLabel("Ready"),
        }
        for label in self.status_labels.values():
            self.status.addPermanentWidget(label)

    def _mode_combo_changed(self) -> None:
        mode = mode_from_value(self.mode_combo.currentData())
        if mode != self.current_mode:
            self.change_mode(mode)

    def change_mode(self, mode: AppMode | str, restore_last: bool = True, save: bool = True) -> None:
        self.current_mode = mode_from_value(mode)
        index = self.mode_combo.findData(self.current_mode.value)
        if index >= 0 and self.mode_combo.currentIndex() != index:
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(index)
            self.mode_combo.blockSignals(False)
        self._rebuild_sidebar()
        page_key = getattr(self.context.config, last_page_attr(self.current_mode), "") if restore_last else ""
        valid_keys = {item.page_key for item in NAVIGATION_MODES[self.current_mode].items}
        if page_key not in valid_keys:
            page_key = NAVIGATION_MODES[self.current_mode].items[0].page_key
        if save:
            self.context.config.ui_last_mode = self.current_mode.value
            save_config(self.context.config)
        self.open_page(page_key, from_mode_change=True)

    def _rebuild_sidebar(self) -> None:
        mode = NAVIGATION_MODES[self.current_mode]
        self.sidebar_title.setText(mode.title)
        self.sidebar_description.setText(mode.description)
        self.sidebar.setVisible(self.current_mode not in {AppMode.FACE_VERIFICATION, AppMode.ALBUM_MANAGEMENT, AppMode.FACE_SWAP})
        self.sidebar_list.clear()
        for nav_item in mode.items:
            title = nav_item.title + ("  Coming soon" if nav_item.coming_soon else "")
            item = QListWidgetItem(title)
            item.setData(Qt.UserRole, nav_item.page_key)
            item.setToolTip(nav_item.description)
            if not nav_item.enabled:
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
                item.setForeground(Qt.gray)
            self.sidebar_list.addItem(item)

    def _sidebar_clicked(self, item: QListWidgetItem) -> None:
        if item.flags() & Qt.ItemIsEnabled:
            self.open_page(str(item.data(Qt.UserRole)))

    def open_page(self, page_key: str, from_mode_change: bool = False) -> None:
        if page_key in self.GLOBAL_PAGES:
            if page_key == "Settings":
                self.open_settings_dialog()
            elif page_key in {"Model Settings", "Model Downloads"}:
                self.open_model_manager(initial=page_key)
            elif page_key == "License Center":
                self.open_license_dialog()
            return
        page_key = self.LEGACY_PAGE_MAP.get(page_key, page_key)
        page = self.page_registry.get(page_key)
        if self.stack.indexOf(page) < 0:
            self.stack.addWidget(page)
        if hasattr(page, "refresh"):
            page.refresh()
        self.stack.setCurrentWidget(page)
        self.current_page_key = page_key
        for row in range(self.sidebar_list.count()):
            item = self.sidebar_list.item(row)
            if item.data(Qt.UserRole) == page_key:
                self.sidebar_list.setCurrentItem(item)
                break
        if not from_mode_change:
            setattr(self.context.config, last_page_attr(self.current_mode), page_key)
            save_config(self.context.config)
        self.set_status(self._page_title(page_key))

    def _page_title(self, page_key: str) -> str:
        for mode in NAVIGATION_MODES.values():
            for item in mode.items:
                if item.page_key == page_key:
                    return item.title
        return page_key

    def show_first_launch(self) -> None:
        wizard = FirstLaunchWizard(self.context.config, self)
        if wizard.exec() == QDialog.Accepted:
            self.context.storage = self.context.storage.__class__(self.context.config.database_path)
            self.change_mode(mode_from_value(self.context.config.ui_last_mode), restore_last=False)
            self.refresh_statusbar()

    def auto_load_model(self) -> None:
        def task():
            self.context.engine.load()
            return self.context.engine

        def done(engine):
            self.refresh_statusbar()
            if engine.is_loaded():
                self.set_status("Model loaded.")
            elif engine.last_error:
                self.set_status(engine.last_error)

        self.run_task("Loading model", task, done, show_dialog=False)

    def open_settings_dialog(self) -> None:
        dialog = SettingsDialog(self.context, self)
        dialog.settingsSaved.connect(self._settings_saved)
        dialog.exec()
        self.refresh_statusbar()

    def _settings_saved(self) -> None:
        self.apply_theme()
        self.refresh_statusbar()

    def apply_theme(self) -> None:
        qt_app = QApplication.instance()
        if qt_app is not None:
            qt_app.setStyleSheet(application_stylesheet(self.context.config.ui_theme))

    def open_model_manager(self, initial: str | None = None) -> None:
        dialog = ModelManagerDialog(self.context, self)
        dialog.modelChanged.connect(self.refresh_statusbar)
        if initial:
            dialog.open_page(initial)
        dialog.exec()
        self.refresh_statusbar()

    def open_license_dialog(self) -> None:
        dialog = LicenseDialog(self.context, self)
        dialog.exec()
        self.refresh_statusbar()

    def set_status(self, message: str) -> None:
        self.status_labels["running"].setText(self._elide(message, 180))

    def refresh_statusbar(self) -> None:
        cfg = self.context.config
        self.model_chip.setVisible(cfg.ui_show_status_chips)
        self.provider_chip.setVisible(cfg.ui_show_status_chips)
        self.license_chip.setVisible(cfg.ui_show_status_chips)
        self.model_chip.setText(self._elide(f"Model: {cfg.model_name}", 150))
        self.provider_chip.setText(self._elide(f"Provider: {cfg.provider}", 130))
        self.license_chip.setText(self._elide(cfg.license_status, 180))
        self.model_chip.setToolTip(cfg.model_name)
        self.provider_chip.setToolTip(cfg.provider)
        self.license_chip.setToolTip(cfg.license_status)
        self.status_labels["model"].setText(self._elide(f"Model: {cfg.model_name}", 160))
        self.status_labels["provider"].setText(f"Provider: {cfg.provider}")
        self.status_labels["database"].setText(self._elide(f"DB: {cfg.database_path}", 280))
        self.status_labels["database"].setToolTip(cfg.database_path)
        self.status_labels["license"].setText(self._elide(f"License: {cfg.license_status}", 220))

    def run_task(self, title: str, fn: Callable, on_result: Callable | None = None, show_dialog: bool = True) -> None:
        worker = Worker(fn)
        worker.setAutoDelete(False)
        self.active_workers.add(worker)
        dialog = StudioProgressDialog(title, self) if show_dialog else None
        if dialog:
            dialog.canceled.connect(worker.cancel)
            worker.signals.progress.connect(dialog.update_progress)
            dialog.show()
        worker.signals.result.connect(lambda result: on_result(result) if on_result else None)
        worker.signals.error.connect(lambda message: QMessageBox.warning(self, title, message))
        worker.signals.error.connect(self.set_status)
        worker.signals.finished.connect(lambda: dialog.close() if dialog else None)
        worker.signals.finished.connect(self.refresh_statusbar)
        worker.signals.finished.connect(lambda: self.active_workers.discard(worker))
        worker.signals.finished.connect(worker.signals.deleteLater)
        self.set_status(title)
        self.thread_pool.start(worker)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.context.config.ui_window_width = max(800, self.width())
        self.context.config.ui_window_height = max(600, self.height())
        self.context.config.ui_sidebar_width = max(200, self.sidebar.width())
        save_config(self.context.config)
        super().closeEvent(event)

    def _elide(self, text: str, width: int) -> str:
        metrics = QFontMetrics(self.font())
        return metrics.elidedText(str(text), Qt.ElideMiddle, width)
