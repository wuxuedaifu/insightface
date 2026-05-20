"""Model settings page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QComboBox, QFormLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QTextEdit, QDoubleSpinBox

from ..core.config import save_config
from ..core.face_engine import FaceEngine, is_cuda_provider_available, providers_from_choice
from ..core.model_downloads import is_model_package_installed, list_installed_swap_models
from .base import BasePage


class ModelSettingsPage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(context, "Model Settings", "Configure model packs, execution provider, thresholds, and runtime checks.", parent)
        form = QFormLayout()
        self.model_combo = QComboBox()
        self.model_packages = ["buffalo_l", "buffalo_m", "buffalo_s", "buffalo_sc", "antelopev2"]
        self.model_combo.addItems([*self.model_packages, "custom model directory"])
        self._update_model_availability()
        self.model_combo.setCurrentText(context.config.model_name if context.config.model_name in self.model_packages else "custom model directory")
        self.custom_dir = QLineEdit(context.config.custom_model_dir)
        self.provider_combo = QComboBox()
        self.provider_combo.addItems(["Auto", "CPU", "CUDA"])
        self.provider_combo.setCurrentText(context.config.provider)
        self._update_provider_availability()
        self.det_combo = QComboBox()
        self.det_combo.addItems(["Auto", "128x128", "320x320", "640x640", "1024x1024"])
        self.det_combo.setCurrentText(context.config.det_size_label)
        self.swap_model_combo = QComboBox()
        self._update_swap_model_choices()
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 1.0)
        self.threshold.setSingleStep(0.01)
        self.threshold.setValue(context.config.recognition_threshold)
        self.workers = QSpinBox()
        self.workers.setRange(1, 16)
        self.workers.setValue(context.config.batch_worker_count)
        self.frame_interval = QSpinBox()
        self.frame_interval.setRange(1, 300)
        self.frame_interval.setValue(context.config.video_frame_interval)
        form.addRow("Model package", self.model_combo)
        form.addRow("Custom model directory", self.custom_dir)
        form.addRow("Provider", self.provider_combo)
        form.addRow("Detection size", self.det_combo)
        form.addRow("Face swap model", self.swap_model_combo)
        form.addRow("Recognition threshold", self.threshold)
        form.addRow("Batch workers", self.workers)
        form.addRow("Video frame interval", self.frame_interval)
        self.content.addLayout(form)
        self.runtime = QTextEdit()
        self.runtime.setReadOnly(True)
        self.content.addWidget(QLabel("Runtime information"))
        self.content.addWidget(self.runtime)
        self.content.addWidget(
            self.row(
                self.button("Save Settings", self.save),
                self.button("Open Model Downloads", lambda: self.window().open_page("Model Downloads")),
                self.button("Test Model Load", self.test_load),
                self.button("Warmup", self.warmup),
            )
        )
        self.refresh()

    def _apply_to_config(self) -> None:
        cfg = self.context.config
        chosen = self.model_combo.currentText()
        cfg.model_name = chosen if chosen != "custom model directory" else self.custom_dir.text().strip()
        cfg.custom_model_dir = self.custom_dir.text().strip()
        provider = self.provider_combo.currentText()
        cfg.provider = "Auto" if provider == "CUDA" and not is_cuda_provider_available() else provider
        if self.det_combo.currentText() == "Auto":
            cfg.det_size = [0, 0]
        else:
            size = self.det_combo.currentText().split("x")
            cfg.det_size = [int(size[0]), int(size[1])]
        cfg.swap_model_path = str(self.swap_model_combo.currentData() or "")
        cfg.recognition_threshold = float(self.threshold.value())
        cfg.batch_worker_count = int(self.workers.value())
        cfg.video_frame_interval = int(self.frame_interval.value())

    def save(self) -> None:
        self._apply_to_config()
        save_config(self.context.config)
        self.set_status("Model settings saved.")
        self.refresh()

    def test_load(self) -> None:
        self._apply_to_config()

        def task():
            engine = FaceEngine(
                model_name=self.context.config.model_name,
                providers=providers_from_choice(self.context.config.provider),
                det_size=self.context.config.det_size_tuple,
                root=self.context.config.model_root,
                custom_model_dir=self.context.config.custom_model_dir,
            )
            engine.load()
            return engine

        def done(engine):
            self.context.engine = engine
            self.window().context.engine = engine
            self.refresh()
            if engine.is_loaded():
                self.set_status("Model loaded successfully.")
            else:
                self.show_error(engine.last_error or "Model load failed.")

        self.run_task("Loading model", task, done)

    def warmup(self) -> None:
        if not self.context.engine.is_loaded():
            self.show_error("Model is not loaded. Please open Models.")
            return
        self.run_task("Model warmup", self.context.engine.warmup, lambda info: self.set_status(f"Warmup complete: {info['warmup_ms']:.1f} ms"))

    def refresh(self) -> None:
        self._update_model_availability()
        self._update_provider_availability()
        self._update_swap_model_choices()
        info = self.context.engine.get_runtime_info()
        self.runtime.setPlainText("\n".join(f"{key}: {value}" for key, value in info.items()))

    def _update_provider_availability(self) -> None:
        cuda_available = is_cuda_provider_available()
        cuda_index = self.provider_combo.findText("CUDA")
        if cuda_index >= 0:
            item = self.provider_combo.model().item(cuda_index)
            if item is not None:
                item.setEnabled(cuda_available)
                item.setToolTip(
                    "CUDAExecutionProvider is available."
                    if cuda_available
                    else "CUDAExecutionProvider is not available. Install a matching onnxruntime-gpu, CUDA runtime, and GPU driver first."
                )
        if self.provider_combo.currentText() == "CUDA" and not cuda_available:
            self.provider_combo.setCurrentText("Auto")
            self.provider_combo.setToolTip("CUDA is unavailable on this machine, so Auto will use CPU.")
        else:
            self.provider_combo.setToolTip("Auto uses CUDA when CUDAExecutionProvider is available, otherwise CPU.")

    def _update_model_availability(self) -> None:
        model = self.model_combo.model()
        for index, package in enumerate(self.model_packages):
            item = model.item(index)
            if item is None:
                continue
            installed = is_model_package_installed(package, self.context.config.model_root)
            item.setEnabled(installed)
            item.setToolTip(
                f"{package} is installed under {self.context.config.model_root}/models."
                if installed
                else f"{package} is not downloaded. Open Models > Downloads to install it."
            )
        custom_item = model.item(len(self.model_packages))
        if custom_item is not None:
            custom_item.setEnabled(True)

    def _update_swap_model_choices(self) -> None:
        current = getattr(self.context.config, "swap_model_path", "")
        paths = list_installed_swap_models(self.context.config.model_root)
        self.swap_model_combo.blockSignals(True)
        self.swap_model_combo.clear()
        if not paths:
            self.swap_model_combo.addItem("No downloaded swap models", "")
            self.swap_model_combo.setEnabled(False)
            self.swap_model_combo.setToolTip("Download inswapper_128.onnx from Models > Downloads first.")
        else:
            self.swap_model_combo.setEnabled(True)
            self.swap_model_combo.setToolTip("Only downloaded swap models are shown.")
            for path in paths:
                label = f"{Path(path).parent.name}/{Path(path).name}"
                self.swap_model_combo.addItem(label, str(path))
            index = self.swap_model_combo.findData(current)
            if index >= 0:
                self.swap_model_combo.setCurrentIndex(index)
            elif current and Path(current).exists():
                self.swap_model_combo.addItem(Path(current).name, current)
                self.swap_model_combo.setCurrentIndex(self.swap_model_combo.count() - 1)
        self.swap_model_combo.blockSignals(False)
