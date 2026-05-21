"""Combined Query and Gallery verification workflow."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QDir, QEvent, QSize, Qt, QUrl, Signal
from PySide6.QtGui import QCursor, QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSizePolicy,
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileSystemModel,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from ..core.constants import IMAGE_EXTENSIONS
from ..core.recognition import compare_embeddings
from ..core.tooltips import apply_button_tooltips, set_button_tooltip
from ..core.utils import list_images, read_image
from ..widgets.threshold_slider import ThresholdSlider
from ..widgets.upload_preview import UploadPreview
from .base import BasePage


class GalleryUploadPanel(QFrame):
    """Clickable and draggable gallery input for image files or folders."""

    pathsChanged = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._paths: list[str] = []
        self.setObjectName("galleryUpload")
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setMinimumHeight(300)
        self.setProperty("hoverActive", False)
        self.setProperty("dragActive", False)
        self.setProperty("hasFiles", False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        header = QHBoxLayout()
        title = QLabel("Gallery")
        title.setStyleSheet("font-size:15px; font-weight:700;")
        self.count_label = QLabel("No images")
        self.count_label.setProperty("role", "muted")
        self.remove_button = QPushButton("Remove Selected")
        self.remove_button.clicked.connect(self.remove_selected)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear)
        set_button_tooltip(self.remove_button)
        set_button_tooltip(self.clear_button)
        header.addWidget(title)
        header.addWidget(self.count_label)
        header.addStretch(1)
        header.addWidget(self.remove_button)
        header.addWidget(self.clear_button)
        layout.addLayout(header)

        self.prompt = QLabel("Click to upload images or a folder, or drag them here")
        self.prompt.setObjectName("uploadPrompt")
        self.prompt.setAlignment(Qt.AlignCenter)
        self.prompt.setWordWrap(True)
        self.prompt.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.prompt, 1)

        self.single_preview = QLabel()
        self.single_preview.setAlignment(Qt.AlignCenter)
        self.single_preview.setMinimumHeight(180)
        self.single_preview.setStyleSheet("border:0; background:transparent;")
        self.single_preview.hide()
        self.single_hint = QLabel("")
        self.single_hint.setObjectName("pathLabel")
        self.single_hint.setAlignment(Qt.AlignCenter)
        self.single_hint.setWordWrap(True)
        self.single_hint.hide()
        layout.addWidget(self.single_preview, 1)
        layout.addWidget(self.single_hint)

        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(56, 56))
        self.list_widget.itemDoubleClicked.connect(self.open_item)
        layout.addWidget(self.list_widget, 1)
        self.list_widget.hide()
        self.remove_button.hide()
        self.clear_button.hide()

        for watched in (self, self.prompt, self.single_preview, self.single_hint, self.list_widget, self.list_widget.viewport()):
            watched.installEventFilter(self)

    def paths(self) -> list[str]:
        return list(self._paths)

    def set_paths(self, paths: list[str], emit: bool = True) -> None:
        resolved = self._expand_paths(paths)
        self._paths = resolved
        self._render()
        if emit:
            self.pathsChanged.emit(self.paths())

    def add_paths(self, paths: list[str]) -> None:
        self.set_paths(self._paths + paths)

    def clear(self) -> None:
        self.set_paths([])

    def remove_selected(self) -> None:
        remove = {item.data(Qt.UserRole) for item in self.list_widget.selectedItems()}
        if not remove:
            return
        self.set_paths([path for path in self._paths if path not in remove])

    def browse(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Select gallery images or folders")
        dialog.resize(760, 520)
        layout = QVBoxLayout(dialog)
        hint = QLabel("Select one or more image files or folders. Folders are scanned recursively.")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        model = QFileSystemModel(dialog)
        model.setRootPath(str(Path.home()))
        model.setFilter(QDir.AllEntries | QDir.NoDotAndDotDot | QDir.Readable)
        model.setNameFilters(["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"])
        model.setNameFilterDisables(False)
        tree = QTreeView(dialog)
        tree.setModel(model)
        tree.setRootIndex(model.index(str(Path.home())))
        tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        tree.setColumnWidth(0, 320)
        tree.doubleClicked.connect(lambda index: tree.setRootIndex(index) if model.isDir(index) else None)
        layout.addWidget(tree, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        apply_button_tooltips(dialog)
        if dialog.exec() == QDialog.Accepted:
            indexes = tree.selectionModel().selectedRows(0)
            paths = [model.filePath(index) for index in indexes]
            if paths:
                self.add_paths(paths)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if event.type() in (QEvent.DragEnter, QEvent.DragMove):
            if event.mimeData().hasUrls() and self._accepted_urls(event.mimeData().urls()):
                self._set_property("dragActive", True)
                event.acceptProposedAction()
                return True
            event.ignore()
            return True
        if event.type() == QEvent.DragLeave:
            self._set_property("dragActive", False)
            return False
        if event.type() == QEvent.Drop:
            self._set_property("dragActive", False)
            paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
            expanded = self._expand_paths(paths)
            if expanded:
                self.add_paths(expanded)
                event.acceptProposedAction()
                return True
            event.ignore()
            return True
        if event.type() == QEvent.Enter:
            self._set_property("hoverActive", True)
            return False
        if event.type() == QEvent.Leave:
            self._update_hover_from_cursor()
            return False
        if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton and watched in {self.prompt, self.single_preview, self.single_hint}:
            self.browse()
            return True
        return super().eventFilter(watched, event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton and not self._paths:
            self.browse()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def enterEvent(self, event) -> None:  # noqa: N802
        self._set_property("hoverActive", True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._update_hover_from_cursor()
        super().leaveEvent(event)

    def open_item(self, item: QListWidgetItem) -> None:
        path = item.data(Qt.UserRole)
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _render(self) -> None:
        self.list_widget.clear()
        self.single_preview.clear()
        for path in self._paths:
            item = QListWidgetItem(Path(path).name)
            icon = self._icon(path, QSize(56, 56))
            if icon is not None:
                item.setIcon(icon)
            item.setToolTip(path)
            item.setData(Qt.UserRole, path)
            self.list_widget.addItem(item)
        has_files = bool(self._paths)
        single = len(self._paths) == 1
        self.prompt.setVisible(not has_files)
        self.single_preview.setVisible(single)
        self.single_hint.setVisible(single)
        self.list_widget.setVisible(has_files and not single)
        self.remove_button.setVisible(has_files and not single)
        self.clear_button.setVisible(has_files)
        count = len(self._paths)
        self.count_label.setText(f"{count} image" + ("" if count == 1 else "s") if count else "No images")
        if single:
            self._update_single_preview()
        self._set_property("hasFiles", has_files)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if len(self._paths) == 1:
            self._update_single_preview()

    def _expand_paths(self, paths: list[str]) -> list[str]:
        collected: list[str] = []
        seen = set()
        for raw_path in paths:
            path = Path(raw_path).expanduser()
            candidates: list[Path]
            if path.is_dir():
                candidates = list_images(path, recursive=True)
            elif path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                candidates = [path]
            else:
                candidates = []
            for candidate in candidates:
                value = str(candidate)
                if value not in seen and value not in collected:
                    collected.append(value)
                    seen.add(value)
        return collected

    def _accepted_urls(self, urls) -> bool:
        for url in urls:
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile()).expanduser()
            if path.is_dir() or (path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
                return True
        return False

    def _set_property(self, name: str, value) -> None:
        self.setProperty(name, value)
        self.style().unpolish(self)
        self.style().polish(self)

    def _update_hover_from_cursor(self) -> None:
        inside = self.rect().contains(self.mapFromGlobal(QCursor.pos()))
        self._set_property("hoverActive", inside)

    @staticmethod
    def _icon(path: str | None, size: QSize) -> QIcon | None:
        if not path:
            return None
        pixmap = QPixmap(str(Path(path)))
        if pixmap.isNull():
            return None
        return QIcon(pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _update_single_preview(self) -> None:
        if len(self._paths) != 1:
            return
        path = self._paths[0]
        pixmap = QPixmap(str(Path(path)))
        if pixmap.isNull():
            self.single_preview.setText(Path(path).name)
        else:
            size = self.single_preview.size()
            target = QSize(max(180, size.width() - 24), max(150, size.height() - 12))
            self.single_preview.setPixmap(pixmap.scaled(target, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.single_hint.setText(f"{Path(path).name}\nClick or drag more images or folders to add to Gallery.")


class VerificationPage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(
            context,
            "Face Recognition",
            "Upload one query image and a gallery image, image set, or folder. One gallery image runs 1:1 compare; multiple gallery images run 1:N gallery search.",
            parent,
        )
        self.query_path = ""
        self.query_image: np.ndarray | None = None
        self.gallery_paths: list[str] = []
        self.results: list[dict] = []

        self.content.addWidget(self.notice("All processing is local by default. Uploaded query and gallery files are not copied or uploaded automatically."))
        splitter = QSplitter(Qt.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.query_input = UploadPreview(
            "Query",
            extensions=[".jpg", ".jpeg", ".png", ".bmp", ".webp"],
            dialog_filter="Images (*.jpg *.jpeg *.png *.bmp *.webp);;All Files (*)",
        )
        self.query_input.pathChanged.connect(lambda path: self.load_query(path) if path else self.clear_query())
        left_layout.addWidget(self.query_input)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.gallery_input = GalleryUploadPanel()
        self.gallery_input.pathsChanged.connect(self.set_gallery_paths)
        right_layout.addWidget(self.gallery_input)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        self.content.addWidget(splitter, 1)

        control_row = QHBoxLayout()
        self.threshold = ThresholdSlider(context.config.recognition_threshold)
        self.run_button = QPushButton("Run Recognition")
        self.run_button.clicked.connect(self.run_verification)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_all)
        set_button_tooltip(self.run_button)
        set_button_tooltip(self.clear_button)
        control_row.addStretch(1)
        control_row.addWidget(self.run_button)
        control_row.addWidget(self.clear_button)
        self.content.addLayout(control_row)
        self.content.addWidget(self.threshold)

        self.result_table = QTableWidget(0, 7)
        self.result_table.setIconSize(QSize(56, 56))
        self.result_table.setHorizontalHeaderLabels(
            ["rank", "thumbnail", "gallery_file", "similarity", "threshold", "decision", "det_score"]
        )
        self.result_table.cellDoubleClicked.connect(self.open_result)
        self.content.addWidget(self.result_table, 1)

    def load_query(self, path: str) -> None:
        image = read_image(path)
        if image is None:
            self.query_input.clear(emit=False)
            self.show_error("Image read failure.")
            return
        self.query_path = path
        self.query_image = image
        self.query_input.set_image(image, path)
        self._update_mode_label()

    def clear_query(self) -> None:
        self.query_path = ""
        self.query_image = None
        self.query_input.clear(emit=False)
        self.query_input.set_faces([])
        self._update_mode_label()

    def set_gallery_paths(self, paths: list[str]) -> None:
        self.gallery_paths = list(paths)
        self.results = []
        self.result_table.setRowCount(0)
        self._update_mode_label()

    def clear_all(self) -> None:
        self.clear_query()
        self.gallery_input.clear()
        self.results = []
        self.result_table.setRowCount(0)
        self.set_status("Cleared.")

    def run_verification(self) -> None:
        if self.query_image is None or not self.query_path:
            self.show_error("Please upload a query image.")
            return
        if not self.gallery_paths:
            self.show_error("Please upload one or more gallery images or folders.")
            return
        if not self.context.engine.is_loaded():
            self.show_error("Model is not loaded. Please open Models.")
            return

        query_image = self.query_image.copy()
        query_path = self.query_path
        gallery_paths = list(self.gallery_paths)
        threshold = self.threshold.value()

        def task(progress=None, is_cancelled=None):
            if len(gallery_paths) == 1:
                gallery_image = read_image(gallery_paths[0])
                if gallery_image is None:
                    raise ValueError(f"Image read failure: {gallery_paths[0]}")
                compare = self.context.engine.compare_images(
                    query_image,
                    gallery_image,
                    threshold=threshold,
                    path1=query_path,
                    path2=gallery_paths[0],
                )
                row = {
                    "rank": 1,
                    "path": gallery_paths[0],
                    "similarity": compare.similarity,
                    "threshold": compare.threshold,
                    "decision": compare.decision,
                    "det_score": compare.face_b.det_score,
                    "bbox": compare.face_b.bbox,
                    "query_bbox": compare.face_a.bbox,
                    "notes": "; ".join(compare.notes),
                }
                return {"query_bbox": compare.face_a.bbox, "results": [row], "mode": "1:1 Compare"}

            query_face = self.context.engine.detect_best_face(query_image, source_path=query_path)
            if query_face is None or query_face.normed_embedding is None:
                raise ValueError("No face detected in the query image or embedding unavailable.")
            rows = []
            for index, path in enumerate(gallery_paths):
                if is_cancelled and is_cancelled():
                    break
                image = read_image(path)
                if image is None:
                    rows.append(self._error_row(path, "Image read failure.", threshold))
                else:
                    face = self.context.engine.detect_best_face(image, source_path=path)
                    if face is None or face.normed_embedding is None:
                        rows.append(self._error_row(path, "No face detected.", threshold))
                    else:
                        result = compare_embeddings(query_face.normed_embedding, face.normed_embedding, threshold)
                        rows.append(
                            {
                                "rank": 0,
                                "path": path,
                                "similarity": float(result["similarity"]),
                                "threshold": threshold,
                                "decision": str(result["decision"]),
                                "det_score": face.det_score,
                                "bbox": face.bbox,
                                "query_bbox": query_face.bbox,
                                "notes": "",
                            }
                        )
                if progress:
                    progress(index + 1, len(gallery_paths), f"Processed {index + 1} of {len(gallery_paths)} gallery images")
            rows.sort(
                key=lambda item: item["similarity"] if isinstance(item.get("similarity"), (int, float)) else -1.0,
                reverse=True,
            )
            for rank, row in enumerate(rows, start=1):
                row["rank"] = rank
            return {"query_bbox": query_face.bbox, "results": rows, "mode": "1:N Gallery Search"}

        def done(payload):
            self.results = payload["results"]
            self.query_input.set_faces([{"bbox": payload["query_bbox"], "label": "Query"}])
            self._populate_results()
            self.set_status(f"{payload['mode']} complete. {len(self.results)} gallery result(s).")

        self.run_task("Running verification", task, done)

    def open_result(self, row: int, column: int) -> None:
        del column
        if row < 0 or row >= len(self.results):
            return
        path = self.results[row].get("path")
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _populate_results(self) -> None:
        self.result_table.setRowCount(len(self.results))
        for row_index, result in enumerate(self.results):
            values = [
                result.get("rank", ""),
                "",
                Path(str(result.get("path", ""))).name if result.get("path") else "",
                f"{float(result.get('similarity', 0.0)):.4f}" if result.get("similarity") is not None else "",
                f"{float(result.get('threshold', 0.0)):.4f}" if result.get("threshold") is not None else "",
                result.get("decision", ""),
                f"{float(result.get('det_score', 0.0)):.4f}" if result.get("det_score") is not None else "",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if col == 1:
                    icon = GalleryUploadPanel._icon(result.get("path"), QSize(56, 56))
                    if icon is not None:
                        item.setIcon(icon)
                item.setToolTip(str(result.get("notes") or result.get("path") or ""))
                item.setData(Qt.UserRole, result.get("path", ""))
                self.result_table.setItem(row_index, col, item)
            self.result_table.setRowHeight(row_index, 64)
        self.result_table.resizeColumnsToContents()

    def _update_mode_label(self) -> None:
        pass

    @staticmethod
    def _error_row(path: str, message: str, threshold: float) -> dict:
        return {
            "rank": 0,
            "path": path,
            "similarity": None,
            "threshold": threshold,
            "decision": message,
            "det_score": None,
            "bbox": None,
            "notes": message,
        }
