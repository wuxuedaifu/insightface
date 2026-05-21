"""Single-page album import, clustering, and review workflow."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QSize, Qt, QUrl, Signal
from PySide6.QtGui import QCursor, QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.clustering import cluster_embeddings_dbscan
from ..core.recognition import cosine_similarity, normalize_embedding
from ..core.tooltips import set_button_tooltip
from ..core.utils import list_images, read_image, save_image, timestamp_for_filename
from .base import BasePage


class AlbumDirectoryList(QListWidget):
    foldersChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("albumDirectoryList")
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.setProperty("hoverActive", False)
        self.setProperty("dragActive", False)
        self.installEventFilter(self)
        self.viewport().installEventFilter(self)

    def add_folder(self, folder: str) -> None:
        folder = str(Path(folder).expanduser())
        if folder and Path(folder).is_dir() and folder not in self.folders():
            self.addItem(folder)
            self.foldersChanged.emit()

    def folders(self) -> list[str]:
        return [self.item(index).text() for index in range(self.count())]

    def dragEnterEvent(self, event) -> None:  # noqa: N802
        if event.mimeData().hasUrls() and any(Path(url.toLocalFile()).is_dir() for url in event.mimeData().urls() if url.isLocalFile()):
            self._drag(True)
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802
        self._drag(False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802
        self._drag(False)
        added = False
        for url in event.mimeData().urls():
            if url.isLocalFile() and Path(url.toLocalFile()).is_dir():
                self.add_folder(url.toLocalFile())
                added = True
        if added:
            event.acceptProposedAction()
        else:
            event.ignore()

    def _drag(self, active: bool) -> None:
        self.setProperty("dragActive", active)
        self.style().unpolish(self)
        self.style().polish(self)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if event.type() == QEvent.Enter:
            self._set_property("hoverActive", True)
            return False
        if event.type() == QEvent.Leave:
            self._update_hover_from_cursor()
            return False
        return super().eventFilter(watched, event)

    def enterEvent(self, event) -> None:  # noqa: N802
        self._set_property("hoverActive", True)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._update_hover_from_cursor()
        super().leaveEvent(event)

    def _set_property(self, name: str, value) -> None:
        self.setProperty(name, value)
        self.style().unpolish(self)
        self.style().polish(self)

    def _update_hover_from_cursor(self) -> None:
        inside = self.rect().contains(self.mapFromGlobal(QCursor.pos()))
        self._set_property("hoverActive", inside)


class AlbumPage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(
            context,
            "Album",
            "Import or refresh local album folders, cluster detected faces, and review the photos in each person group.",
            parent,
        )
        self.clusters: list[dict] = []
        self.cluster_items: dict[int, list[dict]] = {}
        self._loaded_saved_state = False

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        self.content.addWidget(self.notice("All album processing is local. New image files are detected on refresh; existing indexed files are reused for clustering."))
        self.folder_list = AlbumDirectoryList()
        self.folder_list.setMinimumHeight(90)
        self.folder_list.foldersChanged.connect(self._save_directories)
        controls_layout.addWidget(QLabel("Album directories"))
        controls_layout.addWidget(self.folder_list)
        button_row = QHBoxLayout()
        for button in [
            self._button("Add Folder", self.add_folder),
            self._button("Remove Selected", self.remove_selected),
            self._button("Clear", self.clear_directories),
            self._button("Import / Refresh", self.import_refresh),
            self._button("Rebuild All", self.rebuild_all),
        ]:
            button_row.addWidget(button)
        button_row.addStretch(1)
        controls_layout.addLayout(button_row)

        threshold_row = QHBoxLayout()
        self.cluster_threshold = QDoubleSpinBox()
        self.cluster_threshold.setRange(0.01, 0.99)
        self.cluster_threshold.setSingleStep(0.01)
        self.cluster_threshold.setValue(0.28)
        self.match_threshold = QDoubleSpinBox()
        self.match_threshold.setRange(0.01, 0.99)
        self.match_threshold.setSingleStep(0.01)
        self.match_threshold.setValue(0.28)
        self.min_cluster_size = QSpinBox()
        self.min_cluster_size.setRange(2, 50)
        self.min_cluster_size.setValue(2)
        self.algorithm_label = QLabel("Algorithm: DBSCAN")
        threshold_row.addWidget(QLabel("DBSCAN distance threshold"))
        threshold_row.addWidget(self.cluster_threshold)
        threshold_row.addWidget(QLabel("Existing ID duplicate distance"))
        threshold_row.addWidget(self.match_threshold)
        threshold_row.addWidget(QLabel("Min samples"))
        threshold_row.addWidget(self.min_cluster_size)
        threshold_row.addWidget(self.algorithm_label)
        threshold_row.addStretch(1)
        controls_layout.addLayout(threshold_row)
        self.content.addWidget(controls)

        splitter = QSplitter(Qt.Horizontal)
        self.cluster_table = QTableWidget(0, 7)
        self.cluster_table.setIconSize(QSize(56, 56))
        self.cluster_table.setHorizontalHeaderLabels(["ID", "Thumbnail", "Name", "Faces", "Photos", "Avg quality", "Source"])
        self.cluster_table.currentCellChanged.connect(self.cluster_selected)
        splitter.addWidget(self.cluster_table)

        self.photo_table = QTableWidget(0, 3)
        self.photo_table.setIconSize(QSize(96, 72))
        self.photo_table.setHorizontalHeaderLabels(["Thumbnail", "File", "Faces in cluster"])
        self.photo_table.cellDoubleClicked.connect(self.open_photo)
        splitter.addWidget(self.photo_table)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        self.content.addWidget(splitter, 1)

    def _button(self, text: str, callback) -> QPushButton:
        button = QPushButton(text)
        button.clicked.connect(callback)
        set_button_tooltip(button)
        return button

    def add_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select album folder", str(Path.home()))
        if folder:
            self.folder_list.add_folder(folder)

    def remove_selected(self) -> None:
        for item in self.folder_list.selectedItems():
            self.folder_list.takeItem(self.folder_list.row(item))
        self._save_directories()

    def clear_directories(self) -> None:
        self.folder_list.clear()
        self._save_directories()
        self.set_status("Album directories cleared. Existing clustering results are still available.")

    def import_refresh(self) -> None:
        self._run_import_refresh(rebuild=False)

    def rebuild_all(self) -> None:
        folders = [Path(folder) for folder in self.folder_list.folders()]
        if not folders:
            reply = QMessageBox.question(
                self,
                "Rebuild All",
                "No album directories are selected. Rebuild All will clear saved album clustering results. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                self.context.storage.clear_album_results()
                self.clusters = []
                self.cluster_items = {}
                self._populate_clusters()
                self.set_status("Saved album clustering results were cleared.")
            return
        reply = QMessageBox.question(
            self,
            "Rebuild All",
            "Rebuild All will reprocess every image in the selected album directories and replace saved clustering results. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._run_import_refresh(rebuild=True)

    def _run_import_refresh(self, rebuild: bool = False) -> None:
        folders = [Path(folder) for folder in self.folder_list.folders()]
        if not folders:
            self.show_error("Add one or more album directories first.")
            return
        if not self.context.engine.is_loaded():
            self.show_error("Model is not loaded. Please open Models.")
            return
        self._save_directories()
        all_paths = []
        for folder in folders:
            all_paths.extend(str(path) for path in list_images(folder, recursive=True))
        if not all_paths:
            self.show_error("No supported images found in the selected directories.")
            return
        cluster_threshold = float(self.cluster_threshold.value())
        duplicate_threshold = float(self.match_threshold.value())
        min_samples = int(self.min_cluster_size.value())
        existing = set() if rebuild else self.context.storage.existing_media_paths(all_paths)
        new_paths = list(all_paths) if rebuild else [path for path in all_paths if path not in existing]

        def task(progress=None, is_cancelled=None):
            deleted = 0
            if rebuild:
                deleted = self.context.storage.delete_media_items_by_paths(all_paths)
            imported = 0
            faces_saved = 0
            for index, path in enumerate(new_paths):
                if is_cancelled and is_cancelled():
                    break
                image = read_image(path)
                if image is None:
                    if progress:
                        progress(index + 1, len(new_paths), f"Skipped unreadable image: {Path(path).name}")
                    continue
                media_id = self.context.storage.add_media_item(
                    path,
                    "image",
                    width=image.shape[1],
                    height=image.shape[0],
                    file_size=Path(path).stat().st_size if Path(path).exists() else None,
                    mtime=Path(path).stat().st_mtime if Path(path).exists() else None,
                    processed_at=timestamp_for_filename(),
                )
                for face_index, face in enumerate(self.context.engine.detect_faces(image, source_path=path)):
                    if face.normed_embedding is None:
                        continue
                    crop_path = str(Path(self.context.config.crop_dir) / f"album_{media_id}_{face_index}_{timestamp_for_filename()}.png")
                    if face.crop is not None:
                        save_image(crop_path, face.crop)
                    self.context.storage.add_media_face(
                        media_id,
                        face.normed_embedding,
                        crop_path=crop_path,
                        bbox=face.bbox,
                        kps=face.kps,
                        det_score=face.det_score,
                        quality_score=face.quality_score,
                        status="unknown",
                    )
                    faces_saved += 1
                imported += 1
                if progress:
                    progress(index + 1, max(1, len(new_paths)), f"Imported {imported} new images, saved {faces_saved} faces")
            faces = self._faces_for_folders(folders)
            clusters, algorithm = self._cluster_faces(faces, cluster_threshold, duplicate_threshold, min_samples)
            self.context.storage.save_album_results(
                clusters,
                self.cluster_items,
                algorithm,
                cluster_threshold=cluster_threshold,
                duplicate_threshold=duplicate_threshold,
                min_samples=min_samples,
            )
            return {"deleted": deleted, "imported": imported, "faces_saved": faces_saved, "clusters": clusters, "algorithm": algorithm}

        def done(result):
            self.clusters = result["clusters"]
            self.algorithm_label.setText(f"Algorithm: {result['algorithm']}")
            self._populate_clusters()
            if rebuild:
                self.set_status(
                    f"Rebuilt album from scratch: removed {result['deleted']} indexed image(s), "
                    f"processed {result['imported']} image(s), saved {result['faces_saved']} face(s), "
                    f"built {len(self.clusters)} cluster(s)."
                )
            else:
                self.set_status(f"Imported {result['imported']} new image(s), saved {result['faces_saved']} face(s), built {len(self.clusters)} cluster(s).")

        self.run_task("Rebuilding album" if rebuild else "Importing and clustering album", task, done)

    def _faces_for_folders(self, folders: list[Path]) -> list[dict]:
        roots = [folder.resolve() for folder in folders]
        faces = []
        for face in self.context.storage.list_media_faces():
            try:
                media_path = Path(face["media_path"]).resolve()
            except Exception:
                continue
            if any(media_path == root or root in media_path.parents for root in roots):
                if face.get("embedding") is not None:
                    faces.append(face)
        return faces

    def _cluster_faces(
        self,
        faces: list[dict],
        cluster_threshold: float,
        duplicate_threshold: float,
        min_samples: int,
    ) -> tuple[list[dict], str]:
        embeddings = [face["embedding"] for face in faces]
        labels, algorithm = cluster_embeddings_dbscan(
            embeddings,
            distance_threshold=cluster_threshold,
            min_samples=min_samples,
        )
        groups: dict[int, list[dict]] = defaultdict(list)
        next_noise = max(labels, default=-1) + 1
        for face, label in zip(faces, labels):
            if label < 0:
                label = next_noise
                next_noise += 1
            groups[int(label)].append(face)
        existing_people = self.context.storage.list_people()
        max_existing_id = max([int(person["id"]) for person in existing_people], default=0)
        gallery = self.context.storage.load_all_gallery_embeddings()
        next_album_id = max_existing_id + 1
        used_ids = {int(person["id"]) for person in existing_people}
        clusters = []
        self.cluster_items = {}
        for label, items in groups.items():
            vectors = [normalize_embedding(item["embedding"]) for item in items if item.get("embedding") is not None]
            vectors = [vector for vector in vectors if vector is not None]
            if not vectors:
                continue
            centroid = normalize_embedding(np.mean(np.vstack(vectors), axis=0))
            best_person_id = None
            best_person_name = ""
            best_score = -1.0
            for sample in gallery:
                score = cosine_similarity(centroid, sample.get("embedding"))
                if score > best_score:
                    best_score = score
                    best_person_id = sample.get("person_id")
                    best_person_name = sample.get("person_name") or ""
            source = (
                "existing"
                if best_person_id is not None and (1.0 - best_score) <= duplicate_threshold
                else "album"
            )
            if source == "existing":
                cluster_id = int(best_person_id)
                name = best_person_name or f"Person {cluster_id}"
            else:
                while next_album_id in used_ids:
                    next_album_id += 1
                cluster_id = next_album_id
                used_ids.add(cluster_id)
                next_album_id += 1
                name = f"Album Person {cluster_id}"
            representative = max(items, key=lambda item: cosine_similarity(centroid, item.get("embedding")))
            photos = sorted({item["media_path"] for item in items})
            cluster = {
                "id": cluster_id,
                "label": label,
                "name": name,
                "source": source,
                "face_count": len(items),
                "photo_count": len(photos),
                "avg_quality": sum(float(item.get("quality_score") or 0.0) for item in items) / max(1, len(items)),
                "thumbnail_path": representative.get("crop_path") or representative.get("media_path"),
                "photos": photos,
            }
            clusters.append(cluster)
            self.cluster_items[cluster_id] = items
        return sorted(clusters, key=lambda row: (-row["face_count"], row["id"])), algorithm

    def _save_directories(self) -> None:
        self.context.storage.save_album_directories(self.folder_list.folders())

    def refresh(self) -> None:
        if self._loaded_saved_state:
            return
        self._loaded_saved_state = True
        self.folder_list.blockSignals(True)
        self.folder_list.clear()
        for folder in self.context.storage.list_album_directories():
            if Path(folder).expanduser().is_dir():
                self.folder_list.addItem(folder)
        self.folder_list.blockSignals(False)
        self._load_saved_results()

    def _load_saved_results(self) -> None:
        data = self.context.storage.load_album_results()
        clusters = data.get("clusters") if isinstance(data, dict) else None
        if not isinstance(clusters, list):
            return
        faces_by_id = {int(face["id"]): face for face in self.context.storage.list_media_faces() if face.get("id") is not None}
        self.clusters = []
        self.cluster_items = {}
        for cluster in clusters:
            try:
                cluster_id = int(cluster["id"])
            except Exception:
                continue
            face_ids = [int(face_id) for face_id in cluster.get("face_ids", []) if str(face_id).isdigit()]
            self.cluster_items[cluster_id] = [faces_by_id[face_id] for face_id in face_ids if face_id in faces_by_id]
            self.clusters.append(cluster)
        self.algorithm_label.setText(f"Algorithm: {data.get('algorithm', 'DBSCAN')}")
        if data.get("cluster_threshold") is not None:
            self.cluster_threshold.setValue(float(data["cluster_threshold"]))
        if data.get("duplicate_threshold") is not None:
            self.match_threshold.setValue(float(data["duplicate_threshold"]))
        if data.get("min_samples") is not None:
            self.min_cluster_size.setValue(int(data["min_samples"]))
        self._populate_clusters()

    def _populate_clusters(self) -> None:
        self.cluster_table.setRowCount(len(self.clusters))
        for row, cluster in enumerate(self.clusters):
            values = [
                cluster.get("id", ""),
                "",
                cluster.get("name", ""),
                cluster.get("face_count", 0),
                cluster.get("photo_count", 0),
                f"{float(cluster.get('avg_quality') or 0.0):.3f}",
                cluster.get("source", "album"),
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if col == 1:
                    icon = self._icon(cluster.get("thumbnail_path"), QSize(56, 56))
                    if icon:
                        item.setIcon(icon)
                item.setData(Qt.UserRole, cluster.get("id"))
                self.cluster_table.setItem(row, col, item)
            self.cluster_table.setRowHeight(row, 64)
        self.cluster_table.resizeColumnsToContents()
        if self.clusters:
            self.cluster_table.selectRow(0)
            self._populate_photos(int(self.clusters[0].get("id", 0)))
        else:
            self.photo_table.setRowCount(0)

    def cluster_selected(self, current_row: int, current_column: int, previous_row: int, previous_column: int) -> None:
        del current_column, previous_row, previous_column
        if current_row < 0 or current_row >= len(self.clusters):
            self.photo_table.setRowCount(0)
            return
        self._populate_photos(int(self.clusters[current_row].get("id", 0)))

    def _populate_photos(self, cluster_id: int) -> None:
        items = self.cluster_items.get(cluster_id, [])
        grouped: dict[str, int] = defaultdict(int)
        for item in items:
            grouped[item["media_path"]] += 1
        if not grouped:
            for cluster in self.clusters:
                if int(cluster.get("id", -1)) == cluster_id:
                    for path in cluster.get("photos", []):
                        grouped[str(path)] += 1
                    break
        rows = sorted(grouped.items())
        self.photo_table.setRowCount(len(rows))
        for row, (path, count) in enumerate(rows):
            thumb = QTableWidgetItem("")
            icon = self._icon(path, QSize(96, 72))
            if icon:
                thumb.setIcon(icon)
            thumb.setData(Qt.UserRole, path)
            file_item = QTableWidgetItem(path)
            file_item.setData(Qt.UserRole, path)
            self.photo_table.setItem(row, 0, thumb)
            self.photo_table.setItem(row, 1, file_item)
            self.photo_table.setItem(row, 2, QTableWidgetItem(str(count)))
            self.photo_table.setRowHeight(row, 80)
        self.photo_table.resizeColumnsToContents()

    def open_photo(self, row: int, column: int) -> None:
        item = self.photo_table.item(row, 0) or self.photo_table.item(row, 1)
        if item is None:
            return
        path = item.data(Qt.UserRole)
        if path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    @staticmethod
    def _icon(path: str | None, size: QSize) -> QIcon | None:
        if not path:
            return None
        pixmap = QPixmap(str(Path(path)))
        if pixmap.isNull():
            return None
        return QIcon(pixmap.scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
