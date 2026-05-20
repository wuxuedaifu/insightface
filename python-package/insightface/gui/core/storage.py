"""SQLite storage for local people, media, embeddings, and reports."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import numpy as np

from .logging import get_logger
from .recognition import search_gallery
from .utils import safe_json_dumps, utc_now_iso

LOGGER = get_logger("storage")


def embedding_to_blob(embedding: np.ndarray | Iterable[float] | None) -> tuple[Optional[bytes], int]:
    if embedding is None:
        return None, 0
    arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    return arr.tobytes(), int(arr.shape[0])


def blob_to_embedding(blob: bytes | memoryview | None, dim: int | None = None) -> Optional[np.ndarray]:
    if blob is None:
        return None
    arr = np.frombuffer(blob, dtype=np.float32).copy()
    if dim and arr.shape[0] != dim:
        return None
    return arr


class Storage:
    def __init__(self, database_path: str | Path):
        self.database_path = Path(database_path).expanduser()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.database_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            LOGGER.exception("Database operation failed")
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS people (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    display_name TEXT,
                    notes TEXT,
                    tags TEXT,
                    cover_face_sample_id INTEGER,
                    created_at TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS face_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id INTEGER,
                    source_image_path TEXT,
                    crop_path TEXT,
                    embedding BLOB,
                    embedding_dim INTEGER,
                    bbox_json TEXT,
                    kps_json TEXT,
                    det_score REAL,
                    quality_score REAL,
                    model_name TEXT,
                    provider TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS media_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE,
                    media_type TEXT,
                    width INTEGER,
                    height INTEGER,
                    duration_ms INTEGER,
                    file_size INTEGER,
                    mtime REAL,
                    sha256 TEXT,
                    processed_at TEXT,
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS media_faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    media_id INTEGER,
                    frame_index INTEGER,
                    timestamp_ms INTEGER,
                    crop_path TEXT,
                    embedding BLOB,
                    embedding_dim INTEGER,
                    bbox_json TEXT,
                    kps_json TEXT,
                    det_score REAL,
                    quality_score REAL,
                    assigned_person_id INTEGER,
                    predicted_person_id INTEGER,
                    similarity REAL,
                    cluster_id INTEGER,
                    status TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS clusters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    cover_media_face_id INTEGER,
                    notes TEXT,
                    created_at TEXT,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS recognition_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_type TEXT,
                    source_path TEXT,
                    query_face_sample_id INTEGER,
                    predicted_person_id INTEGER,
                    similarity REAL,
                    threshold REAL,
                    status TEXT,
                    payload_json TEXT,
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scenario TEXT,
                    model_name TEXT,
                    provider TEXT,
                    threshold REAL,
                    dataset_summary_json TEXT,
                    metrics_json TEXT,
                    hardware_json TEXT,
                    report_path TEXT,
                    created_at TEXT
                );

                CREATE TABLE IF NOT EXISTS app_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TEXT
                );
                """
            )

    def add_person(
        self,
        name: str,
        display_name: Optional[str] = None,
        notes: str = "",
        tags: str = "",
    ) -> int:
        now = utc_now_iso()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO people (name, display_name, notes, tags, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (name, display_name or name, notes, tags, now, now),
            )
            return int(cur.lastrowid)

    def update_person(self, person_id: int, **fields: Any) -> None:
        allowed = {"name", "display_name", "notes", "tags", "cover_face_sample_id"}
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            return
        updates["updated_at"] = utc_now_iso()
        clause = ", ".join(f"{key}=?" for key in updates)
        values = list(updates.values()) + [person_id]
        with self.connect() as conn:
            conn.execute(f"UPDATE people SET {clause} WHERE id=?", values)

    def delete_person(self, person_id: int, delete_samples: bool = False) -> None:
        with self.connect() as conn:
            if delete_samples:
                conn.execute("DELETE FROM face_samples WHERE person_id=?", (person_id,))
            else:
                conn.execute("UPDATE face_samples SET person_id=NULL WHERE person_id=?", (person_id,))
            conn.execute("UPDATE media_faces SET assigned_person_id=NULL WHERE assigned_person_id=?", (person_id,))
            conn.execute("DELETE FROM people WHERE id=?", (person_id,))

    def merge_people(self, source_person_id: int, target_person_id: int) -> None:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                "UPDATE face_samples SET person_id=?, updated_at=? WHERE person_id=?",
                (target_person_id, now, source_person_id),
            )
            conn.execute(
                "UPDATE media_faces SET assigned_person_id=? WHERE assigned_person_id=?",
                (target_person_id, source_person_id),
            )
            conn.execute("DELETE FROM people WHERE id=?", (source_person_id,))

    def add_face_sample(
        self,
        person_id: Optional[int],
        embedding: np.ndarray | Iterable[float] | None,
        source_image_path: str = "",
        crop_path: str = "",
        bbox: Optional[Iterable[float]] = None,
        kps: Optional[Iterable[Iterable[float]]] = None,
        det_score: float = 0.0,
        quality_score: Optional[float] = None,
        model_name: str = "",
        provider: str = "",
    ) -> int:
        blob, dim = embedding_to_blob(embedding)
        now = utc_now_iso()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO face_samples (
                    person_id, source_image_path, crop_path, embedding, embedding_dim,
                    bbox_json, kps_json, det_score, quality_score, model_name, provider,
                    created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    person_id,
                    source_image_path,
                    crop_path,
                    blob,
                    dim,
                    safe_json_dumps(list(bbox) if bbox is not None else None),
                    safe_json_dumps(list(kps) if kps is not None else None),
                    det_score,
                    quality_score,
                    model_name,
                    provider,
                    now,
                    now,
                ),
            )
            sample_id = int(cur.lastrowid)
            if person_id is not None:
                conn.execute(
                    """
                    UPDATE people
                    SET cover_face_sample_id=COALESCE(cover_face_sample_id, ?), updated_at=?
                    WHERE id=?
                    """,
                    (sample_id, now, person_id),
                )
            return sample_id

    def delete_face_sample(self, sample_id: int) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM face_samples WHERE id=?", (sample_id,))

    def list_people(self) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT p.*,
                       COUNT(fs.id) AS sample_count,
                       AVG(fs.quality_score) AS avg_quality,
                       COALESCE(
                           (
                               SELECT fs_cover.crop_path
                               FROM face_samples fs_cover
                               WHERE fs_cover.id = p.cover_face_sample_id
                                 AND COALESCE(fs_cover.crop_path, '') <> ''
                               LIMIT 1
                           ),
                           (
                               SELECT fs2.crop_path
                               FROM face_samples fs2
                               WHERE fs2.person_id = p.id AND COALESCE(fs2.crop_path, '') <> ''
                               ORDER BY fs2.id ASC
                               LIMIT 1
                           )
                       ) AS cover_crop_path
                FROM people p
                LEFT JOIN face_samples fs ON fs.person_id = p.id
                GROUP BY p.id
                ORDER BY LOWER(COALESCE(p.display_name, p.name))
                """
            ).fetchall()
            return [dict(row) for row in rows]

    def list_face_samples(self, person_id: Optional[int] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM face_samples"
        params: tuple[Any, ...] = ()
        if person_id is not None:
            query += " WHERE person_id=?"
            params = (person_id,)
        query += " ORDER BY created_at DESC, id DESC"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._decode_sample(row) for row in rows]

    def _decode_sample(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = dict(row)
        data["embedding"] = blob_to_embedding(data.get("embedding"), data.get("embedding_dim"))
        for key in ("bbox_json", "kps_json"):
            try:
                data[key.replace("_json", "")] = json.loads(data[key]) if data.get(key) else None
            except Exception:
                data[key.replace("_json", "")] = None
        return data

    def add_media_item(
        self,
        path: str,
        media_type: str = "image",
        width: Optional[int] = None,
        height: Optional[int] = None,
        duration_ms: Optional[int] = None,
        file_size: Optional[int] = None,
        mtime: Optional[float] = None,
        sha256: Optional[str] = None,
        processed_at: Optional[str] = None,
    ) -> int:
        now = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO media_items (
                    path, media_type, width, height, duration_ms, file_size,
                    mtime, sha256, processed_at, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (path, media_type, width, height, duration_ms, file_size, mtime, sha256, processed_at, now),
            )
            row = conn.execute("SELECT id FROM media_items WHERE path=?", (path,)).fetchone()
            return int(row["id"])

    def add_media_face(
        self,
        media_id: int,
        embedding: np.ndarray | Iterable[float] | None,
        crop_path: str = "",
        bbox: Optional[Iterable[float]] = None,
        kps: Optional[Iterable[Iterable[float]]] = None,
        det_score: float = 0.0,
        quality_score: Optional[float] = None,
        frame_index: Optional[int] = None,
        timestamp_ms: Optional[int] = None,
        assigned_person_id: Optional[int] = None,
        predicted_person_id: Optional[int] = None,
        similarity: Optional[float] = None,
        cluster_id: Optional[int] = None,
        status: str = "unknown",
    ) -> int:
        blob, dim = embedding_to_blob(embedding)
        now = utc_now_iso()
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO media_faces (
                    media_id, frame_index, timestamp_ms, crop_path, embedding, embedding_dim,
                    bbox_json, kps_json, det_score, quality_score, assigned_person_id,
                    predicted_person_id, similarity, cluster_id, status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    media_id,
                    frame_index,
                    timestamp_ms,
                    crop_path,
                    blob,
                    dim,
                    safe_json_dumps(list(bbox) if bbox is not None else None),
                    safe_json_dumps(list(kps) if kps is not None else None),
                    det_score,
                    quality_score,
                    assigned_person_id,
                    predicted_person_id,
                    similarity,
                    cluster_id,
                    status,
                    now,
                    now,
                ),
            )
            return int(cur.lastrowid)

    def load_all_gallery_embeddings(self) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT fs.id AS sample_id, fs.person_id, fs.crop_path, fs.embedding,
                       fs.embedding_dim,
                       COALESCE(p.display_name, p.name, 'Unknown') AS person_name
                FROM face_samples fs
                LEFT JOIN people p ON p.id = fs.person_id
                WHERE fs.embedding IS NOT NULL AND fs.person_id IS NOT NULL
                """
            ).fetchall()
            gallery = []
            for row in rows:
                item = dict(row)
                item["embedding"] = blob_to_embedding(item.get("embedding"), item.get("embedding_dim"))
                gallery.append(item)
            return gallery

    def search_embeddings(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        threshold: float = 0.5,
    ):
        return search_gallery(query_embedding, self.load_all_gallery_embeddings(), top_k=top_k, threshold=threshold)

    def log_recognition(
        self,
        source_type: str,
        source_path: str,
        predicted_person_id: Optional[int],
        similarity: float,
        threshold: float,
        status: str,
        payload: Optional[Dict[str, Any]] = None,
        query_face_sample_id: Optional[int] = None,
    ) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO recognition_logs (
                    source_type, source_path, query_face_sample_id, predicted_person_id,
                    similarity, threshold, status, payload_json, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_type,
                    source_path,
                    query_face_sample_id,
                    predicted_person_id,
                    similarity,
                    threshold,
                    status,
                    safe_json_dumps(payload or {}),
                    utc_now_iso(),
                ),
            )
            return int(cur.lastrowid)

    def save_evaluation_run(
        self,
        scenario: str,
        model_name: str,
        provider: str,
        threshold: float,
        dataset_summary: Dict[str, Any],
        metrics: Dict[str, Any],
        hardware: Dict[str, Any],
        report_path: str = "",
        created_at: Optional[str] = None,
    ) -> int:
        with self.connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO evaluation_runs (
                    scenario, model_name, provider, threshold, dataset_summary_json,
                    metrics_json, hardware_json, report_path, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scenario,
                    model_name,
                    provider,
                    threshold,
                    safe_json_dumps(dataset_summary),
                    safe_json_dumps(metrics),
                    safe_json_dumps(hardware),
                    report_path,
                    created_at or utc_now_iso(),
                ),
            )
            return int(cur.lastrowid)

    def list_evaluation_runs(self) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM evaluation_runs ORDER BY created_at DESC, id DESC").fetchall()
            result = []
            for row in rows:
                item = dict(row)
                for key in ("dataset_summary_json", "metrics_json", "hardware_json"):
                    try:
                        item[key.replace("_json", "")] = json.loads(item.get(key) or "{}")
                    except Exception:
                        item[key.replace("_json", "")] = {}
                result.append(item)
            return result

    def delete_evaluation_run(self, run_id: int) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM evaluation_runs WHERE id=?", (run_id,))

    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM app_settings WHERE key=?", (key,)).fetchone()
            return row["value"] if row else default

    def set_setting(self, key: str, value: Any) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO app_settings (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (key, str(value), utc_now_iso()),
            )

    def existing_media_paths(self, paths: Iterable[str]) -> set[str]:
        values = [str(path) for path in paths]
        if not values:
            return set()
        existing: set[str] = set()
        with self.connect() as conn:
            for index in range(0, len(values), 900):
                chunk = values[index : index + 900]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(f"SELECT path FROM media_items WHERE path IN ({placeholders})", chunk).fetchall()
                existing.update(str(row["path"]) for row in rows)
        return existing

    def list_media_faces(self) -> List[Dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT mf.*, mi.path AS media_path, mi.media_type, mi.width, mi.height
                FROM media_faces mf
                JOIN media_items mi ON mi.id = mf.media_id
                WHERE mf.embedding IS NOT NULL
                ORDER BY mf.created_at DESC, mf.id DESC
                """
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["embedding"] = blob_to_embedding(item.get("embedding"), item.get("embedding_dim"))
                for key in ("bbox_json", "kps_json"):
                    try:
                        item[key.replace("_json", "")] = json.loads(item[key]) if item.get(key) else None
                    except Exception:
                        item[key.replace("_json", "")] = None
                result.append(item)
            return result

    def counts(self) -> Dict[str, int]:
        with self.connect() as conn:
            return {
                "people": int(conn.execute("SELECT COUNT(*) FROM people").fetchone()[0]),
                "face_samples": int(conn.execute("SELECT COUNT(*) FROM face_samples").fetchone()[0]),
                "media_items": int(conn.execute("SELECT COUNT(*) FROM media_items").fetchone()[0]),
                "media_faces": int(conn.execute("SELECT COUNT(*) FROM media_faces").fetchone()[0]),
                "evaluation_runs": int(conn.execute("SELECT COUNT(*) FROM evaluation_runs").fetchone()[0]),
            }
