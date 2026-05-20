"""No-code enterprise evaluation routines."""

from __future__ import annotations

import csv
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .constants import DEFAULT_LICENSE_STATUS, IMAGE_EXTENSIONS
from .face_engine import FaceEngine
from .models import EvaluationResult
from .recognition import cosine_similarity
from .utils import list_images, read_image, utc_now_iso


def hardware_info() -> Dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }


def _metrics_at_threshold(rows: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    positives = [row for row in rows if row.get("label") == 1 and row.get("similarity") is not None]
    negatives = [row for row in rows if row.get("label") == 0 and row.get("similarity") is not None]
    tp = sum(1 for row in positives if row["similarity"] >= threshold)
    fn = sum(1 for row in positives if row["similarity"] < threshold)
    fp = sum(1 for row in negatives if row["similarity"] >= threshold)
    tn = sum(1 for row in negatives if row["similarity"] < threshold)
    total = tp + tn + fp + fn
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "FAR": fp / max(1, len(negatives)),
        "FRR": fn / max(1, len(positives)),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }


def recommend_threshold(rows: List[Dict[str, Any]]) -> Optional[float]:
    similarities = sorted({float(row["similarity"]) for row in rows if row.get("similarity") is not None})
    if not similarities:
        return None
    best_threshold = similarities[0]
    best_accuracy = -1.0
    for threshold in similarities:
        metrics = _metrics_at_threshold(rows, threshold)
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_threshold = threshold
    return float(best_threshold)


def tar_at_far(rows: List[Dict[str, Any]], target_far: float) -> Any:
    positives = [row for row in rows if row.get("label") == 1 and row.get("similarity") is not None]
    negatives = [row for row in rows if row.get("label") == 0 and row.get("similarity") is not None]
    if len(negatives) < max(20, int(1 / max(target_far, 1e-9))):
        return "insufficient data"
    thresholds = sorted({float(row["similarity"]) for row in rows if row.get("similarity") is not None}, reverse=True)
    best_tar = 0.0
    for threshold in thresholds:
        far = sum(1 for row in negatives if row["similarity"] >= threshold) / max(1, len(negatives))
        if far <= target_far:
            tar = sum(1 for row in positives if row["similarity"] >= threshold) / max(1, len(positives))
            best_tar = max(best_tar, tar)
    return best_tar


def run_kyc_pairs_evaluation(
    pairs_csv: str | Path,
    engine: FaceEngine,
    threshold: float = 0.5,
    license_status: str = DEFAULT_LICENSE_STATUS,
    progress_callback=None,
    cancel_callback=None,
) -> EvaluationResult:
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    with Path(pairs_csv).open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        pairs = list(reader)
    start_all = time.perf_counter()
    for index, row in enumerate(pairs):
        if cancel_callback and cancel_callback():
            break
        image1_path = row.get("image1_path") or row.get("image1") or ""
        image2_path = row.get("image2_path") or row.get("image2") or ""
        label = int(row.get("label", "0"))
        item: Dict[str, Any] = {"image1_path": image1_path, "image2_path": image2_path, "label": label}
        start = time.perf_counter()
        try:
            img1 = read_image(image1_path)
            img2 = read_image(image2_path)
            if img1 is None or img2 is None:
                raise ValueError("image read failure")
            face1 = engine.detect_best_face(img1, source_path=image1_path)
            face2 = engine.detect_best_face(img2, source_path=image2_path)
            if face1 is None or face2 is None:
                raise ValueError("failed detection")
            if face1.normed_embedding is None or face2.normed_embedding is None:
                raise ValueError("embedding unavailable")
            similarity = cosine_similarity(face1.normed_embedding, face2.normed_embedding)
            item.update(
                {
                    "similarity": similarity,
                    "predicted": 1 if similarity >= threshold else 0,
                    "latency_ms": (time.perf_counter() - start) * 1000.0,
                }
            )
        except Exception as exc:
            item.update({"similarity": None, "predicted": None, "error": str(exc)})
            errors.append({"index": index, "error": str(exc), "row": dict(row)})
        rows.append(item)
        if progress_callback:
            progress_callback(index + 1, len(pairs), f"Processed pair {index + 1}/{len(pairs)}")

    completed = [row for row in rows if row.get("similarity") is not None]
    metrics = _metrics_at_threshold(completed, threshold)
    metrics.update(
        {
            "total_pairs": len(pairs),
            "positive_pairs": sum(1 for row in rows if row.get("label") == 1),
            "negative_pairs": sum(1 for row in rows if row.get("label") == 0),
            "failed_detections": len(errors),
            "TAR@FAR=1e-2": tar_at_far(completed, 1e-2),
            "TAR@FAR=1e-3": tar_at_far(completed, 1e-3),
            "TAR@FAR=1e-4": tar_at_far(completed, 1e-4),
            "average_latency_ms_per_pair": (
                float(np.mean([row["latency_ms"] for row in completed])) if completed else 0.0
            ),
        }
    )
    return EvaluationResult(
        scenario="KYC / 1:1 Verification",
        model_name=engine.model_name,
        provider=", ".join(engine.requested_providers),
        threshold=threshold,
        dataset_summary={
            "pairs_csv": str(pairs_csv),
            "total_pairs": len(pairs),
            "completed_pairs": len(completed),
        },
        metrics=metrics,
        errors=errors,
        latency={
            "total_elapsed_ms": (time.perf_counter() - start_all) * 1000.0,
            "hardware": hardware_info(),
        },
        license_status=license_status,
        created_at=utc_now_iso(),
        raw_results=rows,
        threshold_recommendation=recommend_threshold(completed),
    )


def run_identification_evaluation(
    gallery_folder: str | Path,
    probe_folder: str | Path,
    engine: FaceEngine,
    threshold: float = 0.5,
    ground_truth_csv: Optional[str | Path] = None,
    license_status: str = DEFAULT_LICENSE_STATUS,
    progress_callback=None,
    cancel_callback=None,
) -> EvaluationResult:
    gallery: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    gallery_root = Path(gallery_folder)
    for person_dir in sorted(path for path in gallery_root.iterdir() if path.is_dir()):
        for image_path in sorted(path for path in person_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS):
            try:
                img = read_image(image_path)
                if img is None:
                    raise ValueError("image read failure")
                face = engine.detect_best_face(img, source_path=str(image_path))
                if face is None or face.normed_embedding is None:
                    raise ValueError("no face or embedding")
                gallery.append(
                    {
                        "person_id": person_dir.name,
                        "person_name": person_dir.name,
                        "sample_id": len(gallery) + 1,
                        "crop_path": "",
                        "embedding": face.normed_embedding,
                    }
                )
            except Exception as exc:
                errors.append({"path": str(image_path), "error": str(exc), "stage": "gallery"})

    ground_truth: Dict[str, str] = {}
    if ground_truth_csv:
        with Path(ground_truth_csv).open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                ground_truth[row.get("image_path", "")] = row.get("person_name", "")

    probes = list_images(probe_folder, recursive=True)
    raw: List[Dict[str, Any]] = []
    start_all = time.perf_counter()
    for index, probe in enumerate(probes):
        if cancel_callback and cancel_callback():
            break
        try:
            img = read_image(probe)
            if img is None:
                raise ValueError("image read failure")
            face = engine.detect_best_face(img, source_path=str(probe))
            if face is None or face.normed_embedding is None:
                raise ValueError("no face or embedding")
            from .recognition import search_gallery

            results = search_gallery(face.normed_embedding, gallery, top_k=5, threshold=threshold)
            truth = ground_truth.get(str(probe), ground_truth.get(probe.name, ""))
            row = {
                "probe_path": str(probe),
                "ground_truth": truth,
                "top1": results[0].person_name if results else "Unknown",
                "top1_similarity": results[0].similarity if results else 0.0,
                "top5": [result.person_name for result in results],
                "accepted": bool(results and results[0].similarity >= threshold),
            }
            raw.append(row)
        except Exception as exc:
            errors.append({"path": str(probe), "error": str(exc), "stage": "probe"})
        if progress_callback:
            progress_callback(index + 1, len(probes), f"Processed probe {index + 1}/{len(probes)}")

    truth_rows = [row for row in raw if row.get("ground_truth")]
    top1 = sum(1 for row in truth_rows if row["top1"] == row["ground_truth"]) / max(1, len(truth_rows))
    top5 = sum(1 for row in truth_rows if row["ground_truth"] in row["top5"]) / max(1, len(truth_rows))
    metrics = {
        "gallery_persons": len({row["person_name"] for row in gallery}),
        "gallery_face_samples": len(gallery),
        "probe_images": len(probes),
        "detected_probe_faces": len(raw),
        "Top-1 accuracy": top1 if truth_rows else "ground truth not provided",
        "Top-5 accuracy": top5 if truth_rows else "ground truth not provided",
        "unknown_rejection_rate": sum(1 for row in raw if not row["accepted"]) / max(1, len(raw)),
        "average_search_latency_ms": 0.0,
    }
    return EvaluationResult(
        scenario="Access Control / 1:N Identification",
        model_name=engine.model_name,
        provider=", ".join(engine.requested_providers),
        threshold=threshold,
        dataset_summary={
            "gallery_folder": str(gallery_folder),
            "probe_folder": str(probe_folder),
            "ground_truth_csv": str(ground_truth_csv or ""),
        },
        metrics=metrics,
        errors=errors,
        latency={"total_elapsed_ms": (time.perf_counter() - start_all) * 1000.0, "hardware": hardware_info()},
        license_status=license_status,
        created_at=utc_now_iso(),
        raw_results=raw,
    )
