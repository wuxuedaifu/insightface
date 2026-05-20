"""Face swap model wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class FaceSwapEngine:
    def __init__(self, model_path: str = "", providers: Optional[list[str]] = None):
        self.model_path = model_path
        self.providers = providers or ["CPUExecutionProvider"]
        self.model = None
        self.last_error = ""

    def is_available(self) -> bool:
        return self.model is not None

    def load(self) -> bool:
        path = Path(self.model_path).expanduser()
        if not path.exists():
            self.last_error = (
                "Face swap model not found. Please configure a valid swap model in Models."
            )
            return False
        try:
            import onnxruntime
            from insightface.model_zoo import get_model

            onnxruntime.set_default_logger_severity(3)
            self.model = get_model(str(path), providers=self.providers)
            if self.model is None:
                self.last_error = "Configured face swap model could not be recognized."
                return False
            return True
        except Exception as exc:
            self.last_error = f"Face swap model load failed: {exc}"
            return False

    def swap(self, image: np.ndarray, target_face, source_face) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Face swap model not found. Please configure a valid swap model in Models.")
        return self.model.get(image, target_face, source_face, paste_back=True)
