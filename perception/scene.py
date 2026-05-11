from __future__ import annotations

import os

import numpy as np

from config import OBJECT_CONF_THRESHOLD, YOLO_IMAGE_SIZE, YOLO_MODEL
from memory.store import zone_for_bbox

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional runtime dependency
    YOLO = None


class SceneDetector:
    def __init__(self) -> None:
        self.available = YOLO is not None
        self.model = None
        self.names = {}
        self.error: str | None = None
        self.last_raw_count = 0
        self.last_person_count = 0
        self.last_low_conf_count = 0

    def detect(self, bgr: np.ndarray) -> list[dict]:
        self.error = None
        self.last_raw_count = 0
        self.last_person_count = 0
        self.last_low_conf_count = 0
        if not self._ensure_model():
            return []

        h, w = bgr.shape[:2]
        results = self.model.predict(bgr, imgsz=YOLO_IMAGE_SIZE, verbose=False, device="cpu")
        detections: list[dict] = []
        self.last_raw_count = len(results[0].boxes)
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:
                self.last_person_count += 1
                continue
            conf = float(box.conf[0])
            if conf < OBJECT_CONF_THRESHOLD:
                self.last_low_conf_count += 1
                continue
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
            bbox = [
                max(0.0, min(1.0, x1 / w)),
                max(0.0, min(1.0, y1 / h)),
                max(0.0, min(1.0, x2 / w)),
                max(0.0, min(1.0, y2 / h)),
            ]
            detections.append(
                {
                    "label": str(self.names.get(cls_id, cls_id)),
                    "bbox": bbox,
                    "zone": zone_for_bbox(bbox),
                    "conf": conf,
                }
            )
        return detections

    def _ensure_model(self) -> bool:
        if not self.available:
            self.error = "ultralytics not installed"
            return False
        if self.model is not None:
            return True
        try:
            self.model = YOLO(YOLO_MODEL)
            self.model.to("cpu")
            self.names = self.model.names
            return True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return False
