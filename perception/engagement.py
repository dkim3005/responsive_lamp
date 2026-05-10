from __future__ import annotations

import math
import os
import time

import numpy as np

from config import ENGAGEMENT_PITCH_THRESHOLD_DEG, ENGAGEMENT_YAW_THRESHOLD_DEG

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

try:
    import cv2
    import mediapipe as mp
except Exception:  # pragma: no cover - optional runtime dependency
    cv2 = None
    mp = None


FACE_MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)
LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


class EngagementDetector:
    def __init__(self) -> None:
        self.available = cv2 is not None and mp is not None
        self.cv_available = cv2 is not None
        self._last_t = time.perf_counter()
        self.fps = 0.0
        self.mesh = None
        self.face_cascade = None
        self.error: str | None = None

    def _ensure_mesh(self) -> bool:
        if not self.available:
            self.error = "MediaPipe/OpenCV not installed"
            return False
        if self.mesh is not None:
            return True
        try:
            self.mesh = mp.solutions.face_mesh.FaceMesh(
                refine_landmarks=True,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return False

    def process(self, bgr: np.ndarray) -> dict:
        self._update_fps()
        if not self._ensure_mesh():
            return self._process_haar(bgr, self.error)

        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = self.mesh.process(rgb)
        if not results.multi_face_landmarks:
            return self._process_haar(bgr)

        landmarks = results.multi_face_landmarks[0].landmark
        image_points = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in LANDMARK_IDS], dtype=np.float64)
        focal_length = w
        camera_matrix = np.array([[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        ok, rvec, _ = cv2.solvePnP(
            FACE_MODEL_POINTS,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return self._empty("solvePnP failed")

        rot_mat, _ = cv2.Rodrigues(rvec)
        pitch, yaw, _roll = _rotation_matrix_to_euler(rot_mat)
        xs = [p.x for p in landmarks]
        ys = [p.y for p in landmarks]
        cx = ((min(xs) + max(xs)) / 2.0 - 0.5) * 2.0
        cy = ((min(ys) + max(ys)) / 2.0 - 0.5) * 2.0
        engaged_raw = abs(yaw) < ENGAGEMENT_YAW_THRESHOLD_DEG and abs(pitch) < ENGAGEMENT_PITCH_THRESHOLD_DEG
        return {
            "detected": True,
            "yaw_deg": yaw,
            "pitch_deg": pitch,
            "face_xy": [max(-1.0, min(1.0, cx)), max(-1.0, min(1.0, cy))],
            "face_bbox": [max(0.0, min(xs)), max(0.0, min(ys)), min(1.0, max(xs)), min(1.0, max(ys))],
            "engaged_raw": engaged_raw,
            "fps": self.fps,
            "error": None,
            "method": "facemesh",
        }

    def close(self) -> None:
        if self.mesh is not None:
            self.mesh.close()

    def _empty(self, error: str | None = None) -> dict:
        return {
            "detected": False,
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "face_xy": None,
            "face_bbox": None,
            "engaged_raw": False,
            "fps": self.fps,
            "error": error,
            "method": "none",
        }

    def _update_fps(self) -> None:
        now = time.perf_counter()
        dt = now - self._last_t
        self._last_t = now
        if dt > 0:
            instant = 1.0 / dt
            self.fps = instant if self.fps == 0 else self.fps * 0.85 + instant * 0.15

    def _process_haar(self, bgr: np.ndarray, error: str | None = None) -> dict:
        if cv2 is None:
            return self._empty(error or "OpenCV not installed")
        if self.face_cascade is None:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(70, 70))
        if len(faces) == 0:
            return self._empty(error)
        h, w = bgr.shape[:2]
        x, y, fw, fh = max(faces, key=lambda item: item[2] * item[3])
        cx = ((x + fw / 2) / w - 0.5) * 2.0
        cy = ((y + fh / 2) / h - 0.5) * 2.0
        centered = abs(cx) < 0.45 and abs(cy) < 0.45
        return {
            "detected": True,
            "yaw_deg": cx * 45.0,
            "pitch_deg": cy * 30.0,
            "face_xy": [max(-1.0, min(1.0, cx)), max(-1.0, min(1.0, cy))],
            "face_bbox": [x / w, y / h, (x + fw) / w, (y + fh) / h],
            "engaged_raw": centered,
            "fps": self.fps,
            "error": error,
            "method": "haar",
        }


def _rotation_matrix_to_euler(r: np.ndarray) -> tuple[float, float, float]:
    sy = math.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(r[2, 1], r[2, 2])
        y = math.atan2(-r[2, 0], sy)
        z = math.atan2(r[1, 0], r[0, 0])
    else:
        x = math.atan2(-r[1, 2], r[1, 1])
        y = math.atan2(-r[2, 0], sy)
        z = 0.0
    return math.degrees(x), math.degrees(y), math.degrees(z)
