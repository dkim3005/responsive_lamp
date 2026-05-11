from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np

from config import GAZE_H_THRESHOLD, GAZE_V_THRESHOLD

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

try:
    import cv2
except Exception:
    cv2 = None

try:
    import mediapipe as mp
    from mediapipe.tasks import python as _mp_python
    from mediapipe.tasks.python import vision as _mp_vision
    _TASKS_AVAILABLE = True
except Exception:
    mp = None
    _TASKS_AVAILABLE = False

_MODEL_PATH = str(Path(__file__).parent.parent / "face_landmarker.task")

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

_L_OUTER, _L_INNER, _L_TOP, _L_BOT, _L_IRIS = 33, 133, 159, 145, 468
_R_OUTER, _R_INNER, _R_TOP, _R_BOT, _R_IRIS = 263, 362, 386, 374, 473


def _iris_deviation(landmarks, iris_id, inner_id, outer_id, top_id, bot_id):
    """Return (horizontal, vertical) iris deviation from eye center, each in [-1, 1]."""
    iris = landmarks[iris_id]
    lx = min(landmarks[inner_id].x, landmarks[outer_id].x)
    rx = max(landmarks[inner_id].x, landmarks[outer_id].x)
    ty = landmarks[top_id].y
    by = landmarks[bot_id].y
    eye_w = rx - lx
    eye_h = abs(by - ty)
    if eye_w < 1e-5 or eye_h < 1e-5:
        return 0.0, 0.0
    dev_h = ((iris.x - lx) / eye_w - 0.5) * 2.0
    dev_v = ((iris.y - min(ty, by)) / eye_h - 0.5) * 2.0
    return dev_h, dev_v


class EngagementDetector:
    def __init__(self) -> None:
        self.available = cv2 is not None and _TASKS_AVAILABLE
        self._last_t = time.perf_counter()
        self._start_t = time.perf_counter()
        self.fps = 0.0
        self._landmarker = None
        self.face_cascade = None
        self.error: str | None = None

    def _ensure_landmarker(self) -> bool:
        if not self.available:
            self.error = "MediaPipe Tasks or OpenCV not available"
            return False
        if self._landmarker is not None:
            return True
        try:
            base_options = _mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
            options = _mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=_mp_vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.4,
                min_face_presence_confidence=0.4,
                min_tracking_confidence=0.4,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._landmarker = _mp_vision.FaceLandmarker.create_from_options(options)
            return True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return False

    def process(self, bgr: np.ndarray) -> dict:
        self._update_fps()
        if not self._ensure_landmarker():
            return self._process_haar(bgr, self.error)

        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((time.perf_counter() - self._start_t) * 1000)
            result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        except Exception as exc:
            return self._process_haar(bgr, str(exc))

        if not result.face_landmarks:
            return self._process_haar(bgr)

        landmarks = result.face_landmarks[0]

        # Head pose via solvePnP
        image_points = np.array(
            [(landmarks[i].x * w, landmarks[i].y * h) for i in LANDMARK_IDS],
            dtype=np.float64,
        )
        focal_length = w
        camera_matrix = np.array(
            [[focal_length, 0, w / 2], [0, focal_length, h / 2], [0, 0, 1]],
            dtype=np.float64,
        )
        dist_coeffs = np.zeros((4, 1))
        ok, rvec, _ = cv2.solvePnP(
            FACE_MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return self._empty("solvePnP failed")
        rot_mat, _ = cv2.Rodrigues(rvec)
        pitch, yaw, _roll = _rotation_matrix_to_euler(rot_mat)

        # Iris-based tracking position
        eye_cx = (landmarks[_L_IRIS].x + landmarks[_R_IRIS].x) / 2.0
        eye_cy = (landmarks[_L_IRIS].y + landmarks[_R_IRIS].y) / 2.0
        cx = (eye_cx - 0.5) * 2.0
        cy = (eye_cy - 0.5) * 2.0

        # Engagement: iris must be near center of each eye
        lh, lv = _iris_deviation(landmarks, _L_IRIS, _L_INNER, _L_OUTER, _L_TOP, _L_BOT)
        rh, rv = _iris_deviation(landmarks, _R_IRIS, _R_INNER, _R_OUTER, _R_TOP, _R_BOT)
        avg_h = (abs(lh) + abs(rh)) / 2.0
        avg_v = (abs(lv) + abs(rv)) / 2.0
        engaged_raw = avg_h < GAZE_H_THRESHOLD and avg_v < GAZE_V_THRESHOLD

        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        return {
            "detected": True,
            "yaw_deg": yaw,
            "pitch_deg": pitch,
            "gaze_h": round(avg_h, 3),
            "gaze_v": round(avg_v, 3),
            "face_xy": [max(-1.0, min(1.0, cx)), max(-1.0, min(1.0, cy))],
            "face_bbox": [max(0.0, min(xs)), max(0.0, min(ys)), min(1.0, max(xs)), min(1.0, max(ys))],
            "engaged_raw": engaged_raw,
            "fps": self.fps,
            "error": None,
            "method": "iris-gaze",
        }

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()

    def _empty(self, error: str | None = None) -> dict:
        return {
            "detected": False,
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "gaze_h": None,
            "gaze_v": None,
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
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(70, 70)
        )
        if len(faces) == 0:
            return self._empty(error)
        h, w = bgr.shape[:2]
        x, y, fw, fh = max(faces, key=lambda item: item[2] * item[3])
        cx = ((x + fw / 2) / w - 0.5) * 2.0
        cy = ((y + fh / 2) / h - 0.5) * 2.0
        return {
            "detected": True,
            "yaw_deg": cx * 45.0,
            "pitch_deg": cy * 30.0,
            "gaze_h": None,
            "gaze_v": None,
            "face_xy": [max(-1.0, min(1.0, cx)), max(-1.0, min(1.0, cy))],
            "face_bbox": [x / w, y / h, (x + fw) / w, (y + fh) / h],
            "engaged_raw": False,
            "fps": self.fps,
            "error": error or "face fallback only; gaze not confirmed",
            "method": "haar-face-only",
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
