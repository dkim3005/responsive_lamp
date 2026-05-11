import os

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency during static analysis
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "on"}

CAMERA_FPS = 15
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
MIRROR_CAMERA_PREVIEW = True
CAMERA_CONTROL_FLIP_Y = True
CAMERA_OBJECT_CONTROL_FLIP_Y = _env_bool("CAMERA_OBJECT_CONTROL_FLIP_Y", False)
GAZE_H_THRESHOLD = 0.40
GAZE_V_THRESHOLD = 0.45
ENGAGEMENT_ON_HOLD_SEC = 0.4
ENGAGEMENT_OFF_HOLD_SEC = 1.0
IDLE_AFTER_NO_FACE_SEC = 4.0
SEEK_DELAY_SEC = [3.0, 7.0, 12.0]
ENGAGEMENT_SAMPLE_HZ = 2.0
OBJECT_DETECT_HZ = 1.0
OBJECT_CONF_THRESHOLD = 0.20
OBJECT_IOU_DUP_THRESHOLD = 0.4
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
YOLO_IMAGE_SIZE = int(os.getenv("YOLO_IMAGE_SIZE", "640"))
LAMP_TICK_HZ = 20
WHISPER_MODEL = "tiny.en"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AriaNeural")
ENABLE_EDGE_TTS = _env_bool("ENABLE_EDGE_TTS", False)
DB_PATH = os.getenv("DB_PATH", "memory.db")
HISTORY_MAX_MESSAGES = 20
