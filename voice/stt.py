from __future__ import annotations

import tempfile
import time

from config import WHISPER_MODEL

try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - optional runtime dependency
    WhisperModel = None


class STT:
    def __init__(self, model_name: str = WHISPER_MODEL) -> None:
        self.available = WhisperModel is not None
        self.model_name = model_name
        self.model = None
        self.error: str | None = None

    def transcribe(self, wav_bytes: bytes) -> tuple[str, float]:
        t0 = time.perf_counter()
        if not self._ensure_model():
            return "", (time.perf_counter() - t0) * 1000
        with tempfile.NamedTemporaryFile(suffix=".wav") as fh:
            fh.write(wav_bytes)
            fh.flush()
            segments, _info = self.model.transcribe(fh.name, beam_size=1)
            text = " ".join(segment.text.strip() for segment in segments).strip()
        return text, (time.perf_counter() - t0) * 1000

    def _ensure_model(self) -> bool:
        if not self.available:
            self.error = "faster-whisper not installed"
            return False
        if self.model is not None:
            return True
        try:
            self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
            return True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return False
