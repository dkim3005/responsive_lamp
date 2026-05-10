from __future__ import annotations

import asyncio
import os

from config import ENABLE_EDGE_TTS, TTS_VOICE

try:
    import edge_tts
except Exception:  # pragma: no cover - optional runtime dependency
    edge_tts = None

last_error: str | None = None


async def synthesize(text: str, voice: str = TTS_VOICE) -> bytes:
    global last_error
    last_error = None
    enabled = os.getenv("ENABLE_EDGE_TTS", "1" if ENABLE_EDGE_TTS else "0").lower() in {"1", "true", "yes"}
    if not enabled:
        last_error = "edge-tts disabled; text reply only"
        return b""
    if edge_tts is None:
        last_error = "edge-tts not installed"
        return b""
    try:
        return await asyncio.wait_for(_synthesize_edge(text, voice), timeout=6.0)
    except asyncio.TimeoutError:
        last_error = "edge-tts timed out"
        return b""
    except Exception as exc:
        last_error = f"{type(exc).__name__}: {exc}"
        return b""


async def _synthesize_edge(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(text, voice)
    chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)
