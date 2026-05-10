from __future__ import annotations

from config import TTS_VOICE

try:
    import edge_tts
except Exception:  # pragma: no cover - optional runtime dependency
    edge_tts = None


async def synthesize(text: str, voice: str = TTS_VOICE) -> bytes:
    if edge_tts is None:
        return b""
    communicate = edge_tts.Communicate(text, voice)
    chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)

