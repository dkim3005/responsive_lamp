from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from behavior.fsm import LampFSM
from config import DB_PATH, LAMP_TICK_HZ, OBJECT_DETECT_HZ
from memory.store import MemoryStore, zone_for_bbox
from perception.engagement import EngagementDetector
from perception.scene import SceneDetector
from voice.agent import Agent
from voice.stt import STT
from voice import tts

try:
    import cv2
except Exception:  # pragma: no cover - optional runtime dependency
    cv2 = None

try:
    from pydub import AudioSegment
except Exception:  # pragma: no cover - optional runtime dependency
    AudioSegment = None


FRAME_MAGIC = b"\x01FRM"
AUDIO_MAGIC = b"\x02AUD"

app = FastAPI(title="LeLamp")
app.mount("/static", StaticFiles(directory="lamp/web"), name="static")

store = MemoryStore(DB_PATH)
engagement = EngagementDetector()
scene = SceneDetector()
fsm = LampFSM()
stt = STT()
agent = Agent(store)

frame_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=2)
connections: set[WebSocket] = set()
last_engaged_raw = False
last_face_xy: list[float] | None = None
last_object_detect_t = 0.0
processor_started = False


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("lamp/web/index.html")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    ensure_frame_processor()
    await ws.accept()
    connections.add(ws)
    behavior_task = asyncio.create_task(behavior_loop(ws))
    await send_json(
        ws,
        {
            "type": "log",
            "level": "info",
            "msg": f"Connected. mediapipe={engagement.available} yolo={scene.available} stt={stt.available}",
        },
    )
    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                await handle_binary(msg["bytes"], ws)
            elif "text" in msg and msg["text"] is not None:
                await handle_text(msg["text"], ws)
    except WebSocketDisconnect:
        pass
    finally:
        connections.discard(ws)
        behavior_task.cancel()


def ensure_frame_processor() -> None:
    global processor_started
    if not processor_started:
        asyncio.create_task(frame_processor())
        processor_started = True


async def handle_text(raw: str, ws: WebSocket) -> None:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        await send_json(ws, {"type": "log", "level": "error", "msg": "Invalid JSON"})
        return

    if data.get("type") == "text_input":
        asyncio.create_task(handle_voice_text(data.get("content", ""), ws))
    elif data.get("type") == "mock_engagement":
        await set_mock_engagement(bool(data.get("engaged")), ws)
    elif data.get("type") == "mock_observation":
        label = str(data.get("label") or "cup").strip().lower()
        bbox = data.get("bbox") or [0.66, 0.1, 0.92, 0.35]
        zone = zone_for_bbox(bbox)
        action = store.upsert_observation(label, bbox, zone, 0.99, time.time())
        await broadcast({"type": "memory_event", "label": label, "zone": zone, "action": action, "conf": 0.99})
    elif data.get("type") == "ping":
        await send_json(ws, {"type": "pong", "ts": time.time()})


async def handle_binary(data: bytes, ws: WebSocket) -> None:
    if data.startswith(FRAME_MAGIC):
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
        await frame_queue.put(data[4:])
    elif data.startswith(AUDIO_MAGIC):
        asyncio.create_task(handle_voice_audio(data[4:], ws))


async def frame_processor() -> None:
    global last_engaged_raw, last_face_xy, last_object_detect_t
    while True:
        jpeg = await frame_queue.get()
        if cv2 is None:
            await broadcast({"type": "log", "level": "error", "msg": "OpenCV is not installed; frame ignored"})
            continue
        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            continue

        t0 = time.perf_counter()
        eng = engagement.process(bgr)
        store.log_latency("engagement_loop", (time.perf_counter() - t0) * 1000)
        last_engaged_raw = bool(eng["engaged_raw"])
        last_face_xy = eng["face_xy"]
        await broadcast(
            {
                "type": "engagement",
                "engaged": last_engaged_raw,
                "detected": eng["detected"],
                "face_xy": last_face_xy,
                "fps": round(float(eng.get("fps") or 0.0), 1),
                "error": eng.get("error"),
            }
        )

        now = time.time()
        if now - last_object_detect_t >= 1.0 / OBJECT_DETECT_HZ:
            last_object_detect_t = now
            t1 = time.perf_counter()
            detections = scene.detect(bgr)
            store.log_latency("object_detect", (time.perf_counter() - t1) * 1000)
            for det in detections:
                action = store.upsert_observation(det["label"], det["bbox"], det["zone"], det["conf"], now)
                await broadcast({"type": "memory_event", **det, "action": action})


async def behavior_loop(ws: WebSocket) -> None:
    t0 = time.time()
    while True:
        cmd = fsm.tick(time.time() - t0, last_engaged_raw, last_face_xy)
        await send_json(ws, {"type": "lamp_state", **cmd})
        await asyncio.sleep(1.0 / LAMP_TICK_HZ)


async def set_mock_engagement(engaged: bool, ws: WebSocket) -> None:
    global last_engaged_raw, last_face_xy
    last_engaged_raw = engaged
    last_face_xy = [0.0, 0.0] if engaged else [0.75, 0.0]
    await send_json(
        ws,
        {
            "type": "engagement",
            "engaged": last_engaged_raw,
            "detected": True,
            "face_xy": last_face_xy,
            "fps": 0,
            "error": "mock",
        },
    )


async def handle_voice_audio(audio_bytes: bytes, ws: WebSocket) -> None:
    t0 = time.perf_counter()
    wav = webm_to_wav16k(audio_bytes)
    if not wav:
        await send_json(ws, {"type": "log", "level": "error", "msg": "Audio conversion unavailable; use text input"})
        return
    text, stt_ms = stt.transcribe(wav)
    store.log_latency("stt", stt_ms)
    await send_json(ws, {"type": "transcript", "text": text, "latency_ms": round(stt_ms, 1)})
    await finish_reply(text or "Where is my cup?", ws, t0)


async def handle_voice_text(text: str, ws: WebSocket) -> None:
    t0 = time.perf_counter()
    await send_json(ws, {"type": "transcript", "text": text, "latency_ms": 0})
    await finish_reply(text, ws, t0)


async def finish_reply(text: str, ws: WebSocket, t0: float) -> None:
    llm_t = time.perf_counter()
    reply, llm_ms = agent.chat(text)
    store.log_latency("llm", llm_ms)
    tts_t = time.perf_counter()
    audio = await tts.synthesize(reply)
    tts_ms = (time.perf_counter() - tts_t) * 1000
    store.log_latency("tts", tts_ms)
    e2e_ms = (time.perf_counter() - t0) * 1000
    store.log_latency("e2e_voice", e2e_ms)
    await send_json(
        ws,
        {
            "type": "reply",
            "text": reply,
            "audio_b64": base64.b64encode(audio).decode() if audio else None,
            "latency_ms": round(e2e_ms, 1),
            "llm_ms": round(llm_ms, 1),
            "tts_ms": round(tts_ms, 1),
            "tts_error": tts.last_error,
        },
    )


def webm_to_wav16k(audio_bytes: bytes) -> bytes:
    if AudioSegment is None:
        return b""
    with tempfile.NamedTemporaryFile(suffix=".webm") as src, tempfile.NamedTemporaryFile(suffix=".wav") as dst:
        src.write(audio_bytes)
        src.flush()
        audio = AudioSegment.from_file(src.name)
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(dst.name, format="wav")
        return Path(dst.name).read_bytes()


async def broadcast(payload: dict) -> None:
    dead: list[WebSocket] = []
    for ws in list(connections):
        try:
            await ws.send_json(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        connections.discard(ws)


async def send_json(ws: WebSocket, payload: dict) -> None:
    try:
        await ws.send_json(payload)
    except Exception:
        connections.discard(ws)
