# LeLamp: Expressive 6-DOF Lamp Agent

LeLamp is a browser-driven, Python-backed demo of a social lamp that sees when a user is engaged, reacts with expressive motion/light/sound, remembers visible objects, and answers questions from that memory.

## Architecture

```text
Windows/Chrome browser                         Python/FastAPI backend
webcam + mic ── getUserMedia ── WebSocket ──►  MediaPipe engagement
Three.js 6-DOF lamp ◄── lamp_state JSON ─────  YOLO scene detection
speaker ◄── TTS audio/reply JSON ────────────  SQLite visual memory
text/PTT ───────────────────────────────────►  STT + LLM/tool fallback
```

The browser owns media capture to avoid fragile WSL2 USB webcam/microphone setup. The backend owns perception, memory, behavior state, and conversation orchestration.

## Data Flow

- Camera frames are sent as binary WebSocket messages with `0x01FRM` + JPEG bytes.
- Audio utterances are sent as `0x02AUD` + WebM/Opus bytes after push-to-talk ends.
- The backend emits `lamp_state`, `engagement`, `memory_event`, `transcript`, `reply`, and `log` JSON messages.
- Object memory is stored in SQLite with `label`, normalized `bbox`, `zone`, confidence, first/last seen timestamps, and observation count.

## Design Decisions

| Choice | Alternative | Reason |
|---|---|---|
| Browser media capture | Direct WSL2 USB access | More reliable setup and demo portability |
| Three.js actuator | Physical 6-DOF hardware | Focuses the challenge on perception-to-action architecture |
| MediaPipe FaceMesh | Heavy gaze model | CPU-friendly real-time engagement signal |
| YOLOv8n at low frequency | Larger detector | Keeps object memory from blocking interaction |
| SQLite memory | Vector DB/Postgres | Simple, inspectable, enough for object-location recall |
| Rule-based fallback | Hard dependency on LLM/STT/TTS | Demo still proves memory recall under network/API failure |

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

If optional ML/audio packages or API keys are unavailable, use the built-in mock buttons and text input. The core FSM, memory, and grounded recall path still work.

## Demo Checklist

- Engagement detection: user looks at the lamp, lamp brightens and tracks.
- Attention seeking: user disengages, lamp escalates from wiggle to light pulse to chirp.
- Memory formation: object detection or mock memory stores `cup @ top-right`.
- Memory recall: user asks `Where is my cup?`, lamp answers from SQLite memory with location and latency.

## Evaluation

Run after a demo session:

```bash
python eval/report.py
```

The report prints latency mean/p50/p95 for `engagement_loop`, `object_detect`, `stt`, `llm`, `tts`, and `e2e_voice`.

For engagement reliability, create a CSV with `truth,pred` columns and run:

```bash
python eval/label_engagement.py engagement_labels.csv
```

## Limitations

This submission uses a simulated 6-DOF lamp rather than hardware. The actuator boundary is intentionally narrow: the backend emits joint angles, light, and sound commands that can be mapped to physical servos later. Multi-user identity, emotion detection, and learned personalization are natural next steps.

