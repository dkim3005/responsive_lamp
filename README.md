# LeLamp

A virtual 6-DOF desk lamp that watches you back. It tracks your gaze using iris landmarks, escalates through attention-seeking motions when you look away, and announces objects it spots in the scene. Questions about what it has seen are answered through an LLM grounded in a local visual memory store.

---

## Features

**Gaze tracking.** The lamp follows your eyes in real time using MediaPipe FaceMesh iris landmarks — not head pose. This means it stays locked on you even when you tilt your head, and correctly recognises when you are looking at the screen versus just facing toward it.

**Attention-seeking behaviour.** If you look away, the lamp waits a few seconds before reacting. It then escalates through three stages: a gentle head nod, a full arm wave with a puppy-tilt, and finally a full-body animation where all six joints oscillate at different frequencies. Each stage triggers a spoken prompt. Re-engaging at any point resets the cycle.

**Object memory.** The camera feed is scanned once per second. Any new object that appears in frame is logged to a local SQLite database with its position (described as a 3×3 zone grid: top-left, center-right, etc.) and the lamp physically turns toward it. Ask *"Where is my cup?"* and the LLM queries that database to give a grounded answer.

---

## Setup

Python 3.11 or 3.12 is recommended. The checked-in environment has been tested with Python 3.12 and `mediapipe==0.10.18`.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Add your OpenAI API key to `.env`:

```
OPENAI_API_KEY=sk-...
ENABLE_EDGE_TTS=0
```

`ENABLE_EDGE_TTS=0` is the safer demo default because Microsoft's Edge-TTS endpoint can return 403. The browser still speaks replies with the Web Speech API fallback.

Start the server:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000` in Chrome on the same machine and click **Start Camera**.

---

## Usage

The interface has two inputs: push-to-talk (hold the button or hold Space) and a text field. Both feed into the same conversation pipeline — STT transcription for voice, direct text otherwise.

Example queries the lamp can answer:
- *"Where is my cup?"*
- *"Have you seen my phone?"*
- *"What did you notice a moment ago?"*

The lamp stores everything it detects across the session. Objects seen more recently rank higher in retrieval.

---

## Architecture

```
Chrome (Windows)                    Python backend (WSL2)
  webcam  ──► 0x01FRM + JPEG ────►  MediaPipe iris gaze
  mic     ──► 0x02AUD + WebM  ────►  faster-whisper STT
  Three.js ◄── lamp_state JSON ───  FSM (6-DOF joints + light)
  speaker  ◄── reply + audio  ───   GPT-4o-mini + edge-tts
                                          ↕
                                     SQLite (object log)
```

The browser owns media capture because WSL2 USB device access is unreliable. The backend handles all perception and state logic and pushes commands to the frontend over a single WebSocket connection.

---

## Evaluation

After a session, print latency and engagement reliability statistics from the database:

```bash
python eval/report.py
```

This outputs mean, p50, and p95 for engagement processing, object detection, STT, LLM, TTS, and end-to-end voice latency.

Engagement predictions are sampled automatically while the camera is running. Reliability still needs ground truth: during a demo, press **E** when you are actually looking at the lamp and **D** when you are looking away. The report then computes accuracy, precision, recall, F1, detection rate, and a confusion matrix from those labels.

In the recorded validation run, the system sampled 619 engagement predictions and 14 manual ground-truth checkpoints. The labeled checkpoints produced 1.00 accuracy, 1.00 precision, 1.00 recall, 1.00 F1, and a 1.00 face detection rate:

| truth/pred | engaged | disengaged |
|---|---:|---:|
| engaged | 7 | 0 |
| disengaged | 0 | 7 |


---

## Limitations

**Object detection accuracy.** The system uses YOLOv8n, the smallest model in the YOLOv8 family. It is fast enough to run at 1 Hz on a mid-range CPU, but it is noticeably weaker on partially occluded objects, items at the edges of the frame, and anything outside the 80 COCO classes. A cup behind a laptop, a wallet, or a set of keys will often go undetected. Setting `YOLO_MODEL=yolov8s.pt` improves recall at the cost of roughly 3× inference time.

**Camera coordinate calibration.** Face tracking and object-pointing use separate coordinate transforms because the mirrored preview and the 3D lamp joint model can need different vertical directions. If object-pointing looks vertically inverted on another machine, set `CAMERA_OBJECT_CONTROL_FLIP_Y=1`.

**No physical hardware.** Joint angles, light colour, and sound commands are sent to a Three.js renderer rather than actual servos. The command schema is hardware-agnostic — adding a servo driver layer is the only integration work needed.

**Single-user only.** The engagement model assumes one face in frame. A second person entering the scene will confuse the iris gaze calculation, and the FSM has no concept of speaker identity.

**Lighting sensitivity.** MediaPipe iris detection degrades under low light or strong backlight. In these conditions the system falls back to a Haar cascade face detector, which can only report face position and cannot confirm eye contact.

**TTS latency.** Edge-TTS depends on Microsoft's speech endpoint. Responses typically arrive within one to two seconds, but cold starts and network latency can push this higher. The browser Web Speech API is used as a fallback if the request times out.
