# LeLamp Implementation Spec
> 다른 에이전트(Codex)가 이 문서만 보고 끝까지 구현 가능하도록 작성된 명세서.
> 모든 결정은 사전에 잠겨있다 — TBD 항목 없음. 의문점이 있으면 이 문서의 정의를 따른다.

---

## A. 환경 및 전제

### A.1 환경
- OS: WSL2 (Linux 5.15) on Windows
- 실행 위치: 백엔드는 WSL2 안, 프론트엔드 브라우저는 **Windows의 Chrome**에서 `http://localhost:8000` 접속
- Python: **3.11** (mediapipe/faster-whisper 호환)
- 노드: 불필요 (프론트는 정적 HTML/JS, CDN으로 Three.js 로드)

### A.2 핵심 아키텍처 결정 (WSL2 우회)
**문제**: WSL2에서 USB 웹캠/마이크 직접 접근은 usbipd 셋업이 필요하고 불안정.
**해결**: 브라우저(Windows측 Chrome)가 `getUserMedia`로 카메라·마이크를 캡처해 WebSocket으로 백엔드(WSL2)에 전송. 백엔드는 ML만 담당.

```
[Windows Chrome]                     [WSL2 Python backend]
  webcam ──┐                            ┌── MediaPipe (engagement)
  mic    ──┤                            ├── YOLOv8n (objects)
  display ─┤◄── WebSocket(JSON+binary)──┤── faster-whisper (STT)
  Three.js │                            ├── OpenAI gpt-4o-mini (agent)
  speaker ─┘                            └── edge-tts (TTS)
                                            ↕
                                        SQLite (memory)
```

### A.3 자원 가정
- 노트북: i5-8세대, 통합 그래픽, 8GB+ RAM 가정
- 폰(S20 Ultra): 1일 일정에서는 **사용 안 함**. 시간 남으면 보너스로 IP Webcam 추가.
- LLM: OpenAI API key, 잔액 $9.79. 모델 `gpt-4o-mini`.

---

## B. 시스템 아키텍처

### B.1 프로세스 구성
단일 Python 프로세스 (`server.py`), FastAPI + Uvicorn. 내부적으로 asyncio 태스크 분리:
- `frame_processor_task`: 들어오는 카메라 프레임을 큐에서 꺼내 MediaPipe + YOLO 처리
- `behavior_task`: FSM tick (50ms 주기), 램프 명령 생성 → 브라우저로 push
- `voice_handler_task`: 오디오 청크 → STT → LLM → TTS → 브라우저로 push

### B.2 WebSocket 메시지 프로토콜
단일 엔드포인트 `/ws`, 메시지는 JSON (텍스트) 또는 binary (이미지/오디오).

**Client → Server**

| 형식 | 내용 |
|------|------|
| binary (BLOB) | 첫 4바이트 magic: `0x01 'F' 'R' 'M'` → JPEG 카메라 프레임. 나머지는 JPEG bytes |
| binary (BLOB) | 첫 4바이트 magic: `0x02 'A' 'U' 'D'` → 발화 끝난 직후 webm/opus 오디오. 나머지는 audio bytes |
| JSON | `{"type":"text_input","content":"..."}` (마이크 안 될 때 백업 입력) |
| JSON | `{"type":"ping"}` |

**Server → Client (모두 JSON)**

| type | 페이로드 | 의미 |
|------|---------|------|
| `lamp_state` | `{joints:[j1..j6], light:{intensity:0..1, color:"#rrggbb"}, sound:null|"chirp"|"hum"}` | 50ms마다 |
| `engagement` | `{engaged:bool, face_xy:[x,y]|null, fps:float}` | 프레임 처리 후 |
| `memory_event` | `{label:str, zone:str, action:"insert"|"update"}` | DB 변동 시 |
| `transcript` | `{text:str, latency_ms:float}` | STT 끝 |
| `reply` | `{text:str, audio_b64:str|null, latency_ms:float}` | LLM+TTS 끝 |
| `log` | `{level:"info"|"error", msg:str}` | 디버그 |

### B.3 카메라 프레임 사양
- 해상도: 640×480, JPEG quality 0.6
- 전송률: 15 fps (브라우저에서 throttle)
- 큐 사이즈: 2 (오래된 프레임 drop)

### B.4 오디오 사양
- 포맷: `audio/webm;codecs=opus`, 모노, 16kHz 권장
- 입력 방식: **Push-to-talk 버튼** (스페이스바 또는 마우스). 누르고 있는 동안 녹음, 떼면 전송.
- 백엔드에서 `pydub`로 WAV 16kHz mono PCM 변환 → faster-whisper 입력.

---

## C. 데이터베이스 스키마

`memory.db` (SQLite, 자동 생성)

```sql
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT NOT NULL,
    bbox TEXT NOT NULL,           -- "x1,y1,x2,y2" normalized 0..1
    zone TEXT NOT NULL,           -- "top-left"|"top-center"|...|"bottom-right"
    confidence REAL NOT NULL,
    first_seen REAL NOT NULL,     -- unix timestamp (seconds)
    last_seen REAL NOT NULL,
    count INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_label ON memory(label);
CREATE INDEX IF NOT EXISTS idx_last_seen ON memory(last_seen);

CREATE TABLE IF NOT EXISTS latency_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    metric TEXT NOT NULL,         -- "engagement_loop"|"object_detect"|"stt"|"llm"|"tts"|"e2e_voice"
    value_ms REAL NOT NULL,
    ts REAL NOT NULL
);
```

### C.1 zone 매핑 규칙
프레임을 3×3으로 분할. bbox 중심점이 속한 셀로 결정.
- col: x_center < 1/3 → "left", < 2/3 → "center", else "right"
- row: y_center < 1/3 → "top", < 2/3 → "middle", else "bottom"
- 합치면 `"top-left"`, `"middle-center"` 등 (총 9개)

### C.2 upsert 규칙
같은 라벨 + 직전 60초 이내 + IoU(저장된 bbox, 새 bbox) ≥ 0.4 → UPDATE (`last_seen=now`, `count+=1`, `bbox`는 EMA: `0.7*old + 0.3*new`)
아니면 INSERT.

---

## D. 폴더 구조 (모든 파일 명시)

```
lelamp/
├── .env.example              # OPENAI_API_KEY 템플릿
├── requirements.txt
├── README.md                 # = Writeup
├── server.py                 # FastAPI 엔트리, WebSocket 허브
├── config.py                 # 모든 상수/임계값
├── perception/
│   ├── __init__.py
│   ├── engagement.py         # MediaPipe 기반
│   └── scene.py              # YOLOv8 + zone 매핑
├── behavior/
│   ├── __init__.py
│   ├── fsm.py                # 상태 기계
│   └── motions.py            # 사전정의 모션 키프레임
├── lamp/
│   ├── __init__.py
│   ├── ik.py                 # CCD IK (3-joint position)
│   └── web/
│       ├── index.html
│       ├── lamp.js           # Three.js 씬 + 통신
│       └── style.css
├── memory/
│   ├── __init__.py
│   └── store.py              # SQLite 래퍼
├── voice/
│   ├── __init__.py
│   ├── stt.py                # faster-whisper
│   ├── tts.py                # edge-tts
│   └── agent.py              # OpenAI gpt-4o-mini + tools
├── eval/
│   ├── label_engagement.py   # 라벨링 보조 스크립트
│   └── report.py             # latency_log → CSV/표
└── demo/
    └── recording_script.md   # 영상 촬영 시나리오
```

---

## E. 의존성 (requirements.txt)

```
fastapi==0.115.6
uvicorn[standard]==0.32.1
python-multipart==0.0.20
numpy==1.26.4
opencv-python==4.10.0.84
mediapipe==0.10.18
ultralytics==8.3.40
faster-whisper==1.1.0
edge-tts==7.0.0
pydub==0.25.1
openai==1.58.1
python-dotenv==1.0.1
websockets==14.1
pillow==11.0.0
```

시스템 패키지 (apt): `ffmpeg` (pydub용), `libgl1` (opencv), `libglib2.0-0`.

---

## F. 모듈 명세

### F.1 `config.py`
```python
# 모두 여기서 import. 매직 넘버 금지.
CAMERA_FPS = 15
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
ENGAGEMENT_YAW_THRESHOLD_DEG = 20.0
ENGAGEMENT_PITCH_THRESHOLD_DEG = 15.0
ENGAGEMENT_ON_HOLD_SEC = 0.5    # this many sec of "looking" before engaged=True
ENGAGEMENT_OFF_HOLD_SEC = 1.5
SEEK_DELAY_SEC = [5.0, 10.0, 15.0]   # SEEKING_1/2/3 진입 임계
OBJECT_DETECT_HZ = 1.0
OBJECT_CONF_THRESHOLD = 0.4
OBJECT_IOU_DUP_THRESHOLD = 0.4
LAMP_TICK_HZ = 20               # 50ms
WHISPER_MODEL = "tiny.en"       # CPU 친화
OPENAI_MODEL = "gpt-4o-mini"
TTS_VOICE = "en-US-AriaNeural"
DB_PATH = "memory.db"
LAMP_LINK_LENGTHS = [0.0, 1.0, 0.8, 0.3]   # base, lower, upper, head
```

### F.2 `perception/engagement.py`
**입력**: BGR numpy 이미지 (640x480x3).
**출력**: dict `{detected:bool, yaw_deg:float, pitch_deg:float, face_xy:[float,float]|None, engaged_raw:bool}`.

**알고리즘**:
1. MediaPipe FaceMesh (`refine_landmarks=True`, `max_num_faces=1`).
2. 얼굴 미검출 → 모두 None/False.
3. 6개 키 랜드마크 (코끝 1, 턱 152, 좌안 외측 33, 우안 외측 263, 좌입꼬리 61, 우입꼬리 291)의 2D 픽셀 좌표 추출.
4. 표준 3D 모델 좌표 (mm 단위, 일반적 얼굴 모델)와 `cv2.solvePnP(SOLVEPNP_ITERATIVE)` → rvec.
5. `cv2.Rodrigues` → 회전 행렬 → euler (yaw, pitch, roll).
6. `engaged_raw = abs(yaw) < ENGAGEMENT_YAW_THRESHOLD_DEG and abs(pitch) < ENGAGEMENT_PITCH_THRESHOLD_DEG`.
7. `face_xy`: 얼굴 bbox 중심을 [-1,1] 정규화 (가운데=0,0; 우=+1).

**히스테리시스**는 호출 측(FSM)에서 처리.

**클래스**:
```python
class EngagementDetector:
    def __init__(self): ...
    def process(self, bgr: np.ndarray) -> dict: ...
    def close(self): ...
```

### F.3 `perception/scene.py`
**입력**: BGR 이미지.
**출력**: `list[dict]` — 각 객체 `{label:str, bbox:[x1,y1,x2,y2] normalized, zone:str, conf:float}`.

**알고리즘**:
1. `ultralytics.YOLO("yolov8n.pt")` 1회 로드 (전역).
2. inference → `results[0].boxes`.
3. `cls`가 0(person)인 detection은 제외.
4. `conf >= OBJECT_CONF_THRESHOLD`만.
5. 각 bbox 중심 → zone 계산.

**클래스**:
```python
class SceneDetector:
    def __init__(self): self.model = YOLO("yolov8n.pt")
    def detect(self, bgr) -> list[dict]: ...
```

호출 빈도는 server.py가 `OBJECT_DETECT_HZ`로 throttle.

### F.4 `behavior/fsm.py`
**상태 enum**: `IDLE, ENGAGED, DISENGAGED, SEEKING_1, SEEKING_2, SEEKING_3, OBSERVING`.

**입력 (tick마다)**: `engaged_raw:bool, face_xy:[x,y]|None, t:float`.
**출력**: `{state:str, joints:[6 floats], light:{...}, sound:str|None}`.

**전이 규칙**:
- 어느 상태든 `engaged_raw==True` 0.5초 연속 → `ENGAGED`.
- `ENGAGED`에서 `engaged_raw==False` 1.5초 연속 → `DISENGAGED`.
- `DISENGAGED`에서 5초 → `SEEKING_1`, +5초 → `SEEKING_2`, +5초 → `SEEKING_3`. 각 SEEKING은 모션 끝나면 다시 `DISENGAGED`로.
- 얼굴 자체가 사라진 상태가 3초 지속 → `IDLE`.

**각 상태의 출력**:
- `IDLE`: 기본 자세 (joints=[0, -30°, 60°, 20°, 0, 0]), light intensity=0.5 흰색.
- `ENGAGED`: head를 face_xy 향해 회전 (`head_yaw = -face_x * 30°`, `head_pitch = -face_y * 20°` + 기본 pitch). light intensity=1.0 따뜻한 노랑(#ffd28a).
- `DISENGAGED`: head를 5° 떨굼, light dim 0.4.
- `SEEKING_1`: 2초간 head_yaw에 sin(2πt) × 8° 추가.
- `SEEKING_2`: 3초간 light intensity 0.4↔1.0 sine 펄스.
- `SEEKING_3`: 1초 chirp 사운드 + base_yaw에 sin(πt)×15° 큰 모션.
- 30초마다 비-SEEKING 상태에서 `OBSERVING` 2초 (base_yaw 좌우 천천히 스캔).

**클래스**:
```python
class LampFSM:
    def __init__(self): ...
    def tick(self, t: float, engaged_raw: bool, face_xy) -> dict: ...
```

### F.5 `behavior/motions.py`
사전정의 모션 함수 모음. 각 함수는 `(elapsed:float) -> joint_offsets[6]`.

```python
def head_wiggle(elapsed: float) -> list[float]: ...
def base_scan(elapsed: float) -> list[float]: ...
# etc.
```

FSM이 상태별로 호출.

### F.6 `lamp/ik.py`
**역할**: 머리(head) 목표 위치 (x,y,z) → 3개 조인트(base_yaw, lower_pitch, upper_pitch) 각도 계산.

**알고리즘**: CCD (Cyclic Coordinate Descent), 50줄 이내.
- 링크: `[L0=0, L1=1.0, L2=0.8, L3=0.3]`
- 반복 10회, 끝점이 목표에 가까워질 때까지 마지막 조인트부터 회전.

```python
def solve_ik(target_xyz: tuple) -> tuple[float,float,float]:
    """Returns (base_yaw_deg, lower_pitch_deg, upper_pitch_deg)."""
```

이건 **백엔드에서 호출하지 않고 프론트(lamp.js)에서 구현**해도 OK. 1일 일정에서는 프론트에서 직접 IK 계산하는 게 라운드트립 적음. → **프론트 lamp.js에 IK 포함**, 백엔드는 head 목표 좌표만 보낸다.

→ `lamp/ik.py`는 결국 **프론트엔드 JS로 구현하고 이 파일은 삭제**. 폴더 구조에서 제외.

### F.7 `memory/store.py`
```python
class MemoryStore:
    def __init__(self, db_path: str = DB_PATH): ...
    def upsert_observation(self, label: str, bbox: tuple, zone: str, conf: float, ts: float) -> str:
        """Returns 'insert' or 'update'."""
    def query(self, object_name: str, limit: int = 5) -> list[dict]:
        """Fuzzy match (LIKE %name%), order by last_seen DESC."""
    def log_latency(self, metric: str, value_ms: float): ...
    def close(self): ...
```

### F.8 `voice/stt.py`
```python
from faster_whisper import WhisperModel

class STT:
    def __init__(self, model_name=WHISPER_MODEL):
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
    def transcribe(self, wav_bytes: bytes) -> tuple[str, float]:
        """Returns (text, latency_ms)."""
```
- 입력: WAV 16kHz mono bytes.
- pydub로 webm/opus → WAV 변환은 server.py에서 사전 처리.

### F.9 `voice/tts.py`
```python
import edge_tts

async def synthesize(text: str, voice=TTS_VOICE) -> bytes:
    """Returns MP3 bytes."""
    communicate = edge_tts.Communicate(text, voice)
    chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)
```
브라우저에 base64로 보내 `<audio>` 재생.

### F.10 `voice/agent.py`
```python
SYSTEM_PROMPT = """You are a small, curious desk lamp with a warm friendly personality.
You can see what's happening around you and remember objects you've noticed.
Reply in 1 to 2 short sentences only — you're a lamp, not a chatbot.
When the user asks about an object you might have seen, ALWAYS call query_memory first.
If memory returns nothing, say honestly that you haven't noticed it.
Use casual, warm tone. Don't be overly enthusiastic."""

TOOLS = [{
    "type": "function",
    "function": {
        "name": "query_memory",
        "description": "Search the lamp's visual memory for an object it has seen. Returns location and timestamps.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_name": {"type": "string", "description": "COCO class name like 'cup', 'book', 'laptop', 'bottle'."}
            },
            "required": ["object_name"]
        }
    }
}]

class Agent:
    def __init__(self, store: MemoryStore):
        self.client = OpenAI()
        self.store = store
        self.history = [{"role":"system", "content":SYSTEM_PROMPT}]

    def chat(self, user_text: str) -> tuple[str, float]:
        """Returns (assistant_reply, latency_ms). Handles tool calls in a loop."""
```

**Tool call 처리 루프**:
1. `chat.completions.create(model, messages=history+[user], tools=TOOLS)`.
2. `tool_calls` 있으면 `query_memory(object_name)` 실행 → SQL 결과를 자연어로 포맷:
   `"Found: cup at top-right (last seen 2 min ago, 12 frames)"` 또는 `"No record."`.
3. tool message 포함시켜 다시 호출.
4. 최종 assistant 메시지를 반환. history에 추가.
5. history 길이가 20을 넘으면 system 빼고 앞쪽 자르기.

### F.11 `server.py`
FastAPI 앱. 라우트:
- `GET /` → `lamp/web/index.html` (StaticFiles)
- `GET /static/*` → `lamp/web/`
- `WS /ws` → 메인 WebSocket 핸들러

**의사코드**:
```python
app = FastAPI()
app.mount("/static", StaticFiles(directory="lamp/web"))
engagement = EngagementDetector()
scene = SceneDetector()
fsm = LampFSM()
store = MemoryStore()
stt = STT()
agent = Agent(store)
frame_queue = asyncio.Queue(maxsize=2)
last_object_detect_ts = 0
last_engaged = False
last_face_xy = None

@app.websocket("/ws")
async def ws(websocket):
    await websocket.accept()
    asyncio.create_task(behavior_loop(websocket))
    while True:
        msg = await websocket.receive()
        if "bytes" in msg:
            handle_binary(msg["bytes"], websocket)
        elif "text" in msg:
            data = json.loads(msg["text"])
            if data["type"] == "text_input":
                asyncio.create_task(handle_voice_input_text(data["content"], websocket))

def handle_binary(data, ws):
    if data[:4] == b"\x01FRM":
        # JPEG frame
        if frame_queue.full(): frame_queue.get_nowait()
        frame_queue.put_nowait(data[4:])
    elif data[:4] == b"\x02AUD":
        asyncio.create_task(handle_voice_input(data[4:], ws))

async def frame_processor():
    global last_engaged, last_face_xy, last_object_detect_ts
    while True:
        jpeg = await frame_queue.get()
        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        # engagement
        eng = engagement.process(bgr)
        last_engaged = eng["engaged_raw"]
        last_face_xy = eng["face_xy"]
        await broadcast({"type":"engagement", "engaged":last_engaged, "face_xy":last_face_xy, "fps":...})
        # object detect throttled
        now = time.time()
        if now - last_object_detect_ts > 1.0/OBJECT_DETECT_HZ:
            last_object_detect_ts = now
            for det in scene.detect(bgr):
                action = store.upsert_observation(det["label"], det["bbox"], det["zone"], det["conf"], now)
                await broadcast({"type":"memory_event", "label":det["label"], "zone":det["zone"], "action":action})

async def behavior_loop(ws):
    t0 = time.time()
    while True:
        t = time.time() - t0
        cmd = fsm.tick(t, last_engaged, last_face_xy)
        await ws.send_json({"type":"lamp_state", **cmd})
        await asyncio.sleep(1.0/LAMP_TICK_HZ)

async def handle_voice_input(audio_bytes, ws):
    t0 = time.time()
    wav = webm_to_wav16k(audio_bytes)  # pydub
    text, stt_ms = stt.transcribe(wav)
    await ws.send_json({"type":"transcript", "text":text, "latency_ms":stt_ms})
    reply, llm_ms = agent.chat(text)
    mp3 = await synthesize(reply)
    audio_b64 = base64.b64encode(mp3).decode()
    e2e_ms = (time.time()-t0)*1000
    store.log_latency("e2e_voice", e2e_ms)
    await ws.send_json({"type":"reply", "text":reply, "audio_b64":audio_b64, "latency_ms":e2e_ms})
```

(전체 구현은 단계 G에서)

### F.12 `lamp/web/index.html`
- `<canvas>` (Three.js 렌더 타겟)
- `<video id="cam" hidden autoplay>` (getUserMedia)
- `<canvas id="capture" hidden>` (frame 캡처용)
- `<button id="ptt">Hold to Talk (Space)</button>`
- `<div id="status">` (engaged 상태, FPS, memory event 토스트)
- `<audio id="reply">` (TTS 재생)
- `<script type="importmap">` Three.js CDN
- `<script src="lamp.js" type="module">`

### F.13 `lamp/web/lamp.js`
**구성요소**:

1. **Three.js 씬**
   - 카메라(perspective), 평면 ground, ambient + directional light, lamp 메쉬
   - lamp 메쉬: 4개 box+cylinder 조합으로 Pixar lamp (base, lower_arm, upper_arm, head)
   - 그룹 계층: `base → lower_arm → upper_arm → head`
   - head에 `THREE.SpotLight` 자식으로 부착, 색/세기는 lamp_state로 갱신

2. **CCD IK** (50줄):
   ```js
   function solveIK(targetXYZ, jointAngles, linkLengths) {
     // base_yaw, lower_pitch, upper_pitch, fixed link lengths
     // CCD 10 iters
     return [base_yaw, lower_pitch, upper_pitch];
   }
   ```
   백엔드가 보낸 joints가 head_xy_target 형식이면 IK 적용. 단순히 6개 각도면 그대로 적용. → 백엔드에서 6개 각도 직접 보내기로 결정. IK는 프론트가 ENGAGED 상태에서 face_xy 추적할 때만 옵션으로 사용. **1일 일정 단순화: 백엔드 FSM이 모든 6개 각도 계산. 프론트는 그대로 적용.** IK는 v2.

3. **카메라 캡처 루프**
   ```js
   const stream = await navigator.mediaDevices.getUserMedia({video:{width:640,height:480,frameRate:15}, audio:false});
   video.srcObject = stream;
   setInterval(() => {
     ctx.drawImage(video, 0, 0, 640, 480);
     captureCanvas.toBlob(b => sendFrame(b), 'image/jpeg', 0.6);
   }, 1000/15);

   function sendFrame(blob) {
     blob.arrayBuffer().then(buf => {
       const out = new Uint8Array(4 + buf.byteLength);
       out[0]=0x01; out[1]=0x46; out[2]=0x52; out[3]=0x4D; // "FRM"
       out.set(new Uint8Array(buf), 4);
       ws.send(out);
     });
   }
   ```

4. **Push-to-talk**
   - 별도 `getUserMedia({audio:true})` 스트림.
   - `MediaRecorder(stream, {mimeType:'audio/webm;codecs=opus'})`.
   - 스페이스바 keydown → start, keyup → stop. ondataavailable에 모은 blob을 magic prefix `0x02 'AUD'` 붙여 ws.send.

5. **WebSocket 메시지 처리**
   - `lamp_state`: 조인트/광원/사운드 적용
   - `engagement`: 상태바 업데이트
   - `memory_event`: 토스트 (3초 자동 사라짐)
   - `transcript`: 채팅 영역에 사용자 텍스트 추가
   - `reply`: 채팅 영역에 lamp 텍스트 추가, audio_b64 → `<audio>` 재생

### F.14 `lamp/web/style.css`
- 다크 테마, 채팅 영역 우측, 캔버스 좌측, 토스트 우상단.

---

## G. 구현 순서 (Codex가 따라갈 단계별 가이드)

각 단계는 **(a) 만들/수정할 파일**, **(b) 코드 명세 또는 의사코드**, **(c) 검증 방법** 으로 구성.

### Step 1. 프로젝트 셋업 (목표 30분)

**(a) 파일**:
- `requirements.txt` (E절 그대로)
- `.env.example` (`OPENAI_API_KEY=sk-...`)
- `config.py` (F.1 그대로)
- `lelamp/` 폴더 구조 (D절) — 빈 `__init__.py` 포함

**(b) 명령**:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
sudo apt-get install -y ffmpeg libgl1 libglib2.0-0
cp .env.example .env  # 사용자가 키 채움
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"  # 가중치 다운로드
```

**(c) 검증**: `python -c "import mediapipe, ultralytics, faster_whisper, openai, edge_tts; print('OK')"` 출력.

### Step 2. 최소 FastAPI + 정적 페이지 (목표 30분)

**(a) 파일**:
- `server.py` 최소판: FastAPI, `/`로 index.html 서빙, `/ws` 더미 echo
- `lamp/web/index.html`, `style.css`, `lamp.js` 최소판 (검은 배경 + "Hello Lamp")

**(b) 의사코드**:
```python
# server.py
app = FastAPI()
app.mount("/static", StaticFiles(directory="lamp/web"), name="static")
@app.get("/")
def index(): return FileResponse("lamp/web/index.html")
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    while True:
        m = await websocket.receive_text()
        await websocket.send_text(f"echo:{m}")
```
실행: `uvicorn server:app --reload --host 0.0.0.0 --port 8000`

**(c) 검증**: Windows Chrome `http://localhost:8000` → 텍스트 보임. JS 콘솔에서 `new WebSocket("ws://localhost:8000/ws")` echo 동작.

### Step 3. Three.js 램프 메쉬 (목표 1시간)

**(a) 파일**: `lamp.js` 본격 작성.

**(b) 명세**:
- 씬 setup, OrbitControls
- 6-DOF 램프 그룹 계층:
  ```
  baseGroup (rotation.y = base_yaw)
    └ baseMesh (CylinderGeometry r=0.3 h=0.1)
    └ lowerArmGroup (rotation.x = lower_pitch, position=[0,0.05,0])
        └ lowerArmMesh (BoxGeometry 0.1x1.0x0.1, pivot bottom)
        └ upperArmGroup (rotation.x = upper_pitch, position=[0,1.0,0])
            └ upperArmMesh (BoxGeometry 0.1x0.8x0.1, pivot bottom)
            └ headGroup (rotation.x = head_pitch, rotation.y = head_yaw, rotation.z = head_roll, position=[0,0.8,0])
                └ headMesh (ConeGeometry r=0.2 h=0.3)
                └ spotLight (THREE.SpotLight, target가 head 앞 1m)
  ```
- 슬라이더 6개로 각 조인트 수동 조작 (디버그용, 나중에 숨김)

**(c) 검증**: 슬라이더 움직이면 램프가 움직임. 광원 색/세기 변경 가능.

### Step 4. 백엔드 FSM 골격 + 정해진 모션 (목표 1시간)

**(a) 파일**: `behavior/fsm.py`, `behavior/motions.py`, `server.py`의 `behavior_loop` 추가.

**(b) 명세**: F.4의 LampFSM 구현. 처음에는 `engaged_raw`를 키보드 입력 등으로 mock — 실제 카메라는 다음 스텝.

**(c) 검증**: 페이지에서 mock 토글 버튼으로 engaged 변경 → 램프가 반응. 5/10/15초 SEEKING 진입 확인.

### Step 5. 카메라 프레임 전송 + Engagement (목표 1.5시간)

**(a) 파일**: `lamp.js`에 캡처 루프, `perception/engagement.py`, `server.py`의 `frame_processor` + `handle_binary`.

**(b) 명세**: F.2 + F.11의 frame_processor. 프레임이 큐에 쌓이면 처리, engagement 결과를 `last_engaged`에 저장 → FSM이 사용.

**(c) 검증**:
- 사용자가 카메라 응시 → engaged=True 토스트, 램프가 사용자 향해 회전.
- 시선 옆으로 → 1.5초 후 disengaged.
- 5초 후 head wiggle 발동.

### Step 6. 객체 검출 + 메모리 (목표 1.5시간)

**(a) 파일**: `perception/scene.py`, `memory/store.py`, `server.py`의 객체 검출 throttle 코드.

**(b) 명세**: F.3 + F.7. 1Hz로 YOLO, person 제외, upsert. memory_event 메시지 push.

**(c) 검증**:
- 책상 위 머그컵 카메라에 노출 → 토스트 "saw cup at top-right" 1초 내 출력.
- `sqlite3 memory.db "SELECT * FROM memory"` → 행 존재.

### Step 7. STT + LLM + TTS 풀 루프 (목표 2시간)

**(a) 파일**: `voice/stt.py`, `voice/tts.py`, `voice/agent.py`, `lamp.js`의 PTT, `server.py`의 voice 핸들러.

**(b) 명세**:
- PTT: 스페이스바 누르고 있을 때 녹음 → 떼면 webm blob 전송.
- 백엔드 webm → WAV (pydub). faster-whisper 변환.
- agent.chat(text) → tool call 실행 → 최종 텍스트.
- edge-tts로 MP3, base64 인코딩 → 브라우저 `<audio>` src에 data URL.

**(c) 검증**:
- "Hi" → 짧은 인사 답변.
- 머그컵을 본 후 "Where's my cup?" → tool call → "I saw it at top-right a moment ago" 류 답변.
- 모르는 객체 "Where's the elephant?" → "I haven't seen one" 류.
- 브라우저에서 음성 들림.

### Step 8. 라텐시 측정 + 간이 evaluation (목표 1시간)

**(a) 파일**: `eval/report.py`, `latency_log` 호출 코드 추가.

**(b) 명세**:
- 모든 핵심 단계 (`engagement_loop`, `object_detect`, `stt`, `llm`, `tts`, `e2e_voice`)에 `time.perf_counter` → `store.log_latency(metric, ms)`.
- `report.py`: SQLite 읽어 metric별 mean/p50/p95 표 출력 (Markdown).
- engagement evaluation: 30초 동안 OBS로 화면 녹화하면서 직접 정해진 시나리오 (look-15s / away-15s) 수행. 프레임별 engaged 신호와 ground truth 비교 → confusion matrix. 스크립트 `eval/label_engagement.py`가 도움.

**(c) 검증**: README에 표 들어갈 수치 확보.

### Step 9. 데모 시나리오 + 영상 (목표 1시간)

**(a) 파일**: `demo/recording_script.md`.

**(b) 시나리오 (대본)**:
1. (0–10s) 사용자 카메라 응시. 램프가 사용자 향해 회전, 광원 켜짐. "Engaged" 토스트.
2. (10–30s) 사용자 옆을 봄. 5초 후 head wiggle, 10초 후 light pulse, 15초 후 chirp + 큰 모션.
3. (30–60s) 사용자가 책상에 머그컵, 책, 노트북 등을 차례로 카메라에 노출. memory toast가 뜬다.
4. (60–90s) 사용자가 PTT 버튼 누르고 "Hey lamp, where did you see my mug?" → 답변 음성.

녹화: OBS Studio (무료)로 화면+마이크 녹화, MP4로 출력.

**(c) 검증**: 1080p MP4, 60–120초.

### Step 10. README/Writeup (목표 30분)

**(a) 파일**: `README.md`.

**(b) 섹션**:
1. Overview (1단락)
2. Architecture diagram (이 문서의 ASCII 그대로 또는 Excalidraw PNG)
3. Data flow (high-level: 위 다이어그램 / low-level: WebSocket 메시지 표 from B.2)
4. Design decisions & tradeoffs (이 문서의 6절 표 그대로)
5. How to run (3줄 명령)
6. Evaluation results (Step 8 산출물)
7. Limitations & future work (가상 램프, 단일 카메라, 한국어 X 등)

---

## H. 실행 명령 (최종)

```bash
# 한 번만
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
sudo apt-get install -y ffmpeg libgl1 libglib2.0-0
cp .env.example .env  # OPENAI_API_KEY 채우기

# 매번
source .venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8000
# Windows Chrome에서 http://localhost:8000 열기, 카메라/마이크 권한 허용
```

---

## I. 디자인 트레이드오프 (Writeup용)

| 선택 | 대안 | 이유 |
|------|------|------|
| 가상 램프 (Three.js) | 물리 6-DOF 하드웨어 | 자원 제약. 챌린지 본질은 perception-to-action 파이프라인 |
| 브라우저 미디어 캡처 | WSL2 직접 USB 접근 | usbipd 셋업 회피, 어디서든 동작 |
| 노트북 웹캠 1대 (engagement+scene 동시) | 폰을 두번째 카메라 | 1일 일정. 폰은 v2 |
| MediaPipe FaceMesh + solvePnP | L2CS-Net, 전용 시선추적 | CPU 실시간 가능, 정확도 충분 |
| YOLOv8n | YOLOv8s/DETR | i5 CPU에서 1Hz 안정 |
| 라벨 + IoU 매칭 | CLIP 임베딩 시맨틱 검색 | 1일 일정상 단순화. v2에서 추가 |
| OpenAI gpt-4o-mini | Claude / 로컬 LLM | 사용자 잔액 활용, function calling 안정, 비용 $0.10 미만 |
| edge-tts | pyttsx3, OpenAI TTS | 무료, 자연스러운 음질, 비동기 |
| faster-whisper tiny.en | OpenAI Whisper API | 무료/오프라인, CPU 1초 내 |
| SQLite | Postgres/Chroma | 단일 프로세스, 의존성 0 |
| 백엔드 FSM이 6 조인트 직접 계산 | 프론트 IK | 라운드트립 단순. IK는 v2 |

---

## J. 위험 및 폴백

| 위험 | 폴백 |
|------|------|
| MediaPipe solvePnP가 흔들림 | yaw/pitch에 EMA(α=0.3) 적용 |
| YOLO이 i5에서 너무 느림 (>1초) | 0.5Hz로 throttle 또는 imgsz=320 |
| WSL2에서 ffmpeg 없음 | apt 설치, 또는 Whisper 입력을 raw float32 PCM으로 (브라우저에서 직접 디코드) |
| edge-tts 네트워크 차단 | pyttsx3 + espeak (`apt install espeak-ng`) |
| OpenAI API 장애 | 로컬 답변: 단순 룰 베이스 ("Found X at zone") 사용 |
| 마이크 권한 안 됨 | 텍스트 입력창 추가 (`text_input` 메시지) |
| 카메라 권한 안 됨 | 데모용 정적 비디오 파일 재생 — `<video src="demo.mp4" loop>`로 대체 |

---

## K. Codex가 시작할 때 읽을 순서

1. 이 문서 A, B, C 절 (환경/아키텍처/스키마)
2. F.1 `config.py` 작성 — 모든 상수가 잠긴다
3. Step 1부터 순차 진행. 각 Step의 검증 통과해야 다음으로.
4. 막히면 J절(폴백) 적용
5. 끝나면 Step 9, 10

이 문서 자체를 수정하지 말고 그대로 따라간다. 결정이 필요한 새 사안만 사용자에게 묻는다.
