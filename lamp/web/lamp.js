import * as THREE from "three";

const els = {
  canvas: document.querySelector("#scene"),
  state: document.querySelector("#lamp-state"),
  engagement: document.querySelector("#engagement"),
  fps: document.querySelector("#fps"),
  memory: document.querySelector("#memory"),
  log: document.querySelector("#log"),
  chat: document.querySelector("#chat"),
  toast: document.querySelector("#toast"),
  video: document.querySelector("#cam"),
  capture: document.querySelector("#capture"),
  overlay: document.querySelector("#overlay"),
  detCount: document.querySelector("#det-count"),
  detections: document.querySelector("#detections"),
  audio: document.querySelector("#reply-audio"),
  ptt: document.querySelector("#ptt"),
  textForm: document.querySelector("#text-form"),
  textInput: document.querySelector("#text-input"),
  rememberForm: document.querySelector("#remember-form"),
  rememberLabel: document.querySelector("#remember-label"),
};

const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
ws.binaryType = "arraybuffer";

let cameraTimer = null;
let recorder = null;
let audioChunks = [];
let joints = [0, -30, 60, 20, 0, 0];
let targetJoints = joints.slice();
let lightTarget = { intensity: 0.5, color: "#ffffff" };
let lastDetections = [];
let lastEngagement = null;

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x10100e, 5, 12);

const renderer = new THREE.WebGLRenderer({ canvas: els.canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;

const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
camera.position.set(2.6, 2.0, 4.2);
camera.lookAt(0, 0.8, 0);

scene.add(new THREE.HemisphereLight(0xffefd0, 0x1a2430, 1.2));
const keyLight = new THREE.DirectionalLight(0xffd59a, 1.2);
keyLight.position.set(3, 4, 2);
keyLight.castShadow = true;
scene.add(keyLight);

const floor = new THREE.Mesh(
  new THREE.CircleGeometry(3.2, 80),
  new THREE.MeshStandardMaterial({ color: 0x2a2419, roughness: 0.88 })
);
floor.rotation.x = -Math.PI / 2;
floor.receiveShadow = true;
scene.add(floor);

const lamp = buildLamp();
scene.add(lamp.root);

function buildLamp() {
  const brass = new THREE.MeshStandardMaterial({ color: 0xcf8f2e, metalness: 0.15, roughness: 0.42 });
  const shade = new THREE.MeshStandardMaterial({ color: 0x33302a, metalness: 0.05, roughness: 0.58 });

  const root = new THREE.Group();
  const base = new THREE.Group();
  root.add(base);

  const baseMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.48, 0.56, 0.12, 48), brass);
  baseMesh.castShadow = true;
  baseMesh.receiveShadow = true;
  baseMesh.position.y = 0.06;
  base.add(baseMesh);

  const lower = new THREE.Group();
  lower.position.y = 0.12;
  base.add(lower);
  const lowerArm = armMesh(0.95, brass);
  lower.add(lowerArm);

  const upper = new THREE.Group();
  upper.position.y = 0.95;
  lower.add(upper);
  const upperArm = armMesh(0.78, brass);
  upper.add(upperArm);

  const head = new THREE.Group();
  head.position.y = 0.78;
  upper.add(head);
  const headMesh = new THREE.Mesh(new THREE.ConeGeometry(0.25, 0.45, 36, 1, true), shade);
  headMesh.rotation.x = Math.PI / 2;
  headMesh.position.z = 0.18;
  headMesh.castShadow = true;
  head.add(headMesh);

  const bulb = new THREE.Mesh(
    new THREE.SphereGeometry(0.08, 18, 18),
    new THREE.MeshStandardMaterial({ color: 0xffe1a1, emissive: 0xffc25a, emissiveIntensity: 1.4 })
  );
  bulb.position.z = 0.42;
  head.add(bulb);

  const spot = new THREE.SpotLight(0xffd28a, 1.8, 6, Math.PI / 7, 0.4, 1.2);
  spot.position.set(0, 0, 0.35);
  const target = new THREE.Object3D();
  target.position.set(0, -1.8, 2.6);
  head.add(spot);
  head.add(target);
  spot.target = target;

  return { root, base, lower, upper, head, spot, bulb };
}

function armMesh(length, material) {
  const group = new THREE.Group();
  const left = new THREE.Mesh(new THREE.BoxGeometry(0.055, length, 0.055), material);
  const right = left.clone();
  left.position.set(-0.07, length / 2, 0);
  right.position.set(0.07, length / 2, 0);
  left.castShadow = right.castShadow = true;
  group.add(left, right);
  return group;
}

function resize() {
  const rect = els.canvas.getBoundingClientRect();
  renderer.setSize(rect.width, rect.height, false);
  camera.aspect = rect.width / Math.max(1, rect.height);
  camera.updateProjectionMatrix();
}

function animate() {
  requestAnimationFrame(animate);
  resize();
  joints = joints.map((v, i) => v + (targetJoints[i] - v) * 0.18);
  applyJoints(joints);
  lamp.spot.intensity += (lightTarget.intensity * 2.2 - lamp.spot.intensity) * 0.18;
  lamp.spot.color.set(lightTarget.color);
  lamp.bulb.material.emissive.set(lightTarget.color);
  renderer.render(scene, camera);
}

function applyJoints(j) {
  lamp.base.rotation.y = deg(j[0]);
  lamp.lower.rotation.x = deg(j[1]);
  lamp.upper.rotation.x = deg(j[2]);
  lamp.head.rotation.y = deg(j[3]);
  lamp.head.rotation.x = deg(j[4]);
  lamp.head.rotation.z = deg(j[5]);
}

function deg(v) {
  return (v * Math.PI) / 180;
}

ws.addEventListener("open", () => log("WebSocket connected"));
ws.addEventListener("message", (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === "lamp_state") {
    targetJoints = msg.joints;
    lightTarget = msg.light;
    els.state.textContent = msg.state;
    if (msg.sound === "chirp") chirp();
  } else if (msg.type === "engagement") {
    lastEngagement = msg;
    const method = msg.method ? `/${msg.method}` : "";
    const pose = msg.detected ? ` yaw ${msg.yaw_deg} pitch ${msg.pitch_deg}` : "";
    const state = msg.detected ? (msg.engaged ? "gaze-locked" : "tracking-face") : "no-face";
    els.engagement.textContent = `${state}${method}${pose}`;
    els.fps.textContent = msg.fps ?? 0;
    drawDetections(lastDetections);
  } else if (msg.type === "memory_event") {
    const text = `${msg.action} ${msg.label} @ ${msg.zone}`;
    els.memory.textContent = text;
    toast(text);
  } else if (msg.type === "detections") {
    lastDetections = msg.items || [];
    drawDetections(lastDetections);
    renderDetectionList(lastDetections, msg);
  } else if (msg.type === "transcript") {
    addMsg("user", msg.text || "(no speech detected)");
  } else if (msg.type === "reply") {
    addMsg("lamp", `${msg.text} · ${msg.latency_ms}ms`);
    if (msg.tts_error) log(`tts fallback: ${msg.tts_error}`);
    if (msg.audio_b64) {
      els.audio.src = `data:audio/mp3;base64,${msg.audio_b64}`;
      els.audio.play().catch(() => {});
    }
  } else if (msg.type === "log") {
    log(`${msg.level}: ${msg.msg}`);
  }
});

document.querySelector("#start-camera").addEventListener("click", startCamera);
document.querySelector("#mock-engaged").addEventListener("click", () => send({ type: "mock_engagement", engaged: true }));
document.querySelector("#mock-away").addEventListener("click", () => send({ type: "mock_engagement", engaged: false }));
document.querySelector("#demo-wave").addEventListener("click", () => {
  unlockAudio();
  send({ type: "demo_wave" });
  chirp();
});
document.querySelector("#mock-cup").addEventListener("click", () => send({ type: "mock_observation", label: "cup" }));
els.textForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const content = els.textInput.value.trim();
  if (content) send({ type: "text_input", content });
  els.textInput.value = "";
});
els.rememberForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const label = els.rememberLabel.value.trim().toLowerCase();
  if (!label) return;
  const bbox = lastDetections[0]?.bbox || [0.35, 0.35, 0.65, 0.65];
  send({ type: "manual_observation", label, bbox });
  toast(`remembering visible object as ${label}`);
  els.rememberLabel.value = "";
});

els.ptt.addEventListener("pointerdown", startRecording);
els.ptt.addEventListener("pointerup", stopRecording);
window.addEventListener("keydown", (event) => {
  if (event.code === "Space" && document.activeElement !== els.textInput) {
    event.preventDefault();
    startRecording();
  }
});
window.addEventListener("keyup", (event) => {
  if (event.code === "Space") stopRecording();
});

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 640, height: 480, frameRate: 15 },
    audio: false,
  });
  els.video.srcObject = stream;
  await els.video.play();
  const ctx = els.capture.getContext("2d", { willReadFrequently: true });
  clearInterval(cameraTimer);
  cameraTimer = setInterval(() => {
    if (ws.readyState !== WebSocket.OPEN) return;
    ctx.drawImage(els.video, 0, 0, 640, 480);
    drawDetections(lastDetections);
    els.capture.toBlob((blob) => blob && sendBlob(blob, [0x01, 0x46, 0x52, 0x4d]), "image/jpeg", 0.6);
  }, 1000 / 15);
  log("Camera streaming");
}

async function startRecording() {
  if (recorder?.state === "recording") return;
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioChunks = [];
  recorder = new MediaRecorder(stream, { mimeType: "audio/webm;codecs=opus" });
  recorder.addEventListener("dataavailable", (event) => event.data.size && audioChunks.push(event.data));
  recorder.addEventListener("stop", () => {
    const blob = new Blob(audioChunks, { type: "audio/webm;codecs=opus" });
    sendBlob(blob, [0x02, 0x41, 0x55, 0x44]);
    stream.getTracks().forEach((track) => track.stop());
  });
  recorder.start();
  els.ptt.classList.add("recording");
}

function stopRecording() {
  if (recorder?.state === "recording") {
    recorder.stop();
    els.ptt.classList.remove("recording");
  }
}

function sendBlob(blob, prefix) {
  blob.arrayBuffer().then((buf) => {
    const out = new Uint8Array(prefix.length + buf.byteLength);
    out.set(prefix, 0);
    out.set(new Uint8Array(buf), prefix.length);
    ws.send(out);
  });
}

function send(payload) {
  if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(payload));
}

function addMsg(kind, text) {
  const div = document.createElement("div");
  div.className = `msg ${kind}`;
  div.textContent = text;
  els.chat.append(div);
  els.chat.scrollTop = els.chat.scrollHeight;
}

function log(text) {
  const div = document.createElement("div");
  div.textContent = text;
  els.log.prepend(div);
}

function toast(text) {
  els.toast.textContent = text;
  els.toast.classList.add("show");
  setTimeout(() => els.toast.classList.remove("show"), 2200);
}

function drawDetections(items) {
  const canvas = els.overlay;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 3;
  ctx.font = "18px Georgia, serif";
  ctx.textBaseline = "top";
  for (const item of items) {
    const [x1, y1, x2, y2] = item.bbox;
    const x = (1 - x2) * canvas.width;
    const y = y1 * canvas.height;
    const w = (x2 - x1) * canvas.width;
    const h = (y2 - y1) * canvas.height;
    const label = `${item.label} ${(item.conf * 100).toFixed(0)}%`;
    ctx.strokeStyle = "#e4aa42";
    ctx.fillStyle = "rgba(16, 16, 14, 0.72)";
    ctx.strokeRect(x, y, w, h);
    const textWidth = ctx.measureText(label).width + 12;
    ctx.fillRect(x, Math.max(0, y - 28), textWidth, 26);
    ctx.fillStyle = "#f6eddd";
    ctx.fillText(label, x + 6, Math.max(0, y - 25));
  }
  drawFaceOverlay(ctx, canvas);
}

function drawFaceOverlay(ctx, canvas) {
  if (!lastEngagement?.face_bbox) return;
  const [x1, y1, x2, y2] = lastEngagement.face_bbox;
  const x = (1 - x2) * canvas.width;
  const y = y1 * canvas.height;
  const w = (x2 - x1) * canvas.width;
  const h = (y2 - y1) * canvas.height;
  const label = `${lastEngagement.engaged ? "ENGAGED" : "FACE"} ${lastEngagement.method || ""}`;
  ctx.strokeStyle = lastEngagement.engaged ? "#78ff9f" : "#8fb7ff";
  ctx.fillStyle = "rgba(8, 18, 12, 0.76)";
  ctx.lineWidth = 3;
  ctx.strokeRect(x, y, w, h);
  const textWidth = ctx.measureText(label).width + 12;
  ctx.fillRect(x, y + h + 4, textWidth, 26);
  ctx.fillStyle = "#f6eddd";
  ctx.fillText(label, x + 6, y + h + 7);
}

function renderDetectionList(items, msg) {
  els.detCount.textContent = `${items.length} object${items.length === 1 ? "" : "s"}`;
  if (msg.error) {
    els.detections.textContent = `Detector fallback: ${msg.error}`;
    return;
  }
  if (!items.length) {
    const raw = msg.raw_count ?? 0;
    const people = msg.ignored_person_count ?? 0;
    const low = msg.low_conf_count ?? 0;
    els.detections.textContent = `No stored objects. YOLO saw ${raw} candidates; ignored ${people} person, ${low} low-confidence. Last pass: ${msg.latency_ms ?? 0}ms`;
    return;
  }
  els.detections.textContent = items
    .map((item) => `${item.label} @ ${item.zone} (${(item.conf * 100).toFixed(0)}%)`)
    .join(" · ") + " · Use Remember As to correct the first box.";
}

function chirp() {
  const ctx = getAudioContext();
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.frequency.setValueAtTime(620, ctx.currentTime);
  osc.frequency.exponentialRampToValueAtTime(980, ctx.currentTime + 0.12);
  gain.gain.setValueAtTime(0.0001, ctx.currentTime);
  gain.gain.exponentialRampToValueAtTime(0.12, ctx.currentTime + 0.02);
  gain.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.16);
  osc.connect(gain).connect(ctx.destination);
  osc.start();
  osc.stop(ctx.currentTime + 0.18);
}

let audioContext = null;

function getAudioContext() {
  audioContext ||= new AudioContext();
  return audioContext;
}

function unlockAudio() {
  const ctx = getAudioContext();
  if (ctx.state === "suspended") ctx.resume();
}

animate();
