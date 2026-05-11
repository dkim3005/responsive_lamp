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
  dofValues: document.querySelectorAll("[data-joint]"),
  evalStatus: document.querySelector("#eval-status"),
};

const ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
ws.binaryType = "arraybuffer";

let cameraTimer = null;
let recorder = null;
let audioChunks = [];
let authed = false;
let joints = [0, -30, 60, 0, -25, 0];

const authForm = document.querySelector("#auth-form");
const authInput = document.querySelector("#auth-input");
const authError = document.querySelector("#auth-error");
const llmLock = document.querySelector("#llm-lock");
const llmContent = document.querySelector("#llm-content");

function applyAuthUI() {
  if (authed) {
    llmLock.hidden = true;
    llmContent.hidden = false;
  } else {
    llmLock.hidden = false;
    llmContent.hidden = true;
    authInput.focus();
  }
}
applyAuthUI();

authForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const pw = authInput.value.trim();
  if (!pw) return;
  const msg = JSON.stringify({ type: "auth", password: pw });
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(msg);
  } else {
    ws.addEventListener("open", () => ws.send(msg), { once: true });
  }
});
let targetJoints = joints.slice();
let lightTarget = { intensity: 0.5, color: "#ffffff" };
let lastDetections = [];
let lastEngagement = null;

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x10100e, 6, 16);

const renderer = new THREE.WebGLRenderer({ canvas: els.canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;

const camera = new THREE.PerspectiveCamera(55, 1, 0.1, 100);
camera.position.set(0, 2.5, 7.5);
camera.lookAt(0, 1.2, 0);

scene.add(new THREE.HemisphereLight(0xffefd0, 0x1a2430, 1.4));
const keyLight = new THREE.DirectionalLight(0xffd59a, 1.8);
keyLight.position.set(3, 5, 3);
keyLight.castShadow = true;
scene.add(keyLight);

const floor = new THREE.Mesh(
  new THREE.CircleGeometry(4.4, 96),
  new THREE.MeshStandardMaterial({ color: 0x241e14, roughness: 0.86, metalness: 0.08 })
);
floor.rotation.x = -Math.PI / 2;
floor.receiveShadow = true;
scene.add(floor);

const lamp = buildLamp();
scene.add(lamp.root);

function buildLamp() {
  const brass = new THREE.MeshStandardMaterial({ color: 0xd79635, metalness: 0.35, roughness: 0.32 });
  const dark = new THREE.MeshStandardMaterial({ color: 0x171410, metalness: 0.42, roughness: 0.36 });
  const shade = new THREE.MeshStandardMaterial({ color: 0x2f2a22, metalness: 0.18, roughness: 0.46 });
  const jointMat = new THREE.MeshStandardMaterial({ color: 0xffc35a, emissive: 0x6a3500, emissiveIntensity: 0.18 });


  const root = new THREE.Group();
  root.scale.setScalar(1.32);
  root.position.y = -0.08;

  const base = new THREE.Group();
  root.add(base);

  const baseMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.68, 0.78, 0.18, 72), brass);
  baseMesh.castShadow = true;
  baseMesh.receiveShadow = true;
  baseMesh.position.y = 0.09;
  base.add(baseMesh);
  const baseCap = new THREE.Mesh(new THREE.CylinderGeometry(0.42, 0.48, 0.12, 72), dark);
  baseCap.position.y = 0.2;
  baseCap.castShadow = true;
  base.add(baseCap);
  addDofMarker(base, "J1 Base Yaw", 0xff6b4a, [0.92, 0.22, 0], "y");

  const lower = new THREE.Group();
  lower.position.y = 0.18;
  base.add(lower);
  lower.add(jointSphere(0.18, jointMat));
  addDofMarker(lower, "J2 Lower Pitch", 0xffb000, [-0.42, 0.05, 0], "x");
  const lowerArm = armMesh(1.25, brass, dark);
  lower.add(lowerArm);

  const upper = new THREE.Group();
  upper.position.y = 1.25;
  lower.add(upper);
  upper.add(jointSphere(0.16, jointMat));
  addDofMarker(upper, "J3 Upper Pitch", 0x68d391, [0.42, 0.04, 0], "x");
  const upperArm = armMesh(1.05, brass, dark);
  upper.add(upperArm);

  const headYaw = new THREE.Group();
  headYaw.position.y = 1.05;
  upper.add(headYaw);
  headYaw.add(jointSphere(0.17, jointMat));
  addDofMarker(headYaw, "J4 Head Yaw", 0x45caff, [0.5, 0.12, 0], "y");

  const headPitch = new THREE.Group();
  headYaw.add(headPitch);
  addDofMarker(headPitch, "J5 Head Pitch", 0xa78bfa, [-0.5, 0.12, 0], "x");

  const headRoll = new THREE.Group();
  headPitch.add(headRoll);
  addDofMarker(headRoll, "J6 Head Roll", 0xf472b6, [0, 0.52, 0.2], "z");

  const headMesh = new THREE.Mesh(new THREE.ConeGeometry(0.38, 0.66, 48, 1, true), shade);
  headMesh.rotation.x = Math.PI / 2;
  headMesh.position.z = 0.28;
  headMesh.castShadow = true;
  headRoll.add(headMesh);
  const rim = new THREE.Mesh(new THREE.TorusGeometry(0.38, 0.025, 12, 72), brass);
  rim.position.z = 0.6;
  headRoll.add(rim);

  const bulb = new THREE.Mesh(
    new THREE.SphereGeometry(0.14, 24, 24),
    new THREE.MeshStandardMaterial({ color: 0xffe1a1, emissive: 0xffc25a, emissiveIntensity: 2.4 })
  );
  bulb.position.z = 0.68;
  headRoll.add(bulb);
  const halo = new THREE.Mesh(
    new THREE.SphereGeometry(0.28, 32, 32),
    new THREE.MeshBasicMaterial({ color: 0xffd36a, transparent: true, opacity: 0.22, depthWrite: false })
  );
  halo.position.z = 0.68;
  headRoll.add(halo);


  const spot = new THREE.SpotLight(0xffd28a, 4.2, 8, Math.PI / 6, 0.45, 1.0);
  spot.position.set(0, 0, 0.62);
  const target = new THREE.Object3D();
  target.position.set(0, 0, 3.8);
  headRoll.add(spot);
  headRoll.add(target);
  spot.target = target;

  return { root, base, lower, upper, headYaw, headPitch, headRoll, spot, bulb, halo };
}

function armMesh(length, material, accentMaterial) {
  const group = new THREE.Group();
  const left = new THREE.Mesh(new THREE.BoxGeometry(0.075, length, 0.075), material);
  const right = left.clone();
  left.position.set(-0.13, length / 2, 0);
  right.position.set(0.13, length / 2, 0);
  left.castShadow = right.castShadow = true;
  const brace = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.055, 0.08), accentMaterial);
  brace.position.y = length * 0.52;
  brace.castShadow = true;
  const brace2 = brace.clone();
  brace2.position.y = length * 0.78;
  group.add(left, right, brace, brace2);
  return group;
}

function jointSphere(radius, material) {
  const mesh = new THREE.Mesh(new THREE.SphereGeometry(radius, 32, 20), material);
  mesh.castShadow = true;
  return mesh;
}

function addDofMarker(parent, text, color, position, axis) {
  const marker = new THREE.Group();
  marker.position.set(...position);
  const torus = new THREE.Mesh(
    new THREE.TorusGeometry(0.18, 0.012, 8, 48),
    new THREE.MeshBasicMaterial({ color })
  );
  if (axis === "x") torus.rotation.y = Math.PI / 2;
  if (axis === "y") torus.rotation.x = Math.PI / 2;
  marker.add(torus);
  const label = labelSprite(text, color);
  label.position.set(0, 0.24, 0);
  marker.add(label);
  parent.add(marker);
}

function labelSprite(text, color) {
  const canvas = document.createElement("canvas");
  canvas.width = 420;
  canvas.height = 96;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "rgba(12, 11, 9, 0.82)";
  roundRect(ctx, 8, 14, 404, 58, 18);
  ctx.fill();
  ctx.strokeStyle = `#${color.toString(16).padStart(6, "0")}`;
  ctx.lineWidth = 4;
  ctx.stroke();
  ctx.fillStyle = "#f6eddd";
  ctx.font = "bold 30px Georgia";
  ctx.fillText(text, 28, 52);
  const texture = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false }));
  sprite.scale.set(0.78, 0.18, 1);
  return sprite;
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
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
  joints = joints.map((v, i) => v + (targetJoints[i] - v) * 0.30);
  applyJoints(joints);
  updateDofReadout(joints);
  lamp.spot.intensity += (lightTarget.intensity * 4.8 - lamp.spot.intensity) * 0.18;
  lamp.spot.color.set(lightTarget.color);
  lamp.bulb.material.emissive.set(lightTarget.color);
  lamp.bulb.material.emissiveIntensity = 1.2 + lightTarget.intensity * 3.4;
  lamp.halo.material.color.set(lightTarget.color);
  lamp.halo.material.opacity = 0.08 + lightTarget.intensity * 0.28;
  renderer.render(scene, camera);
}

function applyJoints(j) {
  lamp.base.rotation.y = deg(j[0]);
  lamp.lower.rotation.x = deg(j[1]);
  lamp.upper.rotation.x = deg(j[2]);
  lamp.headYaw.rotation.y = deg(j[3]);
  lamp.headPitch.rotation.x = deg(j[4]);
  lamp.headRoll.rotation.z = deg(j[5]);
}

function deg(v) {
  return (v * Math.PI) / 180;
}

function updateDofReadout(values) {
  els.dofValues.forEach((el) => {
    const index = Number(el.dataset.joint);
    el.textContent = `${values[index].toFixed(0)}°`;
  });
}

ws.addEventListener("open", () => log("WebSocket connected"));
ws.addEventListener("message", (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === "lamp_state") {
    targetJoints = msg.joints;
    lightTarget = msg.light;
    const seekIn = msg.disengaged_for != null ? ` ${msg.disengaged_for}s` : "";
    els.state.textContent = msg.state + seekIn;
    const s = msg.state;
    els.state.className = "state-pill" +
      (s === "ENGAGED"    ? " engaged" :
       s === "SEEKING_3"  ? " seeking-hot" :
       s.startsWith("SEEKING") || s === "DEMO_WAVE" ? " seeking" :
       s === "OBJECT_FOUND" ? " found" : "");
    if (msg.sound === "chirp") chirp();
  } else if (msg.type === "engagement") {
    lastEngagement = msg;
    const method = msg.method ? `/${msg.method}` : "";
    const gaze = msg.gaze_h != null && msg.gaze_v != null ? ` gaze ${msg.gaze_h}/${msg.gaze_v}` : "";
    const pose = msg.detected ? ` yaw ${msg.yaw_deg} pitch ${msg.pitch_deg}${gaze}` : "";
    const state = msg.detected ? (msg.engaged ? "gaze-locked" : "tracking-face") : "no-face";
    els.engagement.textContent = `${state}${method}${pose}`;
    els.fps.textContent = msg.fps ?? 0;
    drawDetections(lastDetections);
  } else if (msg.type === "memory_event") {
    const text = `${msg.label} @ ${msg.zone}`;
    els.memory.textContent = text;
    if (msg.action === "insert") toast(`New: ${text}`);
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
      els.audio.play().catch(() => speak(msg.text));
    } else {
      speak(msg.text);
    }
  } else if (msg.type === "announce") {
    if (msg.audio_b64) {
      els.audio.src = `data:audio/mp3;base64,${msg.audio_b64}`;
      els.audio.play().catch(() => speak(msg.text));
    } else {
      speak(msg.text);
    }
    if (msg.text) toast(msg.text);
  } else if (msg.type === "auth_ok") {
    authed = true;
    applyAuthUI();
  } else if (msg.type === "auth_fail") {
    authError.textContent = "Wrong password";
    authInput.value = "";
    authInput.focus();
  } else if (msg.type === "auth_required") {
    authed = false;
    applyAuthUI();
  } else if (msg.type === "engagement_label_saved") {
    const truth = msg.truth ? "looking" : "away";
    const predicted = msg.predicted ? "looking" : "away";
    els.evalStatus.textContent = `${msg.count} labels`;
    toast(`Label saved: truth ${truth}, predicted ${predicted}`);
  } else if (msg.type === "log") {
    log(`${msg.level}: ${msg.msg}`);
  }
});

document.querySelector("#start-camera").addEventListener("click", startCamera);
document.querySelectorAll("[data-label-engagement]").forEach((button) => {
  button.addEventListener("click", () => markEngagement(button.dataset.labelEngagement === "true"));
});
els.textForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const content = els.textInput.value.trim();
  if (content) send({ type: "text_input", content });
  els.textInput.value = "";
});
els.ptt.addEventListener("pointerdown", startRecording);
els.ptt.addEventListener("pointerup", stopRecording);
window.addEventListener("keydown", (event) => {
  if (event.code === "Space" && document.activeElement !== els.textInput) {
    event.preventDefault();
    startRecording();
  } else if (event.code === "KeyE" && document.activeElement !== els.textInput && !event.repeat) {
    markEngagement(true);
  } else if (event.code === "KeyD" && document.activeElement !== els.textInput && !event.repeat) {
    markEngagement(false);
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
  document.getElementById("cam-start-overlay").classList.add("hidden");
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

function markEngagement(truth) {
  send({ type: "label_engagement", truth });
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
    .join(" · ");
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

function speak(text) {
  if (!text || !window.speechSynthesis) return;
  speechSynthesis.cancel();
  const u = new SpeechSynthesisUtterance(text);
  u.lang = "en-US";
  u.rate = 1.05;
  speechSynthesis.speak(u);
}

animate();
