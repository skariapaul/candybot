// Zen -- candybot's mascot: a CPU-chip character built entirely from Three.js
// primitives (no external GLB asset). Body = a chip-shaped box with gold pin
// details; limbs = capsules on rotating pivot groups; face = a <canvas>
// texture (eyes + mouth) redrawn every frame and applied to the chip's front.
//
// Mouth animation during speech is driven by a pre-computed volume envelope
// published by the orchestrator (see orchestrator/run.py's speak() closure
// and orchestrator/events.py's SpeechEvent) rather than live audio analysis
// in the browser -- audio itself plays server-side through whichever
// audio.profile is active, so Zen never needs to touch actual sound.

import * as THREE from "/vendor/three.module.js";

const FACE_SIZE = 256;
const PIN_COUNT = 9;

let scene, camera, renderer, chipGroup;
let leftArmPivot, rightArmPivot, leftLegPivot, rightLegPivot;
let faceCanvas, faceCtx, faceTexture;
let clock;

let currentFsmState = "IDLE";
let speaking = false;
let speechEnvelope = null;
let speechStartMs = 0;
let speechDurationMs = 0;

let mouthOpenness = 0;
let eyeBlink = 0;
let nextBlinkAt = 0;

function behaviorForState(state) {
  if (state === "GREET" || state === "CAPTURE_NAME" || state === "ASK_ITEM_CHOICE") return "listening";
  if (state === "PICK_AND_HAND" || state === "THANK_YOU") return "happy";
  return "idle";
}

function buildFaceTexture() {
  faceCanvas = document.createElement("canvas");
  faceCanvas.width = FACE_SIZE;
  faceCanvas.height = FACE_SIZE;
  faceCtx = faceCanvas.getContext("2d");
  faceTexture = new THREE.CanvasTexture(faceCanvas);
  return faceTexture;
}

function drawFace(expression) {
  const ctx = faceCtx;
  const w = FACE_SIZE;
  const h = FACE_SIZE;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#0d1b2a";
  ctx.fillRect(0, 0, w, h);

  const eyeY = h * 0.42;
  const eyeR = 22;
  const eyeSpacing = 60;
  ctx.fillStyle = "#5ec8ff";
  for (const side of [-1, 1]) {
    const ex = w / 2 + side * eyeSpacing;
    const scaleY = Math.max(0.08, 1 - eyeBlink);
    ctx.save();
    ctx.translate(ex, eyeY);
    ctx.scale(1, scaleY);
    ctx.beginPath();
    ctx.arc(0, 0, eyeR, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  const mouthY = h * 0.68;
  const mouthW = 90;
  ctx.strokeStyle = "#5ec8ff";
  ctx.lineWidth = 8;
  ctx.lineCap = "round";

  if (expression === "happy") {
    const bend = 14 + mouthOpenness * 20;
    ctx.beginPath();
    ctx.moveTo(w / 2 - mouthW / 2, mouthY);
    ctx.quadraticCurveTo(w / 2, mouthY + bend, w / 2 + mouthW / 2, mouthY);
    ctx.stroke();
  } else if (expression === "listening") {
    ctx.beginPath();
    ctx.arc(w / 2, mouthY, 15, 0, Math.PI * 2);
    ctx.stroke();
  } else {
    const openH = 6 + mouthOpenness * 40;
    ctx.beginPath();
    ctx.ellipse(w / 2, mouthY, mouthW / 2, openH / 2, 0, 0, Math.PI * 2);
    ctx.stroke();
  }

  faceTexture.needsUpdate = true;
}

function makeLimbPivot(x, y, length, material) {
  const pivot = new THREE.Group();
  pivot.position.set(x, y, 0);
  const geo = new THREE.CapsuleGeometry(0.12, length, 4, 8);
  const mesh = new THREE.Mesh(geo, material);
  mesh.position.y = -(length / 2 + 0.12);
  pivot.add(mesh);
  chipGroup.add(pivot);
  return pivot;
}

function buildScene(container) {
  scene = new THREE.Scene();

  const width = container.clientWidth || 1;
  const height = container.clientHeight || 1;
  camera = new THREE.PerspectiveCamera(35, width / height, 0.1, 100);
  camera.position.set(0, 0.3, 6.2);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  container.appendChild(renderer.domElement);

  scene.add(new THREE.AmbientLight(0xffffff, 0.75));
  const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
  dirLight.position.set(2, 3, 4);
  scene.add(dirLight);

  chipGroup = new THREE.Group();
  scene.add(chipGroup);

  const bodyMat = new THREE.MeshStandardMaterial({ color: 0x1c1c1e, metalness: 0.3, roughness: 0.5 });
  const body = new THREE.Mesh(new THREE.BoxGeometry(2.2, 2.2, 0.5), bodyMat);
  chipGroup.add(body);

  buildFaceTexture();
  const face = new THREE.Mesh(
    new THREE.PlaneGeometry(1.7, 1.7),
    new THREE.MeshBasicMaterial({ map: faceTexture, transparent: true })
  );
  face.position.z = 0.26;
  chipGroup.add(face);

  const pinMat = new THREE.MeshStandardMaterial({ color: 0xd4af37, metalness: 0.8, roughness: 0.3 });
  const pinGeo = new THREE.BoxGeometry(0.08, 0.18, 0.08);
  for (const yEdge of [-1.19, 1.19]) {
    for (let i = 0; i < PIN_COUNT; i++) {
      const pin = new THREE.Mesh(pinGeo, pinMat);
      pin.position.set(-1.0 + (2.0 * i) / (PIN_COUNT - 1), yEdge, 0);
      chipGroup.add(pin);
    }
  }

  const limbMat = new THREE.MeshStandardMaterial({ color: 0x3a3a3c, metalness: 0.2, roughness: 0.6 });
  leftArmPivot = makeLimbPivot(-1.25, 0.6, 1.0, limbMat);
  rightArmPivot = makeLimbPivot(1.25, 0.6, 1.0, limbMat);
  leftLegPivot = makeLimbPivot(-0.6, -1.25, 1.0, limbMat);
  rightLegPivot = makeLimbPivot(0.6, -1.25, 1.0, limbMat);

  new ResizeObserver(() => onResize(container)).observe(container);
}

function onResize(container) {
  const width = container.clientWidth || 1;
  const height = container.clientHeight || 1;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}

function tick() {
  requestAnimationFrame(tick);
  const t = clock.getElapsedTime();

  const behavior = speaking ? "talking" : behaviorForState(currentFsmState);

  chipGroup.position.y = Math.sin(t * 1.2) * 0.06;
  chipGroup.rotation.y = Math.sin(t * 0.6) * 0.08;
  chipGroup.rotation.x = 0;

  if (t > nextBlinkAt) {
    eyeBlink = 1;
    setTimeout(() => {
      eyeBlink = 0;
    }, 120);
    nextBlinkAt = t + 2.5 + Math.random() * 2.5;
  }

  let expression = "neutral";

  if (behavior === "listening") {
    expression = "listening";
    leftArmPivot.rotation.z = 0.15 + Math.sin(t * 1.5) * 0.05;
    rightArmPivot.rotation.z = -0.15 - Math.sin(t * 1.5) * 0.05;
    chipGroup.rotation.x = 0.05;
    mouthOpenness = 0;
  } else if (behavior === "happy") {
    expression = "happy";
    rightArmPivot.rotation.z = -1.2 + Math.sin(t * 6) * 0.4;
    leftArmPivot.rotation.z = 0.3 + Math.sin(t * 3) * 0.1;
    chipGroup.position.y += Math.abs(Math.sin(t * 4)) * 0.15;
    mouthOpenness = 0.3;
  } else if (behavior === "talking") {
    expression = "neutral";
    leftArmPivot.rotation.z = 0.2 + Math.sin(t * 2) * 0.08;
    rightArmPivot.rotation.z = -0.2 - Math.sin(t * 2.2) * 0.08;

    const elapsed = performance.now() - speechStartMs;
    if (speechEnvelope && speechEnvelope.length > 0 && elapsed < speechDurationMs) {
      const idx = Math.min(speechEnvelope.length - 1, Math.floor((elapsed / speechDurationMs) * speechEnvelope.length));
      mouthOpenness = speechEnvelope[idx];
    } else {
      mouthOpenness = 0;
      speaking = false;
    }
  } else {
    leftArmPivot.rotation.z = Math.sin(t * 0.8) * 0.06;
    rightArmPivot.rotation.z = -Math.sin(t * 0.8) * 0.06;
    leftLegPivot.rotation.z = 0;
    rightLegPivot.rotation.z = 0;
    mouthOpenness = 0;
  }

  drawFace(expression);
  renderer.render(scene, camera);
}

export function initAvatar(containerId) {
  const container = document.getElementById(containerId);
  if (!container) return;
  clock = new THREE.Clock();
  buildScene(container);
  tick();
}

export function setAvatarState(fsmState) {
  currentFsmState = fsmState;
}

export function playEnvelope(envelope, durationS) {
  speechEnvelope = envelope;
  speechDurationMs = Math.max(1, durationS * 1000);
  speechStartMs = performance.now();
  speaking = true;
}
