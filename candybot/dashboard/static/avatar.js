// Zen -- candybot's mascot: a CPU-chip character built entirely from Three.js
// primitives (no external GLB asset). Body = a chip-shaped box with a blue
// accent trim and gold pin details; limbs = two-segment articulated arms/legs
// (shoulder+elbow, hip+knee, each with a small ball-joint accent) on rotating
// pivot groups; face = a <canvas> texture (eyes + mouth) redrawn every frame
// and applied to a glowing bezel on the chip's front.
//
// Mouth animation during speech is driven by a pre-computed volume envelope
// published by the orchestrator (see orchestrator/run.py's speak() closure
// and orchestrator/events.py's SpeechEvent) rather than live audio analysis
// in the browser -- audio itself plays server-side through whichever
// audio.profile is active, so Zen never needs to touch actual sound.

import * as THREE from "/vendor/three.module.js";

const FACE_SIZE = 256;
const PIN_COUNT = 11;

const COLOR_BODY = 0x1b1f24; // graphite, faint blue tint
const COLOR_TRIM = 0x2a78d6; // accent blue, matches dashboard palette
const COLOR_LIMB_UPPER = 0x24282e; // matches body family
const COLOR_LIMB_LOWER = 0x2a78d6; // accent blue, marks the "flexible" segment
const COLOR_JOINT = 0xd4d8de; // pale metallic ball joints
const COLOR_PIN = 0xd4af37; // gold

let scene, camera, renderer, chipGroup;
let leftShoulder, leftElbow, rightShoulder, rightElbow;
let leftHip, leftKnee, rightHip, rightKnee;
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

  // Screen-like bezel background with a soft radial glow, classier than a flat fill.
  const bg = ctx.createRadialGradient(w / 2, h * 0.52, 10, w / 2, h * 0.52, w * 0.62);
  bg.addColorStop(0, "#132436");
  bg.addColorStop(1, "#070d14");
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(94, 200, 255, 0.35)";
  ctx.lineWidth = 3;
  ctx.strokeRect(6, 6, w - 12, h - 12);

  const eyeY = h * 0.42;
  const eyeR = 22;
  const eyeSpacing = 60;
  for (const side of [-1, 1]) {
    const ex = w / 2 + side * eyeSpacing;
    const scaleY = Math.max(0.08, 1 - eyeBlink);
    ctx.save();
    ctx.translate(ex, eyeY);
    ctx.scale(1, scaleY);
    const glow = ctx.createRadialGradient(0, 0, 0, 0, 0, eyeR);
    glow.addColorStop(0, "#bfe9ff");
    glow.addColorStop(0.55, "#5ec8ff");
    glow.addColorStop(1, "#2a78d6");
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(0, 0, eyeR, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  const mouthY = h * 0.68;
  const mouthW = 90;
  ctx.strokeStyle = "#5ec8ff";
  ctx.lineWidth = 7;
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

function addJoint(parent, radius = 0.1) {
  const joint = new THREE.Mesh(
    new THREE.SphereGeometry(radius, 12, 12),
    new THREE.MeshStandardMaterial({ color: COLOR_JOINT, metalness: 0.6, roughness: 0.35 })
  );
  parent.add(joint);
  return joint;
}

/**
 * Builds a two-segment articulated limb (shoulder/hip pivot -> segment ->
 * elbow/knee pivot -> segment) so poses read as genuinely flexible rather
 * than a single rigid capsule swinging from one point.
 */
function makeArticulatedLimb(x, y, upperLen, lowerLen, isArm) {
  const upperMat = new THREE.MeshStandardMaterial({ color: COLOR_LIMB_UPPER, metalness: 0.35, roughness: 0.5 });
  const lowerMat = new THREE.MeshStandardMaterial({
    color: COLOR_LIMB_LOWER,
    metalness: 0.4,
    roughness: 0.35,
    emissive: 0x0b2e55,
    emissiveIntensity: 0.4,
  });

  const rootPivot = new THREE.Group();
  rootPivot.position.set(x, y, 0);
  addJoint(rootPivot, 0.09);

  const upperMesh = new THREE.Mesh(new THREE.CapsuleGeometry(0.1, upperLen, 4, 8), upperMat);
  upperMesh.position.y = -(upperLen / 2 + 0.1);
  rootPivot.add(upperMesh);

  const midPivot = new THREE.Group();
  midPivot.position.y = -(upperLen + 0.2);
  addJoint(midPivot, 0.08);
  rootPivot.add(midPivot);

  const lowerMesh = new THREE.Mesh(new THREE.CapsuleGeometry(0.085, lowerLen, 4, 8), lowerMat);
  lowerMesh.position.y = -(lowerLen / 2 + 0.08);
  midPivot.add(lowerMesh);

  chipGroup.add(rootPivot);
  return { root: rootPivot, mid: midPivot };
}

function buildScene(container) {
  scene = new THREE.Scene();

  const width = container.clientWidth || 1;
  const height = container.clientHeight || 1;
  camera = new THREE.PerspectiveCamera(35, width / height, 0.1, 100);
  camera.position.set(0, 0.3, 6.4);

  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  container.appendChild(renderer.domElement);

  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const keyLight = new THREE.DirectionalLight(0xffffff, 0.85);
  keyLight.position.set(2, 3, 4);
  scene.add(keyLight);
  // Cool rim light from behind/below for a bit of "tech glow" edge highlight -- classier than flat ambient+key alone.
  const rimLight = new THREE.DirectionalLight(0x5ec8ff, 0.6);
  rimLight.position.set(-2, -1, -3);
  scene.add(rimLight);

  chipGroup = new THREE.Group();
  scene.add(chipGroup);

  const bodyMat = new THREE.MeshStandardMaterial({ color: COLOR_BODY, metalness: 0.45, roughness: 0.4 });
  const body = new THREE.Mesh(new THREE.BoxGeometry(2.2, 2.2, 0.5), bodyMat);
  chipGroup.add(body);

  // Thin accent trim around the body edge -- a slightly larger, slightly
  // thinner frame sitting just behind the body face, reads as a refined edge line.
  const trimMat = new THREE.MeshStandardMaterial({
    color: COLOR_TRIM,
    metalness: 0.3,
    roughness: 0.3,
    emissive: 0x0b2e55,
    emissiveIntensity: 0.5,
  });
  const trim = new THREE.Mesh(new THREE.BoxGeometry(2.3, 2.3, 0.06), trimMat);
  trim.position.z = -0.22;
  chipGroup.add(trim);

  buildFaceTexture();
  const face = new THREE.Mesh(
    new THREE.PlaneGeometry(1.7, 1.7),
    new THREE.MeshBasicMaterial({ map: faceTexture, transparent: true })
  );
  face.position.z = 0.26;
  chipGroup.add(face);

  const pinMat = new THREE.MeshStandardMaterial({
    color: COLOR_PIN,
    metalness: 0.85,
    roughness: 0.25,
    emissive: 0x3a2c05,
    emissiveIntensity: 0.25,
  });
  const pinGeo = new THREE.BoxGeometry(0.06, 0.2, 0.06);
  for (const yEdge of [-1.2, 1.2]) {
    for (let i = 0; i < PIN_COUNT; i++) {
      const pin = new THREE.Mesh(pinGeo, pinMat);
      pin.position.set(-1.05 + (2.1 * i) / (PIN_COUNT - 1), yEdge, 0);
      chipGroup.add(pin);
    }
  }

  const armPair = [makeArticulatedLimb(-1.28, 0.6, 0.62, 0.55, true), makeArticulatedLimb(1.28, 0.6, 0.62, 0.55, true)];
  const legPair = [makeArticulatedLimb(-0.6, -1.28, 0.62, 0.55, false), makeArticulatedLimb(0.6, -1.28, 0.62, 0.55, false)];
  [leftShoulder, rightShoulder] = armPair.map((l) => l.root);
  [leftElbow, rightElbow] = armPair.map((l) => l.mid);
  [leftHip, rightHip] = legPair.map((l) => l.root);
  [leftKnee, rightKnee] = legPair.map((l) => l.mid);

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
    leftShoulder.rotation.z = 0.2 + Math.sin(t * 1.5) * 0.06;
    rightShoulder.rotation.z = -0.2 - Math.sin(t * 1.5) * 0.06;
    leftElbow.rotation.z = -0.15 + Math.sin(t * 1.7) * 0.05;
    rightElbow.rotation.z = 0.15 - Math.sin(t * 1.7) * 0.05;
    chipGroup.rotation.x = 0.05;
    mouthOpenness = 0;
  } else if (behavior === "happy") {
    expression = "happy";
    // Wave: shoulder swings broadly, elbow bends more sharply and out of phase --
    // this two-joint interplay is what makes the gesture read as flexible, not stiff.
    rightShoulder.rotation.z = -1.1 + Math.sin(t * 6) * 0.35;
    rightElbow.rotation.z = -0.6 + Math.sin(t * 6 + 1.1) * 0.5;
    leftShoulder.rotation.z = 0.35 + Math.sin(t * 3) * 0.1;
    leftElbow.rotation.z = -0.2 + Math.sin(t * 3 + 0.6) * 0.15;
    chipGroup.position.y += Math.abs(Math.sin(t * 4)) * 0.15;
    leftKnee.rotation.z = Math.sin(t * 4) * 0.2;
    rightKnee.rotation.z = -Math.sin(t * 4) * 0.2;
    mouthOpenness = 0.3;
  } else if (behavior === "talking") {
    expression = "neutral";
    leftShoulder.rotation.z = 0.22 + Math.sin(t * 2) * 0.07;
    rightShoulder.rotation.z = -0.22 - Math.sin(t * 2.2) * 0.07;
    leftElbow.rotation.z = -0.1 + Math.sin(t * 2.4) * 0.06;
    rightElbow.rotation.z = 0.1 - Math.sin(t * 2.4) * 0.06;

    const elapsed = performance.now() - speechStartMs;
    if (speechEnvelope && speechEnvelope.length > 0 && elapsed < speechDurationMs) {
      const idx = Math.min(speechEnvelope.length - 1, Math.floor((elapsed / speechDurationMs) * speechEnvelope.length));
      mouthOpenness = speechEnvelope[idx];
    } else {
      mouthOpenness = 0;
      speaking = false;
    }
  } else {
    leftShoulder.rotation.z = Math.sin(t * 0.8) * 0.07;
    rightShoulder.rotation.z = -Math.sin(t * 0.8) * 0.07;
    leftElbow.rotation.z = Math.sin(t * 0.9 + 0.4) * 0.05;
    rightElbow.rotation.z = -Math.sin(t * 0.9 + 0.4) * 0.05;
    leftKnee.rotation.z = 0;
    rightKnee.rotation.z = 0;
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
