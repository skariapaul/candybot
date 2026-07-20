import { initAvatar, playEnvelope, setAvatarState } from "/avatar.js";

const STATE_LABELS = {
  IDLE: "Idle",
  GREET: "Greeting",
  CAPTURE_NAME: "Listening for name",
  ASK_ITEM_CHOICE: "Asking: chocolate or candy?",
  PICK_AND_HAND: "Picking up item",
  THANK_YOU: "Thank you!",
  RESET: "Resetting",
};

const stateBadge = document.getElementById("state-badge");
const stateLabel = stateBadge.querySelector(".state-label");
const dialogueStage = document.getElementById("dialogue-stage");
const transcriptLog = document.getElementById("transcript-log");
const cameraStatus = document.getElementById("camera-status");
const cameraFeed = document.getElementById("camera-feed");

const deviceBadge = document.getElementById("device-badge");
const statDeviceNote = document.getElementById("stat-device-note");
const statCpu = document.getElementById("stat-cpu");
const meterCpu = document.getElementById("meter-cpu");
const statRam = document.getElementById("stat-ram");
const meterRam = document.getElementById("meter-ram");
const statGpu = document.getElementById("stat-gpu");
const meterGpu = document.getElementById("meter-gpu");
const statVram = document.getElementById("stat-vram");
const meterVram = document.getElementById("meter-vram");
const statGpuTemp = document.getElementById("stat-gpu-temp");
const statVramDetail = document.getElementById("stat-vram-detail");

function setState(state) {
  stateBadge.dataset.state = state;
  stateLabel.textContent = STATE_LABELS[state] || state;
  setAvatarState(state);
}

function addTranscriptLine(speaker, text) {
  const li = document.createElement("li");
  li.dataset.speaker = speaker;
  const speakerEl = document.createElement("span");
  speakerEl.className = "speaker";
  speakerEl.textContent = speaker === "visitor" ? "Visitor" : "candybot";
  const textEl = document.createElement("span");
  textEl.textContent = text;
  li.append(speakerEl, textEl);
  transcriptLog.appendChild(li);
  transcriptLog.scrollTop = transcriptLog.scrollHeight;
}

// -- Sparklines: rolling history rendered as a thin line + soft area fill,
// fixed-value axis matches the percent metric it sits under (0-100). --
const SPARK_HISTORY = 40;
const SPARK_W = 100;
const SPARK_H = 28;

function createSparkline(svgEl) {
  const history = [];
  function render() {
    if (history.length < 2) {
      svgEl.innerHTML = "";
      return;
    }
    const step = SPARK_W / (SPARK_HISTORY - 1);
    const offsetX = SPARK_W - (history.length - 1) * step;
    const points = history.map((v, i) => {
      const x = (i * step + offsetX).toFixed(1);
      const y = (SPARK_H - (Math.min(100, Math.max(0, v)) / 100) * SPARK_H).toFixed(1);
      return [x, y];
    });
    const linePts = points.map(([x, y]) => `${x},${y}`).join(" ");
    const fillPts = `${points[0][0]},${SPARK_H} ${linePts} ${points[points.length - 1][0]},${SPARK_H}`;
    svgEl.innerHTML =
      `<polyline class="sparkline-fill" points="${fillPts}"></polyline>` +
      `<polyline class="sparkline-line" points="${linePts}"></polyline>`;
  }
  return {
    push(value) {
      if (value == null) return;
      history.push(value);
      if (history.length > SPARK_HISTORY) history.shift();
      render();
    },
  };
}

const sparkCpu = createSparkline(document.getElementById("spark-cpu"));
const sparkRam = createSparkline(document.getElementById("spark-ram"));
const sparkGpu = createSparkline(document.getElementById("spark-gpu"));
const sparkVram = createSparkline(document.getElementById("spark-vram"));

function applyTelemetry(t) {
  if (!t || !t.device) return;

  if (t.device.device === "cuda") {
    deviceBadge.textContent = t.device.gpu_name || "GPU (ROCm)";
    statDeviceNote.textContent = "GPU-accelerated (ROCm)";
  } else {
    deviceBadge.textContent = "CPU";
    statDeviceNote.textContent = "GPU unavailable -- CPU fallback";
  }

  const cpuPct = Math.round(t.cpu?.cpu_percent ?? 0);
  statCpu.textContent = `${cpuPct}%`;
  meterCpu.style.width = `${cpuPct}%`;
  sparkCpu.push(cpuPct);

  const ramPct = Math.round(t.cpu?.ram_percent ?? 0);
  statRam.textContent = `${ramPct}%`;
  meterRam.style.width = `${ramPct}%`;
  sparkRam.push(ramPct);

  if (t.gpu?.available) {
    const gpuPct = Math.round(t.gpu.busy_percent ?? 0);
    statGpu.textContent = `${gpuPct}%`;
    meterGpu.style.width = `${gpuPct}%`;
    sparkGpu.push(gpuPct);
    statGpuTemp.textContent = t.gpu.temp_c != null ? `${t.gpu.temp_c.toFixed(1)}°C` : "—";

    const vramPct = t.gpu.vram_percent != null ? Math.round(t.gpu.vram_percent) : null;
    statVram.textContent = vramPct != null ? `${vramPct}%` : "n/a";
    meterVram.style.width = `${vramPct ?? 0}%`;
    sparkVram.push(vramPct);
    statVramDetail.textContent =
      t.gpu.vram_used_mb != null ? `${Math.round(t.gpu.vram_used_mb)} / ${Math.round(t.gpu.vram_total_mb)} MB` : "—";
  } else {
    statGpu.textContent = "n/a";
    meterGpu.style.width = "0%";
    statGpuTemp.textContent = "—";
    statVram.textContent = "n/a";
    meterVram.style.width = "0%";
    statVramDetail.textContent = "—";
  }
}

function applyDialogue(d) {
  if (!d) return;
  dialogueStage.textContent = d.detail ? `${d.stage} -- ${d.detail}` : d.stage || "—";
}

function applySnapshot(snapshot) {
  setState(snapshot.fsm_state || "IDLE");
  applyDialogue(snapshot.dialogue);
  applyTelemetry(snapshot.telemetry);
  (snapshot.transcript || []).forEach((entry) => addTranscriptLine(entry.speaker, entry.text));
}

function handleMessage(msg) {
  switch (msg.type) {
    case "snapshot":
      applySnapshot(msg.data);
      break;
    case "state_change":
      setState(msg.data.state);
      break;
    case "transcript":
      addTranscriptLine(msg.data.speaker, msg.data.text);
      break;
    case "dialogue":
      applyDialogue(msg.data);
      break;
    case "telemetry":
      applyTelemetry(msg.data);
      break;
    case "speech":
      playEnvelope(msg.data.envelope, msg.data.duration_s);
      break;
  }
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    cameraStatus.textContent = "live";
  };
  ws.onmessage = (event) => {
    try {
      handleMessage(JSON.parse(event.data));
    } catch (err) {
      console.error("Bad message from /ws", err);
    }
  };
  ws.onclose = () => {
    cameraStatus.textContent = "reconnecting…";
    setTimeout(connect, 2000);
  };
  ws.onerror = () => ws.close();
}

cameraFeed.onerror = () => {
  cameraStatus.textContent = "no camera signal";
};
cameraFeed.onload = () => {
  cameraStatus.textContent = "live";
};

initAvatar("avatar-container");
connect();
