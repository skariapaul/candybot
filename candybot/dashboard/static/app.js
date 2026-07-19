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

const statDevice = document.getElementById("stat-device");
const statDeviceNote = document.getElementById("stat-device-note");
const statCpu = document.getElementById("stat-cpu");
const meterCpu = document.getElementById("meter-cpu");
const statRam = document.getElementById("stat-ram");
const meterRam = document.getElementById("meter-ram");
const statGpu = document.getElementById("stat-gpu");
const meterGpu = document.getElementById("meter-gpu");
const statGpuTemp = document.getElementById("stat-gpu-temp");

function setState(state) {
  stateBadge.dataset.state = state;
  stateLabel.textContent = STATE_LABELS[state] || state;
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

function applyTelemetry(t) {
  if (!t || !t.device) return;

  if (t.device.device === "cuda") {
    statDevice.textContent = t.device.gpu_name || "GPU (ROCm)";
    statDeviceNote.textContent = "GPU-accelerated inference";
  } else {
    statDevice.textContent = "CPU";
    statDeviceNote.textContent = "GPU unavailable -- falling back to CPU";
  }

  const cpuPct = Math.round(t.cpu?.cpu_percent ?? 0);
  statCpu.textContent = `${cpuPct}%`;
  meterCpu.style.width = `${cpuPct}%`;

  const ramPct = Math.round(t.cpu?.ram_percent ?? 0);
  statRam.textContent = `${ramPct}%`;
  meterRam.style.width = `${ramPct}%`;

  if (t.gpu?.available) {
    const gpuPct = Math.round(t.gpu.busy_percent ?? 0);
    statGpu.textContent = `${gpuPct}%`;
    meterGpu.style.width = `${gpuPct}%`;
    statGpuTemp.textContent = t.gpu.temp_c != null ? `${t.gpu.temp_c.toFixed(1)}°C` : "—";
  } else {
    statGpu.textContent = "n/a";
    meterGpu.style.width = "0%";
    statGpuTemp.textContent = "—";
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

connect();
