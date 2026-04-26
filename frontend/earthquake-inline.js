
const state = {
  presets: [],
  preset: "nepal",
  magnitude: 7.8,
  depth_km: 15,
  epicenter_x: 0.34,
  epicenter_y: 0.63,
  medium_type: "basin",
  wave_speed: 1.05,
  duration: 18,
  resolution: 60,
  playbackSpeed: 1,
  view: "physics",
  mode: "heat",
  playing: false,
  currentFrame: 0,
  lastTick: 0,
  simulation: null,
  loading: false,
};

const FALLBACK_PRESETS = [
  {
    id: "nepal",
    label: "Nepal Earthquake Scenario",
    magnitude: 7.8,
    depth_km: 15,
    epicenter_x: 0.34,
    epicenter_y: 0.63,
    medium_type: "basin",
    wave_speed: 1.05,
    duration: 18,
    resolution: 60,
  },
  {
    id: "urban",
    label: "Urban City Scenario",
    magnitude: 6.6,
    depth_km: 11,
    epicenter_x: 0.48,
    epicenter_y: 0.56,
    medium_type: "urban",
    wave_speed: 1.0,
    duration: 14,
    resolution: 58,
  },
  {
    id: "disaster",
    label: "High Magnitude Disaster",
    magnitude: 8.9,
    depth_km: 21,
    epicenter_x: 0.28,
    epicenter_y: 0.68,
    medium_type: "mountain",
    wave_speed: 1.18,
    duration: 22,
    resolution: 64,
  },
];

const physicsCanvas = document.getElementById("physicsCanvas");
const aiCanvas = document.getElementById("aiCanvas");
const physicsCtx = physicsCanvas.getContext("2d");
const aiCtx = aiCanvas.getContext("2d");

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function $(id) {
  return document.getElementById(id);
}

function setSliderTrack(el) {
  const min = parseFloat(el.min);
  const max = parseFloat(el.max);
  const val = parseFloat(el.value);
  const pct = ((val - min) / (max - min)) * 100;
  el.style.setProperty("--pct", `${pct.toFixed(1)}%`);
}

function updateControlReadouts() {
  $("magnitudeVal").textContent = Number(state.magnitude).toFixed(1);
  $("depthVal").textContent = `${Number(state.depth_km).toFixed(1)} km`;
  $("speedVal").textContent = `${Number(state.wave_speed).toFixed(2)}x`;
  $("durationVal").textContent = `${Number(state.duration).toFixed(1)} s`;
  $("epicenterText").textContent = `x ${state.epicenter_x.toFixed(2)} / y ${state.epicenter_y.toFixed(2)}`;
}

function syncControlsToState() {
  $("magnitude").value = String(state.magnitude);
  $("depth").value = String(state.depth_km);
  $("waveSpeed").value = String(state.wave_speed);
  $("duration").value = String(state.duration);
  $("mediumType").value = state.medium_type;
  $("playbackSpeed").value = String(state.playbackSpeed);
  ["magnitude", "depth", "waveSpeed", "duration", "timeline"].forEach((id) => {
    const el = $(id);
    if (el) setSliderTrack(el);
  });
  updateControlReadouts();
}

function renderPresetButtons() {
  const grid = $("presetGrid");
  grid.innerHTML = state.presets.map((preset) => `
    <button class="preset-btn ${preset.id === state.preset ? "active" : ""}" data-preset="${preset.id}">
      <div class="label">${preset.id}</div>
      <div class="preset-name">${preset.label}</div>
      <div class="preset-caption">M ${Number(preset.magnitude).toFixed(1)} • ${preset.medium_type} • ${Number(preset.depth_km).toFixed(0)} km depth</div>
    </button>
  `).join("");
}

function applyPreset(presetId) {
  const preset = state.presets.find((item) => item.id === presetId);
  if (!preset) return;
  state.preset = preset.id;
  state.magnitude = Number(preset.magnitude);
  state.depth_km = Number(preset.depth_km);
  state.epicenter_x = Number(preset.epicenter_x);
  state.epicenter_y = Number(preset.epicenter_y);
  state.medium_type = preset.medium_type;
  state.wave_speed = Number(preset.wave_speed);
  state.duration = Number(preset.duration);
  state.resolution = Number(preset.resolution || 60);
  renderPresetButtons();
  syncControlsToState();
  drawPlaceholder();
}

async function loadPresets() {
  try {
    const res = await fetch("/api/earthquake/presets");
    if (!res.ok) throw new Error(`preset request failed with ${res.status}`);
    const data = await res.json();
    state.presets = Array.isArray(data.presets) && data.presets.length ? data.presets : FALLBACK_PRESETS;
  } catch (error) {
    state.presets = FALLBACK_PRESETS;
    $("aiExplanation").textContent = "Using built-in presets because the preset API was unavailable. You can still run the earthquake demo.";
  }
  renderPresetButtons();
  applyPreset(state.preset);
}

function updateSummary(summary = { safe: 0, risky: 0, collapse: 0 }) {
  $("safeCount").textContent = summary.safe ?? 0;
  $("riskyCount").textContent = summary.risky ?? 0;
  $("collapseCount").textContent = summary.collapse ?? 0;
}

function statusClass(status) {
  if (status === "collapse") return "status-collapse";
  if (status === "risky") return "status-risky";
  return "status-safe";
}

function updateBuildingList(buildings = [], frameStates = []) {
  const frameMap = new Map(frameStates.map((item) => [item.id, item]));
  $("buildingList").innerHTML = buildings.map((building) => {
    const current = frameMap.get(building.id) || {};
    const damage = Number(building.damage_score || current.damage || 0);
    const status = current.status || building.status || "safe";
    return `
      <div class="building-card">
        <div style="flex:1;">
          <div class="label">${building.material} • ${building.height} floors</div>
          <div class="title title-md">${building.name}</div>
          <div class="small-copy">Current intensity ${Number(current.intensity || 0).toFixed(2)}</div>
          <div class="damage-bar"><span style="width:${Math.min(100, damage * 100)}%"></span></div>
        </div>
        <span class="status-tag ${statusClass(status)}">${status}</span>
      </div>
    `;
  }).join("");
}

function colorForHeat(value) {
  const v = clamp(value, 0, 1);
  if (v < 0.5) {
    const t = v / 0.5;
    return `rgb(${Math.round(56 + 145 * t)}, ${Math.round(111 + 111 * t)}, ${Math.round(88 - 20 * t)})`;
  }
  const t = (v - 0.5) / 0.5;
  return `rgb(${Math.round(201 + 10 * t)}, ${Math.round(142 - 82 * t)}, ${Math.round(36 - 2 * t)})`;
}

function resizeCanvas(canvas) {
  const wrap = canvas.parentElement;
  if (!wrap || wrap.clientWidth === 0 || wrap.clientHeight === 0) {
    return false;
  }
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(wrap.clientWidth * dpr);
  canvas.height = Math.round(wrap.clientHeight * dpr);
  canvas.style.width = `${wrap.clientWidth}px`;
  canvas.style.height = `${wrap.clientHeight}px`;
  return true;
}

function drawGrid(ctx, canvas, grid, res, mode, epicenter, buildings, frameStates, frame) {
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.save();
  ctx.scale(w / canvas.clientWidth, h / canvas.clientHeight);

  const cssW = canvas.clientWidth;
  const cssH = canvas.clientHeight;
  const cellW = cssW / res;
  const cellH = cssH / res;

  for (let y = 0; y < res; y += 1) {
    for (let x = 0; x < res; x += 1) {
      const index = y * res + x;
      const value = Number(grid[index] || 0);
      if (mode === "waves") {
        const p = Number(frame.p_wave[index] || 0);
        const s = Number(frame.s_wave[index] || 0);
        const a = clamp(Math.abs(p) * 1.8, 0, 1);
        const b = clamp(Math.abs(s) * 1.8, 0, 1);
        const r = Math.round(245 * b + 40 * a);
        const g = Math.round(182 * a + 110 * (1 - value));
        const c = Math.round(255 * a + 30 * b);
        ctx.fillStyle = `rgba(${r}, ${g}, ${c}, ${Math.max(a, b) * 0.72})`;
      } else {
        ctx.fillStyle = colorForHeat(value);
      }
      ctx.fillRect(x * cellW, y * cellH, cellW + 1, cellH + 1);
    }
  }

  ctx.strokeStyle = "rgba(24, 55, 48, 0.12)";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 10; i += 1) {
    const px = (cssW / 10) * i;
    const py = (cssH / 10) * i;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, cssH);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, py);
    ctx.lineTo(cssW, py);
    ctx.stroke();
  }

  const epicenterPx = epicenter.x * cssW;
  const epicenterPy = epicenter.y * cssH;
  ctx.save();
  ctx.strokeStyle = "rgba(33, 31, 77, 0.92)";
  ctx.fillStyle = "rgba(53, 49, 110, 0.92)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(epicenterPx, epicenterPy, 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.beginPath();
  ctx.moveTo(epicenterPx - 14, epicenterPy);
  ctx.lineTo(epicenterPx + 14, epicenterPy);
  ctx.moveTo(epicenterPx, epicenterPy - 14);
  ctx.lineTo(epicenterPx, epicenterPy + 14);
  ctx.stroke();
  ctx.restore();

  ctx.save();
  ctx.lineWidth = 2.4;
  ctx.strokeStyle = "rgba(90, 185, 255, 0.92)";
  ctx.beginPath();
  ctx.arc(epicenterPx, epicenterPy, frame.p_radius * Math.min(cssW, cssH), 0, Math.PI * 2);
  ctx.stroke();
  ctx.strokeStyle = "rgba(255, 157, 77, 0.94)";
  ctx.beginPath();
  ctx.arc(epicenterPx, epicenterPy, frame.s_radius * Math.min(cssW, cssH), 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();

  const states = new Map((frameStates || []).map((item) => [item.id, item]));
  buildings.forEach((building) => {
    const current = states.get(building.id) || {};
    const sway = Number(current.sway || 0);
    const status = current.status || building.status || "safe";
    const x = building.x * cssW + Math.sin((frame.t + building.height) * 1.8) * sway * 0.3;
    const y = building.y * cssH;
    const width = 18 + building.height * 1.5;
    const height = 22 + building.height * 7;
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(Math.sin(frame.t * 1.3 + building.height) * sway * 0.003);
    ctx.fillStyle = status === "collapse" ? "rgba(171, 51, 34, 0.94)" : status === "risky" ? "rgba(201, 142, 36, 0.92)" : "rgba(56, 111, 88, 0.92)";
    ctx.fillRect(-width / 2, -height, width, height);
    ctx.fillStyle = "rgba(255,255,255,0.22)";
    for (let i = 1; i < 4; i += 1) {
      ctx.fillRect(-width / 2 + 3, -height + i * (height / 5), width - 6, 2);
    }
    ctx.restore();
  });

  ctx.restore();
}

function drawAiGrid(ctx, canvas, grid, res, epicenter, buildings) {
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.save();
  ctx.scale(w / canvas.clientWidth, h / canvas.clientHeight);

  const cssW = canvas.clientWidth;
  const cssH = canvas.clientHeight;
  const cellW = cssW / res;
  const cellH = cssH / res;
  for (let y = 0; y < res; y += 1) {
    for (let x = 0; x < res; x += 1) {
      const value = Number(grid[y * res + x] || 0);
      ctx.fillStyle = colorForHeat(value);
      ctx.globalAlpha = 0.9;
      ctx.fillRect(x * cellW, y * cellH, cellW + 1, cellH + 1);
    }
  }

  ctx.globalAlpha = 1;
  ctx.strokeStyle = "rgba(255,255,255,0.5)";
  ctx.setLineDash([6, 6]);
  ctx.strokeRect(14, 14, cssW - 28, cssH - 28);
  ctx.setLineDash([]);

  ctx.fillStyle = "rgba(53, 49, 110, 0.92)";
  ctx.beginPath();
  ctx.arc(epicenter.x * cssW, epicenter.y * cssH, 7, 0, Math.PI * 2);
  ctx.fill();

  buildings.forEach((building) => {
    ctx.fillStyle = "rgba(255,255,255,0.78)";
    ctx.fillRect(building.x * cssW - 10, building.y * cssH - 26, 20, 26);
  });

  ctx.restore();
}

function drawPlaceholder() {
  const ctxs = [physicsCtx, aiCtx];
  const canvases = [physicsCanvas, aiCanvas];
  canvases.forEach((canvas, idx) => {
    if (!resizeCanvas(canvas)) return;
    const ctx = ctxs[idx];
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.save();
    ctx.scale(canvas.width / canvas.clientWidth, canvas.height / canvas.clientHeight);
    ctx.fillStyle = "rgba(255,255,255,0.75)";
    ctx.fillRect(0, 0, canvas.clientWidth, canvas.clientHeight);
    ctx.strokeStyle = "rgba(24, 55, 48, 0.15)";
    for (let i = 0; i <= 10; i += 1) {
      const x = (canvas.clientWidth / 10) * i;
      const y = (canvas.clientHeight / 10) * i;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, canvas.clientHeight);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(canvas.clientWidth, y);
      ctx.stroke();
    }
    ctx.fillStyle = "rgba(24, 55, 48, 0.68)";
    ctx.font = `16px ${getComputedStyle(document.documentElement).getPropertyValue('--sans') || 'sans-serif'}`;
    ctx.fillText("Click to place epicenter and run the earthquake scenario.", 28, 40);
    const x = state.epicenter_x * canvas.clientWidth;
    const y = state.epicenter_y * canvas.clientHeight;
    ctx.fillStyle = "rgba(53, 49, 110, 0.92)";
    ctx.beginPath();
    ctx.arc(x, y, 7, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  });
}

function setView(view) {
  state.view = view;
  document.querySelectorAll("[data-view]").forEach((btn) => btn.classList.toggle("active", btn.dataset.view === view));
  const grid = $("canvasGrid");
  const aiCard = $("aiCard");
  if (view === "split") {
    grid.classList.add("split");
    aiCard.style.display = "";
  } else if (view === "ai") {
    grid.classList.remove("split");
    aiCard.style.display = "";
    document.querySelector(".canvas-card").style.display = "none";
  } else {
    grid.classList.remove("split");
    aiCard.style.display = "none";
    document.querySelector(".canvas-card").style.display = "";
  }
  if (view !== "ai") {
    document.querySelector(".canvas-card").style.display = "";
  }
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll("[data-mode]").forEach((btn) => btn.classList.toggle("active", btn.dataset.mode === mode));
  renderFrame(state.currentFrame);
}

function updateMetrics(metrics = {}) {
  $("maxAmplitude").textContent = Number(metrics.max_wave_amplitude || 0).toFixed(3);
  $("affectedRadius").textContent = `${Number(metrics.affected_radius_km || 0).toFixed(1)} km`;
  $("affectedArea").textContent = `${Number(metrics.affected_area_pct || 0).toFixed(1)}%`;
  $("damagePct").textContent = `${Number(metrics.estimated_damage_pct || 0).toFixed(1)}%`;
}

function renderFrame(index) {
  if (!state.simulation || !state.simulation.frames.length) {
    drawPlaceholder();
    return;
  }

  const frames = state.simulation.frames;
  state.currentFrame = clamp(index, 0, frames.length - 1);
  $("timeline").value = String(state.currentFrame);
  setSliderTrack($("timeline"));

  const frame = frames[state.currentFrame];
  const res = state.simulation.resolution;
  const buildings = state.simulation.buildings || [];
  const epicenter = state.simulation.epicenter;
  const grid = state.mode === "waves" ? frame.heat : frame.heat;

  if (resizeCanvas(physicsCanvas)) {
    drawGrid(physicsCtx, physicsCanvas, grid, res, state.mode, epicenter, buildings, frame.building_states, frame);
  }
  if (resizeCanvas(aiCanvas)) {
    drawAiGrid(aiCtx, aiCanvas, state.simulation.ai_prediction.damage_grid, res, epicenter, buildings);
  }

  $("timelineLabel").textContent = `${Number(frame.t).toFixed(2)} s`;
  $("physicsFrameLabel").textContent = `${state.currentFrame + 1}/${frames.length}`;
  $("pRadiusText").textContent = `${Number(frame.p_radius * state.simulation.map_scale_km).toFixed(1)} km`;
  $("sRadiusText").textContent = `${Number(frame.s_radius * state.simulation.map_scale_km).toFixed(1)} km`;

  updateBuildingList(buildings, frame.building_states);
}

function compareDamageDifference() {
  if (!state.simulation) return "n/a";
  const physics = Number(state.simulation.metrics.critical_area_pct || 0);
  const ai = Number(state.simulation.metrics.estimated_damage_pct || 0);
  return `${Math.abs(ai - physics).toFixed(1)} pts`;
}

function updateAiPanel() {
  if (!state.simulation) return;
  $("aiModeText").textContent = state.simulation.ai_prediction.mode;
  $("aiConfidenceText").textContent = `${Number(state.simulation.ai_prediction.confidence || 0).toFixed(0)}%`;
  $("compareDeltaText").textContent = compareDamageDifference();
  $("aiExplanation").textContent = state.simulation.ai_prediction.explanation;
}

function updateResultsFromSimulation() {
  if (!state.simulation) return;
  updateMetrics(state.simulation.metrics);
  updateSummary(state.simulation.damage_summary);
  updateAiPanel();
  $("timeline").max = String(Math.max(0, state.simulation.frames.length - 1));
  renderFrame(0);
}

async function runSimulation() {
  state.loading = true;
  document.body.classList.add("loading");
  $("runBtn").textContent = "Running...";

  const payload = {
    preset: state.preset,
    magnitude: state.magnitude,
    depth_km: state.depth_km,
    epicenter_x: state.epicenter_x,
    epicenter_y: state.epicenter_y,
    medium_type: state.medium_type,
    wave_speed: state.wave_speed,
    duration: state.duration,
    resolution: state.resolution,
  };

  try {
    const res = await fetch("/api/earthquake/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || `simulation request failed with ${res.status}`);
    }
    const data = await res.json();
    if (!data || !Array.isArray(data.frames)) {
      throw new Error("simulation response did not include frame data");
    }
    state.simulation = data;
    state.playing = false;
    $("playPauseBtn").textContent = "Play";
    updateResultsFromSimulation();
  } catch (error) {
    $("aiExplanation").textContent = `Simulation failed: ${error.message || error}. Restart the FastAPI server so the new /api/earthquake routes are loaded.`;
  } finally {
    state.loading = false;
    document.body.classList.remove("loading");
    $("runBtn").textContent = "Run Simulation";
  }
}

function tick(ts) {
  if (state.playing && state.simulation && state.simulation.frames.length) {
    if (!state.lastTick) state.lastTick = ts;
    const delta = ts - state.lastTick;
    const fpsStep = 1000 / (10 * state.playbackSpeed);
    if (delta >= fpsStep) {
      state.lastTick = ts;
      const next = state.currentFrame + 1;
      if (next >= state.simulation.frames.length) {
        state.playing = false;
        $("playPauseBtn").textContent = "Play";
      } else {
        renderFrame(next);
      }
    }
  } else {
    state.lastTick = ts;
  }
  requestAnimationFrame(tick);
}

function downloadReport() {
  if (!state.simulation) return;
  const report = {
    preset: state.preset,
    controls: {
      magnitude: state.magnitude,
      depth_km: state.depth_km,
      epicenter_x: state.epicenter_x,
      epicenter_y: state.epicenter_y,
      medium_type: state.medium_type,
      wave_speed: state.wave_speed,
      duration: state.duration,
    },
    metrics: state.simulation.metrics,
    damage_summary: state.simulation.damage_summary,
    ai_prediction: state.simulation.ai_prediction,
    buildings: state.simulation.buildings,
  };
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `earthquake-report-${state.preset}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

function handleEpicenterPlacement(event) {
  const rect = physicsCanvas.getBoundingClientRect();
  const x = clamp((event.clientX - rect.left) / rect.width, 0.02, 0.98);
  const y = clamp((event.clientY - rect.top) / rect.height, 0.02, 0.98);
  state.epicenter_x = x;
  state.epicenter_y = y;
  updateControlReadouts();
  drawPlaceholder();
}

function bindControls() {
  [["magnitude", "magnitude"], ["depth", "depth_km"], ["waveSpeed", "wave_speed"], ["duration", "duration"]].forEach(([id, key]) => {
    const el = $(id);
    el.addEventListener("input", () => {
      state[key] = Number(el.value);
      setSliderTrack(el);
      updateControlReadouts();
    });
  });

  $("mediumType").addEventListener("change", (event) => {
    state.medium_type = event.target.value;
  });

  $("playbackSpeed").addEventListener("change", (event) => {
    state.playbackSpeed = Number(event.target.value);
  });

  $("timeline").addEventListener("input", (event) => {
    renderFrame(Number(event.target.value));
  });

  $("runBtn").addEventListener("click", runSimulation);
  $("downloadBtn").addEventListener("click", downloadReport);
  $("playPauseBtn").addEventListener("click", () => {
    if (!state.simulation) return;
    state.playing = !state.playing;
    $("playPauseBtn").textContent = state.playing ? "Pause" : "Play";
  });
  $("resetPlaybackBtn").addEventListener("click", () => {
    state.playing = false;
    $("playPauseBtn").textContent = "Play";
    renderFrame(0);
  });

  document.querySelectorAll("[data-view]").forEach((btn) => {
    btn.addEventListener("click", () => setView(btn.dataset.view));
  });

  document.querySelectorAll("[data-mode]").forEach((btn) => {
    btn.addEventListener("click", () => setMode(btn.dataset.mode));
  });

  $("presetGrid").addEventListener("click", (event) => {
    const btn = event.target.closest("[data-preset]");
    if (!btn) return;
    applyPreset(btn.dataset.preset);
  });

  physicsCanvas.addEventListener("click", handleEpicenterPlacement);
  window.addEventListener("resize", () => renderFrame(state.currentFrame));
}

async function init() {
  bindControls();
  syncControlsToState();
  setView("physics");
  await loadPresets();
  drawPlaceholder();
  requestAnimationFrame(tick);
}

init();
