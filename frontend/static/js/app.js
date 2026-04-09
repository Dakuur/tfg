/**
 * HistoGAT Explorer — main app entry point
 * Handles routing, status bar, debug toggle, and page rendering.
 */
import { API } from "./api.js";
import { renderDashboard }  from "./pages/dashboard.js";
import { renderInference, appendDebugLog } from "./pages/inference.js";
import { renderStatistics } from "./pages/statistics.js";

// ── State ──────────────────────────────────────────────────────────────────────
const STATE = {
  page: "dashboard",
  debug: false,
  status: null,
};

// ── DOM refs ───────────────────────────────────────────────────────────────────
const content    = document.getElementById("content");
const statusBar  = document.getElementById("status-bar");
const debugPanel = document.getElementById("debug-panel");
const debugLog   = document.getElementById("debug-log");
const debugToggle= document.getElementById("debug-toggle");
const reloadBtn  = document.getElementById("reload-btn");

// ── Routing ────────────────────────────────────────────────────────────────────
async function navigate(page) {
  STATE.page = page;

  // Update nav active state
  document.querySelectorAll(".nav-item").forEach(el => {
    el.classList.toggle("active", el.dataset.page === page);
  });

  content.innerHTML = `<div class="loading-spinner"><div class="spinner"></div><p>Cargando…</p></div>`;

  switch (page) {
    case "dashboard":  await renderDashboard(content); break;
    case "inference":  await renderInference(content, STATE.debug); break;
    case "statistics": await renderStatistics(content); break;
  }

  lucide.createIcons();
}

// ── Status bar ─────────────────────────────────────────────────────────────────
async function refreshStatus() {
  try {
    STATE.status = await API.status();
    renderStatusBar(STATE.status);
  } catch {
    statusBar.innerHTML = pill("error", "Backend offline");
  }
}

function renderStatusBar(s) {
  const modelPill = s.model_loaded
    ? pill("ok", `Modelo cargado · epoch ${s.checkpoint?.epoch ?? "?"}`)
    : pill("warn", "Sin modelo");

  const aucPill = s.checkpoint?.val_auc != null
    ? pill("info", `AUC ${s.checkpoint.val_auc.toFixed(3)}`)
    : "";

  const devicePill = pill("info", s.device?.toUpperCase() || "CPU");

  const graphsPill = pill(
    s.num_train_graphs + s.num_val_graphs > 0 ? "ok" : "warn",
    `${s.num_train_graphs + s.num_val_graphs} grafos`
  );

  statusBar.innerHTML = `
    ${modelPill}
    <div class="status-sep"></div>
    ${aucPill}
    ${devicePill}
    <div class="status-sep"></div>
    ${graphsPill}
    <div style="flex:1"></div>
    <span style="font-size:11.5px;color:var(--text3);font-family:var(--mono)">HistoGAT v1.0</span>
  `;
}

function pill(type, text) {
  const dot = type !== "info" ? `<div class="dot"></div>` : "";
  return `<div class="status-pill ${type}">${dot}${text}</div>`;
}

// ── Debug mode ─────────────────────────────────────────────────────────────────
function setDebug(on) {
  STATE.debug = on;
  debugPanel.classList.toggle("hidden", !on);

  if (on) {
    appendDebugLog({ level: "info", msg: "Debug mode activado", t: Date.now() });
    appendDebugLog({ level: "info", msg: `Dispositivo: ${STATE.status?.device?.toUpperCase() || "?"}`, t: Date.now() });
    appendDebugLog({ level: "info", msg: `Modelo: ${STATE.status?.checkpoint?.name || "sin checkpoint"}`, t: Date.now() });
    appendDebugLog({ level: "info", msg: `Grafos: ${(STATE.status?.num_train_graphs ?? 0) + (STATE.status?.num_val_graphs ?? 0)} totales`, t: Date.now() });
  }
}

debugToggle.addEventListener("change", () => setDebug(debugToggle.checked));

document.getElementById("clear-debug-btn")?.addEventListener("click", () => {
  debugLog.innerHTML = "";
});

// ── Nav ────────────────────────────────────────────────────────────────────────
document.querySelectorAll(".nav-item").forEach(el => {
  el.addEventListener("click", () => navigate(el.dataset.page));
});

// ── Reload ─────────────────────────────────────────────────────────────────────
reloadBtn.addEventListener("click", async () => {
  const icon = reloadBtn.querySelector("svg");
  icon?.classList.add("spinning");
  reloadBtn.disabled = true;

  if (STATE.debug) appendDebugLog({ level: "info", msg: "Recargando modelo y grafos…", t: Date.now() });

  try {
    const res = await API.reload();
    if (STATE.debug) appendDebugLog({
      level: res.success ? "success" : "error",
      msg: res.success
        ? `Recarga OK — ${res.num_train} train, ${res.num_val} val`
        : `Error: ${res.error}`,
      t: Date.now(),
    });
    await refreshStatus();
    navigate(STATE.page); // re-render current page
  } catch (e) {
    if (STATE.debug) appendDebugLog({ level: "error", msg: `Reload failed: ${e.message}`, t: Date.now() });
  }

  icon?.classList.remove("spinning");
  reloadBtn.disabled = false;
});

// ── Additional CSS for arch diagram & animations ───────────────────────────────
const extraCSS = `
.arch-layer {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  margin: 2px 0;
}
.arch-name {
  font-weight: 600;
  font-size: 12.5px;
  color: var(--accent-light);
  min-width: 52px;
  font-family: var(--mono);
}
.arch-desc { font-size: 12px; color: var(--text2); flex: 1; }
.arch-meta { font-size: 11px; color: var(--text3); font-family: var(--mono); }
.arch-arrow { text-align: center; color: var(--text3); font-size: 11px; padding: 1px 0; }
.card-warn { border-color: rgba(255,187,68,0.3); }

@keyframes spinning { to { transform: rotate(360deg); } }
.spinning { animation: spinning 0.7s linear infinite; }
`;

const style = document.createElement("style");
style.textContent = extraCSS;
document.head.appendChild(style);

// ── Init ───────────────────────────────────────────────────────────────────────
async function init() {
  lucide.createIcons();
  await refreshStatus();
  await navigate("dashboard");

  // Refresh status every 30s
  setInterval(refreshStatus, 30_000);
}

init();
