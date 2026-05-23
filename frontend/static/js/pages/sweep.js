/**
 * Sweep (Optuna) — visualització en temps real del sweep d'hiperparàmetres.
 *
 * Polling cada 30s. Llegeix:
 *   - /api/sweep/status     → resum global
 *   - /api/sweep/trials     → llista de trials (per la taula top-20 i scatter)
 *   - /api/sweep/best       → millor trial (per la corba ROC)
 *   - /api/sweep/importance → importància hiperparàmetres
 *
 * El frontend NO entrena res. Només llegeix els fitxers de ~/outputs/sweep/.
 */
import { API } from "../api.js";

let _pollTimer = null;

export function stopSweep() {
  if (_pollTimer) { clearInterval(_pollTimer); _pollTimer = null; }
}

export async function renderSweep(container) {
  container.innerHTML = `
    <div class="page-header">
      <h1 class="page-title">Sweep — Optuna</h1>
      <p class="page-sub">Cerca bayesiana del millor model (Sens=100%, Espec màx). Polling cada 30s.</p>
    </div>

    <div id="sw-status-bar" class="section" style="display:flex;gap:24px;flex-wrap:wrap"></div>

    <div class="section">
      <h2 class="section-title">Millor model fins ara</h2>
      <div id="sw-best"></div>
    </div>

    <div class="section">
      <h2 class="section-title">Evolució de l'objectiu</h2>
      <div id="sw-plot-objective" style="height:320px"></div>
    </div>

    <div class="section">
      <h2 class="section-title">Sens vs Spec (per trial)</h2>
      <div id="sw-plot-scatter" style="height:340px"></div>
    </div>

    <div class="section">
      <h2 class="section-title">Corba ROC del millor model (per fold)</h2>
      <div id="sw-plot-roc" style="height:380px"></div>
    </div>

    <div class="section">
      <h2 class="section-title">Importància d'hiperparàmetres</h2>
      <div id="sw-plot-importance" style="height:300px"></div>
    </div>

    <div class="section">
      <h2 class="section-title">Top-20 trials</h2>
      <div class="table-scroll">
        <table id="sw-table" class="data-table">
          <thead>
            <tr>
              <th>#</th><th>Sens</th><th>Spec</th><th>AUC</th>
              <th>Threshold</th><th>BS</th><th>Params</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
  `;

  injectCSS();
  await refresh();
  // Polling cada 30s
  _pollTimer = setInterval(refresh, 30_000);
}

async function refresh() {
  // ── 1) Status header ────────────────────────────────────────────────────
  let status;
  try {
    status = await API.sweepStatus();
  } catch {
    document.getElementById("sw-status-bar").innerHTML =
      `<div class="empty-state">No s'ha pogut connectar amb el backend.</div>`;
    return;
  }

  const statusBar = document.getElementById("sw-status-bar");
  if (!status.exists || status.trials_total === 0) {
    statusBar.innerHTML = `<div class="empty-state">El sweep encara no s'ha iniciat. Llança: <code>cd pt1diagnosis && python sweep.py</code></div>`;
    return;
  }
  statusBar.innerHTML = `
    <div class="metric-card">
      <div class="metric-label">Trials completats</div>
      <div class="metric-value">${status.trials_total}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Status</div>
      <div class="metric-value" style="color:${status.running ? "#4ade80" : "#fbbf24"}">
        ${status.running ? "Running" : "Idle"}
      </div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Millor objectiu</div>
      <div class="metric-value">${status.best?.value?.toFixed(4) ?? "—"}</div>
    </div>
    <div class="metric-card">
      <div class="metric-label">Darrer trial</div>
      <div class="metric-value" style="font-size:13px;font-family:var(--mono)">
        ${status.last_trial_ts?.slice(0,19) ?? "—"}
      </div>
    </div>
  `;

  // ── 2) Best model card ─────────────────────────────────────────────────
  const bestEl = document.getElementById("sw-best");
  if (status.best) {
    const ua = status.best.user_attrs || {};
    bestEl.innerHTML = `
      <div style="display:flex;gap:30px;flex-wrap:wrap;margin-bottom:10px">
        <div><strong>Sens:</strong> ${(ua.sens_mean ?? 0).toFixed(3)}</div>
        <div><strong>Spec:</strong> ${(ua.spec_mean ?? 0).toFixed(3)}</div>
        <div><strong>AUC:</strong> ${(ua.auc_mean ?? 0).toFixed(3)}</div>
        <div><strong>Threshold:</strong> ${(ua.threshold_med ?? 0).toFixed(4)}</div>
        <div><strong>Batch size:</strong> ${ua.batch_size ?? "—"}</div>
        <div><strong>Trial #:</strong> ${status.best.trial_id}</div>
      </div>
      <details>
        <summary style="cursor:pointer;color:var(--text2);font-size:12px">Params del millor trial</summary>
        <pre style="font-size:11.5px;background:var(--bg2);padding:10px;border-radius:6px;overflow-x:auto;margin-top:6px">${escapeHtml(JSON.stringify(status.best.params, null, 2))}</pre>
      </details>
    `;
  } else {
    bestEl.innerHTML = `<div class="empty-state">Cap millor trial encara.</div>`;
  }

  // ── 3) Trials + plots ──────────────────────────────────────────────────
  let trialsResp;
  try { trialsResp = await API.sweepTrials(200); }
  catch { return; }
  const trials = trialsResp.trials || [];

  drawObjectivePlot(trials);
  drawScatterPlot(trials);
  fillTable(trials.slice(0, 20));

  // ROC del millor model
  try {
    const best = await API.sweepBest();
    drawRocPlot(best);
  } catch { /* ignore */ }

  // Importance
  try {
    const imp = await API.sweepImportance();
    drawImportancePlot(imp);
  } catch { /* ignore */ }
}

// ── Plots ────────────────────────────────────────────────────────────────────

function drawObjectivePlot(trials) {
  const sorted = [...trials].sort((a, b) => a.trial_id - b.trial_id);
  const best = [...sorted].sort((a, b) => b.objective - a.objective)[0];

  const traces = [{
    x: sorted.map(t => t.trial_id),
    y: sorted.map(t => t.objective),
    mode: "markers",
    type: "scatter",
    marker: { size: 6, color: "#cc00a8" },
    name: "Objectiu",
  }];
  if (best) {
    traces.push({
      x: [best.trial_id], y: [best.objective],
      mode: "markers",
      marker: { size: 14, color: "#4ade80", symbol: "star",
                line: { color: "#fff", width: 1 } },
      name: "Millor",
    });
  }
  Plotly.newPlot("sw-plot-objective", traces, {
    paper_bgcolor: "#0a0a0a", plot_bgcolor: "#141414",
    xaxis: { title: "Trial #", gridcolor: "#222", color: "#ccc" },
    yaxis: { title: "Spec - penalty(Sens<1)", gridcolor: "#222", color: "#ccc" },
    font: { color: "#ccc", size: 11 },
    margin: { t: 20, b: 50, l: 60, r: 20 },
    showlegend: true,
  }, { displayModeBar: false, responsive: true });
}

function drawScatterPlot(trials) {
  const traces = [{
    x: trials.map(t => t.sens_mean),
    y: trials.map(t => t.spec_mean),
    mode: "markers",
    type: "scatter",
    marker: {
      size: 7,
      color: trials.map(t => t.auc_mean),
      colorscale: "Viridis",
      colorbar: { title: "AUC", tickfont: { color: "#ccc" } },
      line: { color: "#000", width: 0.3 },
    },
    text: trials.map(t => `#${t.trial_id}<br>Obj: ${t.objective.toFixed(3)}`),
    hovertemplate: "Sens=%{x:.3f}<br>Spec=%{y:.3f}<br>%{text}<extra></extra>",
    name: "Trials",
  }];
  // Best point
  if (trials.length) {
    const best = trials[0];  // ja ordenats per objectiu desc
    traces.push({
      x: [best.sens_mean], y: [best.spec_mean],
      mode: "markers",
      marker: { size: 18, color: "#cc00a8", symbol: "star",
                line: { color: "#fff", width: 1 } },
      name: "Millor",
    });
  }
  Plotly.newPlot("sw-plot-scatter", traces, {
    paper_bgcolor: "#0a0a0a", plot_bgcolor: "#141414",
    xaxis: { title: "Sensitivity (CV mean @ t*)", range: [0, 1.02],
             gridcolor: "#222", color: "#ccc" },
    yaxis: { title: "Specificity (CV mean @ t*)", range: [0, 1.02],
             gridcolor: "#222", color: "#ccc" },
    shapes: [{
      type: "line", x0: 1.0, x1: 1.0, y0: 0, y1: 1.0,
      line: { color: "#4ade80", width: 1.5, dash: "dash" },
    }],
    annotations: [{
      x: 1.0, y: 1.02, text: "Sens=1.0 (objectiu)",
      showarrow: false, font: { color: "#4ade80", size: 10 },
      xanchor: "right",
    }],
    font: { color: "#ccc", size: 11 },
    margin: { t: 20, b: 50, l: 60, r: 20 },
  }, { displayModeBar: false, responsive: true });
}

function drawRocPlot(best) {
  const folds = best.folds || [];
  if (!folds.length) {
    document.getElementById("sw-plot-roc").innerHTML =
      `<div class="empty-state">Cap fold guardat encara.</div>`;
    return;
  }
  // Compute ROC curves client-side via sorted thresholds
  const traces = folds.map((f, i) => {
    const { fpr, tpr } = computeRoc(f.probs, f.labels);
    return {
      x: fpr, y: tpr, mode: "lines", type: "scatter",
      line: { width: 1.5, color: i === 0 ? "#cc00a8" : "#888" },
      name: `Fold ${f.fold}`,
    };
  });
  // Diagonal random
  traces.push({
    x: [0, 1], y: [0, 1], mode: "lines",
    line: { color: "#444", width: 1, dash: "dash" },
    showlegend: false, hoverinfo: "skip",
  });
  // Clinical reference points
  const clinical = [
    { name: "JSCCR",  sens: 1.0,   spec: 0.19, color: "#fbbf24" },
    { name: "NCCN",   sens: 0.98,  spec: 0.52, color: "#fbbf24" },
    { name: "ESMO",   sens: 0.98,  spec: 0.50, color: "#fbbf24" },
    { name: "LASSO",  sens: 1.0,   spec: 0.858, color: "#60a5fa" },
  ];
  traces.push({
    x: clinical.map(c => 1 - c.spec),
    y: clinical.map(c => c.sens),
    mode: "markers+text",
    text: clinical.map(c => c.name),
    textposition: "top right",
    marker: { size: 11, color: clinical.map(c => c.color), symbol: "square" },
    name: "Referències",
    textfont: { size: 10, color: "#ccc" },
  });
  Plotly.newPlot("sw-plot-roc", traces, {
    paper_bgcolor: "#0a0a0a", plot_bgcolor: "#141414",
    xaxis: { title: "1 - Specificity (FPR)", range: [0, 1],
             gridcolor: "#222", color: "#ccc" },
    yaxis: { title: "Sensitivity (TPR)", range: [0, 1.02],
             gridcolor: "#222", color: "#ccc" },
    font: { color: "#ccc", size: 11 },
    margin: { t: 20, b: 50, l: 60, r: 20 },
    showlegend: true,
  }, { displayModeBar: false, responsive: true });
}

function drawImportancePlot(impResp) {
  const imp = impResp.importance || {};
  const entries = Object.entries(imp).sort((a, b) => b[1] - a[1]);
  if (!entries.length) {
    document.getElementById("sw-plot-importance").innerHTML =
      `<div class="empty-state">${impResp.note || "Sense dades d'importància encara."}</div>`;
    return;
  }
  Plotly.newPlot("sw-plot-importance", [{
    x: entries.map(e => e[1]),
    y: entries.map(e => e[0]),
    type: "bar",
    orientation: "h",
    marker: { color: "#cc00a8" },
  }], {
    paper_bgcolor: "#0a0a0a", plot_bgcolor: "#141414",
    xaxis: { title: "Importància", gridcolor: "#222", color: "#ccc" },
    yaxis: { color: "#ccc", autorange: "reversed" },
    font: { color: "#ccc", size: 11 },
    margin: { t: 20, b: 50, l: 130, r: 20 },
  }, { displayModeBar: false, responsive: true });
}

function fillTable(trials) {
  const tbody = document.querySelector("#sw-table tbody");
  if (!trials.length) {
    tbody.innerHTML = `<tr><td colspan="7" style="text-align:center;color:var(--text3)">Cap trial encara.</td></tr>`;
    return;
  }
  tbody.innerHTML = trials.map(t => `
    <tr>
      <td>#${t.trial_id}</td>
      <td>${t.sens_mean.toFixed(3)} ± ${t.sens_std.toFixed(3)}</td>
      <td>${t.spec_mean.toFixed(3)} ± ${t.spec_std.toFixed(3)}</td>
      <td>${t.auc_mean.toFixed(3)} ± ${t.auc_std.toFixed(3)}</td>
      <td>${t.threshold_med.toFixed(4)}</td>
      <td>${t.batch_size}</td>
      <td><code style="font-size:10.5px">${shortParams(t.params)}</code></td>
    </tr>
  `).join("");
}

// ── helpers ─────────────────────────────────────────────────────────────────

function computeRoc(probs, labels) {
  // Pairs sorted by score desc
  const pairs = probs.map((p, i) => [p, labels[i]])
                     .sort((a, b) => b[0] - a[0]);
  const P = labels.reduce((s, l) => s + l, 0);
  const N = labels.length - P;
  let tp = 0, fp = 0;
  const fpr = [0], tpr = [0];
  for (const [_, lbl] of pairs) {
    if (lbl === 1) tp++; else fp++;
    fpr.push(N ? fp / N : 0);
    tpr.push(P ? tp / P : 0);
  }
  return { fpr, tpr };
}

function shortParams(p) {
  if (!p) return "";
  const keys = ["architecture", "pooling", "mil", "hidden", "heads", "dropout"];
  return keys.filter(k => k in p)
             .map(k => `${k}=${typeof p[k] === "number" ? p[k].toFixed(3) : p[k]}`)
             .join(" ");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

// ── injected CSS ─────────────────────────────────────────────────────────────
function injectCSS() {
  if (document.getElementById("sweep-css")) return;
  const css = `
    .metric-card {
      background: var(--bg2); border: 1px solid var(--border);
      padding: 10px 16px; border-radius: 8px; min-width: 120px;
    }
    .metric-label { font-size: 11px; color: var(--text3); text-transform: uppercase;
                    letter-spacing: 0.5px; }
    .metric-value { font-size: 22px; font-weight: 600; color: var(--text);
                    margin-top: 4px; font-family: var(--mono); }
    .table-scroll { overflow-x: auto; }
    .data-table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
    .data-table th, .data-table td {
      padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border);
    }
    .data-table th { color: var(--text3); font-weight: 600;
                     font-size: 11px; text-transform: uppercase;
                     letter-spacing: 0.5px; }
    .data-table td { color: var(--text); }
    .data-table tr:hover td { background: var(--bg2); }
  `;
  const style = document.createElement("style");
  style.id = "sweep-css";
  style.textContent = css;
  document.head.appendChild(style);
}
