import { API } from "../api.js";
import { renderGraph, renderAttentionBars } from "../components/graphViz.js";

let _patients  = [];
let _filtered  = [];
let _selectedPid = null;
let _result    = null;
let _debugMode = false;

export async function renderInference(container, debugMode = false) {
  _debugMode = debugMode;

  container.innerHTML = `<div class="loading-spinner"><div class="spinner"></div><p>Carregant pacients…</p></div>`;

  try {
    const data = await API.patients();
    _patients = data.patients || [];
  } catch (e) {
    container.innerHTML = `<div class="empty-state"><p>Error en carregar pacients</p><small>${e.message}</small></div>`;
    return;
  }

  _filtered    = [..._patients];
  _selectedPid = null;
  _result      = null;

  container.innerHTML = buildLayout();
  lucide.createIcons();
  attachEvents(container);
  renderTable(container);
}

function buildLayout() {
  return `
    <div class="page-header">
      <h1 class="page-title">Inferència</h1>
      <p class="page-sub">Selecciona un pacient — la predicció s'agrega sobre totes les seves slides</p>
    </div>

    <div class="two-col" style="align-items:start">
      <!-- Esquerra: selector de pacients -->
      <div>
        <div class="section">
          <div class="section-title"><i data-lucide="users"></i> Seleccionar pacient</div>
          <div class="card" style="padding:14px">
            <div style="display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap">
              <div class="search-wrap" style="flex:1;min-width:140px">
                <i data-lucide="search"></i>
                <input class="search-input" id="patient-search" placeholder="Cerca per pacient, hospital…" />
              </div>
              <select id="split-filter" style="padding:8px 10px;background:var(--bg3);border:1px solid var(--border);border-radius:var(--radius);color:var(--text);font-size:13px">
                <option value="">Tots</option>
                <option value="train">Train</option>
                <option value="val">Val</option>
              </select>
              <select id="label-filter" style="padding:8px 10px;background:var(--bg3);border:1px solid var(--border);border-radius:var(--radius);color:var(--text);font-size:13px">
                <option value="">Totes les classes</option>
                <option value="0">N0</option>
                <option value="1">N1</option>
              </select>
            </div>
            <div class="table-wrap" style="max-height:320px;overflow-y:auto" id="patient-table-wrap"></div>
            <div id="table-count" style="font-size:11.5px;color:var(--text3);margin-top:8px;text-align:right"></div>
          </div>
        </div>

        <!-- Botó d'execució -->
        <div class="section">
          <button class="btn btn-primary" id="run-btn" disabled style="width:100%;justify-content:center">
            <i data-lucide="play-circle"></i> Executar inferència del pacient
          </button>
          <div id="progress-wrap" class="progress-container" style="display:none">
            <div class="progress-label"><span id="progress-label-text">Processant…</span><span id="progress-pct"></span></div>
            <div class="progress-bar"><div class="progress-fill indeterminate" id="progress-fill"></div></div>
          </div>
        </div>
      </div>

      <!-- Dreta: resultat del pacient -->
      <div id="result-area">
        <div class="empty-state" style="padding:40px 20px">
          <i data-lucide="user"></i>
          <p>Selecciona un pacient i executa la inferència</p>
        </div>
      </div>
    </div>

    <!-- Breakdown per slide -->
    <div id="slides-section" style="display:none" class="section">
      <div class="section-title"><i data-lucide="layers"></i> Resultats per slide</div>
      <div class="card" style="padding:14px" id="slides-content"></div>
    </div>

    <!-- Visualització del graf de la slide més rellevant -->
    <div id="viz-section" style="display:none" class="section">
      <div class="section-title">
        <i data-lucide="share-2"></i> Estructura del graf
        <span id="viz-slide-label" style="font-size:12px;color:var(--text3);margin-left:8px"></span>
      </div>
      <div style="margin-bottom:8px;font-size:12px;color:var(--text3)">
        <i data-lucide="info" style="width:13px;height:13px;vertical-align:middle"></i>
        Clic sobre un node per veure el patch d'histopatologia
      </div>
      <div class="two-col" style="align-items:start">
        <div>
          <div class="graph-viz-wrap" id="graph-svg-container" style="height:420px"></div>
          <div class="graph-legend" style="margin-top:6px">
            <div class="legend-item">
              <canvas id="attn-legend-canvas" width="120" height="14" style="border-radius:3px;vertical-align:middle"></canvas>
              <span style="font-size:11px;color:var(--text3);margin-left:4px">baixa → alta atenció</span>
            </div>
            <div class="legend-item" style="margin-left:auto;color:var(--text3)">Scroll per zoom · Arrossega per moure</div>
          </div>
        </div>
        <div>
          <div class="card" id="graph-meta-card"></div>
        </div>
      </div>
    </div>

    <!-- Capes d'atenció -->
    <div id="attention-section" style="display:none" class="section">
      <div class="section-title"><i data-lucide="eye"></i> Capes d'atenció GAT</div>
      <div class="attention-wrap">
        <div class="tabs" id="attn-tabs">
          <button class="tab active" data-layer="layer1">Capa 1</button>
          <button class="tab" data-layer="layer2">Capa 2</button>
          <button class="tab" data-layer="layer3">Capa 3</button>
        </div>
        <div style="padding:16px" id="attn-content"></div>
      </div>
    </div>
  `;
}

function attachEvents(container) {
  container.querySelector("#patient-search").addEventListener("input", () => applyFilters(container));
  container.querySelector("#split-filter").addEventListener("change", () => applyFilters(container));
  container.querySelector("#label-filter").addEventListener("change", () => applyFilters(container));
  container.querySelector("#run-btn").addEventListener("click", () => runInference(container));
}

function applyFilters(container) {
  const q     = container.querySelector("#patient-search").value.toLowerCase();
  const split = container.querySelector("#split-filter").value;
  const label = container.querySelector("#label-filter").value;

  _filtered = _patients.filter(p => {
    if (split && !p.splits.includes(split)) return false;
    if (label !== "" && String(p.label) !== label) return false;
    if (q && !`${p.patient_id} ${p.hospital}`.toLowerCase().includes(q)) return false;
    return true;
  });

  renderTable(container);
}

function renderTable(container) {
  const wrap  = container.querySelector("#patient-table-wrap");
  const count = container.querySelector("#table-count");

  if (!_filtered.length) {
    wrap.innerHTML = `<div class="empty-state" style="padding:30px"><p>Sense resultats</p></div>`;
    count.textContent = "";
    return;
  }

  const rows = _filtered.slice(0, 200).map(p => `
    <tr data-pid="${p.patient_id}" class="${p.patient_id === _selectedPid ? "selected" : ""}">
      <td style="font-size:12px;max-width:130px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${p.patient_id}</td>
      <td style="font-size:11.5px;color:var(--text2)">${p.hospital}</td>
      <td><span class="badge badge-${p.label === 0 ? "n0" : p.label === 1 ? "n1" : "unk"}">${p.label === 0 ? "N0" : p.label === 1 ? "N1" : "?"}</span></td>
      <td class="mono" style="font-size:11.5px;text-align:center">${p.num_slides}</td>
      <td style="font-size:11px;color:var(--text3)">${p.splits.join(", ")}</td>
    </tr>
  `).join("");

  wrap.innerHTML = `
    <table>
      <thead><tr><th>Pacient</th><th>Hospital</th><th>Classe</th><th>Slides</th><th>Split</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;

  count.textContent = `${_filtered.length} pacient${_filtered.length !== 1 ? "s" : ""}`;

  wrap.querySelectorAll("tbody tr").forEach(row => {
    row.addEventListener("click", () => {
      _selectedPid = row.dataset.pid;
      wrap.querySelectorAll("tbody tr").forEach(r => r.classList.remove("selected"));
      row.classList.add("selected");
      container.querySelector("#run-btn").disabled = false;
    });
  });
}

const STEPS = [
  "Carregant slides del pacient…",
  "Executant inferència slide 1…",
  "Executant inferència slide 2…",
  "Agregant prediccions del pacient…",
  "Extraient pesos d'atenció…",
  "Processant resultats…",
];

async function runInference(container) {
  if (!_selectedPid) return;

  const runBtn       = container.querySelector("#run-btn");
  const progressWrap = container.querySelector("#progress-wrap");
  const progressFill = container.querySelector("#progress-fill");
  const progressLabel = container.querySelector("#progress-label-text");
  const progressPct  = container.querySelector("#progress-pct");

  runBtn.disabled = true;
  progressWrap.style.display = "block";
  progressFill.classList.add("indeterminate");
  progressFill.style.width = "";

  let stepIdx = 0;
  const stepInterval = setInterval(() => {
    if (stepIdx < STEPS.length) {
      progressLabel.textContent = STEPS[stepIdx];
      const pct = Math.round((stepIdx / STEPS.length) * 85);
      progressFill.classList.remove("indeterminate");
      progressFill.style.width = pct + "%";
      progressPct.textContent  = pct + "%";
      stepIdx++;
    }
  }, 400);

  appendDebugLog({ level: "info", msg: `Inferència pacient: ${_selectedPid}`, t: Date.now() });

  try {
    _result = await API.inferencePatient(_selectedPid, _debugMode);

    clearInterval(stepInterval);
    progressFill.style.width = "100%";
    progressPct.textContent  = "100%";
    progressLabel.textContent = "Completat!";

    if (_debugMode && _result.debug_log?.length) {
      _result.debug_log.forEach(e => appendDebugLog(e));
    }

    setTimeout(() => {
      progressWrap.style.display = "none";
      progressFill.style.width   = "";
      progressFill.classList.add("indeterminate");
    }, 800);

    renderResult(container, _result);
    renderSlideBreakdown(container, _result);
    await renderGraphViz(container, _result);
    renderAttention(container, _result, "layer3");

  } catch (e) {
    clearInterval(stepInterval);
    progressWrap.style.display = "none";
    appendDebugLog({ level: "error", msg: `Error: ${e.message}`, t: Date.now() });
    container.querySelector("#result-area").innerHTML = `
      <div class="card" style="border-color:var(--red)">
        <div style="color:var(--red);font-weight:600;margin-bottom:6px">Error en la inferència</div>
        <div style="font-size:12.5px;color:var(--text2)">${e.message}</div>
      </div>`;
  }

  runBtn.disabled = false;
}

function renderResult(container, r) {
  const isN1 = r.prediction === 1;
  const cls  = isN1 ? "n1" : "n0";

  const n1Slides = (r.slide_results || []).filter(s => s.pred === 1).length;
  const total    = r.num_slides || 0;

  const reasonHtml = isN1
    ? `<div class="result-reason n1">⚠ ${n1Slides} de ${total} slide${n1Slides !== 1 ? "s" : ""} amb predicció N1</div>`
    : `<div class="result-reason n0">✓ Cap slide amb predicció N1 (${total} slides)</div>`;

  const correctHtml = r.correct !== null
    ? `<span class="${r.correct ? "result-correct" : "result-incorrect"}">${r.correct ? "✓ Correcte" : "✗ Incorrecte"}</span>`
    : "";

  container.querySelector("#result-area").innerHTML = `
    <div class="result-panel">
      <div class="card-title">Predicció del pacient</div>
      <div class="result-prediction ${cls}">${r.label}</div>
      <div style="font-size:13px;color:var(--text2);margin-top:4px">
        P(N1)=${(r.prob_n1 * 100).toFixed(1)}% &nbsp; ${correctHtml}
      </div>
      ${reasonHtml}

      <div class="result-meta">
        <div class="result-meta-row">
          <span>Etiqueta real</span>
          <span class="result-meta-val">
            <span class="badge badge-${r.true_label === 0 ? "n0" : r.true_label === 1 ? "n1" : "unk"}">${r.true_label_name}</span>
          </span>
        </div>
        <div class="result-meta-row"><span>Pacient</span><span class="result-meta-val" style="font-size:12px">${r.patient_id || "—"}</span></div>
        <div class="result-meta-row"><span>Hospital</span><span class="result-meta-val" style="font-size:12px">${r.hospital || "—"}</span></div>
        <div class="result-meta-row"><span>Slides processades</span><span class="result-meta-val mono">${total}</span></div>
      </div>
    </div>`;

  lucide.createIcons();
}

function renderSlideBreakdown(container, r) {
  const section = container.querySelector("#slides-section");
  section.style.display = "";

  const slides = r.slide_results || [];
  if (!slides.length) {
    section.querySelector("#slides-content").innerHTML = `<div class="empty-state"><p>Sense dades de slides</p></div>`;
    return;
  }

  const rows = slides.map(s => {
    const isViz = s.graph_id === r.viz_graph_id;
    const pct   = (s.prob_n1 * 100).toFixed(1);
    const barW  = Math.round(s.prob_n1 * 100);
    return `
      <tr class="${isViz ? "selected" : ""}">
        <td style="font-size:11px;max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:var(--mono)">${s.graph_id.split("/").pop()}</td>
        <td><span class="badge badge-${s.pred === 0 ? "n0" : "n1"}">${s.label_name}</span></td>
        <td style="min-width:100px">
          <div style="display:flex;align-items:center;gap:6px">
            <div style="flex:1;height:6px;background:var(--bg3);border-radius:3px;overflow:hidden">
              <div style="height:100%;width:${barW}%;background:${s.pred === 1 ? "var(--red)" : "var(--green)"};border-radius:3px"></div>
            </div>
            <span class="mono" style="font-size:11px;min-width:36px">${pct}%</span>
          </div>
        </td>
        <td class="mono" style="font-size:11px">${s.num_nodes}</td>
        ${isViz ? `<td style="font-size:10.5px;color:var(--accent-light)">▶ viz</td>` : `<td></td>`}
      </tr>`;
  }).join("");

  section.querySelector("#slides-content").innerHTML = `
    <table>
      <thead><tr><th>Slide</th><th>Predicció</th><th>P(N1)</th><th>Nodes</th><th></th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
}

async function renderGraphViz(container, r) {
  const section = container.querySelector("#viz-section");
  section.style.display = "";

  const label = container.querySelector("#viz-slide-label");
  if (label) label.textContent = `(slide: ${(r.viz_graph_id || "").split("/").pop()})`;

  // Draw attention color legend
  const canvas = container.querySelector("#attn-legend-canvas");
  if (canvas) {
    const ctx = canvas.getContext("2d");
    const grad = ctx.createLinearGradient(0, 0, 120, 0);
    // d3.interpolateTurbo stops: blue→cyan→green→yellow→red
    for (let t = 0; t <= 1; t += 0.1) {
      const c = d3.color(d3.interpolateTurbo(t));
      grad.addColorStop(t, `rgb(${c.r},${c.g},${c.b})`);
    }
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 120, 14);
  }

  const svgContainer = container.querySelector("#graph-svg-container");
  const meta         = container.querySelector("#graph-meta-card");
  const attn3        = r.attention?.layer3;

  // Build slideInfo for patch click
  const slideInfo = {
    hospital:   r.hospital   || "",
    patient_id: r.patient_id || "",
    slide_id:   r.slide_id   || "",
    graph_id:   r.viz_graph_id || "",
    patch_j:    r.patch_j,
    patch_i:    r.patch_i,
  };

  // Try to load slide background
  let bgImageUrl = null;
  if (r.viz_graph_id) {
    bgImageUrl = `/api/slide_bg/${encodeURIComponent(r.viz_graph_id)}`;
    // Preload to check availability (non-blocking)
    const testImg = new Image();
    testImg.src = bgImageUrl;
    // We pass the URL regardless — if it fails, the <image> element just shows nothing
  }

  renderGraph(svgContainer, {
    edge_index:     r.edge_index,
    node_positions: r.node_positions,
    num_nodes:      r.num_nodes,
    feature_norms:  r.feature_norms,
  }, {
    nodeAttention: attn3?.node_attention,
    edgeAttention: attn3 ? { edge_index: attn3.edge_index, weights_mean: attn3.weights_mean } : null,
    height:    420,
    slideInfo,
    bgImageUrl,
  });

  const nodeAttn  = attn3?.node_attention || [];
  const topNodes  = nodeAttn.map((v, i) => ({ i, v })).sort((a, b) => b.v - a.v).slice(0, 5);

  meta.innerHTML = `
    <div class="card-title">Nodes més rellevants (Capa 3)</div>
    <div class="stat-list" style="margin-bottom:14px">
      ${topNodes.map((n, rank) => `
        <div class="stat-row">
          <span class="stat-key">${rank + 1}. Node ${n.i}</span>
          <span class="stat-val">${(n.v * 100).toFixed(1)}%</span>
        </div>`).join("")}
    </div>
    <div class="card-title">Slide visualitzada</div>
    <div class="stat-list">
      <div class="stat-row"><span class="stat-key">Nodes</span><span class="stat-val mono">${r.num_nodes}</span></div>
      <div class="stat-row"><span class="stat-key">Arestes dirigides</span><span class="stat-val mono">${r.num_edges}</span></div>
      <div class="stat-row"><span class="stat-key">Posicions reals</span><span class="stat-val">${r.node_positions ? "✓ Sí" : "No"}</span></div>
      <div class="stat-row"><span class="stat-key">Pooling</span><span class="stat-val accent">${r.pooling_type ?? "—"}</span></div>
      <div class="stat-row"><span class="stat-key">Patches</span><span class="stat-val">${r.patch_j ? "✓ disponibles" : "⚠ cal re-executar build_dataset"}</span></div>
    </div>`;

  lucide.createIcons();
}

function renderAttention(container, r, activeLayer) {
  const section = container.querySelector("#attention-section");
  section.style.display = "";

  section.querySelectorAll(".tab").forEach(tab => {
    tab.classList.toggle("active", tab.dataset.layer === activeLayer);
    tab.onclick = () => renderAttention(container, r, tab.dataset.layer);
  });

  const layerData = r.attention?.[activeLayer];
  if (!layerData) {
    section.querySelector("#attn-content").innerHTML = `<div class="empty-state"><p>Sense dades d'atenció</p></div>`;
    return;
  }

  const content = section.querySelector("#attn-content");

  // SlideInfo for attention mini-graph node clicks
  const slideInfo = {
    hospital:   r.hospital   || "",
    patient_id: r.patient_id || "",
    slide_id:   r.slide_id   || "",
    graph_id:   r.viz_graph_id || "",
    patch_j:    r.patch_j,
    patch_i:    r.patch_i,
  };

  content.innerHTML = `
    <div>
      <div class="section-title" style="margin-bottom:8px">Graf acolorit per atenció (${activeLayer.replace("layer", "Capa ")})</div>
      <div class="graph-viz-wrap" id="attn-graph-mini" style="height:300px"></div>
    </div>
    <div style="margin-top:12px">
      <div class="stat-list">
        <div class="stat-row"><span class="stat-key">Arestes (incl. self-loops)</span><span class="stat-val mono">${layerData.edge_index[0].length}</span></div>
        <div class="stat-row"><span class="stat-key">Caps d'atenció</span><span class="stat-val mono">${layerData.num_heads}</span></div>
        <div class="stat-row"><span class="stat-key">Atenció màxima</span><span class="stat-val mono">${Math.max(...layerData.node_attention).toFixed(4)}</span></div>
        <div class="stat-row"><span class="stat-key">Entropia</span><span class="stat-val mono">${entropy(layerData.weights_mean).toFixed(4)}</span></div>
      </div>
    </div>`;

  renderGraph(content.querySelector("#attn-graph-mini"), {
    edge_index:     r.edge_index,
    node_positions: r.node_positions,
    num_nodes:      r.num_nodes,
    feature_norms:  r.feature_norms,
  }, {
    nodeAttention: layerData.node_attention,
    edgeAttention: { edge_index: layerData.edge_index, weights_mean: layerData.weights_mean },
    height: 300,
    slideInfo,
  });
}

function entropy(vals) {
  const total = vals.reduce((s, v) => s + Math.abs(v), 0);
  if (total === 0) return 0;
  return -vals.reduce((s, v) => {
    const p = Math.abs(v) / total;
    return s + (p > 0 ? p * Math.log2(p) : 0);
  }, 0);
}

export function appendDebugLog(entry) {
  const log = document.querySelector("#debug-log");
  if (!log) return;
  const time = new Date(entry.t || Date.now()).toISOString().slice(11, 23);
  const div  = document.createElement("div");
  div.className = "log-entry";
  div.innerHTML = `<span class="log-time">${time}</span><span class="log-msg ${entry.level || "info"}">${entry.msg}</span>`;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}
