import { API } from "../api.js";
import { renderGraph } from "../components/graphViz.js";

// Set to true to re-enable the WSI background image behind the graph.
// Requires the server to have the *_low.jpg files available.
const SHOW_WSI_BACKGROUND = false;

let _patients      = [];
let _filtered      = [];
let _selectedPid   = null;
let _result        = null;   // patient-level result
let _debugMode     = false;
let _vizGraphId    = null;   // which slide is currently shown in viz

// Viz state
let _vizData       = null;
let _vizBgImageUrl = null;
let _vizWsiExtent  = null;
let _vizSlideInfo  = null;

function _availableLayers() {
  // Retorna les keys disponibles ordenades (layer1, layer2, ...).
  const attn = _vizData?.attention;
  if (!attn) return [];
  return Object.keys(attn)
    .filter(k => /^layer\d+$/.test(k))
    .sort((a, b) => Number(a.replace("layer", "")) - Number(b.replace("layer", "")));
}

function _reRenderGraph(container, scores, layerKey = null) {
  const svgContainer = container.querySelector("#graph-svg-container");
  if (!svgContainer || !_vizData) return;
  // Si no s'ha passat layerKey, agafem l'última capa disponible.
  const layers = _availableLayers();
  const useLayer = layerKey ?? layers[layers.length - 1] ?? null;
  const lyr = useLayer ? _vizData.attention?.[useLayer] : null;
  const effectiveScores = scores ?? lyr?.node_attention ?? null;
  // Per a capes >1 amb DiffPool, els nodes són super-nodes i no hi ha
  // posicions reals → no superposem el WSI ni mostrem patches.
  const layerIdx = useLayer ? Number(useLayer.replace("layer", "")) : 1;
  const isSuperNodeLayer = (_vizData.pooling_type === "diff" && layerIdx > 1);
  renderGraph(svgContainer, {
    edge_index:     lyr?.edge_index ?? _vizData.edge_index,
    node_positions: isSuperNodeLayer ? null : _vizData.node_positions,
    num_nodes:      lyr ? (lyr.node_attention?.length ?? _vizData.num_nodes) : _vizData.num_nodes,
    feature_norms:  isSuperNodeLayer ? null : _vizData.feature_norms,
  }, {
    nodeAttention: effectiveScores,
    edgeAttention: lyr ? { edge_index: lyr.edge_index, weights_mean: lyr.weights_mean } : null,
    height:        440,
    slideInfo:     isSuperNodeLayer ? null : _vizSlideInfo,
    bgImageUrl:    isSuperNodeLayer ? null : _vizBgImageUrl,
    wsiExtent:     isSuperNodeLayer ? null : _vizWsiExtent,
  });
}

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
  _vizGraphId  = null;

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
      <div style="font-size:11.5px;color:var(--text3);margin-top:6px">
        <i data-lucide="mouse-pointer-2" style="width:12px;height:12px;vertical-align:middle"></i>
        Fes clic a qualsevol slide per visualitzar-la
      </div>
    </div>

    <!-- Visualització del graf -->
    <div id="viz-section" style="display:none" class="section">
      <div class="section-title">
        <i data-lucide="share-2"></i> Estructura del graf
        <span id="viz-slide-label" style="font-size:12px;color:var(--text3);margin-left:8px"></span>
        <span id="viz-loading" style="display:none;font-size:11px;color:var(--accent-light);margin-left:8px">carregant…</span>
      </div>
      <div style="margin-bottom:10px;font-size:12px;color:var(--text3);display:flex;gap:10px;align-items:center;flex-wrap:wrap">
        <span><i data-lucide="info" style="width:13px;height:13px;vertical-align:middle"></i>
          Clic sobre un node per veure el patch · Scroll per zoom · Colors: atenció</span>
        <label style="display:inline-flex;gap:6px;align-items:center;color:var(--text)">
          Capa GAT:
          <select id="attn-layer-select" style="padding:3px 6px;background:var(--bg3);border:1px solid var(--border);border-radius:4px;color:var(--text);font-size:11.5px"></select>
        </label>
        <span id="attn-layer-note" style="color:var(--accent-light);font-size:11px"></span>
      </div>
      <div class="two-col" style="align-items:start">
        <div>
          <div class="graph-viz-wrap" id="graph-svg-container" style="height:440px"></div>
          <div class="graph-legend" style="margin-top:6px">
            <div class="legend-item">
              <canvas id="attn-legend-canvas" width="120" height="14" style="border-radius:3px;vertical-align:middle"></canvas>
              <span style="font-size:11px;color:var(--text3);margin-left:6px">baixa → alta atenció</span>
            </div>
            <div class="legend-item" style="margin-left:auto;color:var(--text3)">Arrossega per moure</div>
          </div>
        </div>
        <div>
          <div class="card" id="graph-meta-card"></div>
        </div>
      </div>
    </div>
  `;
}

function attachEvents(container) {
  container.querySelector("#patient-search").addEventListener("input",  () => applyFilters(container));
  container.querySelector("#label-filter").addEventListener("change",   () => applyFilters(container));
  container.querySelector("#run-btn").addEventListener("click", () => runInference(container));
}

function applyFilters(container) {
  const q     = container.querySelector("#patient-search").value.toLowerCase();
  const label = container.querySelector("#label-filter").value;
  _filtered = _patients.filter(p => {
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
    </tr>
  `).join("");

  wrap.innerHTML = `
    <table>
      <thead><tr><th>Pacient</th><th>Hospital</th><th>Classe</th><th>Slides</th></tr></thead>
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

  const runBtn        = container.querySelector("#run-btn");
  const progressWrap  = container.querySelector("#progress-wrap");
  const progressFill  = container.querySelector("#progress-fill");
  const progressLabel = container.querySelector("#progress-label-text");
  const progressPct   = container.querySelector("#progress-pct");

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
    _result     = await API.inferencePatient(_selectedPid, _debugMode);
    _vizGraphId = _result.viz_graph_id;

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
    renderSlideBreakdown(container, _result, container);
    await renderGraphViz(container, _result);

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
  const isN1      = r.prediction === 1;
  const cls       = isN1 ? "n1" : "n0";
  const n1Slides  = (r.slide_results || []).filter(s => s.pred === 1).length;
  const total     = r.num_slides || 0;

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

  _rebuildSlideTable(container, slides, r);
}

function _rebuildSlideTable(container, slides, r) {
  const rows = slides.map(s => {
    const isViz = s.graph_id === _vizGraphId;
    const pct   = (s.prob_n1 * 100).toFixed(1);
    const barW  = Math.round(s.prob_n1 * 100);
    return `
      <tr data-gid="${s.graph_id}" style="cursor:pointer" class="${isViz ? "selected" : ""}">
        <td style="font-size:11px;max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:var(--mono)">
          ${s.graph_id.split("/").pop()}
        </td>
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
        <td style="font-size:10.5px;color:${isViz ? "var(--accent-light)" : "var(--text3)"}">
          ${isViz ? "▶ viz" : "<span style='opacity:0.4'>viz</span>"}
        </td>
      </tr>`;
  }).join("");

  container.querySelector("#slides-content").innerHTML = `
    <table>
      <thead><tr><th>Slide</th><th>Predicció</th><th>P(N1)</th><th>Nodes</th><th></th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;

  // Click a slide row → visualize that slide
  container.querySelector("#slides-content").querySelectorAll("tbody tr").forEach(row => {
    row.addEventListener("click", async () => {
      const gid = row.dataset.gid;
      if (!gid || gid === _vizGraphId) return;
      await vizSlide(container, gid);
    });
  });
}

/** Update ▶ viz marker and row highlight in-place without rebuilding the table */
function _updateVizMarker(container, graphId) {
  container.querySelector("#slides-content").querySelectorAll("tbody tr").forEach(row => {
    const isViz = row.dataset.gid === graphId;
    row.classList.toggle("selected", isViz);
    const lastTd = row.querySelector("td:last-child");
    if (lastTd) {
      lastTd.style.color = isViz ? "var(--accent-light)" : "var(--text3)";
      lastTd.innerHTML   = isViz ? "▶ viz" : "<span style='opacity:0.4'>viz</span>";
    }
  });
}

async function vizSlide(container, graphId) {
  const vizLabel   = container.querySelector("#viz-slide-label");
  const vizLoading = container.querySelector("#viz-loading");
  if (vizLoading) vizLoading.style.display = "";
  if (vizLabel)   vizLabel.textContent = `(carregant…)`;

  try {
    const viz = await API.inference(graphId, false);
    _vizGraphId = graphId;
    _updateVizMarker(container, graphId);
    await _drawViz(container, viz, graphId);
  } catch (e) {
    if (vizLoading) vizLoading.style.display = "none";
    if (vizLabel)   vizLabel.textContent = `(error: ${e.message})`;
  }
}

async function renderGraphViz(container, r) {
  container.querySelector("#viz-section").style.display = "";
  await _drawViz(container, r, r.viz_graph_id);
}

async function _drawViz(container, viz, graphId) {
  _vizData       = viz;
  _vizBgImageUrl = null;
  _vizWsiExtent  = null;

  const vizLabel   = container.querySelector("#viz-slide-label");
  const vizLoading = container.querySelector("#viz-loading");
  const meta       = container.querySelector("#graph-meta-card");

  if (vizLabel)   vizLabel.textContent = `(slide: ${(graphId || "").split("/").pop()})`;
  if (vizLoading) vizLoading.style.display = "none";

  // Attention color legend
  const legendCanvas = container.querySelector("#attn-legend-canvas");
  if (legendCanvas) {
    const ctx  = legendCanvas.getContext("2d");
    const grad = ctx.createLinearGradient(0, 0, 120, 0);
    for (let t = 0; t <= 1; t += 0.1) {
      const c = d3.color(d3.interpolateTurbo(t));
      grad.addColorStop(t, `rgb(${c.r},${c.g},${c.b})`);
    }
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 120, 14);
  }

  _vizSlideInfo = {
    hospital:   viz.hospital   || "",
    patient_id: viz.patient_id || "",
    slide_id:   viz.slide_id   || "",
    graph_id:   graphId || "",
    section_id: viz.section_id ?? null,
    patch_idx:  viz.patch_idx  ?? null,
    patch_j:    viz.patch_j,
    patch_i:    viz.patch_i,
  };

  // Fetch slide metadata (WSI extent) for background alignment.
  // Skipped when SHOW_WSI_BACKGROUND is false — set it to true to re-enable.
  let bgError = null;
  if (SHOW_WSI_BACKGROUND && graphId) {
    try {
      const sm = await API.slideMeta(graphId);
      if (sm.has_bg) {
        _vizBgImageUrl = `/api/slide_bg/${encodeURIComponent(graphId)}`;
        if (sm.j_base != null)
          _vizWsiExtent = { j_base: sm.j_base, i_base: sm.i_base, w: sm.w, h: sm.h };
      } else {
        const sid = (graphId || "").split("/").pop().replace(/\.pt$/, "");
        bgError = `Imatge de fons no disponible: no s'ha trobat <em>${sid}_low.jpg</em>`;
      }
    } catch (e) {
      bgError = `Error en carregar la imatge de fons: ${e.message}`;
    }
  }

  // Inicialitza el selector de capa amb totes les capes disponibles.
  const layerSelect = container.querySelector("#attn-layer-select");
  const layerNote   = container.querySelector("#attn-layer-note");
  const layers      = _availableLayers();
  if (layerSelect) {
    if (layers.length === 0) {
      layerSelect.innerHTML = `<option value="">(no disponible)</option>`;
      layerSelect.disabled  = true;
      if (layerNote) layerNote.textContent = (viz.pooling_type === "diff")
        ? "DiffPool: l'extracció d'atenció per capa no està disponible (super-nodes intermedis)."
        : "";
    } else {
      layerSelect.innerHTML = layers.map(k => `<option value="${k}">${k.replace("layer","Capa ")}</option>`).join("");
      layerSelect.value     = layers[layers.length - 1];   // darrera capa per defecte
      layerSelect.disabled  = false;
      if (layerNote) layerNote.textContent = "";
      layerSelect.onchange = () => {
        const k = layerSelect.value;
        const idx = Number(k.replace("layer",""));
        if (viz.pooling_type === "diff" && idx > 1) {
          if (layerNote) layerNote.textContent = "Capa sobre super-nodes (sense WSI ni patches)";
        } else {
          if (layerNote) layerNote.textContent = "";
        }
        _reRenderGraph(container, null, k);
      };
    }
  }
  _reRenderGraph(container, null, layers[layers.length - 1] ?? null);

  const bgStatusHtml = bgError
    ? `<div class="stat-row"><span class="stat-key">Imatge fons</span><span class="stat-val" style="color:var(--red);font-size:11px">${bgError}</span></div>`
    : _vizBgImageUrl
      ? `<div class="stat-row"><span class="stat-key">Imatge fons</span><span class="stat-val" style="color:var(--green)">✓ _low.jpg</span></div>`
      : "";

  meta.innerHTML = `
    <div class="card-title">Slide visualitzada</div>
    <div class="stat-list">
      <div class="stat-row"><span class="stat-key">Nodes</span><span class="stat-val mono">${viz.num_nodes}</span></div>
      <div class="stat-row"><span class="stat-key">Arestes dirigides</span><span class="stat-val mono">${viz.num_edges}</span></div>
      <div class="stat-row"><span class="stat-key">Posicions reals</span><span class="stat-val">${viz.node_positions ? "✓ Sí" : "No"}</span></div>
      <div class="stat-row"><span class="stat-key">Pooling</span><span class="stat-val accent">${viz.pooling_type ?? "—"}</span></div>
      <div class="stat-row"><span class="stat-key">Patches disponibles</span><span class="stat-val">${viz.patch_idx ? "✓ Sí" : (viz.patch_j ? "⚠ legacy" : "⚠ cal rebuild")}</span></div>
      ${bgStatusHtml}
    </div>`;

  lucide.createIcons();
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
