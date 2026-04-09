import { API } from "../api.js";
import { renderGraph, renderPCA, renderAttentionBars } from "../components/graphViz.js";

let _graphs = [];
let _filtered = [];
let _selectedId = null;
let _result = null;
let _debugMode = false;

export async function renderInference(container, debugMode = false) {
  _debugMode = debugMode;

  container.innerHTML = `<div class="loading-spinner"><div class="spinner"></div><p>Carregant grafs…</p></div>`;

  try {
    const data = await API.graphs();
    _graphs = data.graphs || [];
  } catch (e) {
    container.innerHTML = `<div class="empty-state"><p>Error en carregar grafs</p><small>${e.message}</small></div>`;
    return;
  }

  _filtered = [..._graphs];
  _selectedId = null;
  _result = null;

  container.innerHTML = buildLayout();
  lucide.createIcons();
  attachEvents(container);
  renderTable(container);
}

function buildLayout() {
  return `
    <div class="page-header">
      <h1 class="page-title">Inferència</h1>
      <p class="page-sub">Selecciona un graf, executa el forward pass i explora l'atenció</p>
    </div>

    <div class="two-col" style="align-items:start">
      <!-- Esquerra: selector de grafs -->
      <div>
        <div class="section">
          <div class="section-title"><i data-lucide="database"></i> Seleccionar graf</div>
          <div class="card" style="padding:14px">
            <div style="display:flex;gap:8px;margin-bottom:10px;flex-wrap:wrap">
              <div class="search-wrap" style="flex:1;min-width:140px">
                <i data-lucide="search"></i>
                <input class="search-input" id="graph-search" placeholder="Cerca per pacient, hospital…" />
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
            <div class="table-wrap" style="max-height:320px;overflow-y:auto" id="graph-table-wrap">
            </div>
            <div id="table-count" style="font-size:11.5px;color:var(--text3);margin-top:8px;text-align:right"></div>
          </div>
        </div>

        <!-- Botó d'execució + progrés -->
        <div class="section">
          <button class="btn btn-primary" id="run-btn" disabled style="width:100%;justify-content:center">
            <i data-lucide="play-circle"></i> Executar forward pass
          </button>
          <div id="progress-wrap" class="progress-container" style="display:none">
            <div class="progress-label"><span id="progress-label-text">Processant…</span><span id="progress-pct"></span></div>
            <div class="progress-bar"><div class="progress-fill indeterminate" id="progress-fill"></div></div>
          </div>
        </div>
      </div>

      <!-- Dreta: resultat -->
      <div id="result-area">
        <div class="empty-state" style="padding:40px 20px">
          <i data-lucide="play-circle"></i>
          <p>Selecciona un graf i executa la inferència</p>
        </div>
      </div>
    </div>

    <!-- Visualització del graf (amplada completa) -->
    <div id="viz-section" style="display:none" class="section">
      <div class="section-title"><i data-lucide="share-2"></i> Estructura del graf</div>
      <div class="two-col" style="align-items:start">
        <div>
          <div class="graph-viz-wrap" id="graph-svg-container" style="height:360px"></div>
          <div class="graph-legend">
            <div class="legend-item"><div class="legend-dot" style="background:var(--accent)"></div> Alta atenció</div>
            <div class="legend-item"><div class="legend-dot" style="background:#2a2a2a;border:1px solid #444"></div> Baixa atenció</div>
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

    <!-- Embeddings de nodes PCA -->
    <div id="pca-section" style="display:none" class="section">
      <div class="section-title"><i data-lucide="scatter-chart"></i> Embeddings de nodes (PCA)</div>
      <div class="three-col" id="pca-grid"></div>
    </div>
  `;
}

function attachEvents(container) {
  container.querySelector("#graph-search").addEventListener("input", () => applyFilters(container));
  container.querySelector("#split-filter").addEventListener("change", () => applyFilters(container));
  container.querySelector("#label-filter").addEventListener("change", () => applyFilters(container));
  container.querySelector("#run-btn").addEventListener("click", () => runInference(container));
}

function applyFilters(container) {
  const q = container.querySelector("#graph-search").value.toLowerCase();
  const split = container.querySelector("#split-filter").value;
  const label = container.querySelector("#label-filter").value;

  _filtered = _graphs.filter(g => {
    if (split && g.split !== split) return false;
    if (label !== "" && String(g.label) !== label) return false;
    if (q && !`${g.patient_id} ${g.hospital} ${g.stem} ${g.id}`.toLowerCase().includes(q)) return false;
    return true;
  });

  renderTable(container);
}

function renderTable(container) {
  const wrap = container.querySelector("#graph-table-wrap");
  const count = container.querySelector("#table-count");

  if (!_filtered.length) {
    wrap.innerHTML = `<div class="empty-state" style="padding:30px"><p>Sense resultats</p></div>`;
    count.textContent = "";
    return;
  }

  const rows = _filtered.slice(0, 200).map(g => `
    <tr data-id="${g.id}" class="${g.id === _selectedId ? "selected" : ""}">
      <td><span class="badge badge-${g.split}">${g.split}</span></td>
      <td style="font-size:12px;max-width:120px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${g.patient_id}</td>
      <td><span class="badge badge-${g.label === 0 ? "n0" : g.label === 1 ? "n1" : "unk"}">${g.label === 0 ? "N0" : g.label === 1 ? "N1" : "?"}</span></td>
      <td class="mono" style="font-size:11.5px">${g.num_nodes}</td>
      <td class="mono" style="font-size:11.5px">${g.num_edges}</td>
    </tr>
  `).join("");

  wrap.innerHTML = `
    <table>
      <thead><tr><th>Split</th><th>Pacient</th><th>Classe</th><th>Nodes</th><th>Arestes</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>`;

  count.textContent = `${_filtered.length} graf${_filtered.length !== 1 ? "s" : ""}`;

  wrap.querySelectorAll("tbody tr").forEach(row => {
    row.addEventListener("click", () => {
      _selectedId = row.dataset.id;
      wrap.querySelectorAll("tbody tr").forEach(r => r.classList.remove("selected"));
      row.classList.add("selected");
      container.querySelector("#run-btn").disabled = false;
    });
  });
}

const STEPS = [
  "Carregant graf des del disc…",
  "Preparant tensor de característiques…",
  "GAT Capa 1 — extraient atenció…",
  "GAT Capa 2 — extraient atenció…",
  "GAT Capa 3 — extraient atenció…",
  "Global pooling + MLP head…",
  "Calculant PCA dels embeddings…",
  "Processant resultats…",
];

async function runInference(container) {
  if (!_selectedId) return;

  const runBtn = container.querySelector("#run-btn");
  const progressWrap = container.querySelector("#progress-wrap");
  const progressFill = container.querySelector("#progress-fill");
  const progressLabel = container.querySelector("#progress-label-text");
  const progressPct = container.querySelector("#progress-pct");

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
      progressPct.textContent = pct + "%";
      stepIdx++;
    }
  }, 350);

  appendDebugLog({ level: "info", msg: `Iniciant inferència: ${_selectedId}`, t: Date.now() });

  try {
    _result = await API.inference(_selectedId, _debugMode);

    clearInterval(stepInterval);
    progressFill.style.width = "100%";
    progressPct.textContent = "100%";
    progressLabel.textContent = "Completat!";

    if (_debugMode && _result.debug_log?.length) {
      _result.debug_log.forEach(e => appendDebugLog(e));
    }

    setTimeout(() => {
      progressWrap.style.display = "none";
      progressFill.style.width = "";
      progressFill.classList.add("indeterminate");
    }, 800);

    renderResult(container, _result);
    renderGraphViz(container, _result);
    renderAttention(container, _result, "layer1");
    renderPCASections(container, _result);

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
  const cls = isN1 ? "n1" : "n0";
  const conf = (r.confidence * 100).toFixed(1);
  const correctHtml = r.correct !== null
    ? `<span class="${r.correct ? "result-correct" : "result-incorrect"}">${r.correct ? "✓ Correcte" : "✗ Incorrecte"}</span>`
    : "";

  container.querySelector("#result-area").innerHTML = `
    <div class="result-panel">
      <div class="card-title">Predicció</div>
      <div class="result-prediction ${cls}">${r.label}</div>
      <div style="font-size:13px;color:var(--text2);margin-top:4px">
        Confiança ${conf}% ${correctHtml}
      </div>

      <div class="confidence-bar-wrap">
        <div class="confidence-labels">
          <span>N0 &nbsp;${(r.prob_n0 * 100).toFixed(1)}%</span>
          <span>N1 &nbsp;${(r.prob_n1 * 100).toFixed(1)}%</span>
        </div>
        <div class="conf-bar-track">
          <div class="conf-bar-fill ${cls}" style="width:${r.confidence * 100}%"></div>
        </div>
      </div>

      <div class="result-meta">
        <div class="result-meta-row">
          <span>Etiqueta real</span>
          <span class="result-meta-val">
            <span class="badge badge-${r.true_label === 0 ? "n0" : r.true_label === 1 ? "n1" : "unk"}">${r.true_label_name}</span>
          </span>
        </div>
        <div class="result-meta-row"><span>Pacient</span><span class="result-meta-val" style="font-size:12px">${r.patient_id || "—"}</span></div>
        <div class="result-meta-row"><span>Hospital</span><span class="result-meta-val" style="font-size:12px">${r.hospital || "—"}</span></div>
        <div class="result-meta-row"><span>Nodes</span><span class="result-meta-val mono">${r.num_nodes}</span></div>
        <div class="result-meta-row"><span>Arestes</span><span class="result-meta-val mono">${r.num_edges}</span></div>
      </div>
    </div>`;

  lucide.createIcons();
}

function renderGraphViz(container, r) {
  const section = container.querySelector("#viz-section");
  section.style.display = "";

  const svgContainer = container.querySelector("#graph-svg-container");
  const meta = container.querySelector("#graph-meta-card");

  const attn3 = r.attention?.layer3;

  renderGraph(svgContainer, {
    edge_index: r.edge_index,
    node_positions: r.node_positions,
    num_nodes: r.num_nodes,
    feature_norms: r.feature_norms,
  }, {
    nodeAttention: attn3?.node_attention,
    edgeAttention: attn3 ? { edge_index: attn3.edge_index, weights_mean: attn3.weights_mean } : null,
    height: 360,
  });

  const nodeAttn = attn3?.node_attention || [];
  const topNodes = nodeAttn
    .map((v, i) => ({ i, v }))
    .sort((a, b) => b.v - a.v)
    .slice(0, 5);

  meta.innerHTML = `
    <div class="card-title">Nodes més rellevants (Capa 3)</div>
    <div class="stat-list" style="margin-bottom:14px">
      ${topNodes.map((n, rank) => `
        <div class="stat-row">
          <span class="stat-key">${rank + 1}. Node ${n.i}</span>
          <span class="stat-val">${(n.v * 100).toFixed(1)}%</span>
        </div>`).join("")}
    </div>
    <div class="card-title">Informació del graf</div>
    <div class="stat-list">
      <div class="stat-row"><span class="stat-key">Nodes</span><span class="stat-val mono">${r.num_nodes}</span></div>
      <div class="stat-row"><span class="stat-key">Arestes dirigides</span><span class="stat-val mono">${r.num_edges}</span></div>
      <div class="stat-row"><span class="stat-key">Arestes no dir.</span><span class="stat-val mono">${Math.round(r.num_edges / 2)}</span></div>
      <div class="stat-row"><span class="stat-key">Posicions reals</span><span class="stat-val">${r.node_positions ? "✓ Sí" : "No"}</span></div>
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

  const layerLabel = activeLayer.replace("layer", "Capa ");
  const content = section.querySelector("#attn-content");
  content.innerHTML = `
    <div class="two-col">
      <div>
        <div class="section-title" style="margin-bottom:8px">Atenció mitjana per node</div>
        <div id="attn-bar-chart"></div>
      </div>
      <div>
        <div class="section-title" style="margin-bottom:8px">Graf acolorit per atenció (${layerLabel})</div>
        <div class="graph-viz-wrap" id="attn-graph-mini" style="height:280px"></div>
      </div>
    </div>
    <div style="margin-top:12px">
      <div class="stat-list">
        <div class="stat-row">
          <span class="stat-key">Arestes (incl. self-loops)</span>
          <span class="stat-val mono">${layerData.edge_index[0].length}</span>
        </div>
        <div class="stat-row">
          <span class="stat-key">Caps d'atenció</span>
          <span class="stat-val mono">${layerData.num_heads}</span>
        </div>
        <div class="stat-row">
          <span class="stat-key">Atenció màxima</span>
          <span class="stat-val mono">${Math.max(...layerData.node_attention).toFixed(4)}</span>
        </div>
        <div class="stat-row">
          <span class="stat-key">Entropia (dispersió)</span>
          <span class="stat-val mono">${entropy(layerData.weights_mean).toFixed(4)}</span>
        </div>
      </div>
    </div>`;

  renderAttentionBars(content.querySelector("#attn-bar-chart"), layerData.node_attention, "");

  renderGraph(content.querySelector("#attn-graph-mini"), {
    edge_index: r.edge_index,
    node_positions: r.node_positions,
    num_nodes: r.num_nodes,
    feature_norms: r.feature_norms,
  }, {
    nodeAttention: layerData.node_attention,
    edgeAttention: { edge_index: layerData.edge_index, weights_mean: layerData.weights_mean },
    height: 280,
  });
}

function renderPCASections(container, r) {
  const section = container.querySelector("#pca-section");
  section.style.display = "";
  const grid = container.querySelector("#pca-grid");
  grid.innerHTML = `
    <div><div id="pca-layer1"></div></div>
    <div><div id="pca-layer2"></div></div>
    <div><div id="pca-layer3"></div></div>`;

  for (const [name, label] of [["layer1", "Capa 1"], ["layer2", "Capa 2"], ["layer3", "Capa 3"]]) {
    const pcaData = r.node_embeddings?.[name];
    if (!pcaData) continue;
    const nodeAttn = r.attention?.[name]?.node_attention;
    renderPCA(grid.querySelector(`#pca-${name}`), pcaData, nodeAttn, label);
  }
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
  const div = document.createElement("div");
  div.className = "log-entry";
  div.innerHTML = `<span class="log-time">${time}</span><span class="log-msg ${entry.level || "info"}">${entry.msg}</span>`;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}
