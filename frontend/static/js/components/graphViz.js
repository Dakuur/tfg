/**
 * D3-based graph visualization.
 * Draws nodes and edges over an optional slide background image,
 * colored by attention weight (blue → green → yellow → red).
 * Clicking a node opens a modal with the patch JPG.
 */

// ── Color scale: blue → cyan → green → yellow → red (d3.interpolateTurbo) ────
const _attnColor = d3.scaleSequential().domain([0, 1]).interpolator(d3.interpolateTurbo);

function _edgeColor(v) {
  // Interpolate from near-transparent dark to a warm amber
  const [r, g, b] = [
    Math.round(180 * v + 40 * (1 - v)),
    Math.round(120 * v + 40 * (1 - v)),
    Math.round(20  * v + 40 * (1 - v)),
  ];
  return `rgba(${r},${g},${b},${0.15 + 0.65 * v})`;
}

// ── Patch modal ────────────────────────────────────────────────────────────────
function _ensureModal() {
  if (document.getElementById("patch-modal")) return;
  const modal = document.createElement("div");
  modal.id = "patch-modal";
  modal.style.cssText = `
    display:none;position:fixed;inset:0;background:rgba(0,0,0,.75);
    z-index:9000;align-items:center;justify-content:center;flex-direction:column;gap:12px;
  `;
  modal.innerHTML = `
    <div style="position:relative;max-width:90vw;max-height:90vh">
      <button id="patch-modal-close" style="
        position:absolute;top:-36px;right:0;background:transparent;border:none;
        color:#fff;font-size:24px;cursor:pointer;line-height:1">✕</button>
      <div id="patch-modal-label" style="color:#ccc;font-size:12px;margin-bottom:6px;text-align:center"></div>
      <img id="patch-modal-img" src="" alt="patch"
        style="max-width:90vw;max-height:80vh;border-radius:6px;display:block;border:1px solid #444"/>
      <div id="patch-modal-meta" style="color:#888;font-size:11px;margin-top:6px;text-align:center"></div>
    </div>`;
  document.body.appendChild(modal);
  modal.addEventListener("click", e => { if (e.target === modal) _closeModal(); });
  modal.querySelector("#patch-modal-close").addEventListener("click", _closeModal);
  document.addEventListener("keydown", e => { if (e.key === "Escape") _closeModal(); });
}

function _closeModal() {
  const m = document.getElementById("patch-modal");
  if (m) { m.style.display = "none"; m.querySelector("#patch-modal-img").src = ""; }
}

function _openPatchModal(nodeData, slideInfo) {
  _ensureModal();
  const { hospital, patient_id, slide_id, patch_j, patch_i } = slideInfo;
  const idx = nodeData.id;
  const j   = patch_j?.[idx];
  const i   = patch_i?.[idx];

  const modal  = document.getElementById("patch-modal");
  const img    = modal.querySelector("#patch-modal-img");
  const label  = modal.querySelector("#patch-modal-label");
  const meta   = modal.querySelector("#patch-modal-meta");

  label.textContent = `Node ${idx} — atenció: ${(nodeData.attn * 100).toFixed(1)}%`;

  if (j == null || i == null) {
    meta.textContent = "Coordenades del patch no disponibles (cal re-executar build_dataset)";
    img.src = "";
    modal.style.display = "flex";
    return;
  }

  const url = `/api/patch_image?hospital=${encodeURIComponent(hospital)}&patient_id=${encodeURIComponent(patient_id)}&slide_id=${encodeURIComponent(slide_id)}&j=${j}&i=${i}`;
  img.src = "";
  meta.textContent = `Carregant patch (j=${j}, i=${i})…`;
  modal.style.display = "flex";

  img.onload  = () => { meta.textContent = `j=${j}  i=${i}  |  ${hospital} · ${patient_id} · ${slide_id}`; };
  img.onerror = () => { meta.textContent = "No s'ha pogut carregar el patch (potser no accessible des del servidor)"; };
  img.src = url;
}

// ── Main render function ───────────────────────────────────────────────────────
export function renderGraph(container, data, opts = {}) {
  const {
    nodeAttention = null,
    edgeAttention = null,
    width         = container.clientWidth || 500,
    height        = opts.height || 360,
    slideInfo     = null,   // { hospital, patient_id, slide_id, patch_j, patch_i, graph_id }
    bgImageUrl    = null,   // URL of slide background image (covers node bbox)
  } = opts;

  container.innerHTML = "";

  const { edge_index, node_positions, num_nodes, feature_norms } = data;

  const nodes = Array.from({ length: num_nodes }, (_, i) => ({
    id:   i,
    attn: nodeAttention ? nodeAttention[i] : 0,
    norm: feature_norms ? feature_norms[i] : 1,
  }));

  const rawEdges = [];
  if (edge_index) {
    const seen = new Set();
    const srcs = edge_index[0], dsts = edge_index[1];
    for (let k = 0; k < srcs.length; k++) {
      const key = `${Math.min(srcs[k], dsts[k])}-${Math.max(srcs[k], dsts[k])}`;
      if (!seen.has(key)) { seen.add(key); rawEdges.push({ source: srcs[k], target: dsts[k], idx: k }); }
    }
  }

  let edgeAttnMap = {};
  if (edgeAttention) {
    const { edge_index: attnEI, weights_mean } = edgeAttention;
    for (let k = 0; k < attnEI[0].length; k++) {
      const key = `${Math.min(attnEI[0][k], attnEI[1][k])}-${Math.max(attnEI[0][k], attnEI[1][k])}`;
      edgeAttnMap[key] = (edgeAttnMap[key] || 0) + weights_mean[k];
    }
    const vals = Object.values(edgeAttnMap);
    const maxV = Math.max(...vals, 1e-6);
    for (const k in edgeAttnMap) edgeAttnMap[k] /= maxV;
  }

  const edgesWithAttn = rawEdges.map(e => {
    const key = `${Math.min(e.source, e.target)}-${Math.max(e.source, e.target)}`;
    return { ...e, attn: edgeAttnMap[key] ?? 0.15 };
  });

  const attnVals     = nodes.map(n => n.attn);
  const maxAttn      = Math.max(...attnVals, 1e-6);
  const normAttnVals = attnVals.map(v => v / maxAttn);

  const svg = d3.select(container)
    .append("svg")
    .attr("width", "100%")
    .attr("height", height)
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("preserveAspectRatio", "xMidYMid meet");

  const defs = svg.append("defs");
  defs.append("marker")
    .attr("id", "arrow")
    .attr("markerWidth", 6).attr("markerHeight", 6)
    .attr("refX", 10).attr("refY", 3)
    .attr("orient", "auto")
    .append("path")
    .attr("d", "M0,0 L0,6 L6,3 z")
    .attr("fill", "#f90").attr("opacity", 0.5);

  // Clip path to keep background within bounds
  defs.append("clipPath").attr("id", "graph-clip")
    .append("rect").attr("width", width).attr("height", height);

  const g = svg.append("g").attr("clip-path", "url(#graph-clip)");

  svg.call(d3.zoom()
    .scaleExtent([0.3, 10])
    .on("zoom", event => g.attr("transform", event.transform))
  );

  let scaleX, scaleY, drawGraph;
  const pad = 40;

  if (node_positions && node_positions.length === num_nodes) {
    const xs   = node_positions.map(p => p[0]);
    const ys   = node_positions.map(p => p[1]);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);

    scaleX = d3.scaleLinear().domain([xMin, xMax]).range([pad, width  - pad]);
    scaleY = d3.scaleLinear().domain([yMin, yMax]).range([pad, height - pad]);

    nodes.forEach((n, i) => {
      n.x = scaleX(node_positions[i][0]);
      n.y = scaleY(node_positions[i][1]);
      n.fx = n.x; n.fy = n.y;
    });

    // ── Background slide image ─────────────────────────────────────────────
    if (bgImageUrl) {
      g.append("image")
        .attr("href", bgImageUrl)
        .attr("x", pad).attr("y", pad)
        .attr("width",  width  - 2 * pad)
        .attr("height", height - 2 * pad)
        .attr("preserveAspectRatio", "none")
        .attr("opacity", 0.55);
    }
  }

  // ── Edges ──────────────────────────────────────────────────────────────────
  const edgeLines = g.append("g").attr("class", "edges")
    .selectAll("line")
    .data(edgesWithAttn)
    .join("line")
    .attr("stroke", d => _edgeColor(d.attn))
    .attr("stroke-width", d => 1 + d.attn * 2)
    .attr("opacity", d => 0.3 + d.attn * 0.6);

  // ── Nodes ──────────────────────────────────────────────────────────────────
  const nodeGroup = g.append("g").attr("class", "nodes")
    .selectAll("g")
    .data(nodes)
    .join("g")
    .attr("cursor", "pointer")
    .call(d3.drag()
      .on("start", (event, d) => { d.fx = d.x; d.fy = d.y; })
      .on("drag",  (event, d) => { d.fx = event.x; d.fy = event.y; })
      .on("end",   (event, d) => { if (!node_positions) { d.fx = null; d.fy = null; } })
    )
    .on("click", (event, d) => {
      event.stopPropagation();
      if (slideInfo) _openPatchModal({ ...d, attn: d.attn }, slideInfo);
    })
    .on("mouseover", (event, d) => showTooltip(event, d))
    .on("mousemove", event     => moveTooltip(event))
    .on("mouseout",  ()        => hideTooltip());

  nodeGroup.append("circle")
    .attr("r", d => 5 + normAttnVals[d.id] * 8)
    .attr("fill", d => _attnColor(normAttnVals[d.id]))
    .attr("stroke", d => normAttnVals[d.id] > 0.6 ? "#fff" : "rgba(255,255,255,0.2)")
    .attr("stroke-width", d => normAttnVals[d.id] > 0.6 ? 1.5 : 0.5);

  nodeGroup.append("text")
    .attr("dy", "0.35em")
    .attr("text-anchor", "middle")
    .attr("font-size", "8px")
    .attr("fill", "rgba(255,255,255,0.6)")
    .attr("pointer-events", "none")
    .text(d => d.id);

  function ticked() {
    edgeLines
      .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    nodeGroup.attr("transform", d => `translate(${d.x},${d.y})`);
  }

  drawGraph = function () {
    edgeLines
      .attr("x1", d => nodes[d.source].x).attr("y1", d => nodes[d.source].y)
      .attr("x2", d => nodes[d.target].x).attr("y2", d => nodes[d.target].y);
    nodeGroup.attr("transform", d => `translate(${d.x},${d.y})`);
  };

  if (node_positions && node_positions.length === num_nodes) {
    drawGraph();
  } else {
    // Force simulation fallback (no real positions)
    const sim = d3.forceSimulation(nodes)
      .force("link",      d3.forceLink(edgesWithAttn).id(d => d.id).distance(60))
      .force("charge",    d3.forceManyBody().strength(-120))
      .force("center",    d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide(14))
      .on("tick", ticked);
  }

  // ── Tooltip ────────────────────────────────────────────────────────────────
  const tooltip = d3.select("body .tooltip").node()
    ? d3.select("body .tooltip")
    : d3.select("body").append("div").attr("class", "tooltip").style("display", "none");

  function showTooltip(event, d) {
    const clickHint = slideInfo ? "<br><em>Clic per veure patch</em>" : "";
    tooltip.style("display", "block").html(
      `<strong>Node ${d.id}</strong><br>` +
      `Atenció: ${(d.attn * 100).toFixed(1)}%<br>` +
      `‖feat‖: ${d.norm?.toFixed(2) ?? "—"}` + clickHint
    );
    moveTooltip(event);
  }
  function moveTooltip(event) {
    tooltip.style("left", (event.pageX + 12) + "px").style("top", (event.pageY - 28) + "px");
  }
  function hideTooltip() { tooltip.style("display", "none"); }
}

/** Render attention weight bar chart using Plotly */
export function renderAttentionBars(container, nodeAttention, title = "") {
  const N    = nodeAttention.length;
  const maxA = Math.max(...nodeAttention, 1e-6);
  const normA = nodeAttention.map(v => v / maxA);

  const colors = normA.map(v => {
    // Turbo-like: blue→green→yellow→red
    const c = d3.color(_attnColor(v));
    return `rgba(${c.r},${c.g},${c.b},0.85)`;
  });

  const trace = {
    x: Array.from({ length: N }, (_, i) => `N${i}`),
    y: nodeAttention,
    type: "bar",
    marker: { color: colors },
    hovertemplate: "Node %{x}<br>Atenció: %{y:.4f}<extra></extra>",
  };

  const layout = {
    title: { text: title, font: { color: "#888", size: 12 } },
    paper_bgcolor: "#1c1c1c", plot_bgcolor: "#1c1c1c",
    font: { color: "#888", family: "Inter, sans-serif", size: 11 },
    xaxis: { title: "Node", gridcolor: "#2a2a2a", color: "#555" },
    yaxis: { title: "Pes d'atenció (mitjana)", gridcolor: "#2a2a2a", color: "#555" },
    margin: { l: 50, r: 16, t: 36, b: 50 },
    height: 220,
  };

  Plotly.react(container, [trace], layout, { displayModeBar: false, responsive: true });
}
