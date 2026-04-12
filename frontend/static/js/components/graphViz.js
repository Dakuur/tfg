/**
 * D3-based graph visualization.
 * Nodes colored by attention weight (blue → green → yellow → red).
 * Click on a node to open the patch image modal.
 * Optional background image aligned to WSI coordinates.
 */

const _attnColor = d3.scaleSequential().domain([0, 1]).interpolator(d3.interpolateTurbo);

function _edgeColor(v) {
  const r = Math.round(180 * v + 40 * (1 - v));
  const g = Math.round(120 * v + 40 * (1 - v));
  const b = Math.round(20  * v + 40 * (1 - v));
  return `rgba(${r},${g},${b},${0.15 + 0.65 * v})`;
}

// ── Patch modal ────────────────────────────────────────────────────────────────
function _ensureModal() {
  if (document.getElementById("patch-modal")) return;
  const modal = document.createElement("div");
  modal.id = "patch-modal";
  modal.style.cssText = [
    "display:none", "position:fixed", "inset:0", "background:rgba(0,0,0,.82)",
    "z-index:9000", "align-items:center", "justify-content:center", "flex-direction:column",
  ].join(";");
  modal.innerHTML = `
    <div style="position:relative;max-width:90vw;max-height:90vh;text-align:center">
      <button id="patch-modal-close" style="
        position:absolute;top:-36px;right:0;background:transparent;border:none;
        color:#fff;font-size:26px;cursor:pointer;line-height:1;padding:0 6px">✕</button>
      <div id="patch-modal-label" style="color:#ddd;font-size:13px;margin-bottom:8px;font-weight:500"></div>
      <img id="patch-modal-img" src="" alt="patch"
        style="max-width:90vw;max-height:78vh;border-radius:6px;display:block;
               border:1px solid #444;background:#1a1a1a;min-width:200px;min-height:100px"/>
      <div id="patch-modal-meta" style="color:#777;font-size:11px;margin-top:8px"></div>
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

  const modal = document.getElementById("patch-modal");
  const img   = modal.querySelector("#patch-modal-img");
  const label = modal.querySelector("#patch-modal-label");
  const meta  = modal.querySelector("#patch-modal-meta");

  label.textContent = `Node ${idx}  ·  atenció: ${(nodeData.attn * 100).toFixed(1)}%`;
  modal.style.display = "flex";

  if (j == null || i == null) {
    img.removeAttribute("src");
    meta.textContent = "Coordenades no disponibles — cal re-executar build_dataset.py";
    return;
  }

  const url = `/api/patch_image?hospital=${encodeURIComponent(hospital)}`
            + `&patient_id=${encodeURIComponent(patient_id)}`
            + `&slide_id=${encodeURIComponent(slide_id)}`
            + `&j=${j}&i=${i}`;

  meta.textContent = "Carregant patch…";
  img.src = "";
  img.onload  = () => { meta.textContent = `j=${j}  i=${i}  ·  ${hospital}`; };
  img.onerror = () => { meta.textContent = "No s'ha pogut carregar el patch des del servidor"; };
  img.src = url;
}

// ── Main render function ───────────────────────────────────────────────────────
/**
 * @param {HTMLElement} container
 * @param {object} data  – { edge_index, node_positions, num_nodes, feature_norms }
 * @param {object} opts
 *   nodeAttention  – array of per-node weights (0–1)
 *   edgeAttention  – { edge_index, weights_mean }
 *   height         – SVG height in px
 *   slideInfo      – { hospital, patient_id, slide_id, patch_j, patch_i } for patch modal
 *   bgImageUrl     – URL of background image
 *   wsiExtent      – { j_base, i_base, w, h } in WSI level-0 pixels; if provided,
 *                    the background image is aligned to this region instead of the node bbox
 */
export function renderGraph(container, data, opts = {}) {
  const {
    nodeAttention = null,
    edgeAttention = null,
    width         = container.clientWidth || 500,
    height        = opts.height || 360,
    slideInfo     = null,
    bgImageUrl    = null,
    wsiExtent     = null,   // { j_base, i_base, w, h }
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
      if (!seen.has(key)) { seen.add(key); rawEdges.push({ source: srcs[k], target: dsts[k] }); }
    }
  }

  let edgeAttnMap = {};
  if (edgeAttention) {
    const { edge_index: attnEI, weights_mean } = edgeAttention;
    for (let k = 0; k < attnEI[0].length; k++) {
      const key = `${Math.min(attnEI[0][k], attnEI[1][k])}-${Math.max(attnEI[0][k], attnEI[1][k])}`;
      edgeAttnMap[key] = (edgeAttnMap[key] || 0) + weights_mean[k];
    }
    const maxV = Math.max(...Object.values(edgeAttnMap), 1e-6);
    for (const k in edgeAttnMap) edgeAttnMap[k] /= maxV;
  }

  const edgesWithAttn = rawEdges.map(e => {
    const key = `${Math.min(e.source, e.target)}-${Math.max(e.source, e.target)}`;
    return { ...e, attn: edgeAttnMap[key] ?? 0.15 };
  });

  const attnVals     = nodes.map(n => n.attn);
  const maxAttn      = Math.max(...attnVals, 1e-6);
  const normAttnVals = attnVals.map(v => v / maxAttn);

  // ── SVG ───────────────────────────────────────────────────────────────────────
  const svg = d3.select(container)
    .append("svg")
    .attr("width", "100%").attr("height", height)
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("preserveAspectRatio", "xMidYMid meet");

  svg.append("defs").append("clipPath").attr("id", "graph-clip")
    .append("rect").attr("width", width).attr("height", height);

  const g = svg.append("g").attr("clip-path", "url(#graph-clip)");

  svg.call(d3.zoom()
    .scaleExtent([0.2, 12])
    .on("zoom", event => g.attr("transform", event.transform))
  );

  const pad = 40;

  // ── Position nodes (real WSI coords) ──────────────────────────────────────────
  let scaleX, scaleY;

  if (node_positions && node_positions.length === num_nodes) {
    const xs   = node_positions.map(p => p[0]);
    const ys   = node_positions.map(p => p[1]);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);

    // Uniform scale: same pixels-per-WSI-unit on both axes to avoid distortion
    const xRange = (xMax - xMin) || 1;
    const yRange = (yMax - yMin) || 1;
    const availW = width  - 2 * pad;
    const availH = height - 2 * pad;
    const uScale = Math.min(availW / xRange, availH / yRange);
    const xOff   = pad + (availW - xRange * uScale) / 2;
    const yOff   = pad + (availH - yRange * uScale) / 2;

    scaleX = j => xOff + (j - xMin) * uScale;
    scaleY = i => yOff + (i - yMin) * uScale;

    nodes.forEach((n, i) => {
      n.x = scaleX(node_positions[i][0]);
      n.y = scaleY(node_positions[i][1]);
    });

    // ── Background image ────────────────────────────────────────────────────────
    if (bgImageUrl) {
      let imgX, imgY, imgW, imgH;

      if (wsiExtent && wsiExtent.j_base != null) {
        // Align to the actual slide WSI region using the same uniform scale
        imgX = scaleX(wsiExtent.j_base);
        imgY = scaleY(wsiExtent.i_base);
        imgW = wsiExtent.w * uScale;
        imgH = wsiExtent.h * uScale;
      } else {
        // Fallback: fill the node bounding box
        imgX = xOff; imgY = yOff;
        imgW = xRange * uScale;
        imgH = yRange * uScale;
      }

      g.append("image")
        .attr("href", bgImageUrl)
        .attr("x", imgX).attr("y", imgY)
        .attr("width", imgW).attr("height", imgH)
        .attr("preserveAspectRatio", "none")   // dimensions already respect aspect ratio
        .attr("opacity", 0.55);
    }
  }

  // ── Edges ─────────────────────────────────────────────────────────────────────
  const edgeLines = g.append("g").attr("class", "edges")
    .selectAll("line").data(edgesWithAttn).join("line")
    .attr("stroke", d => _edgeColor(d.attn))
    .attr("stroke-width", d => 1 + d.attn * 2)
    .attr("opacity", d => 0.3 + d.attn * 0.6);

  // ── Nodes (no drag — click only) ───────────────────────────────────────────────
  const nodeGroup = g.append("g").attr("class", "nodes")
    .selectAll("g").data(nodes).join("g")
    .attr("cursor", slideInfo ? "pointer" : "default")
    .on("click", (event, d) => {
      event.stopPropagation();
      if (slideInfo) _openPatchModal(d, slideInfo);
    })
    .on("mouseover", (event, d) => showTooltip(event, d))
    .on("mousemove", event     => moveTooltip(event))
    .on("mouseout",  ()        => hideTooltip());

  nodeGroup.append("circle")
    .attr("r", d => 5 + normAttnVals[d.id] * 8)
    .attr("fill", d => _attnColor(normAttnVals[d.id]))
    .attr("stroke", d => normAttnVals[d.id] > 0.6 ? "rgba(255,255,255,0.8)" : "rgba(255,255,255,0.2)")
    .attr("stroke-width", d => normAttnVals[d.id] > 0.6 ? 1.5 : 0.5);

  nodeGroup.append("text")
    .attr("dy", "0.35em").attr("text-anchor", "middle")
    .attr("font-size", "8px").attr("fill", "rgba(255,255,255,0.5)")
    .attr("pointer-events", "none")
    .text(d => d.id);

  // ── Draw (static real positions OR force simulation) ───────────────────────────
  if (node_positions && node_positions.length === num_nodes) {
    edgeLines
      .attr("x1", d => nodes[d.source].x).attr("y1", d => nodes[d.source].y)
      .attr("x2", d => nodes[d.target].x).attr("y2", d => nodes[d.target].y);
    nodeGroup.attr("transform", d => `translate(${d.x},${d.y})`);
  } else {
    d3.forceSimulation(nodes)
      .force("link",      d3.forceLink(edgesWithAttn).id(d => d.id).distance(60))
      .force("charge",    d3.forceManyBody().strength(-120))
      .force("center",    d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide(14))
      .on("tick", () => {
        edgeLines
          .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
          .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
        nodeGroup.attr("transform", d => `translate(${d.x},${d.y})`);
      });
  }

  // ── Tooltip ───────────────────────────────────────────────────────────────────
  const tooltip = d3.select("body .tooltip").node()
    ? d3.select("body .tooltip")
    : d3.select("body").append("div").attr("class", "tooltip").style("display", "none");

  function showTooltip(event, d) {
    const clickHint = slideInfo ? "<br><em style='color:#aaa'>Clic per veure patch</em>" : "";
    tooltip.style("display", "block").html(
      `<strong>Node ${d.id}</strong><br>Atenció: ${(d.attn * 100).toFixed(1)}%` + clickHint
    );
    moveTooltip(event);
  }
  function moveTooltip(event) {
    tooltip.style("left", (event.pageX + 14) + "px").style("top", (event.pageY - 32) + "px");
  }
  function hideTooltip() { tooltip.style("display", "none"); }
}
