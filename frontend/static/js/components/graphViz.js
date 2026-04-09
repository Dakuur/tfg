/**
 * D3-based graph visualization.
 * Draws nodes and edges, colored by attention weight.
 */
export function renderGraph(container, data, opts = {}) {
  const {
    nodeAttention = null,   // array of per-node attention (0–1 range)
    edgeAttention = null,   // { edge_index, weights_mean } for edge coloring
    width = container.clientWidth || 500,
    height = opts.height || 360,
    onNodeClick = null,
  } = opts;

  // Clear previous
  container.innerHTML = "";

  const { edge_index, node_positions, num_nodes, feature_norms } = data;

  // Build node list
  const nodes = Array.from({ length: num_nodes }, (_, i) => ({
    id: i,
    attn: nodeAttention ? nodeAttention[i] : 0,
    norm: feature_norms ? feature_norms[i] : 1,
  }));

  // Build edge list (use original graph edges, not attention edges which include self-loops)
  const rawEdges = [];
  if (edge_index) {
    const seen = new Set();
    const srcs = edge_index[0], dsts = edge_index[1];
    for (let k = 0; k < srcs.length; k++) {
      const key = `${Math.min(srcs[k], dsts[k])}-${Math.max(srcs[k], dsts[k])}`;
      if (!seen.has(key)) {
        seen.add(key);
        rawEdges.push({ source: srcs[k], target: dsts[k], idx: k });
      }
    }
  }

  // Edge attention lookup
  let edgeAttnMap = {};
  if (edgeAttention) {
    const { edge_index: attnEI, weights_mean } = edgeAttention;
    for (let k = 0; k < attnEI[0].length; k++) {
      const key = `${Math.min(attnEI[0][k], attnEI[1][k])}-${Math.max(attnEI[0][k], attnEI[1][k])}`;
      edgeAttnMap[key] = (edgeAttnMap[key] || 0) + weights_mean[k];
    }
    // Normalize
    const vals = Object.values(edgeAttnMap);
    const maxV = Math.max(...vals, 1e-6);
    for (const k in edgeAttnMap) edgeAttnMap[k] /= maxV;
  }

  const edgesWithAttn = rawEdges.map(e => {
    const key = `${Math.min(e.source, e.target)}-${Math.max(e.source, e.target)}`;
    return { ...e, attn: edgeAttnMap[key] ?? 0.15 };
  });

  // Color scale: dark gray → magenta
  const nodeColor = d3.scaleSequential()
    .domain([0, 1])
    .interpolator(d3.interpolateRgb("#2a2a2a", "#cc00a8"));

  const edgeColor = (v) => `rgba(${204 * v + 42 * (1 - v) | 0}, 0, ${168 * v + 42 * (1 - v) | 0}, ${0.15 + 0.7 * v})`;

  const svg = d3.select(container)
    .append("svg")
    .attr("width", "100%")
    .attr("height", height)
    .attr("viewBox", `0 0 ${width} ${height}`)
    .attr("preserveAspectRatio", "xMidYMid meet");

  // Arrow marker (for directed edges)
  svg.append("defs").append("marker")
    .attr("id", "arrow")
    .attr("markerWidth", 6).attr("markerHeight", 6)
    .attr("refX", 10).attr("refY", 3)
    .attr("orient", "auto")
    .append("path")
    .attr("d", "M0,0 L0,6 L6,3 z")
    .attr("fill", "var(--accent)").attr("opacity", 0.5);

  const g = svg.append("g");

  // Zoom & pan
  svg.call(d3.zoom()
    .scaleExtent([0.3, 6])
    .on("zoom", (event) => g.attr("transform", event.transform))
  );

  let sim;

  if (node_positions && node_positions.length === num_nodes) {
    // Use real WSI coordinates (normalized)
    const xs = node_positions.map(p => p[0]);
    const ys = node_positions.map(p => p[1]);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yMin = Math.min(...ys), yMax = Math.max(...ys);
    const pad = 40;

    const scaleX = d3.scaleLinear().domain([xMin, xMax]).range([pad, width - pad]);
    const scaleY = d3.scaleLinear().domain([yMin, yMax]).range([pad, height - pad]);

    nodes.forEach((n, i) => {
      n.x = scaleX(node_positions[i][0]);
      n.y = scaleY(node_positions[i][1]);
      n.fx = n.x;
      n.fy = n.y;
    });
    drawGraph();
  } else {
    // Force simulation fallback
    sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(edgesWithAttn).id(d => d.id).distance(60))
      .force("charge", d3.forceManyBody().strength(-120))
      .force("center", d3.forceCenter(width / 2, height / 2))
      .force("collision", d3.forceCollide(14))
      .on("tick", ticked);
  }

  // Draw edges
  const edgeLines = g.append("g").attr("class", "edges")
    .selectAll("line")
    .data(edgesWithAttn)
    .join("line")
    .attr("stroke", d => edgeColor(d.attn))
    .attr("stroke-width", d => 1 + d.attn * 2)
    .attr("opacity", d => 0.3 + d.attn * 0.6);

  // Draw nodes
  const attnVals = nodes.map(n => n.attn);
  const maxAttn = Math.max(...attnVals, 1e-6);
  const normAttnVals = attnVals.map(v => v / maxAttn);

  const nodeGroup = g.append("g").attr("class", "nodes")
    .selectAll("g")
    .data(nodes)
    .join("g")
    .attr("cursor", "pointer")
    .call(d3.drag()
      .on("start", (event, d) => {
        if (sim && !event.active) sim.alphaTarget(0.3).restart();
        d.fx = d.x; d.fy = d.y;
      })
      .on("drag", (event, d) => { d.fx = event.x; d.fy = event.y; })
      .on("end", (event, d) => {
        if (sim && !event.active) sim.alphaTarget(0);
        if (!node_positions) { d.fx = null; d.fy = null; }
      })
    )
    .on("click", (event, d) => onNodeClick && onNodeClick(d, event))
    .on("mouseover", (event, d) => showTooltip(event, d))
    .on("mousemove", (event)    => moveTooltip(event))
    .on("mouseout",  ()         => hideTooltip());

  nodeGroup.append("circle")
    .attr("r", d => 5 + normAttnVals[d.id] * 8)
    .attr("fill", d => nodeColor(normAttnVals[d.id]))
    .attr("stroke", d => normAttnVals[d.id] > 0.6 ? "var(--accent)" : "var(--border2)")
    .attr("stroke-width", d => normAttnVals[d.id] > 0.6 ? 1.5 : 0.5);

  nodeGroup.append("text")
    .attr("dy", "0.35em")
    .attr("text-anchor", "middle")
    .attr("font-size", "8px")
    .attr("fill", "var(--text3)")
    .attr("pointer-events", "none")
    .text(d => d.id);

  function ticked() {
    edgeLines
      .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
    nodeGroup.attr("transform", d => `translate(${d.x},${d.y})`);
  }

  function drawGraph() {
    edgeLines
      .attr("x1", d => nodes[d.source].x).attr("y1", d => nodes[d.source].y)
      .attr("x2", d => nodes[d.target].x).attr("y2", d => nodes[d.target].y);
    nodeGroup.attr("transform", d => `translate(${d.x},${d.y})`);
  }

  // Tooltip
  const tooltip = d3.select("body").select(".tooltip").node()
    ? d3.select("body .tooltip")
    : d3.select("body").append("div").attr("class", "tooltip").style("display", "none");

  function showTooltip(event, d) {
    tooltip.style("display", "block").html(
      `<strong>Nodo ${d.id}</strong><br>` +
      `Atención: ${(d.attn * 100).toFixed(1)}%<br>` +
      `‖feat‖: ${d.norm?.toFixed(2) ?? "—"}`
    );
    moveTooltip(event);
  }
  function moveTooltip(event) {
    tooltip.style("left", (event.pageX + 12) + "px").style("top", (event.pageY - 28) + "px");
  }
  function hideTooltip() { tooltip.style("display", "none"); }
}

/** Render PCA scatter plot of node embeddings using Plotly */
export function renderPCA(container, pcaData, nodeAttention = null, title = "") {
  const { coords, variance_explained } = pcaData;
  const N = coords.length;
  const attn = nodeAttention || Array(N).fill(0.5);
  const maxA = Math.max(...attn, 1e-6);
  const normA = attn.map(v => v / maxA);

  // Magenta gradient colors
  const colors = normA.map(v => {
    const r = Math.round(204 * v + 42 * (1 - v));
    const g = Math.round(0);
    const b = Math.round(168 * v + 42 * (1 - v));
    return `rgb(${r},${g},${b})`;
  });

  const trace = {
    x: coords.map(p => p[0]),
    y: coords.map(p => p[1]),
    mode: "markers+text",
    type: "scatter",
    text: Array.from({ length: N }, (_, i) => String(i)),
    textfont: { size: 8, color: "#555" },
    textposition: "top center",
    marker: {
      size: normA.map(v => 7 + v * 10),
      color: colors,
      line: { width: 0.5, color: "#333" },
    },
    hovertemplate: "Nodo %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
  };

  const layout = {
    title: { text: title, font: { color: "#888", size: 12 } },
    paper_bgcolor: "#1c1c1c",
    plot_bgcolor: "#1c1c1c",
    font: { color: "#888", family: "Inter, sans-serif", size: 11 },
    xaxis: {
      title: `PC1 (${(variance_explained[0] * 100).toFixed(1)}%)`,
      gridcolor: "#2a2a2a", zerolinecolor: "#2a2a2a",
    },
    yaxis: {
      title: `PC2 (${(variance_explained[1] * 100).toFixed(1)}%)`,
      gridcolor: "#2a2a2a", zerolinecolor: "#2a2a2a",
    },
    margin: { l: 50, r: 16, t: 36, b: 50 },
    showlegend: false,
    height: 260,
  };

  Plotly.react(container, [trace], layout, { displayModeBar: false, responsive: true });
}

/** Render attention weight bar chart using Plotly */
export function renderAttentionBars(container, nodeAttention, title = "") {
  const N = nodeAttention.length;
  const maxA = Math.max(...nodeAttention, 1e-6);
  const normA = nodeAttention.map(v => v / maxA);

  const colors = normA.map(v => {
    const r = Math.round(204 * v + 68 * (1 - v));
    const b = Math.round(168 * v + 68 * (1 - v));
    return `rgba(${r},0,${b},0.85)`;
  });

  const trace = {
    x: Array.from({ length: N }, (_, i) => `N${i}`),
    y: nodeAttention,
    type: "bar",
    marker: { color: colors },
    hovertemplate: "Nodo %{x}<br>Atención: %{y:.4f}<extra></extra>",
  };

  const layout = {
    title: { text: title, font: { color: "#888", size: 12 } },
    paper_bgcolor: "#1c1c1c",
    plot_bgcolor: "#1c1c1c",
    font: { color: "#888", family: "Inter, sans-serif", size: 11 },
    xaxis: { title: "Nodo", gridcolor: "#2a2a2a", color: "#555" },
    yaxis: { title: "Peso de atención (media)", gridcolor: "#2a2a2a", color: "#555" },
    margin: { l: 50, r: 16, t: 36, b: 50 },
    height: 220,
  };

  Plotly.react(container, [trace], layout, { displayModeBar: false, responsive: true });
}
