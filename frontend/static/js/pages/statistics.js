import { API } from "../api.js";

export async function renderStatistics(container) {
  container.innerHTML = `<div class="loading-spinner"><div class="spinner"></div><p>Calculando estadísticas…</p></div>`;

  let data;
  try {
    data = await API.stats();
  } catch (e) {
    container.innerHTML = `<div class="empty-state"><p>Error al cargar estadísticas</p><small>${e.message}</small></div>`;
    return;
  }

  if (data.error) {
    container.innerHTML = `
      <div class="page-header"><h1 class="page-title">Estadísticas</h1></div>
      <div class="notice">
        <i data-lucide="alert-triangle"></i>
        ${data.error}
      </div>`;
    lucide.createIcons();
    return;
  }

  const acc = data.accuracy != null ? (data.accuracy * 100).toFixed(1) + "%" : "—";
  const auc = data.auc != null ? data.auc.toFixed(4) : "—";
  const total = data.total_samples ?? 0;
  const dist = data.class_distribution || {};

  container.innerHTML = `
    <div class="page-header">
      <h1 class="page-title">Estadísticas del modelo</h1>
      <p class="page-sub">Evaluado sobre el split de validación (${total} muestras)</p>
    </div>

    <!-- KPI cards -->
    <div class="grid-4 section">
      <div class="card">
        <div class="card-title">Accuracy</div>
        <div class="card-value accent">${acc}</div>
        <div class="card-sub">val set</div>
      </div>
      <div class="card">
        <div class="card-title">AUC-ROC</div>
        <div class="card-value accent">${auc}</div>
        <div class="card-sub">área bajo la curva ROC</div>
      </div>
      <div class="card">
        <div class="card-title">N0 (sin metástasis)</div>
        <div class="card-value">${dist.N0 ?? "—"}</div>
        <div class="card-sub">${total ? ((dist.N0 / total) * 100).toFixed(1) + "% del total" : ""}</div>
      </div>
      <div class="card">
        <div class="card-title">N1 (con metástasis)</div>
        <div class="card-value">${dist.N1 ?? "—"}</div>
        <div class="card-sub">${total ? ((dist.N1 / total) * 100).toFixed(1) + "% del total" : ""}</div>
      </div>
    </div>

    <!-- Charts row 1 -->
    <div class="two-col section">
      <div class="card chart-container">
        <div class="card-title" style="margin-bottom:0">Curva Precision-Recall</div>
        <div id="chart-pr"></div>
      </div>
      <div class="card chart-container">
        <div class="card-title" style="margin-bottom:0">Curva ROC</div>
        <div id="chart-roc"></div>
      </div>
    </div>

    <!-- Charts row 2 -->
    <div class="two-col section">
      <div class="card chart-container">
        <div class="card-title" style="margin-bottom:0">Matriz de confusión</div>
        <div id="chart-cm"></div>
      </div>
      <div class="card">
        <div class="card-title">Métricas por clase</div>
        <div id="metrics-table-wrap"></div>
      </div>
    </div>
  `;

  lucide.createIcons();

  // ── Precision-Recall curve ─────────────────────────────────────────────────
  if (data.precision_recall) {
    const { precision, recall } = data.precision_recall;

    // Compute AUC-PR (trapezoidal)
    let aucPR = 0;
    for (let i = 1; i < recall.length; i++) {
      aucPR += Math.abs(recall[i - 1] - recall[i]) * (precision[i] + precision[i - 1]) / 2;
    }

    Plotly.react(container.querySelector("#chart-pr"), [{
      x: recall,
      y: precision,
      mode: "lines",
      fill: "tozeroy",
      fillcolor: "rgba(204,0,168,0.08)",
      line: { color: "#cc00a8", width: 2 },
      name: `AP = ${aucPR.toFixed(3)}`,
      hovertemplate: "Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>",
    }], plotLayout("Recall", "Precision", `AP = ${aucPR.toFixed(3)}`), { displayModeBar: false, responsive: true });
  }

  // ── ROC curve ──────────────────────────────────────────────────────────────
  if (data.roc) {
    const { fpr, tpr } = data.roc;
    Plotly.react(container.querySelector("#chart-roc"), [
      {
        x: fpr, y: tpr,
        mode: "lines",
        fill: "tozeroy",
        fillcolor: "rgba(204,0,168,0.08)",
        line: { color: "#cc00a8", width: 2 },
        name: `AUC = ${auc}`,
        hovertemplate: "FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>",
      },
      {
        x: [0, 1], y: [0, 1],
        mode: "lines",
        line: { color: "#333", width: 1, dash: "dash" },
        name: "Random",
        hoverinfo: "skip",
      },
    ], plotLayout("FPR", "TPR", `AUC = ${auc}`), { displayModeBar: false, responsive: true });
  }

  // ── Confusion matrix ───────────────────────────────────────────────────────
  if (data.confusion_matrix) {
    const cm = data.confusion_matrix; // [[TN, FP], [FN, TP]]
    const labels = ["N0", "N1"];
    const z = cm;
    const zMax = Math.max(...cm.flat()) || 1;

    Plotly.react(container.querySelector("#chart-cm"), [{
      z,
      x: labels,
      y: labels,
      type: "heatmap",
      colorscale: [[0, "#1c1c1c"], [1, "#cc00a8"]],
      showscale: false,
      text: cm.map(row => row.map(v => String(v))),
      texttemplate: "<b>%{text}</b>",
      hovertemplate: "Real: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
    }], {
      paper_bgcolor: "#1c1c1c",
      plot_bgcolor: "#1c1c1c",
      font: { color: "#888", family: "Inter, sans-serif", size: 12 },
      xaxis: { title: "Predicho", color: "#888" },
      yaxis: { title: "Real", color: "#888", autorange: "reversed" },
      margin: { l: 60, r: 20, t: 20, b: 60 },
      height: 280,
      annotations: cm.flatMap((row, i) =>
        row.map((v, j) => ({
          x: labels[j], y: labels[i],
          text: `<b>${v}</b>`,
          font: { color: v / zMax > 0.5 ? "#fff" : "#888", size: 16 },
          showarrow: false,
        }))
      ),
    }, { displayModeBar: false, responsive: true });
  }

  // ── Metrics per class table ────────────────────────────────────────────────
  // Derive from confusion matrix
  const cm = data.confusion_matrix;
  if (cm) {
    const TN = cm[0][0], FP = cm[0][1], FN = cm[1][0], TP = cm[1][1];
    const precN0 = TN / Math.max(TN + FN, 1);
    const recN0  = TN / Math.max(TN + FP, 1);
    const f1N0   = f1(precN0, recN0);
    const precN1 = TP / Math.max(TP + FP, 1);
    const recN1  = TP / Math.max(TP + FN, 1);
    const f1N1   = f1(precN1, recN1);

    container.querySelector("#metrics-table-wrap").innerHTML = `
      <table class="metrics-table">
        <thead>
          <tr><th>Clase</th><th>Precision</th><th>Recall</th><th>F1</th><th>Soporte</th></tr>
        </thead>
        <tbody>
          <tr>
            <td><span class="badge badge-n0">N0</span></td>
            <td class="mono">${precN0.toFixed(3)}</td>
            <td class="mono">${recN0.toFixed(3)}</td>
            <td class="mono">${f1N0.toFixed(3)}</td>
            <td class="mono">${TN + FP}</td>
          </tr>
          <tr>
            <td><span class="badge badge-n1">N1</span></td>
            <td class="mono">${precN1.toFixed(3)}</td>
            <td class="mono">${recN1.toFixed(3)}</td>
            <td class="mono">${f1N1.toFixed(3)}</td>
            <td class="mono">${FN + TP}</td>
          </tr>
          <tr style="border-top:1px solid var(--border2)">
            <td style="color:var(--text3)">Media</td>
            <td class="mono">${((precN0 + precN1) / 2).toFixed(3)}</td>
            <td class="mono">${((recN0 + recN1) / 2).toFixed(3)}</td>
            <td class="mono">${((f1N0 + f1N1) / 2).toFixed(3)}</td>
            <td class="mono">${total}</td>
          </tr>
        </tbody>
      </table>

      <div style="margin-top:16px">
        <div class="card-title">Distribución de clases</div>
        <div id="chart-dist" style="margin-top:8px"></div>
      </div>`;

    // Class distribution donut
    Plotly.react(container.querySelector("#chart-dist"), [{
      values: [dist.N0 ?? 0, dist.N1 ?? 0],
      labels: ["N0", "N1"],
      type: "pie",
      hole: 0.6,
      marker: { colors: ["#00d488", "#ff4466"] },
      textinfo: "label+percent",
      textfont: { color: "#888", size: 12 },
      hovertemplate: "%{label}: %{value} (%{percent})<extra></extra>",
    }], {
      paper_bgcolor: "#1c1c1c",
      plot_bgcolor: "#1c1c1c",
      font: { color: "#888", family: "Inter, sans-serif", size: 12 },
      margin: { l: 20, r: 20, t: 10, b: 20 },
      height: 200,
      showlegend: true,
      legend: { font: { color: "#888" } },
    }, { displayModeBar: false, responsive: true });
  }
}

function plotLayout(xlabel, ylabel, title = "") {
  return {
    title: { text: title, font: { color: "#666", size: 12 } },
    paper_bgcolor: "#1c1c1c",
    plot_bgcolor: "#1c1c1c",
    font: { color: "#888", family: "Inter, sans-serif", size: 11 },
    xaxis: { title: xlabel, gridcolor: "#2a2a2a", color: "#666", range: [-0.02, 1.02] },
    yaxis: { title: ylabel, gridcolor: "#2a2a2a", color: "#666", range: [-0.02, 1.02] },
    margin: { l: 50, r: 20, t: 36, b: 50 },
    height: 280,
    showlegend: true,
    legend: { font: { color: "#888" } },
  };
}

function f1(p, r) {
  return (p + r) > 0 ? 2 * p * r / (p + r) : 0;
}
