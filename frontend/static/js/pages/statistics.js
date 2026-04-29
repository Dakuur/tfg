import { API } from "../api.js";

export async function renderStatistics(container) {
  container.innerHTML = `<div class="loading-spinner"><div class="spinner"></div><p>Calculant estadístiques…</p></div>`;

  let data;
  try {
    data = await API.stats();
  } catch (e) {
    container.innerHTML = `<div class="empty-state"><p>Error en carregar estadístiques</p><small>${e.message}</small></div>`;
    return;
  }

  if (data.error) {
    container.innerHTML = `
      <div class="page-header"><h1 class="page-title">Estadístiques</h1></div>
      <div class="notice">
        <i data-lucide="alert-triangle"></i>
        ${data.error}
      </div>`;
    lucide.createIcons();
    return;
  }

  const acc   = data.accuracy != null ? (data.accuracy * 100).toFixed(1) + "%" : "—";
  const auc   = data.auc != null ? data.auc.toFixed(4) : "—";
  const total = data.total_samples ?? 0;
  const level = data.level === "patient" ? "pacients" : "mostres";
  const dist  = data.class_distribution || {};
  const curAgg = data.aggregation ?? "noisy_or";

  container.innerHTML = `
    <div class="page-header">
      <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:0.75rem">
        <div>
          <h1 class="page-title">Estadístiques del model</h1>
          <p class="page-sub">Avaluat sobre el split de validació (${total} ${level})</p>
        </div>
        <div style="padding-top:0.3rem;font-size:0.82rem;color:#666">
          Agregació: <span style="color:#999;font-family:monospace">${curAgg}</span>
        </div>
      </div>
    </div>

    <div class="grid-4 section">
      <div class="card">
        <div class="card-title">Exactitud</div>
        <div class="card-value accent">${acc}</div>
        <div class="card-sub">val set</div>
      </div>
      <div class="card">
        <div class="card-title">AUC-ROC</div>
        <div class="card-value accent">${auc}</div>
        <div class="card-sub">àrea sota la corba ROC</div>
      </div>
      <div class="card">
        <div class="card-title">N0 (sense metàstasi)</div>
        <div class="card-value">${dist.N0 ?? "—"}</div>
        <div class="card-sub">${total ? ((dist.N0 / total) * 100).toFixed(1) + "% del total" : ""}</div>
      </div>
      <div class="card">
        <div class="card-title">N1 (amb metàstasi)</div>
        <div class="card-value">${dist.N1 ?? "—"}</div>
        <div class="card-sub">${total ? ((dist.N1 / total) * 100).toFixed(1) + "% del total" : ""}</div>
      </div>
    </div>

    <div class="two-col section">
      <div class="card chart-container">
        <div class="card-title" style="margin-bottom:0">Corba Precisió-Recall</div>
        <div id="chart-pr"></div>
      </div>
      <div class="card chart-container">
        <div class="card-title" style="margin-bottom:0">Corba ROC</div>
        <div id="chart-roc"></div>
      </div>
    </div>

    <div class="two-col section">
      <div class="card chart-container">
        <div class="card-title" style="margin-bottom:0">Matriu de confusió</div>
        <div id="chart-cm"></div>
      </div>
      <div class="card">
        <div class="card-title">Mètriques per classe</div>
        <div id="metrics-table-wrap"></div>
      </div>
    </div>
  `;

  lucide.createIcons();

  // ── Corba Precisió-Recall ──────────────────────────────────────────────────
  if (data.precision_recall) {
    const { precision, recall } = data.precision_recall;

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
      hovertemplate: "Recall: %{x:.3f}<br>Precisió: %{y:.3f}<extra></extra>",
    }], plotLayout("Recall", "Precisió", `AP = ${aucPR.toFixed(3)}`), { displayModeBar: false, responsive: true });
  }

  // ── Corba ROC ──────────────────────────────────────────────────────────────
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
        name: "Aleatori",
        hoverinfo: "skip",
      },
    ], plotLayout("FPR", "TPR", `AUC = ${auc}`), { displayModeBar: false, responsive: true });
  }

  // ── Matriu de confusió ─────────────────────────────────────────────────────
  if (data.confusion_matrix) {
    const cm = data.confusion_matrix;
    const labels = ["N0", "N1"];
    const zMax = Math.max(...cm.flat()) || 1;

    // Note: use EITHER texttemplate OR annotations — never both (causes overlap).
    // Annotations are preferred: they allow adaptive text colour per cell.
    Plotly.react(container.querySelector("#chart-cm"), [{
      z: cm,
      x: labels,
      y: labels,
      type: "heatmap",
      colorscale: [[0, "#1c1c1c"], [1, "#cc00a8"]],
      showscale: false,
      hovertemplate: "Real: %{y}<br>Predit: %{x}<br>Count: %{z}<extra></extra>",
    }], {
      paper_bgcolor: "#1c1c1c",
      plot_bgcolor: "#1c1c1c",
      font: { color: "#888", family: "Inter, sans-serif", size: 12 },
      xaxis: { title: "Predit", color: "#888", tickfont: { size: 13 } },
      yaxis: { title: "Real",   color: "#888", tickfont: { size: 13 }, autorange: "reversed" },
      margin: { l: 60, r: 20, t: 20, b: 60 },
      height: 280,
      annotations: cm.flatMap((row, i) =>
        row.map((v, j) => ({
          x: labels[j], y: labels[i],
          text: `<b>${v}</b>`,
          font: { color: v / zMax > 0.5 ? "#fff" : "#aaa", size: 18, family: "Inter, sans-serif" },
          showarrow: false,
          xanchor: "center",
          yanchor: "middle",
        }))
      ),
    }, { displayModeBar: false, responsive: true });
  }

  // ── Taula de mètriques per classe ──────────────────────────────────────────
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
          <tr><th>Classe</th><th>Precisió</th><th>Recall</th><th>F1</th><th>Suport</th></tr>
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
            <td style="color:var(--text3)">Mitjana</td>
            <td class="mono">${((precN0 + precN1) / 2).toFixed(3)}</td>
            <td class="mono">${((recN0 + recN1) / 2).toFixed(3)}</td>
            <td class="mono">${((f1N0 + f1N1) / 2).toFixed(3)}</td>
            <td class="mono">${total}</td>
          </tr>
        </tbody>
      </table>

      <div style="margin-top:16px">
        <div class="card-title">Distribució de classes</div>
        <div id="chart-dist" style="margin-top:8px"></div>
      </div>`;

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
