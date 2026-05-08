import { API } from "../api.js";

export async function renderDashboard(container) {
  container.innerHTML = `<div class="loading-spinner"><div class="spinner"></div><p>Carregant estat…</p></div>`;

  let status, ckptsData;
  try {
    [status, ckptsData] = await Promise.all([API.status(), API.checkpoints()]);
  } catch (e) {
    container.innerHTML = `<div class="empty-state"><p>No s'ha pogut connectar amb el backend.</p><small>${e.message}</small></div>`;
    return;
  }

  const ck          = status.checkpoint;
  const modelLoaded = status.model_loaded;
  const checkpoints = ckptsData.checkpoints || [];

  const archHTML = ck ? `
    <div class="arch-diagram">
      ${archLayer("Conv1", `in=${ck.in_channels}→hidden=${ck.hidden}`, `heads=${ck.heads}, concat`)}
      <div class="arch-arrow">↓ BN · ELU · Dropout</div>
      ${archLayer("Conv2", `${ck.hidden * ck.heads}→${ck.hidden}`, `heads=${ck.heads}, concat`)}
      <div class="arch-arrow">↓ BN · ELU · Dropout</div>
      ${archLayer("Conv3", `${ck.hidden * ck.heads}→${ck.hidden}`, `heads=1, no concat`)}
      <div class="arch-arrow">↓ BN · ELU</div>
      ${archLayer("Pool", poolingLabel(ck.pooling, ck.hidden), "")}
      <div class="arch-arrow">↓</div>
      ${archLayer("MLP", `→${ck.hidden}→2`, "ReLU · Dropout")}
    </div>
  ` : `<div class="empty-state" style="padding:20px"><p>Sense model carregat</p></div>`;

  container.innerHTML = `
    <div class="page-header">
      <h1 class="page-title">Dashboard</h1>
      <p class="page-sub">Vista general del model i les dades disponibles</p>
    </div>

    <!-- Model selector -->
    ${checkpoints.length > 0 ? `
    <div class="section">
      <div class="section-title"><i data-lucide="layers"></i> Seleccionar model
        <span style="margin-left:8px;font-size:11px;color:var(--text3);font-weight:400">${checkpoints.length} models · ordenats per AUC test ↓</span>
      </div>
      <div class="card" style="padding:0;overflow:hidden">
        <div style="max-height:340px;overflow-y:auto">
          <table style="width:100%;border-collapse:collapse;font-size:12px">
            <thead>
              <tr style="background:var(--bg2);position:sticky;top:0;z-index:1">
                <th style="padding:7px 10px;text-align:left;color:var(--text3);font-weight:500;white-space:nowrap"></th>
                <th style="padding:7px 10px;text-align:right;color:var(--text3);font-weight:500;white-space:nowrap">AUC test</th>
                <th style="padding:7px 10px;text-align:right;color:var(--text3);font-weight:500;white-space:nowrap">AUC val</th>
                <th style="padding:7px 10px;text-align:left;color:var(--text3);font-weight:500">Pooling · MIL</th>
                <th style="padding:7px 10px;text-align:left;color:var(--text3);font-weight:500">Nom</th>
                <th style="padding:7px 4px"></th>
              </tr>
            </thead>
            <tbody id="ckpt-table-body">
              ${checkpoints.map((c, i) => {
                const testAuc = c.test_auc != null ? c.test_auc.toFixed(3) : "—";
                const valAuc  = c.val_auc  != null ? c.val_auc.toFixed(3)  : "—";
                const poolMil = [c.pooling, c.aggregation].filter(Boolean).join(" · ") || "—";
                const shortName = c.name.replace(/_best$/, "").replace(/^gs2\//, "");
                const isActive = c.active;
                const rowBg = isActive ? "background:rgba(204,0,168,0.08)" : (i % 2 === 0 ? "" : "background:var(--bg2)");
                const aucColor = c.test_auc != null
                  ? (c.test_auc >= 0.85 ? "var(--green)" : c.test_auc >= 0.70 ? "var(--accent-light)" : "var(--text3)")
                  : "var(--text3)";
                return `<tr class="ckpt-row" data-name="${c.name}" style="cursor:pointer;border-bottom:1px solid var(--border1);${rowBg}">
                  <td style="padding:6px 10px;color:var(--accent-light);font-size:13px">${isActive ? "▶" : ""}</td>
                  <td style="padding:6px 10px;text-align:right;font-family:var(--mono);font-weight:600;color:${aucColor}">${testAuc}</td>
                  <td style="padding:6px 10px;text-align:right;font-family:var(--mono);color:var(--text2)">${valAuc}</td>
                  <td style="padding:6px 10px;color:var(--text2);white-space:nowrap">${poolMil}</td>
                  <td style="padding:6px 10px;color:var(--text3);font-family:var(--mono);font-size:11px;max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${c.name}">${shortName}</td>
                  <td style="padding:6px 4px">
                    <button class="btn-load-ckpt btn btn-ghost" data-name="${c.name}" style="padding:2px 8px;font-size:11px;white-space:nowrap">
                      ${isActive ? "actiu" : "carregar"}
                    </button>
                  </td>
                </tr>`;
              }).join("")}
            </tbody>
          </table>
        </div>
        <div style="padding:8px 12px;background:var(--bg2);border-top:1px solid var(--border1);font-size:11px;color:var(--text3);display:flex;gap:16px">
          <span id="model-select-status"></span>
          <span style="margin-left:auto">Clic a una fila o al botó per carregar el model</span>
        </div>
      </div>
    </div>
    ` : ""}

    <div class="grid-4 section">
      <div class="card ${modelLoaded ? "" : "card-warn"}">
        <div class="card-title">Estat del model</div>
        <div class="card-value ${modelLoaded ? "accent" : ""}" style="font-size:18px">
          ${modelLoaded ? "✓ Carregat" : "✗ Sense model"}
        </div>
        <div class="card-sub">${ck ? ck.name : "—"}</div>
      </div>

      <div class="card">
        <div class="card-title">AUC test</div>
        <div class="card-value accent">${ck?.test_auc != null ? ck.test_auc.toFixed(4) : (ck?.val_auc != null ? ck.val_auc.toFixed(4) : "—")}</div>
        <div class="card-sub">${ck?.test_auc != null ? `val: ${ck.val_auc?.toFixed(4) ?? "—"} · ep. ${ck?.epoch ?? "—"}` : `val AUC · epoch ${ck?.epoch ?? "—"}`}</div>
      </div>

      <div class="card">
        <div class="card-title">Grafs de test</div>
        <div class="card-value">${status.num_test_graphs ?? 0}</div>
        <div class="card-sub">${status.val_stats_ready ? "✓ Estadístiques llestes" : "estadístiques pendents"}</div>
      </div>

      <div class="card">
        <div class="card-title">Agregació MIL</div>
        <div class="card-value accent" style="font-size:13px;font-family:var(--mono)">${status.aggregation ?? "—"}</div>
        <div class="card-sub">mètode de combinació de slides</div>
      </div>
    </div>

    <div class="two-col section">
      <div class="card">
        <div class="card-title">Informació del checkpoint</div>
        ${ck ? `
          <div class="stat-list">
            <div class="stat-row"><span class="stat-key">Fitxer</span><span class="stat-val" style="max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11px">${ck.name}</span></div>
            <div class="stat-row"><span class="stat-key">Epoch</span><span class="stat-val">${ck.epoch ?? "—"}</span></div>
            <div class="stat-row"><span class="stat-key">Val AUC</span><span class="stat-val">${ck.val_auc?.toFixed(4) ?? "—"}</span></div>
            ${ck.val_f1_macro != null ? `<div class="stat-row"><span class="stat-key">Val F1 macro</span><span class="stat-val">${ck.val_f1_macro.toFixed(4)}</span></div>` : ""}
            <div class="stat-row"><span class="stat-key">Paràmetres</span><span class="stat-val">${ck.num_params?.toLocaleString() ?? "—"}</span></div>
            <div class="stat-row"><span class="stat-key">Dispositiu</span><span class="stat-val">${status.device.toUpperCase()}</span></div>
            <div class="stat-row"><span class="stat-key">in_channels</span><span class="stat-val">${ck.in_channels}</span></div>
            <div class="stat-row"><span class="stat-key">hidden</span><span class="stat-val">${ck.hidden}</span></div>
            <div class="stat-row"><span class="stat-key">heads</span><span class="stat-val">${ck.heads}</span></div>
            <div class="stat-row"><span class="stat-key">dropout</span><span class="stat-val">${ck.dropout}</span></div>
            <div class="stat-row"><span class="stat-key">pooling</span><span class="stat-val accent">${ck.pooling ?? "mean_max"}</span></div>
            <div class="stat-row"><span class="stat-key">config YAML</span><span class="stat-val">${ck.has_config ? "✓" : "—"}</span></div>
          </div>
        ` : `<div class="empty-state" style="padding:20px"><p>Sense checkpoint carregat</p></div>`}
      </div>

      <div class="card">
        <div class="card-title">Arquitectura GAT</div>
        ${archHTML}
      </div>
    </div>

    <div class="section">
      <div class="section-title"><i data-lucide="zap"></i> Accions ràpides</div>
      <div style="display:flex;gap:10px;flex-wrap:wrap">
        <button class="btn btn-primary" id="goto-inference-btn">
          <i data-lucide="play-circle"></i> Anar a Inferència
        </button>
        <button class="btn btn-ghost" id="goto-stats-btn">
          <i data-lucide="bar-chart-2"></i> Veure Estadístiques
        </button>
      </div>
    </div>

    <div class="section">
      <div class="section-title"><i data-lucide="folder-search"></i> Rutes de cerca</div>
      <div class="card" style="padding:14px">
        ${renderSearchPaths(status.search_paths)}
      </div>
    </div>

    ${!modelLoaded ? `
      <div class="notice">
        <i data-lucide="alert-triangle"></i>
        No s'ha trobat cap checkpoint. Entrena el model primer amb <code>python pt1diagnosis/PipelineGAT.py</code>.
      </div>
    ` : ""}
    ${!status.num_test_graphs ? `
      <div class="notice" style="margin-top:10px">
        <i data-lucide="database"></i>
        No hi ha grafs a <code>outputs/graphs/</code>. Genera les dades amb <code>python pt1diagnosis/scripts_david/build_dataset.py</code>.
      </div>
    ` : ""}
  `;

  lucide.createIcons();

  // ── model selector ──────────────────────────────────────────────────────────
  const selStatus = container.querySelector("#model-select-status");

  async function loadModel(name) {
    if (!name) return;
    selStatus.style.color = "var(--text3)";
    selStatus.textContent = "Carregant…";
    try {
      await API.selectModel(name);
      selStatus.style.color = "var(--green)";
      selStatus.textContent = "✓ Model carregat";
      setTimeout(() => renderDashboard(container), 400);
    } catch (e) {
      selStatus.style.color = "var(--red)";
      selStatus.textContent = `✗ Error: ${e.message}`;
    }
  }

  container.querySelectorAll(".ckpt-row").forEach(row => {
    row.addEventListener("click", () => loadModel(row.dataset.name));
  });
  container.querySelectorAll(".btn-load-ckpt").forEach(btn => {
    btn.addEventListener("click", e => { e.stopPropagation(); loadModel(btn.dataset.name); });
  });

  // ── navigation shortcuts ────────────────────────────────────────────────────
  container.querySelector("#goto-inference-btn")?.addEventListener("click", () => {
    document.querySelector('.nav-item[data-page="inference"]')?.click();
  });
  container.querySelector("#goto-stats-btn")?.addEventListener("click", () => {
    document.querySelector('.nav-item[data-page="statistics"]')?.click();
  });
}


// ── helpers ───────────────────────────────────────────────────────────────────

function poolingLabel(pooling, hidden) {
  if (!pooling || pooling === "mean_max")
    return `global_mean ⊕ global_max → ${hidden * 2}`;
  if (pooling === "mean") return `global_mean_pool → ${hidden}`;
  if (pooling === "max")  return `global_max_pool  → ${hidden}`;
  if (pooling === "sum")  return `global_add_pool  → ${hidden}`;
  if (pooling === "diff") return `DiffPool (hierarchical) → ${hidden * 2}`;
  return pooling;
}

function renderSearchPaths(sp) {
  if (!sp) return `<span style="color:var(--text3);font-size:12px">No disponible</span>`;

  const ckptOk  = sp.checkpoints_dir_exists;
  const graphsOk = sp.graphs_dir_exists;
  const ckpts   = sp.all_checkpoints || [];

  return `
    <div class="stat-list">
      <div class="stat-row">
        <span class="stat-key" style="display:flex;align-items:center;gap:6px">
          <span style="color:${ckptOk ? "var(--green)" : "var(--red)"}">●</span> Checkpoints
        </span>
        <span class="stat-val mono" style="font-size:11px;color:var(--text2)">${sp.checkpoints_dir}</span>
      </div>
      ${ckpts.length > 0
        ? ckpts.map((name, i) => `
          <div class="stat-row" style="padding-left:16px">
            <span class="stat-key" style="font-size:11.5px">${i === 0 ? "▶ actiu" : `#${i + 1}`}</span>
            <span class="stat-val mono" style="font-size:11px;color:${i === 0 ? "var(--accent-light)" : "var(--text3)"}">${name}</span>
          </div>`).join("")
        : `<div style="padding-left:16px;font-size:11.5px;color:var(--red);margin-top:4px">Cap fitxer .pt trobat</div>`
      }
      <div class="stat-row" style="margin-top:8px">
        <span class="stat-key" style="display:flex;align-items:center;gap:6px">
          <span style="color:${graphsOk ? "var(--green)" : "var(--red)"}">●</span> Grafs
        </span>
        <span class="stat-val mono" style="font-size:11px;color:var(--text2)">${sp.graphs_dir}</span>
      </div>
    </div>`;
}

function archLayer(name, desc, meta) {
  return `
    <div class="arch-layer">
      <span class="arch-name">${name}</span>
      <span class="arch-desc">${desc}</span>
      <span class="arch-meta">${meta}</span>
    </div>`;
}
