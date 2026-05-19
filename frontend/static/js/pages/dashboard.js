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
  const checkpoints = (ckptsData.checkpoints || []).slice().sort((a, b) => {
    const ka = a.cv?.auc_mean ?? a.val_auc ?? -Infinity;
    const kb = b.cv?.auc_mean ?? b.val_auc ?? -Infinity;
    return kb - ka;
  });

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
    ${checkpoints.length > 0 ? (() => {
      const scale = computeAucScale(checkpoints);
      const ticks = scaleTicks(scale);
      return `
    <div class="section">
      <div class="section-title"><i data-lucide="layers"></i> Seleccionar model
        <span style="margin-left:8px;font-size:11px;color:var(--text3);font-weight:400">${checkpoints.length} models · ordenats per AUC val (CV) ↓</span>
      </div>
      <div class="card" style="padding:0;overflow:hidden">
        <div style="max-height:340px;overflow-y:auto">
          <table style="width:100%;border-collapse:collapse;font-size:12px">
            <thead>
              <tr style="background:var(--bg2);position:sticky;top:0;z-index:1">
                <th style="padding:7px 6px;text-align:center;color:var(--text3);font-weight:500;width:30px" title="Model estrella (es carrega per defecte)">★</th>
                <th style="padding:7px 10px;text-align:left;color:var(--text3);font-weight:500;white-space:nowrap"></th>
                <th style="padding:7px 10px;text-align:right;color:var(--text3);font-weight:500;white-space:nowrap">AUC test</th>
                <th style="padding:7px 10px;color:var(--text3);font-weight:500;white-space:nowrap;width:260px">
                  ${renderAxisHeader(ticks)}
                </th>
                <th style="padding:7px 10px;text-align:left;color:var(--text3);font-weight:500">Pooling · MIL</th>
                <th style="padding:7px 10px;text-align:left;color:var(--text3);font-weight:500">Nom</th>
                <th style="padding:7px 4px"></th>
              </tr>
            </thead>
            <tbody id="ckpt-table-body">
              ${checkpoints.map((c, i) => {
                const testAuc = c.test_auc != null ? c.test_auc.toFixed(3) : "—";
                const poolMil = [c.pooling, c.aggregation].filter(Boolean).join(" · ") || "—";
                const shortName = c.name.replace(/_best$/, "").replace(/^gs\d+\//, "");
                const isActive = c.active;
                const isStar   = !!c.star;
                const rowBg = isActive ? "background:rgba(204,0,168,0.08)" : (i % 2 === 0 ? "" : "background:var(--bg2)");
                const aucColor = c.test_auc != null
                  ? (c.test_auc >= 0.85 ? "var(--green)" : c.test_auc >= 0.70 ? "var(--accent-light)" : "var(--text3)")
                  : "var(--text3)";
                const starColor = isStar ? "#ffc94a" : "var(--text3)";
                const starGlyph = isStar ? "★" : "☆";
                return `<tr class="ckpt-row" data-name="${c.name}" style="cursor:pointer;border-bottom:1px solid var(--border1);${rowBg}">
                  <td style="padding:4px 6px;text-align:center">
                    <button class="btn-star" data-name="${c.name}" data-star="${isStar ? 1 : 0}" title="${isStar ? "Treure estrella" : "Marcar com a model estrella (es carregarà per defecte)"}"
                            style="background:none;border:none;cursor:pointer;font-size:18px;color:${starColor};padding:2px;line-height:1">${starGlyph}</button>
                  </td>
                  <td style="padding:6px 10px;color:var(--accent-light);font-size:13px">${isActive ? "▶" : ""}</td>
                  <td style="padding:6px 10px;text-align:right;font-family:var(--mono);font-weight:600;color:${aucColor}">${testAuc}</td>
                  <td style="padding:6px 10px">${renderAucTrack(c, scale, ticks)}</td>
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
        <div style="padding:8px 12px;background:var(--bg2);border-top:1px solid var(--border1);font-size:11px;color:var(--text3);display:flex;gap:16px;align-items:center">
          <span id="model-select-status"></span>
          <span style="display:inline-flex;align-items:center;gap:6px">
            <span style="display:inline-block;width:18px;height:8px;background:rgba(204,0,168,0.22);border:1px solid rgba(204,0,168,0.45);border-radius:2px"></span>
            CV val ± σ
          </span>
          <span style="display:inline-flex;align-items:center;gap:6px">
            <span style="display:inline-block;width:10px;height:10px;background:#3ee089;border:1.5px solid #0a0a0a;border-radius:50%"></span>
            AUC test (si calculat)
          </span>
          <span style="margin-left:auto">Clic a una fila o al botó per carregar el model</span>
        </div>
      </div>
    </div>
    `;
    })() : ""}

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
        <div class="card-sub">${
          ck?.cv?.auc_mean != null
            ? `CV: ${ck.cv.auc_mean.toFixed(4)} ± ${ck.cv.auc_std?.toFixed(4) ?? "—"} (${ck.cv.folds}-fold)`
            : (ck?.test_auc != null ? `val: ${ck.val_auc?.toFixed(4) ?? "—"} · ep. ${ck?.epoch ?? "—"}` : `val AUC · epoch ${ck?.epoch ?? "—"}`)
        }</div>
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
            ${ck.cv?.auc_mean != null ? `<div class="stat-row"><span class="stat-key">CV AUC (${ck.cv.folds}-fold)</span><span class="stat-val accent">${ck.cv.auc_mean.toFixed(4)} ± ${ck.cv.auc_std?.toFixed(4) ?? "—"}</span></div>` : ""}
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
  // ── Star toggle (defineix el model que es carregarà a startup) ─────────────
  container.querySelectorAll(".btn-star").forEach(btn => {
    btn.addEventListener("click", async e => {
      e.stopPropagation();
      const isStar = btn.dataset.star === "1";
      try {
        await API.setStarModel(isStar ? null : btn.dataset.name);
        renderDashboard(container);
      } catch (err) {
        selStatus.style.color = "var(--red)";
        selStatus.textContent = `✗ Star: ${err.message}`;
      }
    });
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

// ── boxplot horitzontal: escala global per a tots els models ─────────────────

function computeAucScale(checkpoints) {
  const vals = [];
  checkpoints.forEach(c => {
    const m = c.cv?.auc_mean, s = c.cv?.auc_std;
    if (m != null) {
      vals.push(m);
      if (s != null) { vals.push(m - s); vals.push(m + s); }
    } else if (c.val_auc != null) {
      vals.push(c.val_auc);
    }
    if (c.test_auc != null) vals.push(c.test_auc);
  });
  if (vals.length === 0) return { min: 0, max: 1 };
  const lo = Math.max(0, Math.min(...vals) - 0.03);
  const hi = Math.min(1, Math.max(...vals) + 0.03);
  return { min: lo, max: hi };
}

// Genera ticks "rodons" (cada 0.05 o 0.1 segons rang) dins de [min, max].
function scaleTicks({ min, max }) {
  const range = max - min;
  const step = range > 0.4 ? 0.1 : 0.05;
  const start = Math.ceil(min / step) * step;
  const ticks = [];
  for (let v = start; v <= max + 1e-9; v += step) ticks.push(+v.toFixed(2));
  return { min, max, step, values: ticks };
}

function pctOf(v, scale) {
  return ((v - scale.min) / (scale.max - scale.min)) * 100;
}

function renderAxisHeader(ticks) {
  const labels = ticks.values.map(t => `
    <span style="position:absolute;left:${pctOf(t, ticks)}%;transform:translateX(-50%);
                 font-family:var(--mono);font-size:9.5px;color:var(--text3)">${t.toFixed(2)}</span>
  `).join("");
  return `<div style="position:relative;height:14px">${labels}</div>`;
}

function renderAucTrack(c, scale, ticks) {
  const m = c.cv?.auc_mean, s = c.cv?.auc_std;
  const test = c.test_auc;
  const val  = c.val_auc;

  // Ticks faints alineats amb la capçalera.
  const tickLines = ticks.values.map(t => `
    <span style="position:absolute;left:${pctOf(t, scale)}%;top:0;bottom:0;width:1px;
                 background:var(--border1);opacity:0.55"></span>
  `).join("");

  // Box: CV mean ± σ  (o petit marcador si només hi ha val_auc).
  let box = "";
  let tooltip = "";
  if (m != null && s != null && s > 0) {
    const lo = pctOf(m - s, scale);
    const hi = pctOf(m + s, scale);
    box = `
      <span style="position:absolute;left:${lo}%;width:${hi - lo}%;top:3px;bottom:3px;
                   background:rgba(204,0,168,0.22);border:1px solid rgba(204,0,168,0.45);
                   border-radius:2px"></span>
      <span style="position:absolute;left:${pctOf(m, scale)}%;top:1px;bottom:1px;width:1.5px;
                   background:rgba(204,0,168,0.85);transform:translateX(-50%)"></span>
    `;
    tooltip = `CV val: ${m.toFixed(4)} ± ${s.toFixed(4)}`;
  } else if (m != null) {
    box = `<span style="position:absolute;left:${pctOf(m, scale)}%;top:1px;bottom:1px;width:2px;
                 background:rgba(204,0,168,0.7);transform:translateX(-50%)"></span>`;
    tooltip = `CV val: ${m.toFixed(4)}`;
  } else if (val != null) {
    box = `<span style="position:absolute;left:${pctOf(val, scale)}%;top:1px;bottom:1px;width:2px;
                 background:rgba(204,0,168,0.7);transform:translateX(-50%)"></span>`;
    tooltip = `val: ${val.toFixed(4)}`;
  } else {
    tooltip = "sense AUC val";
  }

  // Punt del test AUC.
  let dot = "";
  if (test != null) {
    dot = `<span style="position:absolute;left:${pctOf(test, scale)}%;top:50%;
                 width:11px;height:11px;background:#3ee089;border:2px solid #0a0a0a;
                 border-radius:50%;transform:translate(-50%,-50%);
                 box-shadow:0 0 0 1px rgba(62,224,137,0.55), 0 0 6px rgba(62,224,137,0.7);
                 z-index:3"></span>`;
    tooltip += ` · test: ${test.toFixed(4)}`;
  } else {
    tooltip += " · test: no calculat";
  }

  return `
    <div title="${tooltip}" style="position:relative;height:18px;background:var(--bg2);
                                   border-radius:3px">
      ${tickLines}
      ${box}
      ${dot}
    </div>`;
}
