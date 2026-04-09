import { API } from "../api.js";

export async function renderDashboard(container) {
  container.innerHTML = `<div class="loading-spinner"><div class="spinner"></div><p>Cargando estado…</p></div>`;

  let status;
  try {
    status = await API.status();
  } catch (e) {
    container.innerHTML = `<div class="empty-state"><p>No se pudo conectar con el backend.</p><small>${e.message}</small></div>`;
    return;
  }

  const ck = status.checkpoint;
  const modelLoaded = status.model_loaded;

  // ── Architecture overview HTML ─────────────────────────────────────────────
  const archHTML = ck ? `
    <div class="arch-diagram">
      ${archLayer("Conv1", `in=${ck.in_channels}→hidden=${ck.hidden}`, `heads=${ck.heads}, concat`)}
      <div class="arch-arrow">↓ BN · ELU · Dropout</div>
      ${archLayer("Conv2", `${ck.hidden * ck.heads}→${ck.hidden}`, `heads=${ck.heads}, concat`)}
      <div class="arch-arrow">↓ BN · ELU · Dropout</div>
      ${archLayer("Conv3", `${ck.hidden * ck.heads}→${ck.hidden}`, `heads=1, no concat`)}
      <div class="arch-arrow">↓ BN · ELU</div>
      ${archLayer("Pool", "global_mean ⊕ global_max", `→ ${ck.hidden * 2}`)}
      <div class="arch-arrow">↓</div>
      ${archLayer("MLP", `${ck.hidden * 2}→${ck.hidden}→2`, "ReLU · Dropout")}
    </div>
  ` : `<div class="empty-state" style="padding:20px"><p>Sin modelo cargado</p></div>`;

  container.innerHTML = `
    <div class="page-header">
      <h1 class="page-title">Dashboard</h1>
      <p class="page-sub">Vista general del modelo y datos disponibles</p>
    </div>

    <!-- Status cards -->
    <div class="grid-4 section">
      <div class="card ${modelLoaded ? "" : "card-warn"}">
        <div class="card-title">Estado del modelo</div>
        <div class="card-value ${modelLoaded ? "accent" : ""}" style="font-size:18px">
          ${modelLoaded ? "✓ Cargado" : "✗ Sin modelo"}
        </div>
        <div class="card-sub">${ck ? ck.name : "—"}</div>
      </div>

      <div class="card">
        <div class="card-title">Mejor val AUC</div>
        <div class="card-value accent">${ck?.val_auc != null ? ck.val_auc.toFixed(4) : "—"}</div>
        <div class="card-sub">epoch ${ck?.epoch ?? "—"}</div>
      </div>

      <div class="card">
        <div class="card-title">Grafos de entrenamiento</div>
        <div class="card-value">${status.num_train_graphs}</div>
        <div class="card-sub">split train</div>
      </div>

      <div class="card">
        <div class="card-title">Grafos de validación</div>
        <div class="card-value">${status.num_val_graphs}</div>
        <div class="card-sub">${status.val_stats_ready ? "✓ Estadísticas listas" : "stats pendientes"}</div>
      </div>
    </div>

    <!-- Two-col: model info + architecture -->
    <div class="two-col section">
      <div class="card">
        <div class="card-title">Información del checkpoint</div>
        ${ck ? `
          <div class="stat-list">
            <div class="stat-row"><span class="stat-key">Archivo</span><span class="stat-val" style="max-width:160px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:11px">${ck.name}</span></div>
            <div class="stat-row"><span class="stat-key">Epoch</span><span class="stat-val">${ck.epoch ?? "—"}</span></div>
            <div class="stat-row"><span class="stat-key">Val AUC</span><span class="stat-val">${ck.val_auc?.toFixed(4) ?? "—"}</span></div>
            <div class="stat-row"><span class="stat-key">Parámetros</span><span class="stat-val">${ck.num_params?.toLocaleString() ?? "—"}</span></div>
            <div class="stat-row"><span class="stat-key">Dispositivo</span><span class="stat-val">${status.device.toUpperCase()}</span></div>
            <div class="stat-row"><span class="stat-key">in_channels</span><span class="stat-val">${ck.in_channels}</span></div>
            <div class="stat-row"><span class="stat-key">hidden</span><span class="stat-val">${ck.hidden}</span></div>
            <div class="stat-row"><span class="stat-key">heads</span><span class="stat-val">${ck.heads}</span></div>
            <div class="stat-row"><span class="stat-key">dropout</span><span class="stat-val">${ck.dropout}</span></div>
          </div>
        ` : `<div class="empty-state" style="padding:20px"><p>Sin checkpoint cargado</p></div>`}
      </div>

      <div class="card">
        <div class="card-title">Arquitectura GAT</div>
        ${archHTML}
      </div>
    </div>

    <!-- Quick actions -->
    <div class="section">
      <div class="section-title"><i data-lucide="zap"></i> Acciones rápidas</div>
      <div style="display:flex;gap:10px;flex-wrap:wrap">
        <button class="btn btn-primary" id="goto-inference-btn">
          <i data-lucide="play-circle"></i> Ir a Inferencia
        </button>
        <button class="btn btn-ghost" id="goto-stats-btn">
          <i data-lucide="bar-chart-2"></i> Ver Estadísticas
        </button>
      </div>
    </div>

    <!-- Notice if no model/data -->
    ${!modelLoaded ? `
      <div class="notice">
        <i data-lucide="alert-triangle"></i>
        No se encontró ningún checkpoint. Entrena el modelo primero con <code>python train.py</code>.
      </div>
    ` : ""}
    ${status.num_train_graphs === 0 && status.num_val_graphs === 0 ? `
      <div class="notice" style="margin-top:10px">
        <i data-lucide="database"></i>
        No hay grafos en <code>outputs/graphs/</code>. Genera los datos con <code>python scripts/build_dataset.py</code>.
      </div>
    ` : ""}
  `;

  lucide.createIcons();

  // Quick nav
  container.querySelector("#goto-inference-btn")?.addEventListener("click", () => {
    document.querySelector('.nav-item[data-page="inference"]')?.click();
  });
  container.querySelector("#goto-stats-btn")?.addEventListener("click", () => {
    document.querySelector('.nav-item[data-page="statistics"]')?.click();
  });
}

function archLayer(name, desc, meta) {
  return `
    <div class="arch-layer">
      <span class="arch-name">${name}</span>
      <span class="arch-desc">${desc}</span>
      <span class="arch-meta">${meta}</span>
    </div>`;
}
