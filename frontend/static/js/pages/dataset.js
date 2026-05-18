/**
 * Dataset gallery — graella de thumbnails _low.png amb visor a pantalla completa.
 * Pensat per buscar visualment imatges boniques per al TFG.
 */
import { API } from "../api.js";

const COLS = 10;

export async function renderDataset(container) {
  container.innerHTML = `<div class="loading-spinner"><div class="spinner"></div><p>Indexant slides…</p></div>`;

  let data;
  try {
    data = await API.datasetSlides();
  } catch (e) {
    container.innerHTML = `<div class="empty-state"><p>Error: ${e.message}</p></div>`;
    return;
  }

  const slides = data.slides || [];
  const hospitals = [...new Set(slides.map(s => s.hospital))].sort();

  container.innerHTML = `
    <div class="page-header">
      <h1 class="page-title">Dataset — galeria de slides</h1>
      <p class="page-sub">${slides.length} thumbnails (_low.png) · ${hospitals.length} hospitals · clic per veure a pantalla completa</p>
    </div>

    <div class="section" style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
      <label style="font-size:12px;color:var(--text2)">Hospital:</label>
      <select id="ds-hospital" class="model-select" style="min-width:240px">
        <option value="">Tots (${slides.length})</option>
        ${hospitals.map(h => {
          const n = slides.filter(s => s.hospital === h).length;
          return `<option value="${escapeAttr(h)}">${escapeHtml(h)} (${n})</option>`;
        }).join("")}
      </select>
      <input id="ds-search" type="text" placeholder="Cerca per nom/pacient…" class="model-select" style="flex:1;min-width:200px"/>
      <button id="ds-refresh" class="btn btn-ghost" title="Tornar a escanejar el directori RGB_Images">
        <i data-lucide="refresh-cw"></i> Refrescar índex
      </button>
      <span id="ds-count" style="font-size:11.5px;color:var(--text3);margin-left:auto"></span>
    </div>

    <div id="ds-grid" class="dataset-grid"></div>

    <!-- Lightbox -->
    <div id="ds-lightbox" class="ds-lightbox hidden">
      <button class="ds-lb-close" title="Tancar (Esc)"><i data-lucide="x"></i></button>
      <div class="ds-lb-nav">
        <button class="ds-lb-prev" title="Anterior (←)"><i data-lucide="chevron-left"></i></button>
        <button class="ds-lb-next" title="Següent (→)"><i data-lucide="chevron-right"></i></button>
      </div>
      <img id="ds-lb-img" alt=""/>
      <div id="ds-lb-caption" class="ds-lb-caption"></div>
    </div>
  `;

  injectCSS();
  lucide.createIcons();

  const grid       = container.querySelector("#ds-grid");
  const hospitalEl = container.querySelector("#ds-hospital");
  const searchEl   = container.querySelector("#ds-search");
  const refreshBtn = container.querySelector("#ds-refresh");
  const countEl    = container.querySelector("#ds-count");
  const lightbox   = container.querySelector("#ds-lightbox");
  const lbImg      = container.querySelector("#ds-lb-img");
  const lbCap      = container.querySelector("#ds-lb-caption");

  let filtered = slides;
  let currentIdx = -1;

  function applyFilter() {
    const h = hospitalEl.value;
    const q = searchEl.value.trim().toLowerCase();
    filtered = slides.filter(s => {
      if (h && s.hospital !== h) return false;
      if (q && !(`${s.name} ${s.patient}`).toLowerCase().includes(q)) return false;
      return true;
    });
    renderGrid();
  }

  function renderGrid() {
    countEl.textContent = `${filtered.length} mostrats`;
    if (filtered.length === 0) {
      grid.innerHTML = `<div class="empty-state" style="grid-column:1/-1"><p>Cap slide amb aquests filtres</p></div>`;
      return;
    }
    grid.innerHTML = filtered.map((s, i) => `
      <div class="ds-cell" data-idx="${i}" title="${escapeAttr(s.hospital + ' · ' + s.patient + ' · ' + s.name)}">
        <img loading="lazy" src="${API.datasetImageUrl(s.id)}" alt="${escapeAttr(s.name)}"/>
        <div class="ds-cell-label">${escapeHtml(s.name)}</div>
      </div>
    `).join("");
    grid.querySelectorAll(".ds-cell").forEach(cell => {
      cell.addEventListener("click", () => openLightbox(parseInt(cell.dataset.idx, 10)));
    });
  }

  function openLightbox(idx) {
    if (idx < 0 || idx >= filtered.length) return;
    currentIdx = idx;
    const s = filtered[idx];
    lbImg.src = API.datasetImageUrl(s.id);
    lbCap.innerHTML = `<strong>${escapeHtml(s.name)}</strong> · ${escapeHtml(s.hospital)} · pacient ${escapeHtml(s.patient)} <span style="color:var(--text3);margin-left:8px">(${idx + 1}/${filtered.length})</span>`;
    lightbox.classList.remove("hidden");
  }
  function closeLightbox() {
    lightbox.classList.add("hidden");
    lbImg.src = "";
    currentIdx = -1;
  }
  function navLightbox(delta) {
    if (currentIdx < 0) return;
    openLightbox((currentIdx + delta + filtered.length) % filtered.length);
  }

  // Events
  hospitalEl.addEventListener("change", applyFilter);
  let searchTimer;
  searchEl.addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(applyFilter, 200);
  });
  refreshBtn.addEventListener("click", async () => {
    refreshBtn.disabled = true;
    refreshBtn.querySelector("svg")?.classList.add("spinning");
    try {
      const fresh = await API.datasetSlides(true);
      slides.length = 0;
      slides.push(...(fresh.slides || []));
      applyFilter();
    } finally {
      refreshBtn.disabled = false;
      refreshBtn.querySelector("svg")?.classList.remove("spinning");
    }
  });
  container.querySelector(".ds-lb-close")?.addEventListener("click", closeLightbox);
  container.querySelector(".ds-lb-prev")?.addEventListener("click", () => navLightbox(-1));
  container.querySelector(".ds-lb-next")?.addEventListener("click", () => navLightbox(+1));
  lightbox.addEventListener("click", e => { if (e.target === lightbox) closeLightbox(); });
  document.addEventListener("keydown", e => {
    if (lightbox.classList.contains("hidden")) return;
    if (e.key === "Escape")     closeLightbox();
    if (e.key === "ArrowLeft")  navLightbox(-1);
    if (e.key === "ArrowRight") navLightbox(+1);
  });

  renderGrid();
}

// ── helpers ───────────────────────────────────────────────────────────────────

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;",
  }[c]));
}
function escapeAttr(s) { return escapeHtml(s); }

function injectCSS() {
  if (document.getElementById("ds-css")) return;
  const s = document.createElement("style");
  s.id = "ds-css";
  s.textContent = `
    .dataset-grid {
      display: grid;
      grid-template-columns: repeat(${COLS}, 1fr);
      gap: 6px;
      padding: 4px 0 40px;
    }
    .ds-cell {
      background: var(--bg2);
      border: 1px solid var(--border1);
      border-radius: 4px;
      overflow: hidden;
      cursor: pointer;
      transition: border-color .15s, transform .15s;
      aspect-ratio: 1;
      display: flex;
      flex-direction: column;
    }
    .ds-cell:hover {
      border-color: var(--accent);
      transform: scale(1.03);
      z-index: 2;
    }
    .ds-cell img {
      width: 100%;
      flex: 1;
      object-fit: cover;
      background: #000;
      display: block;
    }
    .ds-cell-label {
      font-size: 9.5px;
      color: var(--text3);
      font-family: var(--mono);
      padding: 3px 4px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      border-top: 1px solid var(--border1);
      background: var(--bg1);
    }
    .ds-lightbox {
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.92);
      display: flex;
      align-items: center;
      justify-content: center;
      z-index: 1000;
    }
    .ds-lightbox.hidden { display: none; }
    .ds-lightbox img {
      max-width: 95vw;
      max-height: 88vh;
      object-fit: contain;
      box-shadow: 0 12px 40px rgba(0,0,0,0.7);
    }
    .ds-lb-close, .ds-lb-prev, .ds-lb-next {
      position: absolute;
      background: rgba(0,0,0,0.55);
      border: 1px solid rgba(255,255,255,0.15);
      color: #fff;
      width: 42px;
      height: 42px;
      border-radius: 50%;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      transition: background .15s;
    }
    .ds-lb-close:hover, .ds-lb-prev:hover, .ds-lb-next:hover {
      background: rgba(204,0,168,0.7);
    }
    .ds-lb-close { top: 16px; right: 16px; }
    .ds-lb-prev  { left: 16px; top: 50%; transform: translateY(-50%); }
    .ds-lb-next  { right: 16px; top: 50%; transform: translateY(-50%); }
    .ds-lb-caption {
      position: absolute;
      bottom: 16px;
      left: 50%;
      transform: translateX(-50%);
      color: #fff;
      background: rgba(0,0,0,0.6);
      padding: 8px 16px;
      border-radius: 6px;
      font-size: 13px;
      font-family: var(--mono);
      max-width: 80vw;
      text-align: center;
    }
  `;
  document.head.appendChild(s);
}
