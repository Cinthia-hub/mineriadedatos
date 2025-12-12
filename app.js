// app.js - UI completo con edición usando el mismo formulario existente (no se crea uno nuevo)
// Modificado para mostrar la etiqueta de polaridad (sentiment) inmediatamente cuando el servidor la devuelve,
// y para importar/mostrar sentiment desde site_data/*.json cuando existe.
const PREFERRED_PATH = "site_data/movies_with_reviews.json";
const FALLBACK_PATH = "site_data/movies.json";
const PDF_PATH = "site_data/model_results.pdf";

let movies = [];
let filtered = [];
const PAGE_SIZE = 20;
let currentPage = 1;

// DOM refs
const grid = document.getElementById("grid");
const searchInput = document.getElementById("search");
const genreFilter = document.getElementById("genre-filter");
const sortSelect = document.getElementById("sort");
const pagination = document.getElementById("pagination");
const modal = document.getElementById("modal");
const modalBody = document.getElementById("modal-body");
const modalClose = document.getElementById("modal-close");
const modalBackdrop = document.getElementById("modal-backdrop");
const downloadPdfBtn = document.getElementById("download-pdf");
const globalNotice = document.getElementById("global-notice");

// Confirm modal refs
const confirmModal = document.getElementById("confirm-modal");
const confirmBackdrop = document.getElementById("confirm-backdrop");
const confirmTitleEl = document.getElementById("confirm-title");
const confirmMessageEl = document.getElementById("confirm-message");
const confirmCancelBtn = document.getElementById("confirm-cancel");
const confirmOkBtn = document.getElementById("confirm-ok");

let userReviews = {}; // estructura: { movieKey: [ {id, author, rating, review, ts, db_rowid?, sentiment?, sentiment_compound?} ] }
let currentEdit = null; // { userKey, reviewId, db_rowid, movie_index } o null

// state for confirm modal callback
let _currentConfirm = null;

// PDF polling
let lastPdfMtime = null;
let pdfPollHandle = null;
const PDF_POLL_INTERVAL_MS = 2000;
const PDF_POLL_TIMEOUT_MS = 2 * 60 * 1000; // 2 minutos

// Local storage key
const STORAGE_KEY = "imdb_viewer_user_reviews_v1";

// ----------------- Helpers: storage / notice / fetch -----------------
function loadUserReviews() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    return JSON.parse(raw);
  } catch (e) {
    console.warn("No se pudo cargar user reviews:", e);
    return {};
  }
}
function saveUserReviews() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(userReviews));
  } catch (e) {
    console.warn("No se pudo guardar user reviews:", e);
  }
}
function makeMovieKey(m, idx) {
  return `movie_${idx}`;
}
function showNotice(message, type = "info", timeout = 3500) {
  if (!globalNotice) return;
  globalNotice.textContent = message;
  globalNotice.classList.remove("hidden");
  globalNotice.style.borderLeft = type === "error" ? `4px solid var(--notice-error)` : (type === "success" ? `4px solid var(--notice-success)` : `4px solid transparent`);
  clearTimeout(globalNotice._hideTimer);
  if (timeout > 0) {
    globalNotice._hideTimer = setTimeout(() => {
      globalNotice.classList.add("hidden");
    }, timeout);
  }
}

// ----------------- Confirm modal functions -----------------
function showConfirm({ title = "Confirmar", message = "¿Estás seguro?", onConfirm = null, onCancel = null }) {
  if (!confirmModal) {
    // Fallback to native confirm if modal unavailable
    if (window.confirm(message)) {
      if (typeof onConfirm === "function") onConfirm();
    } else {
      if (typeof onCancel === "function") onCancel();
    }
    return;
  }
  confirmTitleEl.textContent = title;
  confirmMessageEl.textContent = message;
  confirmModal.classList.remove("hidden");
  confirmModal.setAttribute("aria-hidden", "false");
  _currentConfirm = { onConfirm, onCancel };

  // focus management
  confirmCancelBtn.focus();
}

function hideConfirm() {
  if (!confirmModal) return;
  confirmModal.classList.add("hidden");
  confirmModal.setAttribute("aria-hidden", "true");
  _currentConfirm = null;
}

// wire confirm buttons
confirmCancelBtn.addEventListener("click", () => {
  const cb = _currentConfirm && _currentConfirm.onCancel;
  hideConfirm();
  if (typeof cb === "function") cb();
});
confirmBackdrop && confirmBackdrop.addEventListener("click", () => {
  const cb = _currentConfirm && _currentConfirm.onCancel;
  hideConfirm();
  if (typeof cb === "function") cb();
});
confirmOkBtn.addEventListener("click", () => {
  const cb = _currentConfirm && _currentConfirm.onConfirm;
  hideConfirm();
  if (typeof cb === "function") cb();
});
document.addEventListener("keydown", (e)=>{ if (e.key==="Escape") { if (confirmModal && !confirmModal.classList.contains("hidden")) hideConfirm(); } });

// ----------------- PDF status / polling -----------------
function fetchPdfStatus() {
  return fetch("/api/pdf_status")
    .then(r => r.json())
    .catch(() => ({ exists: false, mtime: null, regenerating: false }));
}

function updatePdfLink(mtime) {
  const url = PDF_PATH + (mtime ? `?v=${encodeURIComponent(mtime)}` : "");
  downloadPdfBtn.href = url;
  lastPdfMtime = mtime;
  if (mtime) {
    downloadPdfBtn.setAttribute("aria-disabled", "false");
    downloadPdfBtn.title = "Descargar model_results.pdf";
  } else {
    downloadPdfBtn.setAttribute("aria-disabled", "true");
    downloadPdfBtn.title = "No se encontró model_results.pdf. Genera el PDF corriendo el script y coloca site_data/model_results.pdf";
  }
}

function showPdfGeneratingState(isGenerating) {
  if (isGenerating) {
    downloadPdfBtn.setAttribute("aria-disabled", "true");
    downloadPdfBtn.dataset.generating = "1";
    downloadPdfBtn._oldText = downloadPdfBtn.textContent;
    downloadPdfBtn.textContent = "Generando PDF...";
    downloadPdfBtn.title = "El PDF se está regenerando. Se habilitará cuando termine.";
  } else {
    if (downloadPdfBtn.dataset.generating) {
      delete downloadPdfBtn.dataset.generating;
      downloadPdfBtn.textContent = downloadPdfBtn._oldText || "Descargar grafos";
      downloadPdfBtn._oldText = null;
    }
  }
}

function startPdfPolling(prevMtime) {
  if (pdfPollHandle) return;
  showPdfGeneratingState(true);
  const startAt = Date.now();
  pdfPollHandle = setInterval(() => {
    fetchPdfStatus().then(js => {
      if (js.exists && js.mtime && js.mtime !== prevMtime) {
        clearInterval(pdfPollHandle);
        pdfPollHandle = null;
        updatePdfLink(js.mtime);
        showPdfGeneratingState(false);
        showNotice("PDF actualizado ✓", "success", 2000);
        return;
      }
      if (!js.regenerating) {
        if (js.exists && js.mtime) {
          clearInterval(pdfPollHandle);
          pdfPollHandle = null;
          updatePdfLink(js.mtime);
          showPdfGeneratingState(false);
          showNotice("PDF listo", "success", 1800);
          return;
        }
      }
      if (Date.now() - startAt > PDF_POLL_TIMEOUT_MS) {
        clearInterval(pdfPollHandle);
        pdfPollHandle = null;
        showPdfGeneratingState(false);
        showNotice("Timeout esperando regeneración de PDF.", "error", 3000);
      }
    }).catch(err => {
      console.warn("Error consultando /api/pdf_status:", err);
      if (Date.now() - startAt > PDF_POLL_TIMEOUT_MS) {
        clearInterval(pdfPollHandle);
        pdfPollHandle = null;
        showPdfGeneratingState(false);
      }
    });
  }, PDF_POLL_INTERVAL_MS);
}

function checkPdfAvailability() {
  fetchPdfStatus().then(js => {
    if (js.exists) updatePdfLink(js.mtime); else updatePdfLink(null);
    if (js.regenerating) {
      showPdfGeneratingState(true);
      startPdfPolling(lastPdfMtime);
    } else {
      showPdfGeneratingState(false);
    }
  }).catch(() => { updatePdfLink(null); showPdfGeneratingState(false); });
}

downloadPdfBtn.addEventListener("click", (ev) => {
  if (downloadPdfBtn.getAttribute("aria-disabled") === "true") {
    ev.preventDefault();
    showNotice("No se encontró site_data/model_results.pdf en el servidor. Genera el PDF y coloca site_data/model_results.pdf", "error", 4000);
  }
});

// ----------------- Fetch data & initial load -----------------
function fetchData() {
  return fetch(PREFERRED_PATH)
    .then(res => { if (!res.ok) throw new Error("no file"); return res.json(); })
    .catch(() => fetch(FALLBACK_PATH).then(r => { if (!r.ok) throw new Error("No se pudo cargar movies.json"); return r.json(); }));
}

fetchData().then(d => {
  movies = (d || []).map((m,i)=>{ m._id = i; return m; });
  userReviews = loadUserReviews();

  movies.forEach((m,i) => {
    const srvList = m.user_reviews || m.reviews || [];
    if (Array.isArray(srvList) && srvList.length > 0) {
      const key = makeMovieKey(m, i);
      userReviews[key] = userReviews[key] || [];
      srvList.forEach(r => {
        try {
          const id = r.id != null ? Number(r.id) : (r.db_rowid != null ? Number(r.db_rowid) : null);
          const ts = r.posted || r.ts || r.posted_at || new Date().toISOString();
          const item = {
            id: id || Date.now() + Math.floor(Math.random()*1000),
            author: r.author || r.Author || "Anon",
            rating: (r.rating != null ? Number(r.rating) : null),
            review: r.review || r.REVIEW || "",
            ts: ts,
            db_rowid: id || r.db_rowid || null,
            // include sentiment fields if server provided them
            sentiment: r.sentiment || null,
            sentiment_compound: (r.sentiment_compound != null ? Number(r.sentiment_compound) : null)
          };
          if (!userReviews[key].some(x => String(x.id) === String(item.id))) userReviews[key].push(item);
        } catch(e){ console.warn("Error importando reseña:", e); }
      });
    }
  });

  saveUserReviews();
  initFilters();
  applyFilters();
  checkPdfAvailability();
}).catch(err => {
  grid.innerHTML = `<div class="no-results">No se pudo cargar datos: ${err}</div>`;
  checkPdfAvailability();
});

// ----------------- Render / Filters / Helpers -----------------
function parseNumber(v){ if (v===null||v===undefined) return null; const n=parseFloat(v); return Number.isFinite(n)?n:null; }
function computeAverageRating(m) {
  const key = makeMovieKey(m,m._id);
  const vals = [];
  if (m.positive_review && parseNumber(m.positive_review.rating)!=null) vals.push(parseNumber(m.positive_review.rating));
  if (m.negative_review && parseNumber(m.negative_review.rating)!=null) vals.push(parseNumber(m.negative_review.rating));
  const u = userReviews[key]||[];
  for (const r of u) if (parseNumber(r.rating)!=null) vals.push(parseNumber(r.rating));
  if (vals.length===0) return null;
  const sum = vals.reduce((a,b)=>a+b,0);
  return Math.round((sum/vals.length)*10)/10;
}

function initFilters(){
  const genres=new Set();
  movies.forEach(m=>{ if (m.Genre) m.Genre.split(",").map(s=>s.trim()).forEach(g=>genres.add(g)); });
  const sorted=Array.from(genres).sort();
  sorted.forEach(g=>{ const opt=document.createElement("option"); opt.value=g; opt.textContent=g; genreFilter.appendChild(opt); });
}

function applyFilters(){
  const q=searchInput.value.trim().toLowerCase();
  const genre=genreFilter.value;
  filtered = movies.filter(m=>{
    const hay=[m.Series_Title||"", m.Director||"", m.Star1||"", m.Star2||"", m.Star3||"", m.Star4||""].join(" ").toLowerCase();
    if (q && !hay.includes(q)) return false;
    if (genre) { if (!m.Genre) return false; const gs=m.Genre.split(",").map(s=>s.trim().toLowerCase()); if (!gs.includes(genre.toLowerCase())) return false; }
    return true;
  });
  const s = sortSelect.value;
  filtered.sort((a,b)=>{ if (s==="rating_desc") return (b.IMDB_Rating||0)-(a.IMDB_Rating||0); if (s==="rating_asc") return (a.IMDB_Rating||0)-(b.IMDB_Rating||0); if (s==="year_desc") return (b.Released_Year||0)-(a.Released_Year||0); if (s==="year_asc") return (a.Released_Year||0)-(b.Released_Year||0); return 0; });
  currentPage=1; render();
}

function render(){
  grid.innerHTML="";
  if (!filtered.length) { grid.innerHTML = `<div class="no-results">No se encontraron resultados.</div>`; pagination.innerHTML=""; return; }
  const start=(currentPage-1)*PAGE_SIZE;
  const pageItems = filtered.slice(start, start+PAGE_SIZE);
  pageItems.forEach(m=>{
    const card = document.createElement("article"); card.className="card";
    const poster = document.createElement("div"); poster.className="poster";
    if (m.Poster_Link && m.Poster_Link!=="None") {
      const img=document.createElement("img"); img.src=m.Poster_Link; img.alt=m.Series_Title||"Poster"; img.loading="lazy";
      img.addEventListener("error", ()=>{ try{ img.remove(); }catch(e){} poster.classList.add("no-poster"); });
      poster.appendChild(img);
    } else poster.classList.add("no-poster");
    const meta=document.createElement("div"); meta.className="meta";
    const titleRow=document.createElement("div"); titleRow.className="title-row";
    const h3=document.createElement("h3");
    const a=document.createElement("a"); a.className="movie-link"; a.href="#"; a.textContent=`${m.Series_Title||"—"} (${m.Released_Year||"—"})`;
    a.addEventListener("click", ev=>{ ev.preventDefault(); openModalWithMovie(m); });
    h3.appendChild(a);
    const badge=document.createElement("div"); badge.className="badge"; badge.textContent = m.IMDB_Rating ? String(m.IMDB_Rating) : "—";
    titleRow.appendChild(h3); titleRow.appendChild(badge);
    const small=document.createElement("div"); small.className="small"; small.textContent = `${m.Genre||""} • ${m.Runtime? m.Runtime+" min":""}`.trim();
    const overview=document.createElement("div"); overview.className="overview"; overview.textContent = m.Overview || "";
    const stars=document.createElement("div"); stars.className="stars"; stars.textContent = [m.Director, m.Star1, m.Star2, m.Star3, m.Star4].filter(Boolean).join(" • ");
    const footerRow=document.createElement("div"); footerRow.className="footer-row";
    const left=document.createElement("div"); left.textContent = `Ranking: ${computeAverageRating(m) != null ? computeAverageRating(m) : "—"}`;
    const right=document.createElement("div"); right.className="user-reviews-count"; right.textContent = m.Gross ? `$${m.Gross.toLocaleString?.() || m.Gross}` : "";
    const key = makeMovieKey(m, m._id); const ur = userReviews[key]||[];
    if (ur.length>0){ const indicator=document.createElement("span"); indicator.className="user-review-indicator"; indicator.textContent = `${ur.length} reseña(s)`; right.appendChild(indicator); }
    footerRow.appendChild(left); footerRow.appendChild(right);
    meta.appendChild(titleRow); meta.appendChild(small); meta.appendChild(overview); meta.appendChild(stars); meta.appendChild(footerRow);
    card.appendChild(poster); card.appendChild(meta); grid.appendChild(card);
  });
  renderPagination();
}

function renderPagination(){
  const totalPages = Math.ceil(filtered.length / PAGE_SIZE);
  pagination.innerHTML = "";
  if (totalPages <= 1) return;
  const makeBtn = (text, pg)=>{ const b=document.createElement("button"); b.className="page-btn"; b.textContent=text; b.onclick=()=>{ currentPage=pg; render(); }; return b; };
  if (currentPage>1) pagination.appendChild(makeBtn("« Prev", currentPage-1));
  const start = Math.max(1, currentPage-2); const end = Math.min(totalPages, start+4);
  for (let p=start; p<=end; p++){ const btn=makeBtn(p,p); if (p===currentPage){ btn.style.fontWeight="700"; btn.style.borderColor="rgba(255,255,255,0.12)"; } pagination.appendChild(btn); }
  if (currentPage<totalPages) pagination.appendChild(makeBtn("Next »", currentPage+1));
}

// ----------------- Helper: fetch single movie data from server JSON -----------------
function fetchServerMovie(movie_index) {
  // intenta prefer path primero, si falla usar fallback
  return fetch(PREFERRED_PATH)
    .then(res => { if (!res.ok) throw new Error("no file"); return res.json(); })
    .catch(() => fetch(FALLBACK_PATH).then(r => { if (!r.ok) throw new Error("no movies json"); return r.json(); }))
    .then(arr => {
      if (!Array.isArray(arr)) return null;
      if (movie_index < 0 || movie_index >= arr.length) return null;
      return arr[movie_index];
    })
    .catch(err => {
      console.warn("No se pudo leer JSON del servidor:", err);
      return null;
    });
}

// Importa/mezcla reseñas del servidor para una sola película en userReviews
function importServerReviewsForMovie(m) {
  const movie_index = m._id;
  const key = makeMovieKey(m, movie_index);
  return fetchServerMovie(movie_index).then(serverMovie => {
    if (!serverMovie) {
      showNotice("No se encontraron reseñas en el JSON del servidor para esta película.", "error", 2500);
      return false;
    }
    const srvList = serverMovie.user_reviews || serverMovie.reviews || [];
    if (!Array.isArray(srvList) || srvList.length === 0) {
      showNotice("El servidor no tiene reseñas registradas para esta película.", "info", 1800);
      return false;
    }
    userReviews[key] = userReviews[key] || [];
    const existing = userReviews[key];

    // construir índices para deduplicar por id y db_rowid
    const existingById = new Map();
    const existingByDb = new Map();
    existing.forEach(item => {
      if (item.id != null) existingById.set(String(item.id), item);
      if (item.db_rowid != null) existingByDb.set(String(item.db_rowid), item);
    });

    let imported = 0;
    // añadimos/actualizamos desde servidor
    srvList.forEach(r => {
      try {
        const sid = r.id != null ? String(r.id) : (r.db_rowid != null ? String(r.db_rowid) : null);
        const dbid = r.db_rowid != null ? String(r.db_rowid) : (r.id != null ? String(r.id) : null);
        // Normalize fields as in load
        const author = r.author || r.Author || "Anon";
        const rating = (r.rating != null ? Number(r.rating) : null);
        const reviewText = r.review || r.REVIEW || "";
        const posted = r.posted || r.ts || new Date().toISOString();
        // sentiment fields (if present on server JSON)
        const sentiment = r.sentiment || null;
        const sentiment_compound = (r.sentiment_compound != null ? Number(r.sentiment_compound) : null);
        // Prefer match by db_rowid then id
        let found = null;
        if (dbid && existingByDb.has(dbid)) found = existingByDb.get(dbid);
        else if (sid && existingById.has(sid)) found = existingById.get(sid);

        if (found) {
          // actualizar campos del entry local con datos del servidor
          found.author = author;
          found.rating = rating;
          found.review = reviewText;
          found.ts = posted;
          // ensure db_rowid/id kept
          if (dbid) found.db_rowid = dbid;
          if (sid) found.id = sid;
          // update sentiment if available
          if (sentiment) found.sentiment = sentiment;
          if (sentiment_compound != null) found.sentiment_compound = sentiment_compound;
        } else {
          // insertar nuevo al final (o al inicio?)
          const newItem = {
            id: sid || (Date.now() + Math.floor(Math.random()*1000)),
            db_rowid: dbid || (sid || null),
            author,
            rating,
            review: reviewText,
            ts: posted,
            sentiment,
            sentiment_compound
          };
          userReviews[key].push(newItem);
          imported++;
          // update maps
          if (newItem.id != null) existingById.set(String(newItem.id), newItem);
          if (newItem.db_rowid != null) existingByDb.set(String(newItem.db_rowid), newItem);
        }
      } catch (e) {
        console.warn("Error importando reseña servidor:", e);
      }
    });

    if (imported > 0) {
      saveUserReviews();
      showNotice(`Importadas ${imported} reseña(s) desde el servidor.`, "success", 2200);
    } else {
      showNotice("Reseñas del servidor sincronizadas (sin cambios).", "info", 1400);
    }
    return true;
  });
}

// ----------------- Modal / review rendering and single form reuse -----------------
function openModalWithMovie(m){
  modalBody.innerHTML = "";
  currentEdit = null; // reset any editing state when opening modal

  const title1 = document.createElement("h2");
  title1.className = "modal-title";
  title1.id = "modal-title";
  title1.textContent = `${m.Series_Title || "—"} (${m.Released_Year || "—"})`;
  modalBody.appendChild(title1);

  // Overview (dataset)
  if (m.Overview) {
    const ov = document.createElement("div");
    ov.className = "review-block";
    const h = document.createElement("div");
    h.className = "review-header";
    const sub = document.createElement("div");
    sub.textContent = "Resumen";
    sub.style.fontWeight = "700";
    h.appendChild(sub);
    ov.appendChild(h);
    const bd = document.createElement("div");
    bd.className = "review-body";
    bd.textContent = m.Overview;
    ov.appendChild(bd);
    modalBody.appendChild(ov);
  }

  // Positive review from dataset
  const posBlock = document.createElement("div");
  posBlock.className = "review-block";
  const posHeader = document.createElement("div");
  posHeader.className = "review-header";
  const posTitle = document.createElement("div");
  posTitle.textContent = "Reseña positiva (dataset)";
  posTitle.style.fontWeight = "700";
  posHeader.appendChild(posTitle);
  const posTitle2 = document.createElement("div");
  posTitle2.style.display = "flex";
  posHeader.appendChild(posTitle2);
  if (m.positive_review && m.positive_review.rating != null) {
    const pr = document.createElement("div");
    pr.className = "review-rating";
    pr.textContent = String(m.positive_review.rating);
    posTitle2.appendChild(pr);
  }
  // show sentiment for dataset positive_review if present
  if (m.positive_review && m.positive_review.sentiment) {
    const sb = document.createElement("span");
    sb.textContent = String(m.positive_review.sentiment);
    sb.title = (m.positive_review.sentiment_compound != null ? `score: ${m.positive_review.sentiment_compound}` : "");
    sb.style.marginLeft = "8px";
    sb.style.padding = "2px 6px";
    sb.style.borderRadius = "10px";
    sb.style.color = "#fff";
    const s = m.positive_review.sentiment;
    if (s === "positive") sb.style.background = "#2ca02c";
    else if (s === "negative") sb.style.background = "#d62728";
    else sb.style.background = "#7f7f7f";
    posTitle2.appendChild(sb);
  }
  posBlock.appendChild(posHeader);
  if (m.positive_review) {
    const meta = document.createElement("div");
    meta.className = "review-author";
    meta.textContent = `${m.positive_review.title || ""} — ${m.positive_review.author || "Anon"}`;
    posBlock.appendChild(meta);
    const body = document.createElement("div");
    body.className = "review-body";
    body.textContent = m.positive_review.review || "(sin texto)";
    posBlock.appendChild(body);
  } else {
    const body = document.createElement("div");
    body.className = "review-body";
    body.textContent = "(No hay reseña positiva asignada)";
    posBlock.appendChild(body);
  }
  modalBody.appendChild(posBlock);

  // Negative review from dataset
  const negBlock = document.createElement("div");
  negBlock.className = "review-block";
  const negHeader = document.createElement("div");
  negHeader.className = "review-header";
  const negTitle = document.createElement("div");
  negTitle.textContent = "Reseña negativa (dataset)";
  negTitle.style.fontWeight = "700";
  negHeader.appendChild(negTitle);
  const negTitle2 = document.createElement("div");
  negTitle2.style.display = "flex";
  negHeader.appendChild(negTitle2);
  if (m.negative_review && m.negative_review.rating != null) {
    const nr = document.createElement("div");
    nr.className = "review-rating";
    nr.textContent = String(m.negative_review.rating);
    negTitle2.appendChild(nr);
  }
  // show sentiment for dataset negative_review if present
  if (m.negative_review && m.negative_review.sentiment) {
    const sb = document.createElement("span");
    sb.textContent = String(m.negative_review.sentiment);
    sb.title = (m.negative_review.sentiment_compound != null ? `score: ${m.negative_review.sentiment_compound}` : "");
    sb.style.marginLeft = "8px";
    sb.style.padding = "2px 6px";
    sb.style.borderRadius = "10px";
    sb.style.color = "#fff";
    const s = m.negative_review.sentiment;
    if (s === "positive") sb.style.background = "#2ca02c";
    else if (s === "negative") sb.style.background = "#d62728";
    else sb.style.background = "#7f7f7f";
    negTitle2.appendChild(sb);
  }
  negBlock.appendChild(negHeader);
  if (m.negative_review) {
    const meta = document.createElement("div");
    meta.className = "review-author";
    meta.textContent = `${m.negative_review.title || ""} — ${m.negative_review.author || "Anon"}`;
    negBlock.appendChild(meta);
    const body = document.createElement("div");
    body.className = "review-body";
    body.textContent = m.negative_review.review || "(sin texto)";
    negBlock.appendChild(body);
  } else {
    const body = document.createElement("div");
    body.className = "review-body";
    body.textContent = "(No hay reseña negativa asignada)";
    negBlock.appendChild(body);
  }
  modalBody.appendChild(negBlock);

  // Header row with title + reload button
  const headerRow = document.createElement("div");
  headerRow.style.display = "flex";
  headerRow.style.justifyContent = "space-between";
  headerRow.style.alignItems = "center";
  headerRow.style.gap = "12px";

  const reloadBtn = document.createElement("button");
  reloadBtn.className = "btn btn-ghost";
  reloadBtn.style.marginLeft = "8px";
  reloadBtn.textContent = "Recargar reseñas (servidor)";
  reloadBtn.addEventListener("click", () => {
    showNotice("Recargando reseñas desde el servidor...", "info", 1400);
    importServerReviewsForMovie(m).then(ok => {
      // actualizar lista en modal si ya fue renderizada
      const list = modalBody.querySelector("#user-review-list");
      if (list) {
        list.innerHTML = "";
        const key = makeMovieKey(m, m._id);
        const existing = userReviews[key] || [];
        if (existing.length === 0) {
          const p = document.createElement("div"); p.className="review-body"; p.textContent="(No has añadido reseñas para esta película aún)"; list.appendChild(p);
        } else {
          existing.slice().reverse().forEach(r => list.appendChild(renderUserReviewItem(key, r, m._id)));
        }
        render(); // actualizar tarjeta y promedios
      }
    });
  });

  headerRow.appendChild(reloadBtn);
  modalBody.appendChild(headerRow);

  // Import server reviews automatically when opening modal, then render list
  importServerReviewsForMovie(m).finally(() => {
    // user reviews list
    const userKey = makeMovieKey(m, m._id);
    const userSection = document.createElement("div"); userSection.className="review-block";
    const uh = document.createElement("div"); uh.className="review-header";
    const ut = document.createElement("div"); ut.textContent = "Tus reseñas (local + servidor)"; ut.style.fontWeight="700"; uh.appendChild(ut); userSection.appendChild(uh);

    const listContainer = document.createElement("div"); listContainer.id = "user-review-list";
    const existing = (userReviews[userKey] || []);
    if (existing.length === 0){
      const p = document.createElement("div"); p.className="review-body"; p.textContent="(No has añadido reseñas para esta película aún)"; listContainer.appendChild(p);
    } else {
      existing.slice().reverse().forEach(r => { listContainer.appendChild(renderUserReviewItem(userKey, r, m._id)); });
    }
    userSection.appendChild(listContainer);

    // --- REUSE existing form (same DOM nodes each time) ---
    const form = document.createElement("div"); form.className="user-form";

    const nameRow = document.createElement("div"); nameRow.className="row";
    const nameInput = document.createElement("input"); nameInput.type="text"; nameInput.placeholder="Tu nombre (opcional)"; nameInput.id="user-name"; nameRow.appendChild(nameInput);

    const ratingSelect = document.createElement("select"); ratingSelect.id="user-rating";
    for (let i=10;i>=1;i--) { const opt=document.createElement("option"); opt.value = i; opt.textContent = `${i} / 10`; ratingSelect.appendChild(opt); }
    nameRow.appendChild(ratingSelect);
    form.appendChild(nameRow);

    const ta = document.createElement("textarea"); ta.placeholder="Escribe tu reseña aquí..."; ta.id="user-review-text"; form.appendChild(ta);

    const formRow = document.createElement("div"); formRow.className = "row";
    const saveBtn = document.createElement("button"); saveBtn.className = "btn btn-primary"; saveBtn.textContent = "Guardar reseña";
    const clearBtn = document.createElement("button"); clearBtn.className = "btn btn-ghost"; clearBtn.textContent = "Limpiar formulario";

    // Save handler: if currentEdit -> edit flow; else -> add flow
    saveBtn.addEventListener("click", () => {
      const author = nameInput.value.trim() || "Anon";
      const rating = parseInt(ratingSelect.value, 10);
      const reviewText = ta.value.trim();
      if (!reviewText) {
        showNotice("Escribe una reseña antes de guardar.", "error");
        return;
      }

      if (currentEdit) {
        // EDIT mode: use /api/edit_review
        const payload = {
          movie_index: currentEdit.movie_index,
          review_id: currentEdit.reviewId,
          db_rowid: currentEdit.db_rowid || null,
          author,
          rating,
          review: reviewText
        };
        fetch("/api/edit_review", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        })
        .then(res => res.json().catch(() => ({ ok:false })))
        .then(js => {
          if (js && js.ok) {
            // update local storage
            const arr = userReviews[currentEdit.userKey] || [];
            const idx = arr.findIndex(x => String(x.id) === String(currentEdit.reviewId));
            if (idx >= 0) {
              arr[idx].author = author;
              arr[idx].rating = rating;
              arr[idx].review = reviewText;
              arr[idx].ts = new Date().toISOString();
              // NEW: accept sentiment returned by server so UI updates immediately
              if (js.sentiment) arr[idx].sentiment = js.sentiment;
              if (js.sentiment_compound != null) arr[idx].sentiment_compound = Number(js.sentiment_compound);
            }
            userReviews[currentEdit.userKey] = arr;
            saveUserReviews();
            // re-render list and grid
            const list = document.getElementById("user-review-list");
            list.innerHTML = "";
            const existing2 = userReviews[currentEdit.userKey] || [];
            if (existing2.length === 0) {
              const p = document.createElement("div"); p.className="review-body"; p.textContent="(No has añadido reseñas para esta película aún)"; list.appendChild(p);
            } else {
              existing2.slice().reverse().forEach(rr => list.appendChild(renderUserReviewItem(currentEdit.userKey, rr, currentEdit.movie_index)));
            }
            render();
            showNotice("Reseña editada.", "success", 1600);
            if (js.regeneration_started) {
              showNotice("Regenerando PDF en segundo plano...", "info", 2200);
              startPdfPolling(lastPdfMtime);
            }
            // reset form to add mode
            currentEdit = null;
            saveBtn.textContent = "Guardar reseña";
            clearBtn.textContent = "Limpiar formulario";
            nameInput.value = "";
            ta.value = "";
            ratingSelect.value = "10";
          } else {
            // fallback local edit
            const arr = userReviews[currentEdit.userKey] || [];
            const idx = arr.findIndex(x => String(x.id) === String(currentEdit.reviewId));
            if (idx >= 0) {
              arr[idx].author = author;
              arr[idx].rating = rating;
              arr[idx].review = reviewText;
              arr[idx].ts = new Date().toISOString();
              userReviews[currentEdit.userKey] = arr;
              saveUserReviews();
              const list = document.getElementById("user-review-list"); list.innerHTML = "";
              const existing2 = userReviews[currentEdit.userKey] || [];
              if (existing2.length === 0) {
                const p = document.createElement("div"); p.className="review-body"; p.textContent="(No has añadido reseñas para esta película aún)"; list.appendChild(p);
              } else {
                existing2.slice().reverse().forEach(rr => list.appendChild(renderUserReviewItem(currentEdit.userKey, rr, currentEdit.movie_index)));
              }
              render();
              showNotice("Edición aplicada localmente (no se pudo actualizar el servidor).", "error", 3500);
              currentEdit = null;
              saveBtn.textContent = "Guardar reseña";
              clearBtn.textContent = "Limpiar formulario";
              nameInput.value = "";
              ta.value = "";
              ratingSelect.value = "10";
            } else {
              showNotice("No fue posible editar la reseña en el servidor.", "error", 3000);
            }
          }
        })
        .catch(err => {
          console.warn("Error editando reseña:", err);
          // fallback local
          const arr = userReviews[currentEdit.userKey] || [];
          const idx = arr.findIndex(x => String(x.id) === String(currentEdit.reviewId));
          if (idx >= 0) {
            arr[idx].author = author;
            arr[idx].rating = rating;
            arr[idx].review = reviewText;
            arr[idx].ts = new Date().toISOString();
            userReviews[currentEdit.userKey] = arr;
            saveUserReviews();
            const list = document.getElementById("user-review-list"); list.innerHTML = "";
            const existing2 = userReviews[currentEdit.userKey] || [];
            if (existing2.length === 0) {
              const p = document.createElement("div"); p.className="review-body"; p.textContent="(No has añadido reseñas para esta película aún)"; list.appendChild(p);
            } else {
              existing2.slice().reverse().forEach(rr => list.appendChild(renderUserReviewItem(currentEdit.userKey, rr, currentEdit.movie_index)));
            }
            render();
            showNotice("Edición aplicada localmente (error de red).", "error", 3500);
            currentEdit = null;
            saveBtn.textContent = "Guardar reseña";
            clearBtn.textContent = "Limpiar formulario";
            nameInput.value = "";
            ta.value = "";
            ratingSelect.value = "10";
          }
        });
      } else {
        // ADD mode: same as before
        const key = makeMovieKey(m, m._id);
        const provisional = { id: Date.now(), author, rating, review: reviewText, ts: new Date().toISOString() };
        const payload = { title: m.Series_Title || "", author, review: reviewText, rating, movie_index: m._id };
        fetch("/api/add_review", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) })
          .then(res => res.json().catch(() => ({ ok:false })))
          .then(js => {
            if (js && js.ok) {
              const rowid = js.rowid || js.id || null;
              const newRev = { id: rowid || provisional.id, db_rowid: rowid || null, author, rating, review: reviewText, ts: new Date().toISOString() };
              // NEW: if server returned sentiment, add it immediately so badge shows
              if (js.sentiment) newRev.sentiment = js.sentiment;
              if (js.sentiment_compound != null) newRev.sentiment_compound = Number(js.sentiment_compound);

              userReviews[key] = userReviews[key] || [];
              userReviews[key].push(newRev);
              saveUserReviews();
              const list = document.getElementById("user-review-list");
              if (list.children.length === 1 && list.children[0].classList.contains("review-body") && list.children[0].textContent.includes("No has añadido")) list.innerHTML = "";
              list.insertBefore(renderUserReviewItem(key, newRev, m._id), list.firstChild);
              ta.value = ""; nameInput.value = ""; ratingSelect.value = "10";
              render();
              showNotice("Reseña guardada.", "success", 1600);
              if (js.regeneration_started) { startPdfPolling(lastPdfMtime); showNotice("Regenerando PDF en segundo plano...", "info", 2200); }
            } else {
              userReviews[key] = userReviews[key] || []; userReviews[key].push(provisional); saveUserReviews();
              const list = document.getElementById("user-review-list");
              if (list.children.length === 1 && list.children[0].classList.contains("review-body") && list.children[0].textContent.includes("No has añadido")) list.innerHTML = "";
              list.insertBefore(renderUserReviewItem(key, provisional, m._id), list.firstChild);
              ta.value = ""; nameInput.value = ""; ratingSelect.value = "10";
              render();
              showNotice("Guardado localmente (sin conexión al servidor).", "error", 3000);
            }
          })
          .catch(err => {
            console.warn("Error guardando reseña:", err);
            userReviews[key] = userReviews[key] || []; userReviews[key].push(provisional); saveUserReviews();
            const list = document.getElementById("user-review-list");
            if (list.children.length === 1 && list.children[0].classList.contains("review-body") && list.children[0].textContent.includes("No has añadido")) list.innerHTML = "";
            list.insertBefore(renderUserReviewItem(key, provisional, m._id), list.firstChild);
            ta.value = ""; nameInput.value = ""; ratingSelect.value = "10";
            render();
            showNotice("Guardar localmente: no fue posible contactar al servidor.", "error", 3500);
          });
      }
    });

    // Clear / Cancel button
    clearBtn.addEventListener("click", () => {
      if (currentEdit) {
        // cancel edit
        currentEdit = null;
        saveBtn.textContent = "Guardar reseña";
        clearBtn.textContent = "Limpiar formulario";
        nameInput.value = "";
        ta.value = "";
        ratingSelect.value = "10";
        showNotice("Edición cancelada.", "info", 1200);
      } else {
        nameInput.value = "";
        ta.value = "";
        ratingSelect.value = "10";
      }
    });

    formRow.appendChild(saveBtn);
    formRow.appendChild(clearBtn);
    form.appendChild(formRow);

    // Append sections to modal
    modalBody.appendChild(userSection);
    modalBody.appendChild(form);

    modal.classList.remove("hidden"); modal.setAttribute("aria-hidden","false"); modalClose.focus();
  });
}

// renderUserReviewItem: simple edit button that fills the existing form
function renderUserReviewItem(userKey, r, movie_index){
  const block = document.createElement("div"); block.className="review-block user-review-item";
  const header = document.createElement("div"); header.className="review-header";
  const leftTitle = document.createElement("div"); leftTitle.className="review-author";
  try{ leftTitle.textContent = `${r.author} — ${new Date(r.ts).toLocaleString()}`; } catch(e){ leftTitle.textContent = `${r.author} — ${r.ts||""}`; }
  leftTitle.style.fontWeight="600"; header.appendChild(leftTitle);
  const ratingBadge = document.createElement("div"); ratingBadge.className="review-rating"; ratingBadge.textContent = `${r.rating}`;

  // sentiment badge: if available show colored pill
  let sentimentBadge = null;
  if (r.sentiment) {
    sentimentBadge = document.createElement("span");
    sentimentBadge.textContent = String(r.sentiment);
    sentimentBadge.title = (r.sentiment_compound != null ? `score: ${r.sentiment_compound}` : "");
    sentimentBadge.style.marginLeft = "8px";
    sentimentBadge.style.padding = "2px 6px";
    sentimentBadge.style.borderRadius = "10px";
    sentimentBadge.style.color = "#fff";
    sentimentBadge.style.fontSize = "12px";
    const s = r.sentiment;
    if (s === "positive") sentimentBadge.style.background = "#2ca02c";
    else if (s === "negative") sentimentBadge.style.background = "#d62728";
    else sentimentBadge.style.background = "#7f7f7f";
  }

  // Edit button: populate the same form (no new form)
  const editBtn = document.createElement("button"); editBtn.className="user-review-edit"; editBtn.textContent="Editar"; editBtn.style.background="transparent"; editBtn.style.border="0"; editBtn.style.color="var(--muted)"; editBtn.style.cursor="pointer"; editBtn.style.marginRight="8px";
  editBtn.addEventListener("click", () => {
    // find form controls in the modal (they exist because modal is open)
    const nameInput = modalBody.querySelector("#user-name");
    const ratingSelect = modalBody.querySelector("#user-rating");
    const ta = modalBody.querySelector("#user-review-text");
    const saveBtn = modalBody.querySelector(".user-form .btn-primary") || modalBody.querySelector(".user-form button.btn-primary");
    const clearBtn = modalBody.querySelector(".user-form .btn-ghost") || modalBody.querySelector(".user-form button.btn-ghost");
    if (!nameInput || !ratingSelect || !ta || !saveBtn || !clearBtn) {
      showNotice("Formulario no disponible para editar.", "error");
      return;
    }
    // set editing state
    currentEdit = { userKey, reviewId: r.id, db_rowid: r.db_rowid || null, movie_index };
    nameInput.value = r.author || "";
    try { ratingSelect.value = String(Number(r.rating) || 10); } catch(e) { ratingSelect.value = "10"; }
    ta.value = r.review || "";
    saveBtn.textContent = "Guardar cambios";
    clearBtn.textContent = "Cancelar edición";
    nameInput.focus();
    showNotice("Editando reseña. Haz cambios y pulsa 'Guardar cambios' o 'Cancelar edición'.", "info", 2500);
  });

  const delBtn = document.createElement("button"); delBtn.className="user-review-delete"; delBtn.textContent="Eliminar"; delBtn.style.background="transparent"; delBtn.style.border="0"; delBtn.style.color="var(--muted)"; delBtn.style.cursor="pointer"; delBtn.style.marginRight="8px";
  delBtn.addEventListener("click", ()=>{
    // Show custom confirm modal instead of native confirm()
    showConfirm({
      title: "Eliminar reseña",
      message: "¿Eliminar esta reseña? Esta acción no se puede deshacer. Se intentará borrar del servidor/JSON y de la base de datos si corresponde; en caso de fallo se eliminará localmente.",
      onConfirm: () => {
        const arr = userReviews[userKey] || []; const idx = arr.findIndex(x => String(x.id) === String(r.id));
        if (idx >= 0){
          const dbRowId = r.db_rowid || null;
          if (dbRowId){
            fetch("/api/delete_review", { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({ movie_index: movie_index, review_id: dbRowId, db_rowid: dbRowId })})
              .then(res=>res.json().catch(()=>({ok:false})))
              .then(js=>{
                arr.splice(idx,1); userReviews[userKey]=arr; saveUserReviews();
                const list = modalBody.querySelector("#user-review-list");
                if ((userReviews[userKey]||[]).length===0){ list.innerHTML=''; const p=document.createElement("div"); p.className="review-body"; p.textContent="(No has añadido reseñas para esta película aún)"; list.appendChild(p); }
                else { list.removeChild(block); }
                render(); showNotice("Reseña eliminada.", "success", 1800);
                if (js && js.regeneration_started){ startPdfPolling(lastPdfMtime); showNotice("Regenerando PDF...", "info", 2000); }
              }).catch(err=>{ console.warn("Error borrando:", err); arr.splice(idx,1); userReviews[userKey]=arr; saveUserReviews(); const list=modalBody.querySelector("#user-review-list"); if((userReviews[userKey]||[]).length===0){ list.innerHTML=''; const p=document.createElement("div"); p.className="review-body"; p.textContent="(No has añadido reseñas para esta película aún)"; list.appendChild(p);} else list.removeChild(block); render(); showNotice("Eliminada localmente (sin conexión).", "error", 3000); });
          } else {
            arr.splice(idx,1); userReviews[userKey]=arr; saveUserReviews(); const list=modalBody.querySelector("#user-review-list"); if((userReviews[userKey]||[]).length===0){ list.innerHTML=''; const p=document.createElement("div"); p.className="review-body"; p.textContent="(No has añadido reseñas para esta película aún)"; list.appendChild(p);} else list.removeChild(block); render(); showNotice("Reseña eliminada localmente.", "success", 1600);
          }
        }
      },
      onCancel: () => {
        showNotice("Eliminación cancelada.", "info", 1200);
      }
    });
  });

  const rightContainer = document.createElement("div"); rightContainer.style.display="flex"; rightContainer.style.alignItems="center";
  rightContainer.appendChild(editBtn); rightContainer.appendChild(delBtn); rightContainer.appendChild(ratingBadge);
  if (sentimentBadge) rightContainer.appendChild(sentimentBadge);
  header.appendChild(rightContainer); block.appendChild(header);
  const body = document.createElement("div"); body.className="review-body"; body.textContent = r.review; block.appendChild(body);
  return block;
}

// ----------------- Modal events & listeners -----------------
function closeModal(){ modal.classList.add("hidden"); modal.setAttribute("aria-hidden","true"); modalBody.innerHTML=""; currentEdit = null; }
modalClose.addEventListener("click", closeModal);
modalBackdrop.addEventListener("click", closeModal);
document.addEventListener("keydown", (e)=>{ if (e.key==="Escape") closeModal(); });
searchInput.addEventListener("input", ()=>applyFilters());
genreFilter.addEventListener("change", ()=>applyFilters());
sortSelect.addEventListener("change", ()=>applyFilters());