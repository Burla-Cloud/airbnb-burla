"use strict";

const DATA_BASE = "./data";

const SECTION_COLORS = {
  worst_tv_placements: "#E0533A",
  messiest_listings:   "#B97A1E",
  mirror_selfies:      "#7C5CC9",
  plant_maximalists:   "#0E6E5C",
};

const SECTION_LABELS = {
  worst_tv_placements: "Worst TV placements",
  messiest_listings:   "Messiest listings",
  mirror_selfies:      "Mirror moments",
  plant_maximalists:   "Plant maximalists",
};

// Per-section CLIP prompt copy for the modal "How it was scored" line.
const SCORE_PROMPTS = {
  worst_tv_placements: 'CLIP score against "a TV mounted high on the wall above a fireplace"',
  messiest_listings:   'CLIP score against "a messy cluttered room with stuff everywhere"',
  mirror_selfies:      'CLIP score against "a photographer reflected in a mirror taking a photo"',
  plant_maximalists:   'CLIP score against "a room full of houseplants"',
};

// Human-readable hypothesis titles + bucket labels for the correlations grid.
const HYPOTHESIS_META = {
  brightness_quartile: {
    title: "Brighter photos earn more demand",
    buckets: { q1: "Darkest", q2: "Q2", q3: "Q3", q4: "Brightest" },
  },
  cleaning_fee_ratio_bucket: {
    title: "Cleaning-fee ratio vs demand",
    buckets: { single: "All listings" },
  },
  messiness_quartile: {
    title: "Messier photos vs demand",
    buckets: { q1: "Tidiest", q2: "Q2", q3: "Q3", q4: "Messiest" },
  },
  plant_count_bucket: {
    title: "More houseplants vs demand",
    buckets: { "0": "0 plants", "1": "1 plant", "2-3": "2-3 plants", "4+": "4+ plants" },
  },
  tv_too_high: {
    title: "TV mounted high vs demand",
    buckets: { "False": "Normal height", "True": "Mounted too high" },
  },
};

function fmt(n) {
  if (typeof n !== "number" || !isFinite(n)) return "--";
  if (n >= 1_000_000_000) return (n / 1_000_000_000).toFixed(1) + "B";
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  if (n >= 100) return Math.round(n).toString();
  return n.toFixed(1);
}

const ACRONYMS = new Set([
  "DC", "USA", "UK", "MSA", "NYC", "DR", "BC", "QC", "NSW", "VIC", "QLD", "UAE", "EU",
  "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN",
  "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV",
  "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
  "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]);

const PREPOSITIONS = new Set([
  "of", "and", "the", "in", "on", "at", "to",
  "de", "del", "la", "le", "los", "las", "y",
  "do", "da", "dos", "das",
  "von", "van", "der", "den",
  "el", "il", "li",
]);

function titleCase(s) {
  if (!s) return "";
  const words = String(s).split(/[\s\-_]+/).filter(Boolean);
  return words.map((w, i) => {
    const lo = w.toLowerCase();
    const up = w.toUpperCase();
    const isLast = i === words.length - 1;
    if (isLast && ACRONYMS.has(up)) return up;
    if (i > 0 && PREPOSITIONS.has(lo)) return lo;
    if (ACRONYMS.has(up)) return up;
    return lo.charAt(0).toUpperCase() + lo.slice(1);
  }).join(" ");
}

function placeLabel(it) {
  const cleanCity = (it.city || "").trim();
  const cleanCountry = (it.country || "").trim();
  const c = /^nan$/i.test(cleanCity) ? "" : titleCase(cleanCity);
  const co = /^nan$/i.test(cleanCountry) ? "" : titleCase(cleanCountry);
  if (c && co) return `${c}, ${co}`;
  return c || co || "";
}

function cleanReviewText(s) {
  if (!s) return "";
  return String(s)
    .replace(/<br\s*\/?\s*>/gi, " ")
    .replace(/<\/?p>/gi, " ")
    .replace(/&nbsp;/gi, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function escapeHTML(s) {
  return String(s == null ? "" : s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

async function loadJSON(name) {
  try {
    const r = await fetch(`${DATA_BASE}/${name}.json`, { cache: "no-store" });
    if (!r.ok) return null;
    return await r.json();
  } catch (e) {
    return null;
  }
}

// Dedupe an items array by listing_id first, then by image url. Both can yield
// duplicates: same listing surfaces twice if the upstream pipeline left dupes,
// and same hero photo surfaces across multi-listing hosts.
function dedupeItems(items) {
  const out = [];
  const seenListings = new Set();
  const seenImages = new Set();
  for (const it of items || []) {
    const lid = it.listing_id != null ? String(it.listing_id) : "";
    const img = (it.image_url || it.thumbnail_url || "").trim();
    if (lid && seenListings.has(lid)) continue;
    if (img && seenImages.has(img)) continue;
    if (lid) seenListings.add(lid);
    if (img) seenImages.add(img);
    out.push(it);
  }
  return out;
}

function paintStats(stats) {
  if (!stats) return;
  const map = {
    n_listings: fmt(stats.n_listings),
    n_photo_manifest_rows: fmt(stats.n_photo_manifest_rows),
    n_cpu_images: fmt(stats.n_cpu_images),
    n_gpu_images: fmt(stats.n_gpu_images),
    n_reviews: fmt(stats.n_reviews),
    peak_workers: fmt(stats.peak_workers),
    wall_time_hours: stats.wall_time_hours
      ? stats.wall_time_hours.toFixed(1)
      : "--",
    estimated_cost_usd_dollar: stats.estimated_cost_usd
      ? "$" + Math.round(stats.estimated_cost_usd)
      : "--",
  };
  for (const [k, v] of Object.entries(map)) {
    const el = document.querySelector(`[data-stat="${k}"]`);
    if (el) el.textContent = v;
  }
}

function paintGrid(sectionId, payload) {
  const grid = document.querySelector(`[data-section="${sectionId}"]`);
  if (!grid || !payload) return;
  const items = dedupeItems(payload.items || []);
  grid.innerHTML = "";
  for (const it of items) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "item";
    card.setAttribute("aria-label",
      `Open ${placeLabel(it) || "listing"} in lightbox`);
    const thumb = document.createElement("div");
    thumb.className = "thumb";
    const img = it.image_url || it.thumbnail_url;
    if (img) thumb.style.backgroundImage = `url("${img}")`;
    const overlay = document.createElement("span");
    overlay.className = "thumb__overlay";
    overlay.innerHTML =
      `<svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <polyline points="15 3 21 3 21 9"></polyline>
        <polyline points="9 21 3 21 3 15"></polyline>
        <line x1="21" y1="3" x2="14" y2="10"></line>
        <line x1="3" y1="21" x2="10" y2="14"></line>
      </svg>`;
    overlay.setAttribute("aria-label", "Expand photo");
    thumb.appendChild(overlay);
    card.appendChild(thumb);
    const meta = document.createElement("div");
    meta.className = "meta";
    const city = document.createElement("span");
    city.className = "city";
    city.textContent = placeLabel(it) || "Unknown";
    meta.appendChild(city);
    card.appendChild(meta);
    card.addEventListener("click", () => openPhotoModal(it, sectionId));
    grid.appendChild(card);
  }
  // Update the count chip (if the section has one) to match dedup count.
  const head = grid.parentElement && grid.parentElement.querySelector(".section__count");
  if (head) {
    head.textContent = `${items.length} listing${items.length === 1 ? "" : "s"}`;
  }
}

function paintReviews(payload) {
  const root = document.querySelector('[data-section="funniest_reviews"]');
  if (!root || !payload || !payload.items) return;
  root.innerHTML = "";
  for (const it of payload.items) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "card";
    card.setAttribute("aria-label", "Open full review");
    const headline = it.one_line && it.one_line.length
      ? it.one_line
      : (it.category || "Funniest review");
    const cat = (it.category || "").replace(/_/g, " ");
    const place = placeLabel(it);
    const date = it.date
      ? new Date(it.date).toLocaleDateString("en-US", { month: "short", year: "numeric" })
      : "";
    const datePlace = [place, date].filter(Boolean).join(" \u00b7 ");
    const cleanComment = cleanReviewText(it.comment || "");
    const trimmed = cleanComment.slice(0, 320);
    const more = cleanComment.length > 320;
    card.innerHTML = `
      <h3 class="one-line">${escapeHTML(headline)}</h3>
      <p class="quote">${escapeHTML(trimmed)}${more ? "&hellip;" : ""}</p>
      <div class="footer-line">
        <span class="category">${escapeHTML(cat)}</span>
        <span>${escapeHTML(datePlace)}</span>
      </div>
      <span class="card__cta">Read full review &rarr;</span>
    `;
    card.addEventListener("click", () => openReviewModal(it));
    root.appendChild(card);
  }
}

function paintCorrelations(payload) {
  const root = document.getElementById("corr-list");
  if (!root || !payload || !payload.hypotheses) return;
  root.innerHTML = "";
  const sorted = [...payload.hypotheses].sort((a, b) => {
    const av = a.verdict === "accepted" ? 0 : 1;
    const bv = b.verdict === "accepted" ? 0 : 1;
    return av - bv;
  });
  for (const h of sorted) {
    const block = document.createElement("section");
    block.className = "corr" + (h.verdict === "accepted" ? " accepted-corr" : "");
    const buckets = h.buckets || [];
    if (!buckets.length) continue;
    if (buckets.length < 2) continue;
    const meta = HYPOTHESIS_META[h.hypothesis] || {};
    const title = meta.title || titleCase(h.hypothesis);
    const labelFor = (raw) => {
      if (meta.buckets && meta.buckets[String(raw)] != null) return meta.buckets[String(raw)];
      return titleCase(String(raw).replace(/_/g, " "));
    };
    const allLow = Math.min(...buckets.map((b) => b.ci_low));
    const allHigh = Math.max(...buckets.map((b) => b.ci_high));
    const span = Math.max(1e-6, allHigh - allLow);
    const bars = buckets.map((b) => {
      const left = ((b.ci_low - allLow) / span) * 100;
      const width = ((b.ci_high - b.ci_low) / span) * 100;
      const med = ((b.median - allLow) / span) * 100;
      return `
        <div class="corr-bar">
          <span class="label">${escapeHTML(labelFor(b.bucket))}</span>
          <span class="ci-track">
            <span class="ci-fill" style="left:${left}%; width:${Math.max(width, 0.5)}%"></span>
            <span class="ci-median" style="left:${med}%"></span>
          </span>
          <span class="n">n=${fmt(b.n)}</span>
        </div>
      `;
    }).join("");
    const verdictLabel = h.verdict === "accepted" ? "Accepted" : "Rejected";
    block.innerHTML = `
      <h3>
        <span>${escapeHTML(title)}</span>
        <span class="verdict ${h.verdict}">${verdictLabel}</span>
      </h3>
      ${h.reason ? `<p class="reason">${escapeHTML(h.reason)}</p>` : ""}
      <div class="corr-bars">${bars}</div>
    `;
    root.appendChild(block);
  }
}

function paintMap(payload) {
  if (!payload || !payload.points || !payload.points.length) return;
  if (typeof L === "undefined") return;
  const map = L.map("map", {
    worldCopyJump: true,
    scrollWheelZoom: false,
    attributionControl: true,
    zoomControl: true,
  }).setView([30, 10], 2);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}{r}.png", {
    attribution: "&copy; OpenStreetMap &copy; CARTO",
    subdomains: "abcd",
    maxZoom: 18,
  }).addTo(map);

  const layers = {};
  for (const k of Object.keys(SECTION_COLORS)) {
    layers[k] = L.layerGroup().addTo(map);
  }
  const escAttr = (s) => String(s || "").replace(/"/g, "&quot;").replace(/</g, "&lt;");

  for (const p of payload.points) {
    if (typeof p.lat !== "number" || typeof p.lng !== "number") continue;
    if (!SECTION_COLORS[p.type]) continue;
    const color = SECTION_COLORS[p.type];
    const lid = p.listing_id != null ? String(p.listing_id) : "";
    const url = p.listing_url || (lid ? `https://www.airbnb.com/rooms/${lid}` : "");
    const popupHTML = `
      <strong>${SECTION_LABELS[p.type] || p.type}</strong>
      <br/>
      ${url
        ? `<a href="${escAttr(url)}" target="_blank" rel="noopener noreferrer">Open listing #${escAttr(lid)} on Airbnb &nearr;</a>`
        : `listing #${escAttr(lid)}`}
    `;
    L.circleMarker([p.lat, p.lng], {
      radius: 5,
      color: color,
      fillColor: color,
      fillOpacity: 0.85,
      weight: 1,
      opacity: 0.95,
    })
      .bindPopup(popupHTML)
      .addTo(layers[p.type]);
  }

  const legend = document.getElementById("map-legend");
  if (!legend) return;

  const activeFilters = new Set(Object.keys(SECTION_COLORS));
  legend.innerHTML = Object.entries(SECTION_COLORS)
    .map(
      ([k, c]) =>
        `<button type="button" class="legend-chip is-active" data-key="${k}" aria-pressed="true">
          <span class="swatch" style="background:${c}"></span>
          <span class="legend-label">${SECTION_LABELS[k]}</span>
        </button>`
    )
    .join("") + `<button type="button" class="legend-reset" data-action="reset">Reset</button>`;

  const updateChip = (chip, active) => {
    chip.classList.toggle("is-active", active);
    chip.setAttribute("aria-pressed", active ? "true" : "false");
  };

  legend.addEventListener("click", (ev) => {
    const target = ev.target.closest("[data-key], [data-action]");
    if (!target) return;
    if (target.dataset.action === "reset") {
      for (const k of Object.keys(SECTION_COLORS)) {
        if (!activeFilters.has(k)) {
          activeFilters.add(k);
          map.addLayer(layers[k]);
        }
        const chip = legend.querySelector(`[data-key="${k}"]`);
        if (chip) updateChip(chip, true);
      }
      return;
    }
    const key = target.dataset.key;
    if (!key || !layers[key]) return;
    if (activeFilters.has(key)) {
      activeFilters.delete(key);
      map.removeLayer(layers[key]);
      updateChip(target, false);
    } else {
      activeFilters.add(key);
      map.addLayer(layers[key]);
      updateChip(target, true);
    }
    if (activeFilters.size === 0) {
      for (const k of Object.keys(SECTION_COLORS)) {
        activeFilters.add(k);
        map.addLayer(layers[k]);
        const chip = legend.querySelector(`[data-key="${k}"]`);
        if (chip) updateChip(chip, true);
      }
    }
  });
}

function setupNav() {
  const nav = document.getElementById("nav");
  if (!nav) return;
  const onScroll = () => {
    if (window.scrollY > 4) nav.classList.add("is-scrolled");
    else nav.classList.remove("is-scrolled");
  };
  onScroll();
  window.addEventListener("scroll", onScroll, { passive: true });
}

// ============================================================================
// Modal infrastructure
// ============================================================================

let _modalEl = null;
let _restoreFocusEl = null;

function ensureModalEl() {
  if (_modalEl) return _modalEl;
  const m = document.createElement("div");
  m.className = "modal";
  m.setAttribute("aria-hidden", "true");
  m.setAttribute("role", "dialog");
  m.setAttribute("aria-modal", "true");
  m.innerHTML = `
    <div class="modal__backdrop" data-modal-close></div>
    <div class="modal__panel" role="document">
      <button type="button" class="modal__close" data-modal-close aria-label="Close">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">
          <line x1="18" y1="6" x2="6" y2="18"></line>
          <line x1="6" y1="6" x2="18" y2="18"></line>
        </svg>
      </button>
      <div class="modal__body"></div>
    </div>
  `;
  document.body.appendChild(m);
  m.addEventListener("click", (e) => {
    if (e.target.closest("[data-modal-close]")) closeModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && m.classList.contains("is-open")) closeModal();
  });
  _modalEl = m;
  return m;
}

function openModal(html) {
  const m = ensureModalEl();
  m.querySelector(".modal__body").innerHTML = html;
  m.classList.add("is-open");
  m.setAttribute("aria-hidden", "false");
  document.body.classList.add("no-scroll");
  _restoreFocusEl = document.activeElement;
  m.querySelector(".modal__close").focus();
}

function closeModal() {
  if (!_modalEl) return;
  _modalEl.classList.remove("is-open");
  _modalEl.setAttribute("aria-hidden", "true");
  document.body.classList.remove("no-scroll");
  if (_restoreFocusEl && typeof _restoreFocusEl.focus === "function") {
    _restoreFocusEl.focus();
  }
}

function openPhotoModal(it, sectionId) {
  const img = it.image_url || it.thumbnail_url || "";
  const place = placeLabel(it) || "Unknown location";
  const sectionLabel = SECTION_LABELS[sectionId] || "";
  const promptCopy = SCORE_PROMPTS[sectionId] || "";
  const score = (typeof it.score === "number") ? it.score.toFixed(3) : null;
  const listingUrl = it.listing_url || "";
  const html = `
    <figure class="modal__figure">
      <img src="${escapeHTML(img)}" alt="${escapeHTML(place)} photo" />
    </figure>
    <div class="modal__caption">
      <div class="modal__caption-top">
        <div>
          <div class="modal__eyebrow">${escapeHTML(sectionLabel)}</div>
          <h3 class="modal__title">${escapeHTML(place)}</h3>
          ${it.name ? `<p class="modal__sub">${escapeHTML(it.name)}</p>` : ""}
        </div>
        ${listingUrl
          ? `<a class="modal__cta" href="${escapeHTML(listingUrl)}" target="_blank" rel="noopener">
              View on Airbnb
              <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                <polyline points="15 3 21 3 21 9"></polyline>
                <line x1="10" y1="14" x2="21" y2="3"></line>
                <path d="M21 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5"></path>
              </svg>
             </a>`
          : ""}
      </div>
      ${score
        ? `<div class="modal__score">
            <span class="modal__score-num">${escapeHTML(score)}</span>
            <span class="modal__score-text">
              ${escapeHTML(promptCopy)}.
              Higher means a closer cosine match. Top-1% of all photos in the index.
            </span>
          </div>`
        : ""}
    </div>
  `;
  openModal(html);
}

function openReviewModal(it) {
  const headline = it.one_line && it.one_line.length
    ? it.one_line
    : (it.category || "Funniest review");
  const cat = (it.category || "").replace(/_/g, " ");
  const place = placeLabel(it);
  const date = it.date
    ? new Date(it.date).toLocaleDateString("en-US",
        { year: "numeric", month: "long", day: "numeric" })
    : "";
  const datePlace = [place, date].filter(Boolean).join(" \u00b7 ");
  // Prefer the full untruncated comment if the upstream pipeline included one.
  const fullText = cleanReviewText(it.comment_full || it.comment || "");
  const listingUrl = it.listing_url || "";
  const truncated = !it.comment_full && (it.comment || "").length >= 600;
  const html = `
    <div class="review-modal">
      <div class="modal__eyebrow">${escapeHTML(cat || "Funniest reviews")}</div>
      <h3 class="modal__title">${escapeHTML(headline)}</h3>
      <p class="modal__sub">${escapeHTML(datePlace)}</p>
      <div class="review-modal__body">
        <p>${escapeHTML(fullText)}</p>
        ${truncated
          ? `<p class="review-modal__note">Inside Airbnb capped this comment at 600 characters; that is the full text we have.</p>`
          : ""}
      </div>
      ${listingUrl
        ? `<a class="modal__cta" href="${escapeHTML(listingUrl)}" target="_blank" rel="noopener">
            View listing on Airbnb
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <polyline points="15 3 21 3 21 9"></polyline>
              <line x1="10" y1="14" x2="21" y2="3"></line>
              <path d="M21 14v5a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5"></path>
            </svg>
           </a>`
        : ""}
    </div>
  `;
  openModal(html);
}

(async function main() {
  setupNav();
  ensureModalEl();
  const [stats, tv, messy, mirror, plants, reviews, corr, world] =
    await Promise.all([
      loadJSON("homepage_stats"),
      loadJSON("worst_tv_placements"),
      loadJSON("messiest_listings"),
      loadJSON("mirror_selfies"),
      loadJSON("plant_maximalists"),
      loadJSON("funniest_reviews"),
      loadJSON("correlations"),
      loadJSON("world_map"),
    ]);

  paintStats(stats);
  paintGrid("worst_tv_placements", tv);
  paintGrid("messiest_listings", messy);
  paintGrid("mirror_selfies", mirror);
  paintGrid("plant_maximalists", plants);
  paintReviews(reviews);
  paintCorrelations(corr);
  paintMap(world);
})();
