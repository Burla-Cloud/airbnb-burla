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
  mirror_selfies:      "Mirror selfies",
  plant_maximalists:   "Plant maximalists",
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
    buckets: { "False": "TV at normal height", "True": "TV mounted too high" },
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
  // Country / region
  "DC", "USA", "UK", "MSA", "NYC", "DR", "BC", "QC", "NSW", "VIC", "QLD", "UAE", "EU",
  // US states (Inside Airbnb often emits city slugs that include state abbreviations)
  "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN",
  "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV",
  "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
  "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]);

function titleCase(s) {
  if (!s) return "";
  return String(s)
    .split(/[\s\-_]+/)
    .filter(Boolean)
    .map((w) => {
      const up = w.toUpperCase();
      if (ACRONYMS.has(up)) return up;
      const lo = w.toLowerCase();
      if (["of", "and", "the", "in", "on", "de", "la", "le", "y", "del"].includes(lo)) return lo;
      return lo.charAt(0).toUpperCase() + lo.slice(1);
    })
    .join(" ");
}

function placeLabel(it) {
  const cleanCity = (it.city || "").trim();
  const cleanCountry = (it.country || "").trim();
  // The pipeline emits literal "nan" strings when source values were missing.
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

async function loadJSON(name) {
  try {
    const r = await fetch(`${DATA_BASE}/${name}.json`, { cache: "no-store" });
    if (!r.ok) return null;
    return await r.json();
  } catch (e) {
    return null;
  }
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
  if (!grid || !payload || !payload.items) return;
  grid.innerHTML = "";
  for (const it of payload.items) {
    const a = document.createElement("a");
    a.className = "item";
    a.href = it.listing_url || "#";
    a.target = "_blank";
    a.rel = "noopener";
    const thumb = document.createElement("div");
    thumb.className = "thumb";
    const img = it.image_url || it.thumbnail_url;
    if (img) {
      thumb.style.backgroundImage = `url("${img}")`;
    }
    a.appendChild(thumb);
    const meta = document.createElement("div");
    meta.className = "meta";
    const city = document.createElement("span");
    city.className = "city";
    city.textContent = placeLabel(it);
    const score = document.createElement("span");
    score.className = "score";
    score.textContent = (it.score || 0).toFixed(2);
    meta.appendChild(city);
    meta.appendChild(score);
    a.appendChild(meta);
    grid.appendChild(a);
  }
}

function paintReviews(payload) {
  const root = document.querySelector('[data-section="funniest_reviews"]');
  if (!root || !payload || !payload.items) return;
  root.innerHTML = "";
  for (const it of payload.items) {
    const card = document.createElement("article");
    card.className = "card";
    const headline = it.one_line && it.one_line.length
      ? it.one_line
      : (it.category || "Funniest review");
    const cat = (it.category || "").replace(/_/g, " ");
    const place = placeLabel(it);
    const date = it.date ? new Date(it.date).toLocaleDateString("en-US", { month: "short", year: "numeric" }) : "";
    const datePlace = [place, date].filter(Boolean).join(" · ");
    const cleanComment = cleanReviewText(it.comment || "");
    const trimmed = cleanComment.slice(0, 460);
    const more = cleanComment.length > 460;
    card.innerHTML = `
      <h3 class="one-line">${escapeHTML(headline)}</h3>
      <p class="quote">${escapeHTML(trimmed)}${more ? "&hellip;" : ""}</p>
      <div class="footer-line">
        <span class="category">${escapeHTML(cat)}</span>
        <span>${escapeHTML(datePlace)}</span>
      </div>
    `;
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
    // Skip degenerate hypotheses where the source field couldn't be split into
    // multiple buckets (e.g. cleaning-fee ratio collapses to a single bucket
    // because the raw price/fee fields are mostly null in Inside Airbnb dumps).
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
    attribution: '&copy; OpenStreetMap &copy; CARTO',
    subdomains: "abcd",
    maxZoom: 18,
  }).addTo(map);
  for (const p of payload.points) {
    if (typeof p.lat !== "number" || typeof p.lng !== "number") continue;
    if (!SECTION_COLORS[p.type]) continue; // skip categories we no longer surface
    const color = SECTION_COLORS[p.type];
    L.circleMarker([p.lat, p.lng], {
      radius: 5,
      color: color,
      fillColor: color,
      fillOpacity: 0.85,
      weight: 1,
      opacity: 0.95,
    })
      .bindPopup(
        `<strong>${SECTION_LABELS[p.type] || p.type}</strong><br/>listing #${p.listing_id}`
      )
      .addTo(map);
  }
  const legend = document.getElementById("map-legend");
  if (legend) {
    legend.innerHTML = Object.entries(SECTION_COLORS)
      .map(
        ([k, c]) =>
          `<span><span class="swatch" style="background:${c}"></span>${SECTION_LABELS[k]}</span>`
      )
      .join("");
  }
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

function escapeHTML(s) {
  return String(s == null ? "" : s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

(async function main() {
  setupNav();
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
