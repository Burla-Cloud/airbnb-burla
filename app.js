"use strict";

const DATA_BASE = "./data";

const SECTION_COLORS = {
  worst_tv_placements: "#ff5a5f",
  messiest_listings: "#f9a826",
  mirror_selfies: "#9c5dff",
  plant_maximalists: "#3ddc97",
  insane_cleaning_fees: "#00a699",
  funniest_reviews: "#5fa8ff",
};

const SECTION_LABELS = {
  worst_tv_placements: "Worst TV placements",
  messiest_listings: "Messiest listings",
  mirror_selfies: "Mirror selfies",
  plant_maximalists: "Plant maximalists",
  insane_cleaning_fees: "Insane cleaning fees",
  funniest_reviews: "Funniest reviews",
};

function fmt(n) {
  if (typeof n !== "number" || !isFinite(n)) return "--";
  if (n >= 1_000_000_000) return (n / 1_000_000_000).toFixed(1) + "B";
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(1) + "M";
  if (n >= 1_000) return (n / 1_000).toFixed(1) + "K";
  if (n >= 100) return Math.round(n).toString();
  return n.toFixed(1);
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
  const items = payload.items;
  for (const it of items) {
    const a = document.createElement("a");
    a.className = "item";
    a.href = it.listing_url || "#";
    a.target = "_blank";
    a.rel = "noopener";
    const thumb = document.createElement("div");
    thumb.className = "thumb";
    if (it.image_url) {
      thumb.style.backgroundImage = `url("${it.image_url}")`;
      thumb.style.backgroundSize = "cover";
      thumb.style.backgroundPosition = "center";
    } else if (it.thumbnail_url) {
      thumb.style.backgroundImage = `url("${it.thumbnail_url}")`;
      thumb.style.backgroundSize = "cover";
      thumb.style.backgroundPosition = "center";
    }
    a.appendChild(thumb);
    const meta = document.createElement("div");
    meta.className = "meta";
    const city = document.createElement("span");
    city.className = "city";
    city.textContent = `${it.city || "--"}${it.country ? ", " + it.country : ""}`;
    const score = document.createElement("span");
    score.className = "score";
    score.textContent = `score ${(it.score || 0).toFixed(2)}`;
    meta.appendChild(city);
    meta.appendChild(score);
    a.appendChild(meta);
    grid.appendChild(a);
  }
}

function paintFeesTable(payload) {
  const tbody = document.querySelector('[data-section="insane_cleaning_fees"]');
  if (!tbody || !payload || !payload.items) return;
  tbody.innerHTML = "";
  for (const it of payload.items.slice(0, 100)) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span class="city">${it.city || "--"}${it.country ? ", " + it.country : ""}</span></td>
      <td>$${Math.round(it.price)}</td>
      <td>$${Math.round(it.cleaning_fee)}</td>
      <td class="ratio">${it.fee_ratio.toFixed(1)}x</td>
      <td><a href="${it.listing_url || "#"}" target="_blank" rel="noopener">listing</a></td>
    `;
    tbody.appendChild(tr);
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
    const cat = it.category || "";
    card.innerHTML = `
      <h3 class="one-line">${escapeHTML(headline)}</h3>
      <p class="quote">"${escapeHTML((it.comment || "").slice(0, 480))}${(it.comment || "").length > 480 ? "..." : ""}"</p>
      <div class="footer-line">
        <span class="category">${escapeHTML(cat)}</span>
        <span>${it.city || ""}${it.country ? ", " + it.country : ""} · ${it.date || ""}</span>
      </div>
    `;
    root.appendChild(card);
  }
}

function paintCorrelations(payload) {
  const root = document.getElementById("corr-list");
  if (!root || !payload || !payload.hypotheses) return;
  root.innerHTML = "";
  for (const h of payload.hypotheses) {
    const block = document.createElement("section");
    block.className = "corr";
    const buckets = h.buckets || [];
    if (!buckets.length) continue;
    const allLow = Math.min(...buckets.map((b) => b.ci_low));
    const allHigh = Math.max(...buckets.map((b) => b.ci_high));
    const span = Math.max(1e-6, allHigh - allLow);
    const bars = buckets.map((b) => {
      const left = ((b.ci_low - allLow) / span) * 100;
      const width = ((b.ci_high - b.ci_low) / span) * 100;
      const med = ((b.median - allLow) / span) * 100;
      return `
        <div class="corr-bar">
          <span class="label">${escapeHTML(b.bucket)}</span>
          <span class="ci-track">
            <span class="ci-fill" style="left:${left}%; width:${Math.max(width, 0.5)}%"></span>
            <span class="ci-median" style="left:${med}%"></span>
          </span>
          <span class="n">n=${fmt(b.n)}</span>
        </div>
      `;
    }).join("");
    block.innerHTML = `
      <h3>
        <span>${escapeHTML(h.hypothesis)}</span>
        <span class="verdict ${h.verdict}">${escapeHTML(h.verdict)}</span>
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
  const map = L.map("map", { worldCopyJump: true, scrollWheelZoom: false }).setView([20, 0], 2);
  L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
    attribution: '&copy; OpenStreetMap &copy; CARTO',
    subdomains: "abcd",
    maxZoom: 18,
  }).addTo(map);
  for (const p of payload.points) {
    if (typeof p.lat !== "number" || typeof p.lng !== "number") continue;
    const color = SECTION_COLORS[p.type] || "#cccccc";
    L.circleMarker([p.lat, p.lng], {
      radius: 4,
      color,
      fillColor: color,
      fillOpacity: 0.7,
      weight: 1,
    })
      .bindPopup(`<b>${SECTION_LABELS[p.type] || p.type}</b><br/>listing #${p.listing_id}`)
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

function escapeHTML(s) {
  return String(s || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

(async function main() {
  const [stats, tv, messy, mirror, plants, fees, reviews, corr, world] =
    await Promise.all([
      loadJSON("homepage_stats"),
      loadJSON("worst_tv_placements"),
      loadJSON("messiest_listings"),
      loadJSON("mirror_selfies"),
      loadJSON("plant_maximalists"),
      loadJSON("insane_cleaning_fees"),
      loadJSON("funniest_reviews"),
      loadJSON("correlations"),
      loadJSON("world_map"),
    ]);

  paintStats(stats);
  paintGrid("worst_tv_placements", tv);
  paintGrid("messiest_listings", messy);
  paintGrid("mirror_selfies", mirror);
  paintGrid("plant_maximalists", plants);
  paintFeesTable(fees);
  paintReviews(reviews);
  paintCorrelations(corr);
  paintMap(world);
})();
