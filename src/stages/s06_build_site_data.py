"""Stage 6: build every data/outputs/*.json the site reads, from shared parquets.

Runs a single Burla worker that reads listings, CLIP scores, Haiku-validated
TVs / pets / rooms, reviews, and bootstrap correlations from
``/workspace/shared``, computes per-section top-K, and returns everything as
small JSON-able dicts. The local stage writes those dicts to
``data/outputs/*.json``, applies ``data/manual_blocklist.json``, mirrors the
result into ``site/data/``, and merges the runtime log.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from dotenv import load_dotenv

# Hoist for cloudpickle bundling on Burla workers.
import numpy as _np  # noqa: F401
import pandas as _pd  # noqa: F401
import pyarrow as _pa  # noqa: F401
import pyarrow.parquet as _pq  # noqa: F401

from ..config import (
    OUTPUT_DIR, OUTPUT_TOP_K, RUNTIME_LOG_PATH, SHARED_ROOT, REPO_ROOT,
)
from ..lib.budget import BudgetTracker
from ..lib.io import ensure_dir, register_src_for_burla, write_json, read_json

import os as _os
import pandas as pd
import pyarrow.parquet as pq
import traceback as _tb

@dataclass
class ArtifactsArgs:
    listings_path: str
    images_cpu_path: str
    images_gpu_path: str
    reviews_scored_path: str
    correlations_path: str
    photo_manifest_path: str
    pets_validated_path: str
    rooms_categories_path: str
    tv_validated_path: str
    top_k: dict


def build_artifacts(args: ArtifactsArgs) -> dict:
    """Run on Burla: read all shared parquets, build small per-section dicts."""
    out = {"ok": False, "sections": {}, "stats": {}, "world_map": [], "error": None}
    try:

        listings_cols = ["listing_id", "city", "country", "region", "snapshot_date",
                         "name", "price_usd", "demand_proxy",
                         "latitude", "longitude", "picture_url",
                         "listing_url", "reviews_per_month"]
        # listings_demand.parquet (when present) carries occupancy_365, the
        # primary demand proxy now. Fall back to listings_clean.parquet if the
        # calendar stage has not run yet.
        listings_path = args.listings_path
        if not _os.path.exists(listings_path):
            fallback = listings_path.replace("listings_demand.parquet",
                                             "listings_clean.parquet")
            if _os.path.exists(fallback):
                listings_path = fallback
        try:
            extra_cols = ["occupancy_365", "occupancy_weekend", "weekend_premium",
                          "lead_time_open", "price_volatility"]
            listings = pd.read_parquet(
                listings_path, columns=listings_cols + extra_cols,
            )
        except Exception:
            listings = pd.read_parquet(listings_path, columns=listings_cols)
        listings["price_usd"] = pd.to_numeric(listings["price_usd"], errors="coerce")
        listings["price"] = listings["price_usd"]
        listings["demand_proxy"] = pd.to_numeric(listings["demand_proxy"], errors="coerce")
        if "occupancy_365" in listings.columns:
            listings["demand_proxy"] = pd.to_numeric(
                listings["occupancy_365"], errors="coerce"
            ).fillna(listings["demand_proxy"])

        # The TV / pets / rooms sections all read pre-validated parquets from
        # s05c, so we don't need the raw CLIP/YOLO columns anymore. We only
        # need row counts for the homepage stats, which we get from parquet
        # metadata to avoid loading 1.7M rows for two counters.
        def _row_count(path: str) -> int:
            try:
                return int(pq.read_metadata(path).num_rows)
            except Exception:
                return 0

        out["stats"]["n_listings"] = int(len(listings))
        out["stats"]["n_listings_with_demand"] = int(listings["demand_proxy"].notna().sum())
        out["stats"]["n_cpu_images"] = _row_count(args.images_cpu_path)
        out["stats"]["n_gpu_images"] = _row_count(args.images_gpu_path)
        out["stats"]["n_photo_manifest_rows"] = _row_count(args.photo_manifest_path)
        reviews_raw_path = (
            args.reviews_scored_path.rsplit("/", 1)[0] + "/reviews_raw.parquet"
        )
        out["stats"]["n_reviews"] = _row_count(reviews_raw_path)

        def _attach_listing(df, score_col):
            j = df.merge(listings, on="listing_id", how="left").dropna(subset=["picture_url"])
            j[score_col] = pd.to_numeric(j[score_col], errors="coerce")
            j = j.dropna(subset=[score_col])
            return j

        def _serialize(j: "pd.DataFrame", score_col: str, k: int, extra=None):
            rows = []
            for _, r in j.head(k).iterrows():
                row = {
                    "listing_id": int(r["listing_id"]),
                    "city": str(r.get("city", "")),
                    "country": str(r.get("country", "")),
                    "name": str(r.get("name", ""))[:140],
                    "score": float(r[score_col]),
                    "image_url": str(r.get("image_url", "")),
                    "thumbnail_url": str(r.get("picture_url", "")),
                    "listing_url": str(r.get("listing_url", "")),
                    "demand_proxy": float(r.get("demand_proxy"))
                        if pd.notna(r.get("demand_proxy")) else None,
                    "lat": float(r["latitude"]) if pd.notna(r["latitude"]) else None,
                    "lng": float(r["longitude"]) if pd.notna(r["longitude"]) else None,
                }
                if extra:
                    for k2 in extra:
                        v = r.get(k2)
                        if pd.notna(v):
                            try:
                                row[k2] = float(v)
                            except (TypeError, ValueError):
                                row[k2] = str(v)
                rows.append(row)
            return rows

        # Worst TV placements: read tv_validated.parquet from s05c (only rows
        # where Haiku Vision confirmed above_fireplace or unusually_high).
        tv_section = {
            "title": "Worst TV placements across every public Airbnb",
            "subtitle": "CLIP shortlisted candidates, Haiku Vision said yes this is mounted absurdly",
            "n": 0, "items": [],
        }
        try:
            tvv = pd.read_parquet(args.tv_validated_path)
            if len(tvv):
                tvv["tv_score"] = pd.to_numeric(
                    tvv.get("haiku_score"), errors="coerce"
                ).fillna(0)
                tvv = tvv.sort_values("tv_score", ascending=False)
                tvv = tvv.drop_duplicates(subset=["listing_id"], keep="first")
                tv_j = _attach_listing(tvv, "tv_score")
                if "image_url" in tv_j.columns:
                    tv_j = tv_j.drop_duplicates(subset=["image_url"], keep="first")
                if "one_line" in tv_j.columns:
                    tv_j["one_line"] = tv_j["one_line"].fillna("").astype(str)
                tv_section["n"] = int(min(len(tv_j), args.top_k["worst_tv_placements"]))
                tv_section["items"] = _serialize(
                    tv_j, "tv_score", args.top_k["worst_tv_placements"],
                    extra=["tv_placement", "one_line", "haiku_score"],
                )
        except Exception as e:
            tv_section["error"] = str(e)[:200]
        out["sections"]["worst_tv_placements"] = tv_section

        # Pets in photos: read pets_validated.parquet from s05c. Each row is
        # a Haiku-confirmed real cat or dog, ranked by haiku_score.
        pets_section = {
            "title": "Cats and dogs Claude said are actually real",
            "subtitle": "CLIP found candidates, Haiku Vision said YES this is a real animal",
            "n": 0, "items": [],
        }
        try:
            petsv = pd.read_parquet(args.pets_validated_path)
            if len(petsv):
                petsv["pet_score"] = pd.to_numeric(
                    petsv.get("haiku_score"), errors="coerce"
                ).fillna(0)
                petsv = petsv.sort_values("pet_score", ascending=False)
                petsv = petsv.drop_duplicates(subset=["listing_id"], keep="first")
                pet_j = _attach_listing(petsv, "pet_score")
                if "image_url" in pet_j.columns:
                    pet_j = pet_j.drop_duplicates(subset=["image_url"], keep="first")
                if "one_line" in pet_j.columns:
                    pet_j["one_line"] = pet_j["one_line"].fillna("").astype(str)
                pets_section["n"] = int(min(len(pet_j), args.top_k["pets_in_photos"]))
                pets_section["items"] = _serialize(
                    pet_j, "pet_score", args.top_k["pets_in_photos"],
                    extra=["animal_type", "one_line", "haiku_score"],
                )
        except Exception as e:
            pets_section["error"] = str(e)[:200]
        out["sections"]["pets_in_photos"] = pets_section

        room_titles = {
            "hectic_kitchen": ("The most hectic kitchens",
                               "Haiku Vision said this kitchen is genuinely chaotic"),
            "drug_den_vibes": ("Listings with drug-den vibes",
                               "Haiku Vision said this room gives unmistakable did-someone-just-leave energy"),
        }
        room_section_keys = {
            "hectic_kitchen": "hectic_kitchens",
            "drug_den_vibes": "drug_den_vibes",
        }
        for cat_key, section_id in room_section_keys.items():
            out["sections"][section_id] = {
                "title": room_titles[cat_key][0],
                "subtitle": room_titles[cat_key][1],
                "n": 0, "items": [],
            }
        try:
            if _os.path.exists(args.rooms_categories_path):
                rooms = pd.read_parquet(args.rooms_categories_path)
                if len(rooms):
                    rooms["haiku_score"] = pd.to_numeric(
                        rooms.get("haiku_score"), errors="coerce"
                    ).fillna(0)
                    for cat_key, section_id in room_section_keys.items():
                        sub = rooms[rooms["category"] == cat_key].copy()
                        if not len(sub):
                            continue
                        sub = sub.sort_values("haiku_score", ascending=False)
                        sub = sub.drop_duplicates(subset=["listing_id"], keep="first")
                        rj = _attach_listing(sub, "haiku_score")
                        if "image_url" in rj.columns:
                            rj = rj.drop_duplicates(subset=["image_url"], keep="first")
                        if "one_line" in rj.columns:
                            rj["one_line"] = rj["one_line"].fillna("").astype(str)
                        top_k = args.top_k.get(section_id, 40)
                        out["sections"][section_id] = {
                            "title": room_titles[cat_key][0],
                            "subtitle": room_titles[cat_key][1],
                            "n": int(min(len(rj), top_k)),
                            "items": _serialize(rj, "haiku_score", top_k,
                                                extra=["one_line", "haiku_score"]),
                        }
        except Exception as e:
            for cat_key, section_id in room_section_keys.items():
                out["sections"][section_id]["error"] = str(e)[:200]

        try:
            rev = pd.read_parquet(args.reviews_scored_path)
            if "claude_humor_score" in rev.columns:
                rev["claude_humor_score"] = pd.to_numeric(
                    rev["claude_humor_score"], errors="coerce"
                ).fillna(0)
                top_reviews = rev.sort_values(
                    "claude_humor_score", ascending=False
                ).head(args.top_k["funniest_reviews"])
            else:
                top_reviews = rev.sort_values("tier1_score", ascending=False).head(
                    args.top_k["funniest_reviews"]
                )
            top_reviews = top_reviews.merge(
                listings[["listing_id", "city", "country", "picture_url", "listing_url"]],
                on="listing_id", how="left",
            )
            review_rows = []
            for _, r in top_reviews.iterrows():
                full_comment = str(r.get("comments", ""))
                review_rows.append({
                    "review_id": int(r.get("review_id", 0)),
                    "listing_id": int(r.get("listing_id", 0)),
                    "city": str(r.get("city", "")),
                    "country": str(r.get("country", "")),
                    "date": str(r.get("date", ""))[:10],
                    "comment": full_comment[:600],
                    "comment_full": full_comment[:8000],
                    "category": str(r.get("claude_category", "") or ""),
                    "humor_score": float(r["claude_humor_score"])
                        if "claude_humor_score" in r and pd.notna(r["claude_humor_score"]) else None,
                    "one_line": str(r.get("claude_one_line", "") or ""),
                    "thumbnail_url": str(r.get("picture_url", "") or ""),
                    "listing_url": str(r.get("listing_url", "") or ""),
                })
            out["sections"]["funniest_reviews"] = {
                "title": "Funniest reviews from 50M",
                "n": len(review_rows), "items": review_rows,
            }
        except Exception as e:
            out["sections"]["funniest_reviews"] = {
                "title": "Funniest reviews from 50M",
                "n": 0, "items": [], "error": str(e)[:200]
            }

        try:
            corr = pd.read_parquet(args.correlations_path)
            grouped = []
            for hyp, sub in corr.groupby("hypothesis"):
                grouped.append({
                    "hypothesis": str(hyp),
                    "verdict": str(sub.iloc[0]["verdict"]),
                    "reason": str(sub.iloc[0]["reason"]),
                    "buckets": [
                        {
                            "bucket": str(b["bucket"]),
                            "n": int(b["n"]),
                            "median": float(b["median"]),
                            "ci_low": float(b["ci_low"]),
                            "ci_high": float(b["ci_high"]),
                        }
                        for _, b in sub.iterrows()
                    ],
                })
            out["sections"]["correlations"] = {
                "title": "5 hypotheses, bootstrapped 95% CIs",
                "hypotheses": grouped,
            }
        except Exception as e:
            out["sections"]["correlations"] = {"title": "Correlations", "hypotheses": [], "error": str(e)[:200]}

        world = []
        for section_id, section in out["sections"].items():
            if section_id == "correlations":
                continue
            for item in section.get("items", []):
                if item.get("lat") is None or item.get("lng") is None:
                    continue
                lid_str = str(item.get("listing_id", 0))
                world.append({
                    "type": section_id,
                    "lat": float(item["lat"]),
                    "lng": float(item["lng"]),
                    "listing_id": lid_str,
                    "listing_url": item.get("listing_url")
                        or f"https://www.airbnb.com/rooms/{lid_str}",
                })
        out["world_map"] = world
        out["ok"] = True
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["traceback"] = _tb.format_exc()[:1000]
    return out


def main() -> None:
    load_dotenv()
    register_src_for_burla()
    from burla import remote_parallel_map

    # Prefer the calendar-enriched listings_demand.parquet (has occupancy_365)
    # if Stage 7 has run. If it is missing, the worker silently falls back to
    # listings_clean.parquet so the pipeline still produces an artifact set.
    listings_shared = f"{SHARED_ROOT}/listings_demand.parquet"
    cpu_shared = f"{SHARED_ROOT}/images_cpu.parquet"
    gpu_shared = f"{SHARED_ROOT}/images_gpu.parquet"
    reviews_shared = f"{SHARED_ROOT}/reviews_scored.parquet"
    corr_shared = f"{SHARED_ROOT}/correlations.parquet"
    photo_manifest_shared = f"{SHARED_ROOT}/photo_manifest.parquet"
    pets_validated_shared = f"{SHARED_ROOT}/pets_validated.parquet"
    rooms_categories_shared = f"{SHARED_ROOT}/room_categories.parquet"
    tv_validated_shared = f"{SHARED_ROOT}/tv_validated.parquet"

    print("[s06] building artifacts on shared FS ...", flush=True)
    t0 = time.time()
    with BudgetTracker("s06_artifacts", n_inputs=1, func_cpu=16) as bt:
        bt.set_workers(1)
        [r] = remote_parallel_map(
            build_artifacts,
            [ArtifactsArgs(
                listings_path=listings_shared,
                images_cpu_path=cpu_shared,
                images_gpu_path=gpu_shared,
                reviews_scored_path=reviews_shared,
                correlations_path=corr_shared,
                photo_manifest_path=photo_manifest_shared,
                pets_validated_path=pets_validated_shared,
                rooms_categories_path=rooms_categories_shared,
                tv_validated_path=tv_validated_shared,
                top_k=dict(OUTPUT_TOP_K),
            )],
            func_cpu=16, func_ram=64, max_parallelism=1, grow=True, spinner=False,
        )
        bt.set_succeeded(1 if r.get("ok") else 0)
        bt.set_failed(0 if r.get("ok") else 1)

    if not r.get("ok"):
        raise SystemExit(f"[s06] failed: {r.get('error')}")

    raw_log = read_json(RUNTIME_LOG_PATH) or {}
    stages = raw_log.get("stages", []) if isinstance(raw_log, dict) else []
    total_wall = sum(stage.get("wall_seconds", 0) for stage in stages) / 3600.0
    total_usd = sum(stage.get("estimated_usd", 0) for stage in stages)
    peak_workers = max((stage.get("n_workers", 0) for stage in stages), default=0)

    runtime_summary = {
        "wall_time_hours": total_wall,
        "estimated_cost_usd": total_usd,
        "peak_workers": peak_workers,
        "stages": stages,
        "completed_at": time.time(),
    }

    ensure_dir(OUTPUT_DIR)
    sections = r["sections"]
    stats = r["stats"]
    # Keep n_reviews from build_artifacts (real review count). funniest_reviews
    # is just the top-K we surfaced.
    stats["n_reviews_funniest_top_k"] = sections.get("funniest_reviews", {}).get("n", 0)

    for section_id, section in sections.items():
        write_json(OUTPUT_DIR / f"{section_id}.json", section)

    write_json(OUTPUT_DIR / "world_map.json", {
        "title": "Every flagged Airbnb in the demo, on a Leaflet map",
        "n": len(r["world_map"]), "points": r["world_map"],
    })
    write_json(OUTPUT_DIR / "homepage_stats.json", {
        "n_listings": stats.get("n_listings", 0),
        "n_photo_manifest_rows": stats.get("n_photo_manifest_rows", 0),
        "n_cpu_images": stats.get("n_cpu_images", 0),
        "n_gpu_images": stats.get("n_gpu_images", 0),
        "n_reviews": stats.get("n_reviews", 0),
        "wall_time_hours": runtime_summary["wall_time_hours"],
        "estimated_cost_usd": runtime_summary["estimated_cost_usd"],
        "peak_workers": runtime_summary["peak_workers"],
    })
    write_json(OUTPUT_DIR / "runtime_log.json", runtime_summary)

    # Apply the human-curated blocklist on top of whatever Haiku surfaced.
    # See data/manual_blocklist.json + scripts/apply_manual_blocklist.py.
    try:
        import subprocess as _sub
        _sub.run(
            ["python", "-m", "scripts.apply_manual_blocklist"],
            check=True,
        )
    except Exception as _exc:  # noqa: BLE001 -- best-effort, don't fail the stage
        print(f"[s06] manual blocklist sync failed (non-fatal): {_exc}", flush=True)

    site_data = REPO_ROOT / "site" / "data"
    site_data.mkdir(parents=True, exist_ok=True)
    for j in OUTPUT_DIR.glob("*.json"):
        (site_data / j.name).write_bytes(j.read_bytes())

    elapsed = time.time() - t0
    print(f"[s06] DONE. {len(sections)} section files written to {OUTPUT_DIR} in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
