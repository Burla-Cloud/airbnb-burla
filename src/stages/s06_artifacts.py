"""Stage 6: build all data/outputs/*.json and viral_summary.md from shared parquets.

Runs a single Burla worker that reads listings + images_cpu + images_gpu +
reviews_scored from /workspace/shared, computes per-section top-K, and returns
everything as small JSON-able dicts. The local stage writes all
``data/outputs/*.json``, generates ``viral_summary.md`` from those JSON files,
and merges the runtime log.
"""
from __future__ import annotations

import json
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
    wtf_haiku_path: str
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

        cpu_cols = ["listing_id", "image_idx", "image_url", "download_ok",
                    "brightness", "edge_density",
                    "clip_messy_room", "clip_tv_above_fireplace",
                    "clip_lots_of_plants",
                    "clip_pet_dog", "clip_pet_cat", "clip_pet_on_furniture"]
        try:
            cpu = pd.read_parquet(args.images_cpu_path, columns=cpu_cols)
        except Exception:
            # If the CPU parquet predates the pet prompts, drop them and retry.
            fallback = [c for c in cpu_cols if not c.startswith("clip_pet_")]
            cpu = pd.read_parquet(args.images_cpu_path, columns=fallback)
            for c in ("clip_pet_dog", "clip_pet_cat", "clip_pet_on_furniture"):
                cpu[c] = 0.0
        cpu = cpu[cpu["download_ok"].astype(bool)]
        try:
            gpu = pd.read_parquet(
                args.images_gpu_path,
                columns=["listing_id", "image_idx", "image_url",
                         "tv_detected", "tv_above_50pct", "tv_bbox",
                         "potted_plant_count",
                         "cat_detected", "dog_detected", "pet_detected", "pet_count"],
            )
        except Exception:
            try:
                gpu = pd.read_parquet(
                    args.images_gpu_path,
                    columns=["listing_id", "image_idx", "image_url",
                             "tv_detected", "tv_above_50pct", "tv_bbox",
                             "potted_plant_count"],
                )
                for c in ("cat_detected", "dog_detected", "pet_detected"):
                    gpu[c] = False
                gpu["pet_count"] = 0
            except Exception:
                gpu = pd.DataFrame(columns=[
                    "listing_id", "image_idx", "image_url",
                    "tv_detected", "tv_above_50pct", "tv_bbox", "potted_plant_count",
                    "cat_detected", "dog_detected", "pet_detected", "pet_count",
                ])

        out["stats"]["n_listings"] = int(len(listings))
        out["stats"]["n_listings_with_demand"] = int(listings["demand_proxy"].notna().sum())
        out["stats"]["n_cpu_images"] = int(len(cpu))
        out["stats"]["n_gpu_images"] = int(len(gpu))

        try:
            out["stats"]["n_photo_manifest_rows"] = int(
                pq.read_metadata(args.photo_manifest_path).num_rows
            )
        except Exception:
            out["stats"]["n_photo_manifest_rows"] = 0

        try:
            reviews_raw_path = args.reviews_scored_path.rsplit("/", 1)[0] + "/reviews_raw.parquet"
            out["stats"]["n_reviews"] = int(
                pq.read_metadata(reviews_raw_path).num_rows
            )
        except Exception:
            out["stats"]["n_reviews"] = 0

        listings_idx = listings.set_index("listing_id")

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

        # Worst TV placements: prefer the Haiku-validated tv_validated.parquet
        # (only rows where placement is "above_fireplace" or "unusually_high").
        # Fall back to the legacy YOLO+CLIP join if the validated parquet is missing.
        tv_section = None
        try:
            if _os.path.exists(args.tv_validated_path):
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
                    tv_section = {
                        "title": "Worst TV placements across every public Airbnb",
                        "subtitle": "CLIP shortlisted candidates, Haiku Vision said yes this is mounted absurdly",
                        "n": int(min(len(tv_j), args.top_k["worst_tv_placements"])),
                        "items": _serialize(tv_j, "tv_score",
                                            args.top_k["worst_tv_placements"],
                                            extra=["tv_placement", "one_line",
                                                   "haiku_score"]),
                    }
        except Exception as e:
            tv_section = {
                "title": "Worst TV placements",
                "n": 0, "items": [], "error": str(e)[:200],
            }
        if tv_section is None:
            worst_tv_src = gpu[gpu["tv_above_50pct"].fillna(False).astype(bool)].copy()
            if len(worst_tv_src):
                cpu_score = cpu[["listing_id", "image_idx", "clip_tv_above_fireplace"]]
                worst_tv_src = worst_tv_src.merge(
                    cpu_score, on=["listing_id", "image_idx"], how="left"
                )
                worst_tv_src["clip_tv_above_fireplace"] = worst_tv_src[
                    "clip_tv_above_fireplace"
                ].fillna(0)
                worst_tv_src = worst_tv_src.sort_values(
                    "clip_tv_above_fireplace", ascending=False
                )
                worst_tv_src = worst_tv_src.drop_duplicates(
                    subset=["listing_id"], keep="first"
                )
                j = _attach_listing(worst_tv_src, "clip_tv_above_fireplace")
                if "image_url" in j.columns:
                    j = j.drop_duplicates(subset=["image_url"], keep="first")
                tv_section = {
                    "title": "Worst TV placements in 1.1M Airbnb listings",
                    "n": int(min(len(j), args.top_k["worst_tv_placements"])),
                    "items": _serialize(j, "clip_tv_above_fireplace",
                                        args.top_k["worst_tv_placements"],
                                        extra=["tv_bbox"]),
                }
            else:
                tv_section = {
                    "title": "Worst TV placements", "n": 0, "items": []
                }
        out["sections"]["worst_tv_placements"] = tv_section

        messy = cpu.sort_values("clip_messy_room", ascending=False)
        messy = messy.drop_duplicates(subset=["listing_id"], keep="first")
        j = _attach_listing(messy, "clip_messy_room")
        if "image_url" in j.columns:
            j = j.drop_duplicates(subset=["image_url"], keep="first")
        out["sections"]["messiest_listings"] = {
            "title": "Messiest Airbnb photos in 1.1M listings",
            "n": int(min(len(j), args.top_k["messiest_listings"])),
            "items": _serialize(j, "clip_messy_room", args.top_k["messiest_listings"]),
        }

        plant_src = cpu.sort_values("clip_lots_of_plants", ascending=False)
        plant_src = plant_src.drop_duplicates(subset=["listing_id"], keep="first")
        if len(gpu):
            plant_counts = gpu.groupby("listing_id")["potted_plant_count"].max().reset_index()
            plant_src = plant_src.merge(plant_counts, on="listing_id", how="left")
        j = _attach_listing(plant_src, "clip_lots_of_plants")
        if "image_url" in j.columns:
            j = j.drop_duplicates(subset=["image_url"], keep="first")
        out["sections"]["plant_maximalists"] = {
            "title": "The most plant-maximalist Airbnbs",
            "n": int(min(len(j), args.top_k["plant_maximalists"])),
            "items": _serialize(j, "clip_lots_of_plants", args.top_k["plant_maximalists"],
                                extra=["potted_plant_count"]),
        }

        # Pets in photos: read pets_validated.parquet (Haiku Vision said YES,
        # this is a real animal). Rank by haiku_score and cap at top_k. Fall back
        # to the old CLIP+YOLO heuristic only if the validated parquet is missing
        # so the pipeline still produces a section during partial runs.
        pets_section = None
        try:
            if _os.path.exists(args.pets_validated_path):
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
                    pets_section = {
                        "title": "Cats and dogs Claude said are actually real",
                        "subtitle": "CLIP found candidates, Haiku Vision said YES this is a real animal",
                        "n": int(min(len(pet_j), args.top_k["pets_in_photos"])),
                        "items": _serialize(pet_j, "pet_score",
                                            args.top_k["pets_in_photos"],
                                            extra=["animal_type", "one_line",
                                                   "haiku_score"]),
                    }
        except Exception as e:
            pets_section = {
                "title": "Cats and dogs Claude said are actually real",
                "n": 0, "items": [], "error": str(e)[:200],
            }
        if pets_section is None:
            cpu["clip_pet_max"] = cpu[["clip_pet_dog", "clip_pet_cat",
                                        "clip_pet_on_furniture"]].max(axis=1)
            pet_src = cpu.sort_values("clip_pet_max", ascending=False).copy()
            pet_src = pet_src.drop_duplicates(subset=["listing_id"], keep="first")
            if len(gpu):
                pet_yolo = gpu.groupby("listing_id").agg(
                    yolo_pet_detected=("pet_detected", "any"),
                    yolo_pet_count_max=("pet_count", "max"),
                ).reset_index()
                pet_src = pet_src.merge(pet_yolo, on="listing_id", how="left")
            else:
                pet_src["yolo_pet_detected"] = False
                pet_src["yolo_pet_count_max"] = 0
            pet_src["yolo_pet_detected"] = pet_src["yolo_pet_detected"].fillna(False).astype(bool)
            pet_src["pet_score"] = pet_src["clip_pet_max"] + pet_src["yolo_pet_detected"].astype(float) * 0.5
            pet_src = pet_src.sort_values("pet_score", ascending=False)
            pet_j = _attach_listing(pet_src, "pet_score")
            if "image_url" in pet_j.columns:
                pet_j = pet_j.drop_duplicates(subset=["image_url"], keep="first")
            pets_section = {
                "title": "Cats, dogs, and the occasional surprise",
                "n": int(min(len(pet_j), args.top_k["pets_in_photos"])),
                "items": _serialize(pet_j, "pet_score", args.top_k["pets_in_photos"],
                                    extra=["clip_pet_max", "yolo_pet_detected",
                                           "yolo_pet_count_max"]),
            }
        out["sections"]["pets_in_photos"] = pets_section

        room_titles = {
            "ugly_bathroom": ("Ugly bathrooms a host actually photographed",
                              "Haiku Vision said this bathroom is genuinely grimy or sad"),
            "hectic_kitchen": ("The most hectic kitchens",
                               "Haiku Vision said this kitchen is genuinely chaotic"),
            "drug_den_vibes": ("Listings with drug-den vibes",
                               "Haiku Vision said this room gives unmistakable did-someone-just-leave energy"),
        }
        room_section_keys = {
            "ugly_bathroom": "ugly_bathrooms",
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

        # WTF clusters: read the wtf_haiku.parquet from Stage 5b. Each row is
        # a Haiku-confirmed absurd photo with cluster + caption.
        try:
            wtf = pd.read_parquet(args.wtf_haiku_path)
            if len(wtf):
                wtf = wtf.merge(listings, on="listing_id", how="left")
                wtf["clip_max"] = pd.to_numeric(wtf.get("clip_max"), errors="coerce")
                wtf["haiku_score"] = pd.to_numeric(wtf.get("haiku_score"), errors="coerce")
                clusters = []
                for cname, sub in wtf.sort_values("haiku_score", ascending=False).groupby("kept_cluster"):
                    items = []
                    for _, r in sub.head(args.top_k["wtf_photos_per_cluster"]).iterrows():
                        items.append({
                            "listing_id": int(r.get("listing_id", 0)),
                            "city": str(r.get("city", "")),
                            "country": str(r.get("country", "")),
                            "name": str(r.get("name", ""))[:140],
                            "image_url": str(r.get("image_url", "")),
                            "thumbnail_url": str(r.get("picture_url", "")),
                            "listing_url": str(r.get("listing_url", "")),
                            "one_line": str(r.get("one_line", ""))[:160],
                            "haiku_score": float(r["haiku_score"]) if pd.notna(r.get("haiku_score")) else 0.0,
                            "demand_proxy": float(r["demand_proxy"]) if pd.notna(r.get("demand_proxy")) else None,
                            "lat": float(r["latitude"]) if pd.notna(r.get("latitude")) else None,
                            "lng": float(r["longitude"]) if pd.notna(r.get("longitude")) else None,
                        })
                    clusters.append({
                        "cluster": str(cname),
                        "n": len(items),
                        "items": items,
                    })
                clusters.sort(key=lambda c: c["n"], reverse=True)
                out["sections"]["wtf_clusters"] = {
                    "title": "Photos that made us go 'wait, what'",
                    "n_clusters": len(clusters),
                    "n_photos": int(sum(c["n"] for c in clusters)),
                    "clusters": clusters,
                }
            else:
                out["sections"]["wtf_clusters"] = {
                    "title": "Photos that made us go 'wait, what'",
                    "n_clusters": 0, "n_photos": 0, "clusters": [],
                }
        except Exception as e:
            out["sections"]["wtf_clusters"] = {
                "title": "Photos that made us go 'wait, what'",
                "n_clusters": 0, "n_photos": 0, "clusters": [],
                "error": str(e)[:200],
            }

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
            if section_id == "wtf_clusters":
                for cluster in section.get("clusters", []):
                    for item in cluster.get("items", []):
                        if item.get("lat") is not None and item.get("lng") is not None:
                            lid_str = str(item.get("listing_id", 0))
                            listing_url = item.get("listing_url") or f"https://www.airbnb.com/rooms/{lid_str}"
                            world.append({
                                "type": "wtf_clusters",
                                "lat": float(item["lat"]),
                                "lng": float(item["lng"]),
                                "listing_id": lid_str,
                                "listing_url": listing_url,
                            })
                continue
            for item in section.get("items", []):
                if item.get("lat") is not None and item.get("lng") is not None:
                    lid = item.get("listing_id", 0)
                    lid_str = str(lid)
                    listing_url = item.get("listing_url") or f"https://www.airbnb.com/rooms/{lid_str}"
                    world.append({
                        "type": section_id,
                        "lat": float(item["lat"]),
                        "lng": float(item["lng"]),
                        "listing_id": lid_str,
                        "listing_url": listing_url,
                    })
        out["world_map"] = world
        out["ok"] = True
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["traceback"] = _tb.format_exc()[:1000]
    return out


_VIRAL_TEMPLATE = """# Airbnb x Burla -- viral summary

(All numbers below are computed from data/outputs/*.json. Regenerated every run.)

## Headline numbers

- {n_listings:,} Airbnb listings worldwide (Inside Airbnb, latest snapshot per city)
- {n_photo_manifest_rows:,} photo URLs scraped from public listing pages
- {n_cpu_images:,} images CLIP-scored on Burla CPU
- {n_gpu_images:,} images run through YOLOv8 on Burla A100s
- {n_reviews:,} reviews heuristic-scored, top {n_tier3:,} sent through Claude

## What we found

### TVs in places no one should mount a TV

Top-{n_tv} listings where YOLO confirmed a TV in the upper half of the photo
and CLIP rated the image high on "TV mounted above a fireplace."

### Messiest photos a host actually posted

Top-{n_messy} listings, ranked by CLIP score against "a messy cluttered room
with stuff everywhere."

### Plant-maximalist Airbnbs

Top-{n_plants} listings combining CLIP "room full of houseplants" with YOLO
potted plant counts.

### Pets in photos

Top-{n_pets} listings where YOLO confirmed a cat or dog and CLIP scored high
on the pet prompts.

### Photos that made us go "wait, what"

{n_wtf_photos} photos across {n_wtf_clusters} Haiku-named clusters of genuinely
absurd, unsettling, or out-of-place hosting choices. Top clusters: {wtf_top_clusters}.

### The funniest reviews

Top-{n_funny} reviews surfaced by 3-tier funnel (heuristic -> embedding cluster
-> Claude humor score).

## What held up under bootstrap

{accepted_findings}

## What did not survive

{rejected_findings}

## Replication

Repo: airbnb-burla
Runtime: {wall_time_hours:.1f} hours wall time, peak {peak_workers} Burla workers.
"""


def _build_viral_summary(sections: dict, stats: dict, runtime: dict) -> str:
    funny = sections.get("funniest_reviews", {})
    correlations = sections.get("correlations", {}).get("hypotheses", [])
    accepted = [c["hypothesis"] for c in correlations if c.get("verdict") == "accepted"]
    rejected = [c["hypothesis"] for c in correlations if c.get("verdict") == "rejected"]

    wtf_section = sections.get("wtf_clusters", {})
    wtf_clusters = wtf_section.get("clusters", [])
    wtf_top_clusters = ", ".join(c["cluster"] for c in wtf_clusters[:5]) or "(none yet)"

    def _bullet(items):
        return "\n".join(f"- {x}" for x in items) if items else "- (none)"

    return _VIRAL_TEMPLATE.format(
        n_listings=stats.get("n_listings", 0),
        n_photo_manifest_rows=stats.get("n_photo_manifest_rows", 0),
        n_cpu_images=stats.get("n_cpu_images", 0),
        n_gpu_images=stats.get("n_gpu_images", 0),
        n_reviews=stats.get("n_reviews", 0),
        n_tier3=funny.get("n", 0),
        n_tv=sections.get("worst_tv_placements", {}).get("n", 0),
        n_messy=sections.get("messiest_listings", {}).get("n", 0),
        n_plants=sections.get("plant_maximalists", {}).get("n", 0),
        n_pets=sections.get("pets_in_photos", {}).get("n", 0),
        n_wtf_clusters=wtf_section.get("n_clusters", 0),
        n_wtf_photos=wtf_section.get("n_photos", 0),
        wtf_top_clusters=wtf_top_clusters,
        n_funny=funny.get("n", 0),
        accepted_findings=_bullet(accepted),
        rejected_findings=_bullet(rejected),
        wall_time_hours=runtime.get("wall_time_hours", 0.0),
        peak_workers=runtime.get("peak_workers", 0),
    )


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
    wtf_shared = f"{SHARED_ROOT}/wtf_haiku.parquet"
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
                wtf_haiku_path=wtf_shared,
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

    md = _build_viral_summary(sections, stats, runtime_summary)
    (OUTPUT_DIR / "viral_summary.md").write_text(md, encoding="utf-8")

    site_data = REPO_ROOT / "site" / "data"
    site_data.mkdir(parents=True, exist_ok=True)
    for j in OUTPUT_DIR.glob("*.json"):
        (site_data / j.name).write_bytes(j.read_bytes())

    elapsed = time.time() - t0
    print(f"[s06] DONE. {len(sections)} section files written to {OUTPUT_DIR} in {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
