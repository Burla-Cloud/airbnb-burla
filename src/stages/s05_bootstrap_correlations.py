"""Stage 5: bootstrap median CIs for 5 hypotheses against demand_proxy.

Joins the listing-level features from listings + images_cpu + images_gpu on
shared FS, derives 5 bucketed hypothesis variables, and runs a 1000-resample
bootstrap to compute the median demand_proxy and a 95% CI for each bucket.

A finding is rejected if any bucket has n < ``MIN_BUCKET_N`` or if the per-bucket
CIs all overlap. The full per-bucket table goes to the manifest; the small
summary parquet is downloaded to ``data/interim/correlations.parquet``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from dotenv import load_dotenv

# Hoist for cloudpickle bundling on Burla workers. The worker calls
# pd.read_parquet which needs pyarrow + pandas available at unpickle time.
import os
import os as _os
import traceback as _tb

import numpy as np
import pandas as pd
import pyarrow as _pa  # noqa: F401
import pyarrow.parquet as _pq  # noqa: F401

from ..config import (
    BOOTSTRAP_RESAMPLES, CORRELATIONS_PATH, HYPOTHESES, MIN_BUCKET_N, SHARED_ROOT,
)
from ..lib.budget import BudgetTracker
from ..lib.io import ensure_dir, register_src_for_burla, write_json

@dataclass
class CorrelateArgs:
    listings_path: str
    images_cpu_path: str
    images_gpu_path: str
    wtf_haiku_path: str
    output_path: str
    hypotheses: list
    bootstrap_resamples: int
    min_bucket_n: int


def correlate_all(args: CorrelateArgs) -> dict:
    """Run on Burla. Read shared parquets, compute per-listing features,
    bootstrap CIs per hypothesis, write a small parquet to shared FS, and
    return a JSON-able summary (which is small, ~50 buckets x ~10 cols)."""
    out = {"ok": False, "n_listings": 0, "n_listings_after_join": 0,
           "rejected": [], "accepted": [], "rows": [],
           "output_path": args.output_path, "error": None}
    try:

        # Prefer listings_demand.parquet (has occupancy_365) but fall back if
        # Stage 7 has not run yet.
        listings_path = args.listings_path
        if not _os.path.exists(listings_path):
            fallback = listings_path.replace("listings_demand.parquet",
                                             "listings_clean.parquet")
            if _os.path.exists(fallback):
                listings_path = fallback
        listings_cols = ["listing_id", "city", "country", "region", "snapshot_date",
                         "demand_proxy", "price_usd",
                         "latitude", "longitude"]
        try:
            listings = pd.read_parquet(
                listings_path,
                columns=listings_cols + ["occupancy_365"],
            )
        except Exception:
            listings = pd.read_parquet(listings_path, columns=listings_cols)
        out["n_listings"] = int(len(listings))
        # If occupancy_365 is present we use it as the primary demand proxy;
        # fallback to reviews_per_month-derived demand_proxy where it is null.
        if "occupancy_365" in listings.columns:
            listings["demand_proxy"] = pd.to_numeric(
                listings["occupancy_365"], errors="coerce"
            ).fillna(pd.to_numeric(listings["demand_proxy"], errors="coerce"))
        listings = listings.dropna(subset=["demand_proxy"]).reset_index(drop=True)

        cpu_cols = ["listing_id", "image_idx", "download_ok",
                    "brightness", "edge_density",
                    "clip_messy_room", "clip_lots_of_plants",
                    "clip_tv_above_fireplace",
                    "clip_pet_dog", "clip_pet_cat", "clip_pet_on_furniture",
                    "clip_wtf_absurd_object", "clip_wtf_unsettling_decor",
                    "clip_wtf_unusual_scene", "clip_wtf_does_not_belong"]
        try:
            cpu = pd.read_parquet(args.images_cpu_path, columns=cpu_cols)
        except Exception:
            base = ["listing_id", "image_idx", "download_ok", "brightness",
                    "edge_density", "clip_messy_room", "clip_lots_of_plants",
                    "clip_tv_above_fireplace"]
            cpu = pd.read_parquet(args.images_cpu_path, columns=base)
            for c in cpu_cols:
                if c not in cpu.columns:
                    cpu[c] = 0.0
        cpu = cpu[cpu["download_ok"].astype(bool)]
        cpu["clip_pet_max"] = cpu[["clip_pet_dog", "clip_pet_cat",
                                    "clip_pet_on_furniture"]].max(axis=1)
        cpu["clip_wtf_max"] = cpu[["clip_wtf_absurd_object", "clip_wtf_unsettling_decor",
                                    "clip_wtf_unusual_scene", "clip_wtf_does_not_belong"]].max(axis=1)
        per_listing_cpu = cpu.groupby("listing_id").agg(
            mean_brightness=("brightness", "mean"),
            messy_score_max=("clip_messy_room", "max"),
            plants_score_max=("clip_lots_of_plants", "max"),
            tv_above_score_max=("clip_tv_above_fireplace", "max"),
            pet_score_max=("clip_pet_max", "max"),
            wtf_score_max=("clip_wtf_max", "max"),
        ).reset_index()

        gpu_cols = ["listing_id", "tv_detected", "tv_above_50pct",
                    "potted_plant_count", "pet_detected", "pet_count"]
        try:
            gpu = pd.read_parquet(args.images_gpu_path, columns=gpu_cols)
        except Exception:
            try:
                base = ["listing_id", "tv_detected", "tv_above_50pct",
                        "potted_plant_count"]
                gpu = pd.read_parquet(args.images_gpu_path, columns=base)
                gpu["pet_detected"] = False
                gpu["pet_count"] = 0
            except Exception:
                gpu = pd.DataFrame(columns=gpu_cols)
        if len(gpu):
            per_listing_gpu = gpu.groupby("listing_id").agg(
                tv_detected_any=("tv_detected", "any"),
                tv_too_high=("tv_above_50pct", "any"),
                plant_count_max=("potted_plant_count", "max"),
                yolo_pet_detected=("pet_detected", "any"),
                yolo_pet_count_max=("pet_count", "max"),
            ).reset_index()
        else:
            per_listing_gpu = pd.DataFrame(columns=[
                "listing_id", "tv_detected_any", "tv_too_high",
                "plant_count_max", "yolo_pet_detected", "yolo_pet_count_max",
            ])

        # WTF: any photo of this listing made it through Haiku as is_absurd?
        try:
            wtf = pd.read_parquet(args.wtf_haiku_path,
                                  columns=["listing_id", "is_absurd"])
            wtf_listings = set(
                wtf[wtf["is_absurd"].astype(bool)]["listing_id"].astype(int).tolist()
            )
        except Exception:
            wtf_listings = set()

        df = listings.merge(per_listing_cpu, on="listing_id", how="left")
        df = df.merge(per_listing_gpu, on="listing_id", how="left")
        df["price_usd"] = pd.to_numeric(df["price_usd"], errors="coerce")
        df["plant_count_max"] = df["plant_count_max"].fillna(0).astype(int)
        df["tv_too_high"] = df["tv_too_high"].fillna(False).astype(bool)
        df["tv_detected_any"] = df["tv_detected_any"].fillna(False).astype(bool)
        df["yolo_pet_detected"] = df["yolo_pet_detected"].fillna(False).astype(bool)
        df["pet_score_max"] = df["pet_score_max"].fillna(0.0)
        df["wtf_score_max"] = df["wtf_score_max"].fillna(0.0)
        # has_pet = YOLO confirmed OR strong CLIP pet match (>0.25 cosine sim
        # against ViT-B/32 prompts is a fairly tight threshold).
        df["has_pet"] = (df["yolo_pet_detected"] |
                         (df["pet_score_max"] >= 0.25)).astype(bool)
        df["is_wtf"] = df["listing_id"].isin(wtf_listings)

        for col, q in [("mean_brightness", "brightness_quartile"),
                       ("messy_score_max", "messiness_quartile")]:
            mask = df[col].notna()
            try:
                df.loc[mask, q] = pd.qcut(
                    df.loc[mask, col], q=4,
                    labels=["q1", "q2", "q3", "q4"], duplicates="drop",
                ).astype(str)
            except ValueError:
                df.loc[mask, q] = "single"

        df["plant_count_bucket"] = pd.cut(
            df["plant_count_max"], bins=[-0.5, 0, 1, 3, 1e6],
            labels=["0", "1", "2-3", "4+"],
        ).astype(str)

        out["n_listings_after_join"] = int(len(df))

        rng = np.random.default_rng(42)
        rows = []
        for hyp_var, target in args.hypotheses:
            if hyp_var == "tv_too_high":
                groups = df[df["tv_detected_any"]].groupby("tv_too_high")[target]
            elif hyp_var in ("has_pet", "is_wtf"):
                groups = df.groupby(hyp_var)[target]
            else:
                groups = df.groupby(hyp_var)[target]

            buckets = []
            for name, g in groups:
                vals = g.dropna().to_numpy()
                n = int(len(vals))
                if n == 0:
                    continue
                med = float(np.median(vals))
                if n >= 5:
                    samples = rng.choice(vals, size=(int(args.bootstrap_resamples), n), replace=True)
                    medians = np.median(samples, axis=1)
                    lo = float(np.percentile(medians, 2.5))
                    hi = float(np.percentile(medians, 97.5))
                else:
                    lo = hi = med
                buckets.append({
                    "hypothesis": hyp_var,
                    "target": target,
                    "bucket": str(name),
                    "n": n,
                    "median": med,
                    "ci_low": lo,
                    "ci_high": hi,
                })

            small_n = [b for b in buckets if b["n"] < int(args.min_bucket_n)]
            overlapping = False
            for i in range(len(buckets)):
                for j in range(i + 1, len(buckets)):
                    a, b = buckets[i], buckets[j]
                    if not (a["ci_high"] < b["ci_low"] or b["ci_high"] < a["ci_low"]):
                        overlapping = True
                        break
                if overlapping:
                    break

            verdict = "accepted"
            reason = ""
            if small_n:
                verdict = "rejected"
                reason = f"buckets with n<{int(args.min_bucket_n)}: " + ",".join(
                    str(b["bucket"]) for b in small_n
                )
            elif overlapping or len(buckets) < 2:
                verdict = "rejected"
                reason = "overlapping CIs across all bucket pairs" if overlapping else "single bucket"

            for b in buckets:
                b["verdict"] = verdict
                b["reason"] = reason
                rows.append(b)
            (out["accepted"] if verdict == "accepted" else out["rejected"]).append({
                "hypothesis": hyp_var,
                "n_buckets": len(buckets),
                "reason": reason,
            })

        result_df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        result_df.to_parquet(args.output_path, compression="zstd", index=False)
        out["rows"] = result_df.to_dict("records")
        out["ok"] = True
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["traceback"] = _tb.format_exc()[:1000]
    return out


def main() -> None:
    load_dotenv()
    register_src_for_burla()
    from burla import remote_parallel_map

    # Prefer listings_demand.parquet (has occupancy_365); the worker falls
    # back to listings_clean.parquet automatically if Stage 7 has not run.
    listings_shared = f"{SHARED_ROOT}/listings_demand.parquet"
    cpu_shared = f"{SHARED_ROOT}/images_cpu.parquet"
    gpu_shared = f"{SHARED_ROOT}/images_gpu.parquet"
    wtf_shared = f"{SHARED_ROOT}/wtf_haiku.parquet"
    correlations_shared = f"{SHARED_ROOT}/correlations.parquet"

    print("[s05] computing correlations on shared FS ...", flush=True)
    t0 = time.time()
    with BudgetTracker("s05_correlate", n_inputs=1, func_cpu=16) as bt:
        bt.set_workers(1)
        [r] = remote_parallel_map(
            correlate_all,
            [CorrelateArgs(
                listings_path=listings_shared,
                images_cpu_path=cpu_shared,
                images_gpu_path=gpu_shared,
                wtf_haiku_path=wtf_shared,
                output_path=correlations_shared,
                hypotheses=list(HYPOTHESES),
                bootstrap_resamples=BOOTSTRAP_RESAMPLES,
                min_bucket_n=MIN_BUCKET_N,
            )],
            func_cpu=16, func_ram=64, max_parallelism=1, grow=True, spinner=False,
        )
        bt.set_succeeded(1 if r.get("ok") else 0)
        bt.set_failed(0 if r.get("ok") else 1)

    if not r.get("ok"):
        raise SystemExit(f"[s05] failed: {r.get('error')}")

    elapsed = time.time() - t0
    print(f"[s05]   {len(r['rows'])} bucket rows in {elapsed:.1f}s "
          f"({len(r['accepted'])} hypotheses accepted, {len(r['rejected'])} rejected)", flush=True)
    for a in r["accepted"]:
        print(f"[s05]   ACCEPTED {a['hypothesis']} ({a['n_buckets']} buckets)", flush=True)
    for rej in r["rejected"]:
        print(f"[s05]   REJECTED {rej['hypothesis']}: {rej['reason']}", flush=True)

    ensure_dir(CORRELATIONS_PATH.parent)
    pd.DataFrame(r["rows"]).to_parquet(CORRELATIONS_PATH, compression="zstd", index=False)
    manifest_path = CORRELATIONS_PATH.with_suffix(".manifest.json")
    write_json(manifest_path, {
        "ok": True,
        "shared_path": correlations_shared,
        "n_listings_after_join": r["n_listings_after_join"],
        "accepted": r["accepted"],
        "rejected": r["rejected"],
        "elapsed_seconds": elapsed,
        "completed_at": time.time(),
    })
    print(f"[s05] DONE. Manifest at {manifest_path}, parquet at {CORRELATIONS_PATH}", flush=True)


if __name__ == "__main__":
    main()
