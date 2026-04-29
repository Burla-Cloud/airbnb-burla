"""Top-level Burla worker functions for listings + images.

Burla pickles these functions across workers, so they must be plain
module functions with no closures and accept a single ``@dataclass`` arg.
"""
from __future__ import annotations

import gzip
import io
import os
import random
import time
import traceback
from dataclasses import dataclass, asdict
from typing import Optional

import requests

# Required worker dependencies.
import numpy as _np  # noqa: F401
import pandas as _pd  # noqa: F401
import pyarrow as _pa  # noqa: F401
import pyarrow.parquet as _pq  # noqa: F401
import PIL  # noqa: F401
import PIL.Image  # noqa: F401
import torch as _torch  # noqa: F401
import open_clip as _open_clip  # noqa: F401

# Pull config values at module-import time so cloudpickle bundles them with
# this module when register_src_for_burla() pickles src.* by value. Worker
# functions reference these via the module-level names below, not via fresh
# `from src.config import ...` statements (which would fail on the worker
# because src/ is not on the worker's PYTHONPATH).
from ..config import (
    CLIP_MODEL as _CLIP_MODEL,
    CLIP_PRETRAINED as _CLIP_PRETRAINED,
    CLIP_PROMPTS as _CLIP_PROMPTS,
    YOLO_MODEL as _YOLO_MODEL,
    YOLO_TARGET_CLASSES as _YOLO_TARGET_CLASSES,
)


# ============================================================================
# Stage 0: validate one Inside Airbnb city
# ============================================================================

@dataclass
class ValidateCityArgs:
    city: str
    country: str
    region: str
    snapshot_date: str
    listings_url: str
    reviews_url: str
    calendar_url: str = ""


def validate_city(args: ValidateCityArgs) -> dict:
    """HEAD-check listings + reviews + calendar, count listings, sample 5 images."""
    out = {
        **asdict(args),
        "listings_ok": False,
        "listings_bytes": 0,
        "n_listings": 0,
        "reviews_ok": False,
        "reviews_bytes": 0,
        "calendar_ok": False,
        "calendar_bytes": 0,
        "sample_image_ok_ratio": 0.0,
        "error": None,
    }
    try:
        out["listings_ok"], out["listings_bytes"] = _head(args.listings_url)
        if args.reviews_url:
            out["reviews_ok"], out["reviews_bytes"] = _head(args.reviews_url)
        if args.calendar_url:
            out["calendar_ok"], out["calendar_bytes"] = _head(args.calendar_url)

        if out["listings_ok"]:
            sample_picture_urls, n_listings = _sample_listings(args.listings_url, n_sample=5)
            out["n_listings"] = n_listings
            n_ok = 0
            for u in sample_picture_urls:
                ok, _ = _head(u, timeout=10)
                if ok:
                    n_ok += 1
            denom = max(1, len(sample_picture_urls))
            out["sample_image_ok_ratio"] = n_ok / denom
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["traceback"] = traceback.format_exc()[:1000]
    return out


def _head(url: str, timeout: int = 30) -> tuple[bool, int]:
    try:
        r = requests.head(
            url, timeout=timeout, allow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (compatible; airbnb-burla/0.1)"},
        )
        return (r.status_code == 200, int(r.headers.get("Content-Length", 0)))
    except Exception:
        return (False, 0)


def _sample_listings(listings_url: str, n_sample: int = 5) -> tuple[list[str], int]:
    """Stream-decompress the listings.csv.gz, return (sample_picture_urls, n_listings).

    For accuracy on the row count we read the whole file (CSV.gz is small,
    typically 1-50 MB per city). The picture_url sample is uniform random.
    """
    import csv
    r = requests.get(
        listings_url, timeout=300,
        headers={"User-Agent": "Mozilla/5.0 (compatible; airbnb-burla/0.1)"},
    )
    r.raise_for_status()
    raw = gzip.decompress(r.content).decode("utf-8", errors="replace")
    reader = csv.DictReader(io.StringIO(raw))
    rows = []
    for i, row in enumerate(reader):
        rows.append(row.get("picture_url") or "")
    n_listings = len(rows)
    if n_listings == 0:
        return [], 0
    rng = random.Random(42)
    pool = [u for u in rows if u and u.startswith("http")]
    if not pool:
        return [], n_listings
    k = min(n_sample, len(pool))
    sample = rng.sample(pool, k)
    return sample, n_listings


# ============================================================================
# Stage 1: download + clean per-city listings (writes to /workspace/shared)
# ============================================================================

@dataclass
class DownloadCityArgs:
    city: str
    country: str
    region: str
    snapshot_date: str
    listings_url: str
    shared_root: str  # e.g. /workspace/shared/airbnb/listings
    city_slug: str    # filename-safe key, includes snapshot date


def download_and_clean_city(args: DownloadCityArgs) -> dict:
    """Download listings.csv.gz for one (city, snapshot), clean, write parquet.

    Returns row count and a small sample summary; the cleaned parquet itself
    lives on /workspace/shared so Stage 2a can load all cities' parquets at once.
    The output parquet carries the snapshot_date column so the merge step can
    keep the latest snapshot per listing_id and emit a snapshot_history parquet
    for trajectory analysis.
    """
    out = {
        "city": args.city,
        "country": args.country,
        "region": args.region,
        "snapshot_date": args.snapshot_date,
        "city_slug": args.city_slug,
        "n_rows": 0,
        "ok": False,
        "error": None,
    }
    try:
        import pandas as pd
        os.makedirs(args.shared_root, exist_ok=True)

        out_path = os.path.join(args.shared_root, f"{args.city_slug}.parquet")
        if os.path.exists(out_path):
            try:
                existing = pd.read_parquet(out_path, columns=["listing_id"])
                out["n_rows"] = int(len(existing))
                out["ok"] = True
                out["shared_path"] = out_path
                out["resumed"] = True
                return out
            except Exception:
                pass

        r = requests.get(
            args.listings_url, timeout=600,
            headers={"User-Agent": "Mozilla/5.0 (compatible; airbnb-burla/0.1)"},
        )
        r.raise_for_status()
        raw = gzip.decompress(r.content)

        df = pd.read_csv(io.BytesIO(raw), low_memory=False)

        df["listing_id"] = df["id"].astype("int64")
        df["price_usd"] = df["price"].apply(_parse_price_inline)
        df["cleaning_fee_usd"] = df.get("cleaning_fee", pd.Series([None] * len(df))).apply(_parse_price_inline)
        df["reviews_per_month"] = pd.to_numeric(df.get("reviews_per_month"), errors="coerce")
        df["number_of_reviews"] = pd.to_numeric(df.get("number_of_reviews"), errors="coerce").fillna(0).astype("int64")
        df["latitude"] = pd.to_numeric(df.get("latitude"), errors="coerce")
        df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
        df["accommodates"] = pd.to_numeric(df.get("accommodates"), errors="coerce")
        df["bedrooms"] = pd.to_numeric(df.get("bedrooms"), errors="coerce")
        df["bathrooms"] = pd.to_numeric(df.get("bathrooms"), errors="coerce")

        cleaning_safe = df["cleaning_fee_usd"].fillna(0)
        nightly_safe = df["price_usd"].replace(0, pd.NA)
        df["cleaning_fee_ratio"] = (cleaning_safe / nightly_safe).astype(float)
        df["demand_proxy"] = df["reviews_per_month"].fillna(0)

        keep = [
            "listing_id", "name", "host_id",
            "neighbourhood_cleansed", "latitude", "longitude",
            "room_type", "property_type",
            "price_usd", "cleaning_fee_usd", "cleaning_fee_ratio",
            "accommodates", "bedrooms", "bathrooms",
            "reviews_per_month", "number_of_reviews", "demand_proxy",
            "picture_url", "listing_url",
        ]
        keep = [c for c in keep if c in df.columns]
        slim = df[keep].copy()
        slim["city"] = args.city
        slim["country"] = args.country
        slim["region"] = args.region
        slim["snapshot_date"] = args.snapshot_date

        out_path = os.path.join(args.shared_root, f"{args.city_slug}.parquet")
        slim.to_parquet(out_path, compression="zstd", index=False)
        out["n_rows"] = int(len(slim))
        out["ok"] = True
        out["shared_path"] = out_path
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["traceback"] = traceback.format_exc()[:1000]
    return out


def _parse_price_inline(value) -> Optional[float]:
    """Worker-side copy of parse_price (avoids importing the lib package)."""
    if value is None:
        return None
    try:
        if value != value:
            return None
    except TypeError:
        pass
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if not s or s.lower() in ("nan", "null", "none"):
        return None
    s = s.lstrip("$\u20ac\u00a3\u00a5").strip()
    s = s.replace(",", "").replace(" ", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


# ============================================================================
# Stage 1 reduce: merge per-city parquets into one shared parquet
# ============================================================================

@dataclass
class MergeListingsArgs:
    shared_root: str
    output_path: str
    history_path: str = ""  # optional: full multi-snapshot parquet


def merge_listings_parquets(args: MergeListingsArgs) -> dict:
    """One Burla worker merges all per-(city, snapshot) parquets.

    Two outputs:
    - ``output_path``: latest snapshot per listing_id (one row per listing).
    - ``history_path`` (optional): every (listing_id, snapshot_date) row, used
      by trajectory analyses and the calendar stage.
    """
    out = {
        "ok": False, "n_files": 0, "n_rows": 0, "n_cities": 0,
        "n_history_rows": 0,
        "output_path": args.output_path, "history_path": args.history_path,
        "schema": [], "sample_rows": [],
        "n_with_picture_url": 0, "error": None,
    }
    try:
        import glob
        import pandas as pd
        files = sorted(glob.glob(os.path.join(args.shared_root, "*.parquet")))
        out["n_files"] = len(files)
        if not files:
            out["error"] = f"no parquets at {args.shared_root}"
            return out
        dfs = [pd.read_parquet(f) for f in files]
        big = pd.concat(dfs, ignore_index=True)
        big["listing_id"] = big["listing_id"].astype("int64")
        if "snapshot_date" not in big.columns:
            big["snapshot_date"] = ""
        big["snapshot_date"] = big["snapshot_date"].astype(str)
        out["n_history_rows"] = int(len(big))

        if args.history_path:
            os.makedirs(os.path.dirname(args.history_path), exist_ok=True)
            history_cols = [
                c for c in (
                    "listing_id", "snapshot_date", "city", "country", "region",
                    "price_usd", "reviews_per_month", "number_of_reviews",
                    "demand_proxy", "accommodates", "bedrooms",
                    "picture_url", "listing_url",
                ) if c in big.columns
            ]
            history_df = big[history_cols].copy()
            for _col in history_df.columns:
                if history_df[_col].dtype == "object":
                    history_df[_col] = history_df[_col].astype("string")
            history_df.to_parquet(
                args.history_path, compression="zstd", index=False,
            )

        big = big.sort_values(["listing_id", "snapshot_date"]) \
                 .drop_duplicates(subset=["listing_id"], keep="last")
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

        for _col in big.columns:
            if big[_col].dtype == "object":
                big[_col] = big[_col].astype("string")
        big.to_parquet(args.output_path, compression="zstd", index=False)
        out["ok"] = True
        out["n_rows"] = int(len(big))
        out["n_cities"] = int(big["city"].nunique()) if "city" in big.columns else 0
        out["schema"] = [(c, str(big[c].dtype)) for c in big.columns]
        if "picture_url" in big.columns:
            out["n_with_picture_url"] = int(big["picture_url"].notna().sum())
        out["sample_rows"] = big.head(50).to_dict("records")
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["traceback"] = traceback.format_exc()[:1000]
    return out


# ============================================================================
# Stage 1b: download calendar.csv.gz for one (city, snapshot) tuple
# ============================================================================

@dataclass
class DownloadCalendarArgs:
    city: str
    country: str
    region: str
    snapshot_date: str
    calendar_url: str
    shared_root: str  # /workspace/shared/airbnb/calendar_v2
    city_slug: str    # filename-safe, includes snapshot_date


def download_and_compress_calendar(args: DownloadCalendarArgs) -> dict:
    """Download calendar.csv.gz (year of forward day-level availability + price),
    compute per-listing summary stats, write a slim parquet to shared FS.

    Calendar files are large (sometimes 100M+ rows raw). We aggregate to one
    row per listing per snapshot in the worker so the downstream calendar stage
    only joins ~1M rows per snapshot, not 100M.
    """
    out = {
        "city": args.city, "snapshot_date": args.snapshot_date,
        "city_slug": args.city_slug,
        "ok": False, "n_listings": 0, "n_calendar_rows": 0,
        "shared_path": None, "error": None,
    }
    try:
        if not args.calendar_url:
            out["error"] = "no_calendar_url"
            return out
        import pandas as pd
        os.makedirs(args.shared_root, exist_ok=True)

        out_path = os.path.join(args.shared_root, f"{args.city_slug}.parquet")
        if os.path.exists(out_path):
            try:
                existing = pd.read_parquet(out_path, columns=["listing_id"])
                out["ok"] = True
                out["n_listings"] = int(len(existing))
                out["shared_path"] = out_path
                out["resumed"] = True
                return out
            except Exception:
                pass

        r = requests.get(
            args.calendar_url, timeout=900,
            headers={"User-Agent": "Mozilla/5.0 (compatible; airbnb-burla/0.1)"},
        )
        if r.status_code != 200:
            out["error"] = f"http_{r.status_code}"
            return out
        raw = gzip.decompress(r.content)
        df = pd.read_csv(io.BytesIO(raw), low_memory=False)
        out["n_calendar_rows"] = int(len(df))
        if "listing_id" not in df.columns:
            out["error"] = "no_listing_id_column"
            return out

        df["listing_id"] = pd.to_numeric(df["listing_id"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["listing_id"])
        df["listing_id"] = df["listing_id"].astype("int64")
        df["available"] = df.get("available", "").astype(str).str.lower().isin(("t", "true"))
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        else:
            df["date"] = pd.NaT
        df["price_usd"] = df.get("price").apply(_parse_price_inline)
        df["weekday"] = df["date"].dt.dayofweek
        df["is_weekend"] = df["weekday"].isin((4, 5))
        if "minimum_nights" in df.columns:
            df["minimum_nights"] = pd.to_numeric(df["minimum_nights"], errors="coerce")
        else:
            df["minimum_nights"] = float("nan")
        snapshot_dt = pd.to_datetime(args.snapshot_date, errors="coerce")
        df["days_from_snapshot"] = (df["date"] - snapshot_dt).dt.days

        grouped = df.groupby("listing_id")
        agg = grouped.agg(
            n_days=("date", "count"),
            n_days_available=("available", "sum"),
            n_weekend_days=("is_weekend", "sum"),
            n_weekend_available=("available", lambda s: int((s & df.loc[s.index, "is_weekend"]).sum())),
            mean_price=("price_usd", "mean"),
            median_price=("price_usd", "median"),
            std_price=("price_usd", "std"),
            min_minimum_nights=("minimum_nights", "min"),
            max_minimum_nights=("minimum_nights", "max"),
        ).reset_index()
        agg["snapshot_date"] = args.snapshot_date
        agg["city"] = args.city
        agg["country"] = args.country
        agg["region"] = args.region

        # Lead-time-open: how many days from snapshot until the first
        # available night? Lower = more open, higher = booked far out.
        first_open = (
            df.loc[df["available"]]
            .groupby("listing_id")["days_from_snapshot"]
            .min()
            .rename("lead_time_open")
        )
        agg = agg.merge(first_open, on="listing_id", how="left")

        out_path = os.path.join(args.shared_root, f"{args.city_slug}.parquet")
        agg.to_parquet(out_path, compression="zstd", index=False)
        out.update({
            "ok": True,
            "n_listings": int(len(agg)),
            "shared_path": out_path,
        })
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["traceback"] = traceback.format_exc()[:1000]
    return out


@dataclass
class MergeCalendarArgs:
    shared_root: str
    output_path: str


def merge_calendar_parquets(args: MergeCalendarArgs) -> dict:
    """Merge per-(city, snapshot) calendar summaries into one parquet.

    Keeps every (listing_id, snapshot_date) row so the calendar stage can
    compute occupancy_365, weekend_premium, etc per-listing-per-snapshot.
    """
    out = {
        "ok": False, "n_files": 0, "n_rows": 0, "n_listings": 0,
        "output_path": args.output_path, "error": None,
    }
    try:
        import glob
        import pandas as pd
        files = sorted(glob.glob(os.path.join(args.shared_root, "*.parquet")))
        out["n_files"] = len(files)
        if not files:
            out["error"] = f"no parquets at {args.shared_root}"
            return out
        big = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        big = big.drop_duplicates(subset=["listing_id", "snapshot_date"])
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

        for _col in big.columns:
            if big[_col].dtype == "object":
                big[_col] = big[_col].astype("string")
        big.to_parquet(args.output_path, compression="zstd", index=False)
        out.update({
            "ok": True, "n_rows": int(len(big)),
            "n_listings": int(big["listing_id"].nunique()),
        })
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["traceback"] = traceback.format_exc()[:1000]
    return out


# ============================================================================
# Stage 2b: download + CLIP-score one image
# ============================================================================

@dataclass
class CpuImageArgs:
    listing_id: int
    city_slug: str
    image_idx: int
    image_url: str


_CLIP_STATE: dict = {}

# Pre-staged CLIP weights on the Burla shared filesystem. We download these
# once via scripts/preload_clip_weights.py to avoid 80 workers per node racing
# on a 605 MB HuggingFace download (which used to OOM the per-node disk).
_CLIP_SHARED_WEIGHTS = "/workspace/shared/airbnb/clip_weights/openai.bin"
_CLIP_LOCAL_WEIGHTS = "/tmp/clip_openai.bin"
_CLIP_NODE_LOCK = "/tmp/clip_openai.lock"


def _ensure_clip():
    """Lazy-load the CLIP model + tokenizer + prompt embeddings into worker globals.

    Loading strategy: copy the pre-staged weights file from /workspace/shared
    (GCSFuse) to /tmp on the local node once per node, then point open_clip at
    the local file. We use a node-local fcntl lock so only one of the ~80
    workers per node does the copy, the rest just read from /tmp.
    """
    if "model" in _CLIP_STATE:
        return _CLIP_STATE
    import fcntl
    import shutil
    import torch
    import open_clip

    # Each Burla node runs ~80 of these workers concurrently. PyTorch defaults
    # to using all available CPUs per process, which means 80 workers x 80
    # threads = 6400 threads fighting for 80 cores. Pin each worker to 1 thread
    # so they actually run in parallel.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"

    last_err: Exception | None = None
    for attempt in range(6):
        try:
            with open(_CLIP_NODE_LOCK, "w") as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                if not (
                    os.path.exists(_CLIP_LOCAL_WEIGHTS)
                    and os.path.getsize(_CLIP_LOCAL_WEIGHTS) > 100_000_000
                ):
                    if not os.path.exists(_CLIP_SHARED_WEIGHTS):
                        raise RuntimeError(
                            f"CLIP weights missing at {_CLIP_SHARED_WEIGHTS}; "
                            "run scripts/preload_clip_weights.py first"
                        )
                    tmp = _CLIP_LOCAL_WEIGHTS + ".part"
                    shutil.copyfile(_CLIP_SHARED_WEIGHTS, tmp)
                    os.replace(tmp, _CLIP_LOCAL_WEIGHTS)
                fcntl.flock(lock_file, fcntl.LOCK_UN)

            model, _, preprocess = open_clip.create_model_and_transforms(
                _CLIP_MODEL, pretrained=_CLIP_LOCAL_WEIGHTS,
            )
            model.to(device).eval()
            tokenizer = open_clip.get_tokenizer(_CLIP_MODEL)
            prompts = list(_CLIP_PROMPTS.values())
            keys = list(_CLIP_PROMPTS.keys())
            with torch.no_grad():
                text_tokens = tokenizer(prompts).to(device)
                text_emb = model.encode_text(text_tokens)
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            _CLIP_STATE.update({
                "device": device, "model": model, "preprocess": preprocess,
                "text_emb": text_emb, "keys": keys,
            })
            return _CLIP_STATE
        except Exception as e:
            last_err = e
            time.sleep(min(30.0, 3.0 * (2 ** attempt)) + random.uniform(0, 2.0))
    raise RuntimeError(
        f"_ensure_clip failed after 6 attempts: {type(last_err).__name__}: {last_err}"
    )


@dataclass
class CpuImageBatchArgs:
    batch_id: int
    photo_manifest_path: str  # parquet path on /workspace/shared
    row_start: int
    row_end: int
    output_root: str          # /workspace/shared/airbnb/images_cpu


def cpu_score_image_batch(args: CpuImageBatchArgs) -> dict:
    """Read a row-range from the photo manifest parquet, score every image,
    write a per-batch parquet to shared FS, return summary stats."""
    out = {
        "batch_id": args.batch_id,
        "n_inputs": 0, "n_ok": 0, "n_failed": 0,
        "shared_path": None, "elapsed_seconds": 0.0, "error": None,
    }
    started = time.time()
    shared_path = os.path.join(args.output_root, f"batch_{args.batch_id:06d}.parquet")
    if os.path.exists(shared_path):
        try:
            import pandas as pd
            existing = pd.read_parquet(shared_path, columns=["download_ok"])
            out["n_inputs"] = int(len(existing))
            out["n_ok"] = int(existing["download_ok"].sum())
            out["n_failed"] = int(out["n_inputs"] - out["n_ok"])
            out["shared_path"] = shared_path
            out["resumed"] = True
            out["elapsed_seconds"] = time.time() - started
            return out
        except Exception:
            pass
    try:
        import pandas as pd
        manifest = pd.read_parquet(
            args.photo_manifest_path,
            columns=["listing_id", "image_idx", "image_url"],
        )
        chunk = manifest.iloc[args.row_start: args.row_end].reset_index(drop=True)
        out["n_inputs"] = int(len(chunk))
        rows = []
        for _, row in chunk.iterrows():
            r = cpu_score_image(CpuImageArgs(
                listing_id=int(row["listing_id"]),
                city_slug="",
                image_idx=int(row["image_idx"]),
                image_url=str(row["image_url"]),
            ))
            rows.append(r)
            if r.get("download_ok"):
                out["n_ok"] += 1
            else:
                out["n_failed"] += 1

        if rows:
            os.makedirs(args.output_root, exist_ok=True)
            shared_path = os.path.join(args.output_root, f"batch_{args.batch_id:06d}.parquet")
            import pandas as pd
            pd.DataFrame(rows).to_parquet(shared_path, compression="zstd", index=False)
            out["shared_path"] = shared_path
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["traceback"] = traceback.format_exc()[:1000]
    out["elapsed_seconds"] = time.time() - started
    return out


@dataclass
class MergeImagesCpuArgs:
    shared_root: str
    output_path: str


def merge_images_cpu(args: MergeImagesCpuArgs) -> dict:
    out = {"ok": False, "n_files": 0, "n_rows": 0, "n_listings": 0,
           "n_download_ok": 0, "output_path": args.output_path, "error": None}
    try:
        import glob
        import pandas as pd
        files = sorted(glob.glob(os.path.join(args.shared_root, "batch_*.parquet")))
        out["n_files"] = len(files)
        if not files:
            out["error"] = f"no batch parquets at {args.shared_root}"
            return out
        dfs = [pd.read_parquet(f) for f in files]
        big = pd.concat(dfs, ignore_index=True)
        big = big.drop_duplicates(subset=["listing_id", "image_idx"])
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        big.to_parquet(args.output_path, compression="zstd", index=False)
        out["ok"] = True
        out["n_rows"] = int(len(big))
        out["n_listings"] = int(big["listing_id"].nunique())
        out["n_download_ok"] = int(big["download_ok"].fillna(False).sum())
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["traceback"] = traceback.format_exc()[:1000]
    return out


def cpu_score_image(args: CpuImageArgs) -> dict:
    """Download one image, compute CLIP scores + brightness + edge density."""
    out = {
        "listing_id": args.listing_id,
        "city_slug": args.city_slug,
        "image_idx": args.image_idx,
        "image_url": args.image_url,
        "download_ok": False,
        "width": None,
        "height": None,
        "brightness": None,
        "edge_density": None,
        "error": None,
    }
    try:
        import numpy as np
        from PIL import Image

        r = requests.get(
            args.image_url, timeout=30,
            headers={"User-Agent": "Mozilla/5.0 (compatible; airbnb-burla/0.1)"},
        )
        if r.status_code != 200:
            out["error"] = f"http_{r.status_code}"
            return out
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        out["download_ok"] = True
        out["width"], out["height"] = img.size

        np_img = np.asarray(img.resize((128, 128)), dtype=np.float32) / 255.0
        out["brightness"] = float(np_img.mean())
        gx = np.abs(np.diff(np_img.mean(axis=2), axis=1)).mean()
        gy = np.abs(np.diff(np_img.mean(axis=2), axis=0)).mean()
        out["edge_density"] = float((gx + gy) / 2.0)

        state = _ensure_clip()
        import torch
        with torch.no_grad():
            tensor = state["preprocess"](img).unsqueeze(0).to(state["device"])
            img_emb = state["model"].encode_image(tensor)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            sims = (img_emb @ state["text_emb"].T).squeeze(0).cpu().tolist()
        for k, v in zip(state["keys"], sims):
            out[f"clip_{k}"] = float(v)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return out


# ============================================================================
# Stage 3: GPU object detection (YOLOv8) on selected images
# ============================================================================

@dataclass
class GpuImageArgs:
    listing_id: int
    city_slug: str
    image_idx: int
    image_url: str


@dataclass
class GpuImageBatchArgs:
    batch_id: int
    rows: list           # list of dicts with listing_id, image_idx, image_url
    output_root: str     # /workspace/shared/airbnb/images_gpu


_YOLO_STATE: dict = {}

# Pre-staged YOLOv8 weights, downloaded once via scripts/preload_yolo_weights.py
# to avoid each GPU node racing on the GitHub release URL on first call.
_YOLO_SHARED_WEIGHTS = "/workspace/shared/airbnb/yolo_weights/yolov8n.pt"
_YOLO_LOCAL_WEIGHTS = "/tmp/yolov8n.pt"
_YOLO_NODE_LOCK = "/tmp/yolov8n.lock"


def _ensure_yolo():
    """Lazy-load YOLOv8 on the GPU worker.

    Burla's GPU image does not ship ultralytics. We pip-install it once per
    worker process, then load weights from /tmp (copied from the shared FS by
    the first worker on each node, protected by an fcntl lock).
    """
    if "model" in _YOLO_STATE:
        return _YOLO_STATE

    import importlib
    import subprocess
    import sys
    import fcntl
    import shutil

    # Burla's GPU image ships torch 2.11.0+cu130 but the host has CUDA driver
    # 12.4, so torch.cuda.is_available() is False. We swap in torch+cu124
    # (compatible with the driver) and install opencv-python-headless to
    # avoid libGL.so.1 issues. ultralytics 8.4.41 supports numpy>=2.
    try:
        importlib.import_module("cv2")
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "--no-input",
             "opencv-python-headless"],
        )
        importlib.invalidate_caches()

    try:
        import torch as _t
        cuda_ok = _t.cuda.is_available()
    except Exception:
        cuda_ok = False
    if not cuda_ok:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet", "--no-input",
                 "torch==2.5.1", "--index-url",
                 "https://download.pytorch.org/whl/cu124"],
            )
            importlib.invalidate_caches()
        except subprocess.CalledProcessError:
            pass

    try:
        importlib.import_module("ultralytics")
    except ModuleNotFoundError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "--no-input",
             "ultralytics>=8.3.50"],
        )
        importlib.invalidate_caches()
    from ultralytics import YOLO

    if os.path.exists(_YOLO_SHARED_WEIGHTS):
        with open(_YOLO_NODE_LOCK, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            if not (
                os.path.exists(_YOLO_LOCAL_WEIGHTS)
                and os.path.getsize(_YOLO_LOCAL_WEIGHTS) > 1_000_000
            ):
                tmp = _YOLO_LOCAL_WEIGHTS + ".part"
                shutil.copyfile(_YOLO_SHARED_WEIGHTS, tmp)
                os.replace(tmp, _YOLO_LOCAL_WEIGHTS)
            fcntl.flock(lock_file, fcntl.LOCK_UN)
        model = YOLO(_YOLO_LOCAL_WEIGHTS)
    else:
        model = YOLO(_YOLO_MODEL)
    import torch
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _YOLO_STATE["model"] = model
    _YOLO_STATE["device"] = device
    return _YOLO_STATE


def gpu_detect_image_batch(args: GpuImageBatchArgs) -> dict:
    """Run YOLOv8 on a batch of images; return summary, write parquet to shared FS."""
    out = {
        "batch_id": args.batch_id, "n_inputs": len(args.rows),
        "n_ok": 0, "n_failed": 0,
        "shared_path": None, "elapsed_seconds": 0.0, "error": None,
    }
    started = time.time()
    shared_path = os.path.join(args.output_root, f"batch_{args.batch_id:06d}.parquet")
    if os.path.exists(shared_path):
        try:
            import pandas as pd
            existing = pd.read_parquet(shared_path, columns=["listing_id"])
            out["n_inputs"] = int(len(existing))
            out["n_ok"] = int(out["n_inputs"])
            out["shared_path"] = shared_path
            out["resumed"] = True
            out["elapsed_seconds"] = time.time() - started
            return out
        except Exception:
            pass
    try:
        rows = []
        for r in args.rows:
            rec = gpu_detect_image(GpuImageArgs(
                listing_id=int(r["listing_id"]),
                city_slug=str(r.get("city_slug", "")),
                image_idx=int(r["image_idx"]),
                image_url=str(r["image_url"]),
            ))
            rows.append(rec)
            if rec.get("error"):
                out["n_failed"] += 1
            else:
                out["n_ok"] += 1
        if rows:
            os.makedirs(args.output_root, exist_ok=True)
            shared_path = os.path.join(args.output_root, f"batch_{args.batch_id:06d}.parquet")
            import json as _json
            import pandas as pd
            df = pd.DataFrame(rows)
            if "tv_bbox" in df.columns:
                df["tv_bbox"] = df["tv_bbox"].apply(
                    lambda v: _json.dumps(v) if v is not None else None
                )
            df.to_parquet(shared_path, compression="zstd", index=False)
            out["shared_path"] = shared_path
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["traceback"] = traceback.format_exc()[:1000]
    out["elapsed_seconds"] = time.time() - started
    return out


@dataclass
class MergeImagesGpuArgs:
    shared_root: str
    output_path: str


def merge_images_gpu(args: MergeImagesGpuArgs) -> dict:
    out = {"ok": False, "n_files": 0, "n_rows": 0, "n_listings": 0,
           "output_path": args.output_path, "error": None}
    try:
        import glob
        import pandas as pd
        files = sorted(glob.glob(os.path.join(args.shared_root, "batch_*.parquet")))
        out["n_files"] = len(files)
        if not files:
            out["error"] = f"no batch parquets at {args.shared_root}"
            return out
        big = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        big = big.drop_duplicates(subset=["listing_id", "image_idx"])
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        big.to_parquet(args.output_path, compression="zstd", index=False)
        out["ok"] = True
        out["n_rows"] = int(len(big))
        out["n_listings"] = int(big["listing_id"].nunique())
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["traceback"] = traceback.format_exc()[:1000]
    return out


def gpu_detect_image(args: GpuImageArgs) -> dict:
    out = {
        "listing_id": args.listing_id,
        "city_slug": args.city_slug,
        "image_idx": args.image_idx,
        "image_url": args.image_url,
        "tv_detected": False,
        "tv_bbox": None,
        "tv_above_50pct": False,
        "person_detected": False,
        "potted_plant_count": 0,
        "couch_detected": False,
        "bed_detected": False,
        "cat_detected": False,
        "dog_detected": False,
        "pet_detected": False,
        "pet_count": 0,
        "n_objects": 0,
        "error": None,
    }
    try:
        from PIL import Image

        r = requests.get(
            args.image_url, timeout=30,
            headers={"User-Agent": "Mozilla/5.0 (compatible; airbnb-burla/0.1)"},
        )
        if r.status_code != 200:
            out["error"] = f"http_{r.status_code}"
            return out
        img = Image.open(io.BytesIO(r.content)).convert("RGB")
        h = img.height

        state = _ensure_yolo()
        device = state.get("device", "cpu")
        results = state["model"].predict(img, verbose=False, device=device)
        if not results:
            return out
        boxes = results[0].boxes
        if boxes is None:
            return out

        cls = boxes.cls.cpu().tolist() if boxes.cls is not None else []
        xyxy = boxes.xyxy.cpu().tolist() if boxes.xyxy is not None else []
        out["n_objects"] = len(cls)

        for c, box in zip(cls, xyxy):
            c = int(c)
            if c == _YOLO_TARGET_CLASSES["tv"]:
                out["tv_detected"] = True
                out["tv_bbox"] = [float(x) for x in box]
                tv_top = box[1]
                if tv_top / max(1, h) < 0.50:
                    out["tv_above_50pct"] = True
            elif c == _YOLO_TARGET_CLASSES["person"]:
                out["person_detected"] = True
            elif c == _YOLO_TARGET_CLASSES["potted plant"]:
                out["potted_plant_count"] += 1
            elif c == _YOLO_TARGET_CLASSES["couch"]:
                out["couch_detected"] = True
            elif c == _YOLO_TARGET_CLASSES["bed"]:
                out["bed_detected"] = True
            elif c == _YOLO_TARGET_CLASSES["cat"]:
                out["cat_detected"] = True
                out["pet_detected"] = True
                out["pet_count"] += 1
            elif c == _YOLO_TARGET_CLASSES["dog"]:
                out["dog_detected"] = True
                out["pet_detected"] = True
                out["pet_count"] += 1
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return out
