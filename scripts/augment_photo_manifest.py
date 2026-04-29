"""Augment photo_manifest.parquet with primary photos from listings_clean.parquet.

Stage 2a's deep scrape only got ~309K listings worth of photos before the
1000-worker run started getting heavily throttled by Datadome. To salvage a
demo-scale image dataset, we union those scraped rows with the primary
``picture_url`` column that Inside Airbnb publishes for every listing.

Result: every listing has at least one image (the primary), plus up to ~25
additional CDN URLs for the subset we successfully scraped.
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import os
import glob
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import traceback

sys.path.insert(0, ".")

from dotenv import load_dotenv

# Hoist for Burla worker dependency detection.
import numpy as _np  # noqa: F401
import pandas as _pd  # noqa: F401
import pyarrow as _pa  # noqa: F401
import pyarrow.parquet as _pq  # noqa: F401

from src.config import SHARED_ROOT
from src.lib.io import register_src_for_burla


@dataclass
class AugmentArgs:
    listings_parquet: str
    scraped_manifest: str
    output_path: str


def augment(args: AugmentArgs) -> dict:
    """Run on Burla. Reads both parquets, unions, dedupes, writes back."""

    out = {"ok": False, "error": None}
    try:
        scraped_table = pq.read_table(args.scraped_manifest)
        scraped = pd.DataFrame({
            "listing_id": scraped_table.column("listing_id").to_pylist(),
            "image_idx": scraped_table.column("image_idx").to_pylist(),
            "image_url": scraped_table.column("image_url").to_pylist(),
            "title": scraped_table.column("title").to_pylist() if "title" in scraped_table.column_names else [None] * scraped_table.num_rows,
            "scraped_at": scraped_table.column("scraped_at").to_pylist() if "scraped_at" in scraped_table.column_names else [time.time()] * scraped_table.num_rows,
        })

        shared_root = os.path.dirname(args.listings_parquet)
        listing_files = sorted(glob.glob(os.path.join(shared_root, "listings", "*.parquet")))

        ids: list = []
        urls: list = []
        n_files_ok = 0
        n_files_fail = 0
        for f in listing_files:
            try:
                t = pq.read_table(f, columns=["listing_id", "picture_url"])
                ids.extend(t.column("listing_id").to_pylist())
                urls.extend(t.column("picture_url").to_pylist())
                n_files_ok += 1
            except Exception:
                n_files_fail += 1

        listings = pd.DataFrame({"listing_id": ids, "picture_url": urls})
        listings = listings.drop_duplicates(subset=["listing_id"])
        listings = listings[listings["picture_url"].notna()]
        listings = listings[listings["picture_url"].astype(str).str.startswith("http")].copy()
        listings["image_idx"] = 0
        listings = listings.rename(columns={"picture_url": "image_url"})
        listings["title"] = None
        listings["scraped_at"] = float(time.time())
        listings = listings[["listing_id", "image_idx", "image_url", "title", "scraped_at"]]
        listings["listing_id"] = pd.to_numeric(listings["listing_id"], errors="coerce").astype("Int64")
        listings = listings[listings["listing_id"].notna()]
        listings["listing_id"] = listings["listing_id"].astype("int64")

        scraped["image_idx"] = pd.to_numeric(scraped["image_idx"], errors="coerce").fillna(0).astype("int64") + 1
        scraped["listing_id"] = pd.to_numeric(scraped["listing_id"], errors="coerce").astype("Int64")
        scraped = scraped[scraped["listing_id"].notna()]
        scraped["listing_id"] = scraped["listing_id"].astype("int64")

        big = pd.concat([listings, scraped[listings.columns.tolist()]], ignore_index=True)
        big = big.drop_duplicates(subset=["listing_id", "image_url"])
        big = big.sort_values(["listing_id", "image_idx"]).reset_index(drop=True)

        out_table = pa.Table.from_pandas(big, preserve_index=False)
        pq.write_table(out_table, args.output_path, compression="zstd")

        out.update(
            ok=True,
            n_rows=int(len(big)),
            n_listings=int(big["listing_id"].nunique()),
            n_primary=int(len(listings)),
            n_scraped=int(len(scraped)),
            n_files_ok=n_files_ok,
            n_files_fail=n_files_fail,
            output_path=args.output_path,
        )
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        out["traceback"] = traceback.format_exc()[:1500]
    return out


def main() -> None:
    load_dotenv()
    register_src_for_burla()
    from burla import remote_parallel_map

    listings_parquet = f"{SHARED_ROOT}/listings_clean.parquet"
    scraped_manifest = f"{SHARED_ROOT}/photo_manifest.parquet"
    output_path = f"{SHARED_ROOT}/photo_manifest.parquet"

    print(f"[augment] augmenting {scraped_manifest} with primary photos from {listings_parquet}", flush=True)
    [res] = remote_parallel_map(
        augment,
        [AugmentArgs(
            listings_parquet=listings_parquet,
            scraped_manifest=scraped_manifest,
            output_path=output_path,
        )],
        func_cpu=8, func_ram=32, max_parallelism=1, grow=True, spinner=True,
    )
    if not res.get("ok"):
        print(f"[augment] traceback: {res.get('traceback','')}", flush=True)
        raise SystemExit(f"[augment] failed: {res.get('error')}")
    print(f"[augment]   {res['n_rows']:,} rows total: "
          f"{res['n_primary']:,} primary + {res['n_scraped']:,} scraped, "
          f"covering {res['n_listings']:,} listings "
          f"(parquet ok/fail = {res.get('n_files_ok')}/{res.get('n_files_fail')})", flush=True)
    print(f"[augment] DONE.", flush=True)


if __name__ == "__main__":
    main()
