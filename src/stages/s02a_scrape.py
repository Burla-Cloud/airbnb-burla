"""Stage 2a: scrape photo manifests from airbnb.com/rooms/<id>.

Reads listing_ids from /workspace/shared/airbnb/listings_clean.parquet on Burla
(via a small lister worker), batches them, fans out scrape_batch tasks across
~1000 parallel workers at 0.5 req/sec/worker, then reduces.

Run with --sample N to scrape just N listings as a sanity check first; the
sample run halts the pipeline if success rate < SCRAPE_MIN_SUCCESS_RATE.
"""
from __future__ import annotations

import argparse
import time

from dotenv import load_dotenv

from ..config import (
    LISTINGS_BATCH_SIZE, PHOTO_MANIFEST_INCLUDE_HISTORY, PHOTO_MANIFEST_PATH,
    SCRAPE_MAX_PARALLELISM, SCRAPE_MIN_SUCCESS_RATE,
    SCRAPE_REQ_PER_SEC_PER_WORKER, SCRAPE_RETRY_LIMIT,
    SHARED_PHOTOS, SHARED_ROOT,
)
from ..lib.budget import BudgetTracker
from ..lib.io import ensure_dir, register_src_for_burla, write_json
from ..tasks.scrape_tasks import (
    ScrapeBatchArgs, scrape_batch,
    MergePhotosArgs, merge_photo_batches,
    ListListingIdsArgs, list_listing_ids,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0,
                        help="Scrape this many random listings as a sanity check (0 = full run)")
    args = parser.parse_args()

    load_dotenv()
    register_src_for_burla()

    if not args.sample:
        from ..lib.io import read_json
        local_manifest_path = PHOTO_MANIFEST_PATH.with_suffix(".manifest.json")
        existing = read_json(local_manifest_path)
        if existing and existing.get("ok") and existing.get("n_rows"):
            print(
                f"[s02a] manifest exists ({existing['n_rows']:,} rows, "
                f"{existing.get('n_listings', 0):,} listings); skipping "
                f"(delete {local_manifest_path} to force rerun).",
                flush=True,
            )
            return

    from burla import remote_parallel_map

    listings_parquet = f"{SHARED_ROOT}/listings_clean.parquet"

    print(f"[s02a] enumerating listing_ids from {listings_parquet} ...", flush=True)
    [lst] = remote_parallel_map(
        list_listing_ids,
        [ListListingIdsArgs(listings_parquet_path=listings_parquet, sample_n=args.sample)],
        func_cpu=4, func_ram=16, max_parallelism=1, grow=True, spinner=False,
    )
    listing_ids = lst["listing_ids"]
    print(f"[s02a]   {len(listing_ids):,} listings selected (of {lst['n_total']:,} total)", flush=True)

    batches: list[ScrapeBatchArgs] = []
    shared_root = SHARED_PHOTOS + ("_sample" if args.sample else "")
    batch_size = LISTINGS_BATCH_SIZE
    if args.sample and args.sample <= 5000:
        batch_size = max(25, args.sample // 20)
    for batch_id, start in enumerate(range(0, len(listing_ids), batch_size)):
        chunk = listing_ids[start:start + batch_size]
        batches.append(ScrapeBatchArgs(
            batch_id=batch_id,
            listing_ids=chunk,
            shared_root=shared_root,
            req_per_sec_per_worker=SCRAPE_REQ_PER_SEC_PER_WORKER,
            retry_limit=SCRAPE_RETRY_LIMIT,
        ))
    print(f"[s02a]   built {len(batches):,} batches of up to {batch_size} listings each", flush=True)

    n_workers = min(SCRAPE_MAX_PARALLELISM, len(batches))
    if args.sample:
        n_workers = min(n_workers, max(20, len(batches)))
    print(f"[s02a] launching scrape on max {n_workers} parallel workers ...", flush=True)
    t0 = time.time()

    with BudgetTracker("s02a_scrape", n_inputs=len(batches), func_cpu=1) as bt:
        bt.set_workers(n_workers)
        results: list[dict] = remote_parallel_map(
            scrape_batch,
            batches,
            func_cpu=1,
            func_ram=2,
            max_parallelism=n_workers,
            grow=True,
            spinner=False,
        )

        n_ok_total = sum(int(r.get("n_ok", 0)) for r in results)
        n_empty = sum(int(r.get("n_empty", 0)) for r in results)
        n_blocked = sum(int(r.get("n_blocked", 0)) for r in results)
        n_failed = sum(int(r.get("n_failed", 0)) for r in results)
        n_listings_seen = sum(int(r.get("n_listings", 0)) for r in results)
        n_photos = sum(int(r.get("n_total_photos", 0)) for r in results)
        success_rate = n_ok_total / max(1, n_listings_seen)

        bt.set_succeeded(n_ok_total + n_empty)
        bt.set_failed(n_blocked + n_failed)
        bt.note(
            n_listings_seen=n_listings_seen,
            n_total_photos=n_photos,
            success_rate=round(success_rate, 4),
            sample_run=bool(args.sample),
        )

    elapsed = time.time() - t0
    print(f"[s02a]   ok={n_ok_total:,} empty={n_empty:,} blocked={n_blocked:,} "
          f"failed={n_failed:,} success_rate={success_rate:.2%} "
          f"photos={n_photos:,} elapsed={elapsed:.1f}s", flush=True)

    if args.sample:
        report = {
            "sample_n": args.sample,
            "n_listings_seen": n_listings_seen,
            "success_rate": success_rate,
            "n_ok": n_ok_total,
            "n_blocked": n_blocked,
            "n_failed": n_failed,
            "n_total_photos": n_photos,
            "elapsed_seconds": elapsed,
            "completed_at": time.time(),
        }
        from ..config import OUTPUT_DIR
        ensure_dir(OUTPUT_DIR)
        write_json(OUTPUT_DIR / "scrape_sample_report.json", report)
        if success_rate < SCRAPE_MIN_SUCCESS_RATE:
            raise SystemExit(
                f"[s02a] sample success rate {success_rate:.2%} < {SCRAPE_MIN_SUCCESS_RATE:.0%} "
                f"floor; halting before full run."
            )
        print(f"[s02a] sample looks good; rerun without --sample for the full scrape.", flush=True)
        return

    print("[s02a] reducing batch parquets into one photo manifest ...", flush=True)
    shared_merged = f"{SHARED_ROOT}/photo_manifest.parquet"
    history_path = f"{SHARED_ROOT}/listings_history.parquet" if PHOTO_MANIFEST_INCLUDE_HISTORY else ""
    [merge] = remote_parallel_map(
        merge_photo_batches,
        [MergePhotosArgs(
            shared_root=shared_root,
            output_path=shared_merged,
            listings_history_path=history_path,
        )],
        func_cpu=8, func_ram=32, max_parallelism=1, grow=True, spinner=False,
    )
    if not merge.get("ok"):
        raise SystemExit(f"[s02a] merge failed: {merge.get('error')}")
    print(f"[s02a]   merged {merge['n_rows']:,} photo rows across "
          f"{merge['n_listings']:,} listings from {merge['n_files']:,} batches "
          f"(history added {merge.get('n_history_added', 0):,}, "
          f"after url-dedupe {merge.get('n_after_url_dedupe', merge['n_rows']):,})",
          flush=True)

    ensure_dir(PHOTO_MANIFEST_PATH.parent)
    manifest_path = PHOTO_MANIFEST_PATH.with_suffix(".manifest.json")
    write_json(manifest_path, {
        "ok": True,
        "shared_path": shared_merged,
        "n_rows": merge["n_rows"],
        "n_listings": merge["n_listings"],
        "n_input_listings": len(listing_ids),
        "success_rate": success_rate,
        "n_blocked": n_blocked,
        "n_failed": n_failed,
        "elapsed_seconds": elapsed,
        "completed_at": time.time(),
    })
    print(f"[s02a] DONE. Manifest at {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
