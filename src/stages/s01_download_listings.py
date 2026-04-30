"""Stage 1: download every (city, snapshot) listings + calendar in parallel.

Reads the validation report from Stage 0, builds one DownloadCityArgs per
passing (city, snapshot) tuple, and fans them out across Burla. Each worker:

- writes ``<city_slug>__<snapshot_date>.parquet`` to ``/workspace/shared/airbnb/listings_v2/``
- carries the snapshot_date column

Then we kick off the calendar download in parallel: one worker per (city, snapshot)
that downloads ``calendar.csv.gz``, aggregates to per-listing summary stats
(occupancy_365, weekend_premium, lead_time_open) and writes a slim parquet to
``/workspace/shared/airbnb/calendar_v2/``.

Reduce: one merge worker concatenates everything, keeps the latest snapshot per
listing_id for the main analysis parquet, and saves a snapshot_history parquet
with every snapshot retained for the calendar/trajectory stages.
"""
from __future__ import annotations

import re
import time

from dotenv import load_dotenv

from ..config import (
    LISTINGS_CLEAN_PATH, SHARED_CALENDAR, SHARED_LISTINGS, SHARED_ROOT,
    VALIDATION_REPORT_PATH,
)
from ..lib.budget import BudgetTracker
from ..lib.io import read_json, register_src_for_burla, write_json, ensure_dir, input_hash
from ..tasks.image_tasks import (
    DownloadCityArgs, download_and_clean_city,
    DownloadCalendarArgs, download_and_compress_calendar,
    MergeListingsArgs, merge_listings_parquets,
    MergeCalendarArgs, merge_calendar_parquets,
)


def _city_slug(country: str, region: str, city: str, snapshot_date: str) -> str:
    raw = f"{country}__{region}__{city}__{snapshot_date}"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-").lower()


def main() -> None:
    load_dotenv()
    register_src_for_burla()
    report = read_json(VALIDATION_REPORT_PATH)
    if not report:
        raise SystemExit(f"[s01] No validation report at {VALIDATION_REPORT_PATH}; run Stage 0 first.")
    passing = report.get("passing", [])
    if not passing:
        raise SystemExit("[s01] Validation report has zero passing tuples; halting.")

    args_list = [
        DownloadCityArgs(
            city=r["city"],
            country=r["country"],
            region=r["region"],
            snapshot_date=r["snapshot_date"],
            listings_url=r["listings_url"],
            shared_root=SHARED_LISTINGS,
            city_slug=_city_slug(r["country"], r["region"], r["city"], r["snapshot_date"]),
        )
        for r in passing
    ]
    cal_args_list = [
        DownloadCalendarArgs(
            city=r["city"],
            country=r["country"],
            region=r["region"],
            snapshot_date=r["snapshot_date"],
            calendar_url=r.get("calendar_url", ""),
            shared_root=SHARED_CALENDAR,
            city_slug=_city_slug(r["country"], r["region"], r["city"], r["snapshot_date"]),
        )
        for r in passing if r.get("calendar_url") and r.get("calendar_ok", True)
    ]
    h = input_hash([a.city_slug for a in args_list])

    manifest_path = LISTINGS_CLEAN_PATH.with_suffix(".manifest.json")
    existing = read_json(manifest_path)
    if existing and existing.get("ok"):
        if existing.get("input_hash") == h:
            print(f"[s01] manifest matches input_hash={h}; skipping (delete {manifest_path} to force).", flush=True)
        else:
            print(
                f"[s01] manifest exists with input_hash={existing.get('input_hash')} (current={h}); "
                f"reusing existing outputs (delete {manifest_path} to force a full rerun).",
                flush=True,
            )
        return

    from burla import remote_parallel_map

    n_tuples = len(args_list)
    n_workers = min(300, n_tuples)
    print(f"[s01] downloading {n_tuples} (city, snapshot) listings on {n_workers} parallel workers ...", flush=True)
    t0 = time.time()

    with BudgetTracker("s01_listings", n_inputs=n_tuples, func_cpu=1) as bt:
        bt.set_workers(n_workers)
        results: list[dict] = remote_parallel_map(
            download_and_clean_city,
            args_list,
            func_cpu=1,
            func_ram=4,
            max_parallelism=n_workers,
            grow=True,
            spinner=False,
        )
        n_ok = sum(1 for r in results if r.get("ok"))
        n_rows = sum(int(r.get("n_rows", 0)) for r in results if r.get("ok"))
        bt.set_succeeded(n_ok)
        bt.set_failed(n_tuples - n_ok)
        bt.note(rows_per_tuple=n_rows / max(1, n_ok), total_rows=n_rows)
        print(f"[s01]   wrote {n_rows:,} rows from {n_ok}/{n_tuples} tuples to {SHARED_LISTINGS}/", flush=True)
        if n_ok < n_tuples:
            for r in results[:5]:
                if not r.get("ok"):
                    print(f"[s01]   FAIL {r.get('city')}/{r.get('snapshot_date')}: {r.get('error')}", flush=True)

    print(f"[s01] downloading {len(cal_args_list)} calendars in parallel ...", flush=True)
    cal_workers = min(300, max(1, len(cal_args_list)))
    with BudgetTracker("s01_listings", n_inputs=len(cal_args_list), func_cpu=2) as bt:
        bt.set_workers(cal_workers)
        cal_results: list[dict] = remote_parallel_map(
            download_and_compress_calendar,
            cal_args_list,
            func_cpu=2,
            func_ram=8,
            max_parallelism=cal_workers,
            grow=True,
            spinner=False,
        )
        n_cal_ok = sum(1 for r in cal_results if r.get("ok"))
        n_cal_listings = sum(int(r.get("n_listings", 0)) for r in cal_results if r.get("ok"))
        bt.set_succeeded(n_cal_ok)
        bt.set_failed(len(cal_args_list) - n_cal_ok)
        bt.note(total_calendar_listings=n_cal_listings)
        print(
            f"[s01]   wrote calendar summaries for {n_cal_listings:,} listings "
            f"from {n_cal_ok}/{len(cal_args_list)} tuples to {SHARED_CALENDAR}/",
            flush=True,
        )

    print("[s01] reducing per-(city, snapshot) parquets into one ...", flush=True)
    shared_merged = f"{SHARED_ROOT}/listings_clean.parquet"
    shared_history = f"{SHARED_ROOT}/listings_history.parquet"
    [merge_result] = remote_parallel_map(
        merge_listings_parquets,
        [MergeListingsArgs(
            shared_root=SHARED_LISTINGS,
            output_path=shared_merged,
            history_path=shared_history,
        )],
        func_cpu=8,
        func_ram=64,
        max_parallelism=1,
        grow=True,
        spinner=False,
    )
    if not merge_result.get("ok"):
        raise SystemExit(f"[s01] merge failed: {merge_result.get('error')}")

    print("[s01] reducing calendar parquets ...", flush=True)
    shared_calendar_merged = f"{SHARED_ROOT}/calendar_summary.parquet"
    [cal_merge] = remote_parallel_map(
        merge_calendar_parquets,
        [MergeCalendarArgs(
            shared_root=SHARED_CALENDAR,
            output_path=shared_calendar_merged,
        )],
        func_cpu=8,
        func_ram=32,
        max_parallelism=1,
        grow=True,
        spinner=False,
    )
    if not cal_merge.get("ok"):
        print(f"[s01] WARNING calendar merge failed: {cal_merge.get('error')}", flush=True)

    elapsed = time.time() - t0
    print(
        f"[s01]   merged {merge_result['n_rows']:,} latest-per-listing rows "
        f"({merge_result['n_history_rows']:,} history rows) "
        f"from {merge_result['n_files']} files in {elapsed:.1f}s",
        flush=True,
    )

    ensure_dir(LISTINGS_CLEAN_PATH.parent)
    write_json(manifest_path, {
        "input_hash": h,
        "ok": True,
        "shared_path": shared_merged,
        "history_path": shared_history,
        "calendar_path": shared_calendar_merged if cal_merge.get("ok") else "",
        "n_rows": merge_result["n_rows"],
        "n_history_rows": merge_result["n_history_rows"],
        "n_calendar_rows": cal_merge.get("n_rows", 0),
        "n_calendar_listings": cal_merge.get("n_listings", 0),
        "n_cities": merge_result["n_cities"],
        "n_with_picture_url": merge_result.get("n_with_picture_url", 0),
        "schema": merge_result.get("schema", []),
        "sample_rows": merge_result.get("sample_rows", []),
        "n_workers": n_workers,
        "elapsed_seconds": elapsed,
        "completed_at": time.time(),
    })
    print(f"[s01] DONE. Manifest at {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
