"""Stage 0: validate every Inside Airbnb city in parallel on Burla.

Reads the index page locally, fans out one ``validate_city`` task per city,
aggregates results into ``data/outputs/validation_report.json``, and prints
the pass/fail breakdown so the user can spot bad cities before Stage 1.
"""
from __future__ import annotations

import os
import time

from dotenv import load_dotenv

from ..config import (
    MAX_SNAPSHOTS_PER_CITY, MIN_LISTINGS_PER_CITY, MAX_SAMPLE_IMAGE_FAIL_RATIO,
    VALIDATION_REPORT_PATH,
)
from ..lib.budget import BudgetTracker
from ..lib.inside_airbnb import discover_all_cities
from ..lib.io import register_src_for_burla, write_json
from ..tasks.image_tasks import ValidateCityArgs, validate_city


def main() -> None:
    load_dotenv()
    register_src_for_burla()
    if os.environ.get("FORCE_S00") != "1" and os.path.exists(VALIDATION_REPORT_PATH):
        try:
            from ..lib.io import read_json
            existing = read_json(VALIDATION_REPORT_PATH)
            if existing and existing.get("n_passing", 0) > 0:
                print(
                    f"[s00] reusing existing validation report at {VALIDATION_REPORT_PATH} "
                    f"(n_passing={existing.get('n_passing')}). Set FORCE_S00=1 to rerun.",
                    flush=True,
                )
                return
        except Exception:
            pass
    print("[s00] discovering Inside Airbnb (city, snapshot) tuples ...", flush=True)
    cities = discover_all_cities(max_snapshots_per_city=MAX_SNAPSHOTS_PER_CITY)
    unique_cities = len({(c.country, c.region, c.city) for c in cities})
    print(f"[s00] {len(cities)} (city, snapshot) tuples across {unique_cities} unique cities", flush=True)

    args_list = [
        ValidateCityArgs(
            city=c.city, country=c.country, region=c.region,
            snapshot_date=c.snapshot_date,
            listings_url=c.listings_url, reviews_url=c.reviews_url,
            calendar_url=c.calendar_url,
        )
        for c in cities
    ]

    from burla import remote_parallel_map

    n_workers = min(300, len(args_list))
    print(f"[s00] launching remote_parallel_map across {n_workers} workers ...", flush=True)
    started = time.time()

    with BudgetTracker("s00_validate", n_inputs=len(args_list), func_cpu=1) as bt:
        results: list[dict] = remote_parallel_map(
            validate_city,
            args_list,
            func_cpu=1,
            func_ram=2,
            max_parallelism=n_workers,
            grow=True,
            spinner=False,
        )
        elapsed = time.time() - started
        print(f"[s00]   {len(results)}/{len(args_list)} cities returned "
              f"({elapsed:.1f}s elapsed)", flush=True)
        bt.set_workers(n_workers)
        bt.set_succeeded(sum(1 for r in results if not r.get("error")))
        bt.set_failed(sum(1 for r in results if r.get("error")))

    cities_passing: list[dict] = []
    cities_failing: list[dict] = []
    for r in results:
        n = r.get("n_listings", 0)
        ratio = r.get("sample_image_ok_ratio", 0.0)
        passes = (
            r.get("listings_ok") and
            n >= MIN_LISTINGS_PER_CITY and
            (1.0 - ratio) <= MAX_SAMPLE_IMAGE_FAIL_RATIO
        )
        target = cities_passing if passes else cities_failing
        target.append(r)

    report = {
        "generated_at": time.time(),
        "n_total": len(results),
        "n_passing": len(cities_passing),
        "n_failing": len(cities_failing),
        "n_unique_cities": unique_cities,
        "max_snapshots_per_city": MAX_SNAPSHOTS_PER_CITY,
        "min_listings_per_city": MIN_LISTINGS_PER_CITY,
        "max_sample_image_fail_ratio": MAX_SAMPLE_IMAGE_FAIL_RATIO,
        "passing": sorted(cities_passing, key=lambda r: -r.get("n_listings", 0)),
        "failing": cities_failing,
    }
    write_json(VALIDATION_REPORT_PATH, report)

    print(f"[s00] DONE. {len(cities_passing)}/{len(results)} (city, snapshot) tuples passed.", flush=True)
    print(f"[s00] Report: {VALIDATION_REPORT_PATH}", flush=True)
    if cities_failing:
        print("[s00] Failing tuples (first 10):", flush=True)
        for r in cities_failing[:10]:
            why = r.get("error") or f"n_listings={r.get('n_listings')} img_ok={r.get('sample_image_ok_ratio')}"
            print(f"[s00]   {r['country']}/{r['city']}/{r.get('snapshot_date')} -> {why}", flush=True)


if __name__ == "__main__":
    main()
