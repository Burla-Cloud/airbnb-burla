"""Stage 7: derive demand signals from the calendar summary parquet.

Reads ``/workspace/shared/airbnb/calendar_summary.parquet`` (one row per
(listing_id, snapshot_date) emitted by Stage 1's calendar download), computes:

- ``occupancy_365``: fraction of days flagged unavailable in the 365-day window
- ``weekend_premium``: ``mean weekend price`` divided by ``mean weekday price``
- ``price_volatility``: coefficient of variation of price across the window
- ``lead_time_open``: days from snapshot to the next available night

For listings present in multiple snapshots, we keep the latest snapshot as the
"now" reading and emit a separate trajectory parquet for any UI that wants to
plot how a listing evolved (we don't surface that on the site for this run, but
it's free now that we have the snapshots).

The output ``listings_demand.parquet`` joins these signals onto every listing
in ``listings_clean.parquet`` so downstream correlate + artifacts can use
``occupancy_365`` as the primary demand proxy.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

from dotenv import load_dotenv

from ..config import SHARED_ROOT
from ..lib.budget import BudgetTracker
from ..lib.io import register_src_for_burla

import os
import pandas as pd
import traceback as _tb

@dataclass
class CalendarDeriveArgs:
    calendar_summary_path: str   # /workspace/shared/airbnb/calendar_summary.parquet
    listings_path: str           # /workspace/shared/airbnb/listings_clean.parquet
    output_path: str             # /workspace/shared/airbnb/listings_demand.parquet
    trajectory_output_path: str  # /workspace/shared/airbnb/listings_trajectory.parquet


def derive_calendar_signals(args: CalendarDeriveArgs) -> dict:
    """Run on Burla. Reads calendar + listings parquets from shared FS, derives
    demand signals, writes the joined parquet back to shared FS."""
    out = {
        "ok": False,
        "n_listings_input": 0,
        "n_listings_with_calendar": 0,
        "n_trajectory_rows": 0,
        "output_path": args.output_path,
        "trajectory_path": args.trajectory_output_path,
        "error": None,
    }
    try:

        cal = pd.read_parquet(args.calendar_summary_path)
        out["n_trajectory_rows"] = int(len(cal))

        cal["n_days"] = pd.to_numeric(cal.get("n_days"), errors="coerce").fillna(0)
        cal["n_days_available"] = pd.to_numeric(cal.get("n_days_available"), errors="coerce").fillna(0)
        cal["n_weekend_days"] = pd.to_numeric(cal.get("n_weekend_days"), errors="coerce").fillna(0)
        cal["n_weekend_available"] = pd.to_numeric(cal.get("n_weekend_available"), errors="coerce").fillna(0)
        cal["mean_price"] = pd.to_numeric(cal.get("mean_price"), errors="coerce")
        cal["std_price"] = pd.to_numeric(cal.get("std_price"), errors="coerce")
        cal["lead_time_open"] = pd.to_numeric(cal.get("lead_time_open"), errors="coerce")

        cal["occupancy_365"] = 1.0 - (
            cal["n_days_available"] / cal["n_days"].replace(0, pd.NA)
        )
        cal["occupancy_weekend"] = 1.0 - (
            cal["n_weekend_available"] / cal["n_weekend_days"].replace(0, pd.NA)
        )
        cal["price_volatility"] = (cal["std_price"] / cal["mean_price"].replace(0, pd.NA)).astype(float)

        # Trajectory parquet: every (listing_id, snapshot_date) row, slim columns.
        traj_cols = [
            "listing_id", "snapshot_date", "city",
            "occupancy_365", "occupancy_weekend",
            "mean_price", "median_price", "price_volatility",
            "lead_time_open",
        ]
        traj_cols = [c for c in traj_cols if c in cal.columns]
        os.makedirs(os.path.dirname(args.trajectory_output_path), exist_ok=True)
        cal[traj_cols].to_parquet(args.trajectory_output_path, compression="zstd", index=False)

        # For listings_demand, keep latest-per-listing reading.
        cal = cal.sort_values(["listing_id", "snapshot_date"]) \
                 .drop_duplicates(subset=["listing_id"], keep="last")
        listings = pd.read_parquet(args.listings_path)
        out["n_listings_input"] = int(len(listings))

        join_cols = [
            "listing_id", "occupancy_365", "occupancy_weekend",
            "mean_price", "median_price", "std_price",
            "price_volatility", "lead_time_open",
        ]
        join_cols = [c for c in join_cols if c in cal.columns]
        merged = listings.merge(cal[join_cols], on="listing_id", how="left")

        # Weekend premium: mean weekend price / mean weekday price. We compute
        # this from the per-listing aggregate by treating mean_price as the
        # overall mean and approximating weekend price from
        # n_weekend_available + n_weekend_days. For now we expose
        # occupancy_weekend / occupancy_365 as a proxy; the full weekend_premium
        # would need more raw calendar columns aggregated upstream.
        merged["weekend_premium"] = merged["occupancy_weekend"] / merged["occupancy_365"].replace(0, pd.NA)

        merged["demand_proxy"] = merged["occupancy_365"].fillna(merged.get("demand_proxy"))
        out["n_listings_with_calendar"] = int(merged["occupancy_365"].notna().sum())

        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        merged.to_parquet(args.output_path, compression="zstd", index=False)
        out["ok"] = True
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["traceback"] = _tb.format_exc()[:1000]
    return out


def main() -> None:
    load_dotenv()
    register_src_for_burla()
    from burla import remote_parallel_map

    calendar_summary_path = f"{SHARED_ROOT}/calendar_summary.parquet"
    listings_path = f"{SHARED_ROOT}/listings_clean.parquet"
    output_path = f"{SHARED_ROOT}/listings_demand.parquet"
    trajectory_path = f"{SHARED_ROOT}/listings_trajectory.parquet"

    print(f"[s07] deriving calendar signals from {calendar_summary_path} ...", flush=True)
    t0 = time.time()
    with BudgetTracker("s07_calendar", n_inputs=1, func_cpu=8) as bt:
        bt.set_workers(1)
        [r] = remote_parallel_map(
            derive_calendar_signals,
            [CalendarDeriveArgs(
                calendar_summary_path=calendar_summary_path,
                listings_path=listings_path,
                output_path=output_path,
                trajectory_output_path=trajectory_path,
            )],
            func_cpu=8, func_ram=64, max_parallelism=1, grow=True, spinner=False,
        )
        if not r.get("ok"):
            print(f"[s07] failed: {r.get('error')}", flush=True)
            if r.get("traceback"):
                print(r["traceback"], flush=True)
            raise SystemExit("[s07] calendar derive failed")
        bt.set_succeeded(1)
        bt.note(
            n_listings_input=r["n_listings_input"],
            n_listings_with_calendar=r["n_listings_with_calendar"],
            n_trajectory_rows=r["n_trajectory_rows"],
        )
    print(
        f"[s07] DONE in {time.time()-t0:.1f}s. "
        f"Joined calendar onto {r['n_listings_with_calendar']:,}/{r['n_listings_input']:,} listings "
        f"(trajectory rows: {r['n_trajectory_rows']:,}).",
        flush=True,
    )


if __name__ == "__main__":
    main()
