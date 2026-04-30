"""Stage 2b: download every photo + CLIP-score + brightness + edges.

Reads /workspace/shared/airbnb/photo_manifest.parquet on Burla, batches the
rows, dispatches ``cpu_score_image_batch`` workers (each handles ~CPU_IMAGE_BATCH_SIZE
images so CLIP only loads once per worker), then reduces.

Run with --sample N to score N images first; the sample run prints a cost
projection so the user can halt before the full ~25-35M run.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from dotenv import load_dotenv

from ..config import (
    CPU_IMAGE_BATCH_SIZE, CPU_IMAGE_MAX_PARALLELISM,
    IMAGES_CPU_PATH, SHARED_IMAGES_CPU, SHARED_ROOT,
)
from ..lib.budget import BudgetTracker, estimate_burla_cpu_usd
from ..lib.io import ensure_dir, register_src_for_burla, write_json
from ..tasks.image_tasks import (
    CpuImageBatchArgs, cpu_score_image_batch,
    MergeImagesCpuArgs, merge_images_cpu,
)


import pyarrow.parquet as pq
import os, glob

@dataclass
class CountManifestArgs:
    photo_manifest_path: str


def count_manifest_rows(args: CountManifestArgs) -> dict:
    """Run on Burla. Counts rows in the photo manifest parquet on shared FS."""
    n = pq.read_metadata(args.photo_manifest_path).num_rows
    return {"n_rows": int(n)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=0,
                        help="CLIP-score this many images as a sanity check (0 = full run)")
    args = parser.parse_args()

    load_dotenv()
    register_src_for_burla()
    from burla import remote_parallel_map

    photo_manifest = f"{SHARED_ROOT}/photo_manifest.parquet"
    print(f"[s02b] counting rows in {photo_manifest} ...", flush=True)

    from ..config import PHOTO_MANIFEST_PATH
    from ..lib.io import read_json
    local_manifest = read_json(PHOTO_MANIFEST_PATH.with_suffix(".manifest.json"))
    if local_manifest and local_manifest.get("n_rows"):
        n_rows = int(local_manifest["n_rows"])
        print(f"[s02b]   read row count from local manifest: {n_rows:,}", flush=True)
    else:
        [count] = remote_parallel_map(
            count_manifest_rows,
            [CountManifestArgs(photo_manifest_path=photo_manifest)],
            func_cpu=2, func_ram=4, max_parallelism=1, grow=True, spinner=False,
        )
        n_rows = count["n_rows"]
    total_rows = n_rows
    if args.sample:
        n_rows = min(n_rows, args.sample)
    print(f"[s02b]   manifest has {total_rows:,} rows; processing {n_rows:,}", flush=True)

    output_root = SHARED_IMAGES_CPU + ("_sample" if args.sample else "")
    batches: list[CpuImageBatchArgs] = []
    for batch_id, start in enumerate(range(0, n_rows, CPU_IMAGE_BATCH_SIZE)):
        end = min(start + CPU_IMAGE_BATCH_SIZE, n_rows)
        batches.append(CpuImageBatchArgs(
            batch_id=batch_id,
            photo_manifest_path=photo_manifest,
            row_start=start,
            row_end=end,
            output_root=output_root,
        ))
    n_workers = min(CPU_IMAGE_MAX_PARALLELISM, len(batches))
    print(f"[s02b]   built {len(batches):,} batches of {CPU_IMAGE_BATCH_SIZE} images, "
          f"max {n_workers} parallel workers", flush=True)

    @dataclass
    class CountShardsArgs:
        output_root: str

    def count_shards(a: CountShardsArgs) -> int:
        return len(glob.glob(os.path.join(a.output_root, "batch_*.parquet")))

    [n_shards] = remote_parallel_map(
        count_shards,
        [CountShardsArgs(output_root=output_root)],
        func_cpu=1, func_ram=1, max_parallelism=1, grow=True, spinner=False,
    )
    print(f"[s02b]   {n_shards:,} of {len(batches):,} batch parquets already on shared FS", flush=True)
    t0 = time.time()
    n_ok = n_failed = n_seen = 0
    if n_shards >= len(batches):
        print("[s02b]   all batches already complete; skipping remote_parallel_map.", flush=True)
        results: list[dict] = []
        n_seen = n_rows
        n_ok = n_rows
    else:
        with BudgetTracker("s02b_images_cpu", n_inputs=n_rows, func_cpu=1) as bt:
            bt.set_workers(n_workers)
            results = remote_parallel_map(
                cpu_score_image_batch,
                batches,
                func_cpu=1,
                func_ram=4,
                max_parallelism=n_workers,
                grow=True,
                spinner=False,
            )
            n_ok = sum(int(r.get("n_ok", 0)) for r in results)
            n_failed = sum(int(r.get("n_failed", 0)) for r in results)
            n_seen = sum(int(r.get("n_inputs", 0)) for r in results)
            errs = [r.get("error") for r in results if r.get("error")]
            if errs:
                print(f"[s02b]   {len(errs)} batches errored. first 3:", flush=True)
                for e in errs[:3]:
                    print(f"[s02b]     {e}", flush=True)
                tracebacks = [r.get("traceback") for r in results if r.get("traceback")]
                if tracebacks:
                    print(f"[s02b]   first traceback:\n{tracebacks[0]}", flush=True)
            bt.set_succeeded(n_ok)
            bt.set_failed(n_failed)
            bt.note(success_rate=n_ok / max(1, n_seen))

    elapsed = time.time() - t0
    print(f"[s02b]   {n_ok:,}/{n_seen:,} images scored ({n_ok/max(1, n_seen):.2%}) "
          f"in {elapsed:.1f}s", flush=True)

    if args.sample:
        full_rows = count["n_rows"]
        cpu_minutes_per_row = (n_workers * elapsed / 60.0) / max(1, n_seen)
        full_cpu_minutes = cpu_minutes_per_row * full_rows
        full_usd = full_cpu_minutes * 0.001
        full_minutes = full_cpu_minutes / max(1, n_workers)
        from ..config import OUTPUT_DIR
        ensure_dir(OUTPUT_DIR)
        write_json(OUTPUT_DIR / "stage02b_sample_report.json", {
            "sample_n": args.sample,
            "n_seen": n_seen, "n_ok": n_ok, "n_failed": n_failed,
            "elapsed_seconds": elapsed,
            "projected_full_rows": full_rows,
            "projected_full_minutes": full_minutes,
            "projected_full_usd_estimate": full_usd,
            "completed_at": time.time(),
        })
        print(f"[s02b] sample done. Full run projection: {full_rows:,} rows -> "
              f"~{full_minutes:.1f} min, ~${full_usd:.2f} (rough)", flush=True)
        return

    print("[s02b] reducing batch parquets ...", flush=True)
    shared_merged = f"{SHARED_ROOT}/images_cpu.parquet"
    [merge] = remote_parallel_map(
        merge_images_cpu,
        [MergeImagesCpuArgs(shared_root=output_root, output_path=shared_merged)],
        func_cpu=8, func_ram=64, max_parallelism=1, grow=True, spinner=False,
    )
    if not merge.get("ok"):
        raise SystemExit(f"[s02b] merge failed: {merge.get('error')}")
    print(f"[s02b]   merged {merge['n_rows']:,} image rows from {merge['n_files']:,} batches", flush=True)

    ensure_dir(IMAGES_CPU_PATH.parent)
    manifest_path = IMAGES_CPU_PATH.with_suffix(".manifest.json")
    write_json(manifest_path, {
        "ok": True,
        "shared_path": shared_merged,
        "n_rows": merge["n_rows"],
        "n_listings": merge["n_listings"],
        "n_download_ok": merge["n_download_ok"],
        "elapsed_seconds": elapsed,
        "n_workers": n_workers,
        "completed_at": time.time(),
    })
    print(f"[s02b] DONE. Manifest at {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
