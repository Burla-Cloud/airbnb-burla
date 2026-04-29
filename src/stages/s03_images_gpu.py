"""Stage 3: GPU object detection on the top-K images per CLIP axis.

Reads /workspace/shared/airbnb/images_cpu.parquet (~25M rows). On Burla we pick
the top-K rows per ``TOP_N_PER_AXIS`` axis (e.g. top 20k images by
``clip_tv_above_fireplace``), union and dedupe, then run YOLOv8 in parallel on
A100_40G GPUs (``GPU_MAX_PARALLELISM`` workers, batches of ``GPU_BATCH_SIZE``).
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

from dotenv import load_dotenv

import numpy as _np  # noqa: F401
import pandas as _pd  # noqa: F401
import pyarrow as _pa  # noqa: F401
import pyarrow.parquet as _pq  # noqa: F401

from ..config import (
    GPU_BATCH_SIZE, GPU_MAX_PARALLELISM,
    IMAGES_GPU_PATH, SHARED_IMAGES_GPU, SHARED_ROOT,
    TOP_N_PER_AXIS,
)
from ..lib.budget import BudgetTracker
from ..lib.io import ensure_dir, register_src_for_burla, write_json
from ..tasks.image_tasks import (
    GpuImageBatchArgs, gpu_detect_image_batch,
    MergeImagesGpuArgs, merge_images_gpu,
)


import traceback
import pandas as pd
import pyarrow as pa  # noqa: F401
import pyarrow.parquet as pq
import os, glob

@dataclass
class TopKImagesArgs:
    images_cpu_path: str
    top_n_per_axis: dict


def select_top_k_images(args: TopKImagesArgs) -> dict:
    """Run on Burla. Pick the top-K rows per CLIP axis from the CPU images parquet,
    union and dedupe by (listing_id, image_idx), and return them as a list of dicts."""
    out = {"ok": False, "rows": [], "per_axis_counts": {},
           "n_total_cpu_rows": 0, "n_selected": 0, "error": None}
    try:

        cols = ["listing_id", "image_idx", "image_url", "download_ok", "city_slug"]
        score_cols = [k for k in args.top_n_per_axis.keys()]
        table = pq.read_table(args.images_cpu_path, columns=cols + score_cols)
        df = table.to_pandas()
        df = df[df["download_ok"].astype(bool)]
        n_total = int(len(df))

        selected = []
        per_axis_counts = {}
        for axis, k in args.top_n_per_axis.items():
            sub = df.nlargest(int(k), axis)
            per_axis_counts[axis] = int(len(sub))
            selected.append(sub)

        union = pd.concat(selected, ignore_index=True).drop_duplicates(
            subset=["listing_id", "image_idx"]
        )
        union = union[["listing_id", "image_idx", "image_url", "city_slug"]]
        union["listing_id"] = union["listing_id"].astype(int)
        union["image_idx"] = union["image_idx"].astype(int)
        union["image_url"] = union["image_url"].astype(str)
        union["city_slug"] = union["city_slug"].fillna("").astype(str)

        out["ok"] = True
        out["n_total_cpu_rows"] = n_total
        out["n_selected"] = int(len(union))
        out["per_axis_counts"] = per_axis_counts
        out["rows"] = union.to_dict("records")
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        out["traceback"] = traceback.format_exc()[:2000]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=0,
                        help="Cap selected image count (0 = use TOP_N_PER_AXIS as-is)")
    args = parser.parse_args()

    load_dotenv()
    register_src_for_burla()
    from burla import remote_parallel_map

    images_cpu_path = f"{SHARED_ROOT}/images_cpu.parquet"
    print(f"[s03] selecting top-K images per axis from {images_cpu_path} ...", flush=True)
    [picked] = remote_parallel_map(
        select_top_k_images,
        [TopKImagesArgs(images_cpu_path=images_cpu_path,
                        top_n_per_axis=TOP_N_PER_AXIS)],
        func_cpu=8, func_ram=64, max_parallelism=1, grow=True, spinner=False,
    )
    if not picked.get("ok"):
        print(f"[s03] select_top_k_images failed: {picked.get('error')}", flush=True)
        if picked.get("traceback"):
            print(picked["traceback"], flush=True)
        raise SystemExit("[s03] cannot continue without top-K image selection")
    rows = picked["rows"]
    if args.max_images and len(rows) > args.max_images:
        rows = rows[: args.max_images]

    n_images = len(rows)
    print(f"[s03]   selected {n_images:,} images "
          f"(of {picked['n_total_cpu_rows']:,} CPU-scored)", flush=True)
    for axis, n in picked["per_axis_counts"].items():
        print(f"[s03]     {axis}: {n:,}", flush=True)

    batches: list[GpuImageBatchArgs] = []
    for i in range(0, n_images, GPU_BATCH_SIZE):
        batches.append(GpuImageBatchArgs(
            batch_id=i // GPU_BATCH_SIZE,
            rows=rows[i: i + GPU_BATCH_SIZE],
            output_root=SHARED_IMAGES_GPU,
        ))
    n_workers = min(GPU_MAX_PARALLELISM, len(batches))
    print(f"[s03]   built {len(batches):,} batches of {GPU_BATCH_SIZE}, "
          f"max {n_workers} A100 workers", flush=True)

    @dataclass
    class _CountGpuShardsArgs:
        output_root: str

    def _count_gpu_shards(a: _CountGpuShardsArgs) -> int:
        return len(glob.glob(os.path.join(a.output_root, "batch_*.parquet")))

    [n_existing_shards] = remote_parallel_map(
        _count_gpu_shards,
        [_CountGpuShardsArgs(output_root=SHARED_IMAGES_GPU)],
        func_cpu=1, func_ram=1, max_parallelism=1, grow=True, spinner=False,
    )
    print(f"[s03]   {n_existing_shards:,} of {len(batches):,} GPU batch parquets already on shared FS", flush=True)
    t0 = time.time()
    if n_existing_shards >= len(batches):
        print("[s03]   all batches already complete; skipping remote_parallel_map.", flush=True)
        results: list[dict] = []
        n_ok = n_images
        n_failed = 0
    else:
        with BudgetTracker("s03_images_gpu", n_inputs=n_images, func_cpu=8) as bt:
            bt.set_workers(n_workers)
            results = remote_parallel_map(
                gpu_detect_image_batch,
                batches,
                func_cpu=8,
                func_ram=64,
                func_gpu="A100_40G",
                max_parallelism=n_workers,
                grow=True,
                spinner=False,
            )
            n_ok = sum(int(r.get("n_ok", 0)) for r in results)
            n_failed = sum(int(r.get("n_failed", 0)) for r in results)
            bt.set_succeeded(n_ok)
            bt.set_failed(n_failed)
            bt.note(success_rate=n_ok / max(1, n_images))

    elapsed = time.time() - t0
    print(f"[s03]   {n_ok:,}/{n_images:,} images detected ({n_ok/max(1,n_images):.2%}) "
          f"in {elapsed:.1f}s", flush=True)

    print("[s03] reducing batch parquets ...", flush=True)
    shared_merged = f"{SHARED_ROOT}/images_gpu.parquet"
    [merge] = remote_parallel_map(
        merge_images_gpu,
        [MergeImagesGpuArgs(shared_root=SHARED_IMAGES_GPU, output_path=shared_merged)],
        func_cpu=8, func_ram=64, max_parallelism=1, grow=True, spinner=False,
    )
    if not merge.get("ok"):
        raise SystemExit(f"[s03] merge failed: {merge.get('error')}")
    print(f"[s03]   merged {merge['n_rows']:,} rows from {merge['n_files']:,} batches", flush=True)

    ensure_dir(IMAGES_GPU_PATH.parent)
    manifest_path = IMAGES_GPU_PATH.with_suffix(".manifest.json")
    write_json(manifest_path, {
        "ok": True,
        "shared_path": shared_merged,
        "n_rows": merge["n_rows"],
        "n_listings": merge["n_listings"],
        "per_axis_counts": picked["per_axis_counts"],
        "n_workers": n_workers,
        "elapsed_seconds": elapsed,
        "completed_at": time.time(),
    })
    print(f"[s03] DONE. Manifest at {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
