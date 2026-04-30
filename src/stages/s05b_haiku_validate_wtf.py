"""Stage 5b: take the top-K CLIP-flagged "WTF" photos and rerank with Claude Haiku.

Reads ``/workspace/shared/airbnb/images_cpu.parquet``, picks the top
``WTF_TOP_K_FOR_HAIKU`` photos by ``max(clip_wtf_*)`` (deduped by
``listing_id``), batches them into ``WTF_HAIKU_BATCH_SIZE`` photos per Haiku
call, runs Haiku in parallel via ``remote_parallel_map`` with up to
``WTF_HAIKU_MAX_PARALLELISM`` workers.

The Haiku call uses a strict JSON schema. We then drop clusters smaller than
``WTF_MIN_LABEL_CLUSTER_SIZE``, keep ``WTF_TOP_PHOTOS_PER_CLUSTER`` per cluster
ranked by Haiku score, and write ``wtf_haiku.parquet`` to the shared FS so
``s05_bootstrap_correlations`` can use the absurd-photo flag as a bucket
variable in the correlation analysis.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv

# Hoist for cloudpickle bundling.
import numpy as _np  # noqa: F401
import pandas as _pd  # noqa: F401
import pyarrow as _pa  # noqa: F401
import pyarrow.parquet as _pq  # noqa: F401

from ..config import (
    ANTHROPIC_MAX_TOKENS, ANTHROPIC_MODEL, SHARED_ROOT, SHARED_WTF_HAIKU,
    WTF_CLIP_PROMPT_KEYS, WTF_HAIKU_BATCH_SIZE, WTF_HAIKU_MAX_PARALLELISM,
    WTF_MIN_LABEL_CLUSTER_SIZE, WTF_TOP_K_FOR_HAIKU, WTF_TOP_PHOTOS_PER_CLUSTER,
)
from ..lib.budget import BudgetTracker
from ..lib.io import register_src_for_burla
from ..tasks.wtf_tasks import (
    WtfHaikuBatchArgs, WtfMergeArgs,
    wtf_haiku_score_batch, merge_wtf_haiku,
)


import pandas as pd
import traceback as _tb

@dataclass
class SelectWtfArgs:
    images_cpu_path: str
    top_k: int
    clip_keys: list


def select_wtf_candidates(args: SelectWtfArgs) -> dict:
    """Run on Burla. Read images_cpu.parquet, compute clip_max across all
    wtf prompt keys, dedupe by listing_id keeping the highest-scoring photo,
    return the top-K rows."""
    out = {"ok": False, "rows": [], "n_total": 0, "n_selected": 0, "error": None}
    try:
        wanted_cols = ["listing_id", "image_idx", "image_url", "download_ok"]
        wanted_cols += [f"clip_{k}" for k in args.clip_keys]
        df = pd.read_parquet(args.images_cpu_path, columns=wanted_cols)
        df = df[df["download_ok"].astype(bool)]
        out["n_total"] = int(len(df))

        score_cols = [f"clip_{k}" for k in args.clip_keys]
        df["clip_max"] = df[score_cols].max(axis=1)
        df = df.sort_values("clip_max", ascending=False)
        df = df.drop_duplicates(subset=["listing_id"], keep="first")
        df = df.head(int(args.top_k)).reset_index(drop=True)

        df["image_id"] = df.index.astype(int)
        rows = df[["image_id", "listing_id", "image_idx", "image_url",
                   "clip_max"]].to_dict("records")
        out["ok"] = True
        out["rows"] = rows
        out["n_selected"] = len(rows)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["traceback"] = _tb.format_exc()[:1000]
    return out


def main() -> None:
    load_dotenv()
    register_src_for_burla()
    from burla import remote_parallel_map

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("[s05b] ANTHROPIC_API_KEY not set; cannot run Haiku rerank")

    cpu_shared = f"{SHARED_ROOT}/images_cpu.parquet"
    print(f"[s05b] selecting top-{WTF_TOP_K_FOR_HAIKU:,} WTF candidates from {cpu_shared} ...", flush=True)
    [picked] = remote_parallel_map(
        select_wtf_candidates,
        [SelectWtfArgs(
            images_cpu_path=cpu_shared,
            top_k=WTF_TOP_K_FOR_HAIKU,
            clip_keys=list(WTF_CLIP_PROMPT_KEYS),
        )],
        func_cpu=8, func_ram=64, max_parallelism=1, grow=True, spinner=False,
    )
    if not picked.get("ok"):
        print(f"[s05b] select failed: {picked.get('error')}", flush=True)
        if picked.get("traceback"):
            print(picked["traceback"], flush=True)
        raise SystemExit("[s05b] cannot continue without WTF candidates")
    rows = picked["rows"]
    print(f"[s05b]   selected {len(rows):,} of {picked['n_total']:,} CPU images", flush=True)

    if not rows:
        print("[s05b] no candidates; writing empty wtf_haiku.parquet and exiting", flush=True)
        return

    batches: list[WtfHaikuBatchArgs] = []
    for i in range(0, len(rows), WTF_HAIKU_BATCH_SIZE):
        batches.append(WtfHaikuBatchArgs(
            batch_id=i // WTF_HAIKU_BATCH_SIZE,
            rows=rows[i: i + WTF_HAIKU_BATCH_SIZE],
            output_root=SHARED_WTF_HAIKU,
            anthropic_api_key=api_key,
            model=ANTHROPIC_MODEL,
            max_tokens=ANTHROPIC_MAX_TOKENS,
        ))
    n_workers = min(WTF_HAIKU_MAX_PARALLELISM, len(batches))
    print(f"[s05b]   {len(batches):,} batches of {WTF_HAIKU_BATCH_SIZE}, "
          f"max {n_workers} Haiku workers", flush=True)

    t0 = time.time()
    with BudgetTracker("s07_wtf_haiku", n_inputs=len(rows), func_cpu=2) as bt:
        bt.set_workers(n_workers)
        results: list[dict] = remote_parallel_map(
            wtf_haiku_score_batch, batches,
            func_cpu=2, func_ram=8,
            max_parallelism=n_workers, grow=True, spinner=False,
        )
        n_ok = sum(int(r.get("n_ok", 0)) for r in results)
        n_failed = sum(int(r.get("n_failed", 0)) for r in results)
        bt.set_succeeded(n_ok)
        bt.set_failed(n_failed)
        bt.note(success_rate=n_ok / max(1, len(rows)))

    print(f"[s05b]   haiku rerank: {n_ok:,}/{len(rows):,} scored "
          f"in {time.time()-t0:.1f}s", flush=True)

    output_path = f"{SHARED_ROOT}/wtf_haiku.parquet"
    [merge] = remote_parallel_map(
        merge_wtf_haiku,
        [WtfMergeArgs(
            shared_root=SHARED_WTF_HAIKU,
            output_path=output_path,
            min_cluster_size=WTF_MIN_LABEL_CLUSTER_SIZE,
            top_per_cluster=WTF_TOP_PHOTOS_PER_CLUSTER,
        )],
        func_cpu=8, func_ram=64, max_parallelism=1, grow=True, spinner=False,
    )
    if not merge.get("ok"):
        raise SystemExit(f"[s05b] merge failed: {merge.get('error')}")
    print(
        f"[s05b]   merged: {merge['n_absurd']:,} absurd of {merge['n_rows']:,} scored, "
        f"{merge['n_clusters']:,} clusters >= {WTF_MIN_LABEL_CLUSTER_SIZE}, "
        f"kept {merge['n_kept']:,}",
        flush=True,
    )
    print(f"[s05b] DONE. wtf_haiku.parquet at {output_path}", flush=True)


if __name__ == "__main__":
    main()
