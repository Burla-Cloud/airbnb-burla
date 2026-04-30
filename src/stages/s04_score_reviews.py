"""Stage 4: 3-tier review scoring funnel (heuristic -> embed/cluster -> Claude).

Tier 1: heuristic on every review (~50M rows). Output: top 200k by tier1_score.
Tier 2: embed top 200k with sentence-transformers, MiniBatch-KMeans 30 clusters.
        Output: top 10k by combined_score (tier1 + cluster rarity boost).
Tier 3: Claude Haiku categorizes each of the top 10k. Output: reviews_scored.parquet.

All bulk data lives on /workspace/shared. We pull only a small manifest locally.
"""
from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass

from dotenv import load_dotenv

from ..config import (
    ANTHROPIC_MAX_TOKENS, ANTHROPIC_MODEL,
    REVIEW_HEURISTIC_KEYWORDS, REVIEW_HUMOR_CATEGORIES,
    REVIEW_TIER1_BATCH_SIZE, REVIEW_TIER1_MAX_PARALLELISM,
    REVIEW_TIER2_NUM_CLUSTERS, REVIEW_TIER2_TOP_K,
    REVIEW_TIER3_BATCH_SIZE, REVIEW_TIER3_MAX_PARALLELISM, REVIEW_TIER3_TOP_K,
    REVIEWS_SCORED_PATH, SHARED_REVIEWS, SHARED_REVIEWS_TIER1,
    SHARED_REVIEWS_TIER2, SHARED_REVIEWS_TIER3, SHARED_ROOT,
    VALIDATION_REPORT_PATH,
)
from ..lib.budget import BudgetTracker
from ..lib.io import ensure_dir, read_json, register_src_for_burla, write_json
from ..tasks.review_tasks import (
    ClaudeBatchArgs, ClusterTier2Args, EmbedTier2Args,
    IngestReviewsArgs, MergeClaudeArgs, MergeReviewsArgs,
    RechunkReviewsArgs, Tier1HeuristicArgs, TopKHeuristicArgs,
    claude_score_batch, cluster_and_rerank_tier2, embed_reviews_batch,
    heuristic_score_batch, ingest_reviews_for_city, merge_and_top_k_tier1,
    merge_claude, merge_reviews, rechunk_reviews_for_tier1,
)


import pyarrow.parquet as pq
import pandas as pd

def _city_slug(country: str, region: str, city: str, snapshot_date: str) -> str:
    raw = f"{country}__{region}__{city}__{snapshot_date}"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-").lower()


@dataclass
class CountParquetArgs:
    path: str


def count_parquet_rows(args: CountParquetArgs) -> dict:
    """Top-level so Burla can pickle it. Counts rows in a parquet on shared FS."""
    return {"n": int(pq.read_metadata(args.path).num_rows)}


@dataclass
class ReadTier3InputArgs:
    path: str
    cap: int


def read_tier3_input(args: ReadTier3InputArgs) -> dict:
    """Top-level so Burla can pickle it. Reads tier-3 input parquet, returns rows as list of dicts."""
    df = pd.read_parquet(args.path, columns=["review_id", "comments"])
    if args.cap and len(df) > args.cap:
        df = df.iloc[: args.cap]
    return {"rows": df.to_dict("records")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Reuse existing /workspace/shared/airbnb/reviews_raw.parquet")
    parser.add_argument("--skip-tier1", action="store_true")
    parser.add_argument("--skip-tier2", action="store_true")
    parser.add_argument("--skip-tier3", action="store_true")
    parser.add_argument("--sample-tier3", type=int, default=0,
                        help="Cap Claude calls (0 = REVIEW_TIER3_TOP_K)")
    args = parser.parse_args()

    load_dotenv()
    register_src_for_burla()
    if os.environ.get("FORCE_S04") != "1":
        manifest_path = REVIEWS_SCORED_PATH.with_suffix(".manifest.json")
        if manifest_path.exists():
            existing = read_json(manifest_path)
            if existing and existing.get("ok") and existing.get("n_claude_rows", 0) > 0:
                print(
                    f"[s04] manifest exists ({existing.get('n_claude_rows', 0):,} Claude-scored "
                    f"rows at {existing.get('shared_path')}); skipping. "
                    f"Set FORCE_S04=1 or delete {manifest_path} to force.",
                    flush=True,
                )
                return
    from burla import remote_parallel_map

    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not args.skip_tier3 and not anthropic_key:
        raise SystemExit("[s04] ANTHROPIC_API_KEY missing; halting before tier 3.")

    raw_path = f"{SHARED_ROOT}/reviews_raw.parquet"

    if not args.skip_ingest:
        report = read_json(VALIDATION_REPORT_PATH)
        if not report:
            raise SystemExit(f"[s04] no validation report at {VALIDATION_REPORT_PATH}")
        # Reviews are append-only on Inside Airbnb: each new snapshot is a
        # superset of the previous one. Pulling every snapshot per city would
        # download tens of GB of duplicates we then have to dedupe. We pick
        # the latest reviews-bearing snapshot per (country, region, city).
        passing_rows = [r for r in report.get("passing", []) if r.get("reviews_url")]
        latest: dict[tuple, dict] = {}
        for r in passing_rows:
            key = (r["country"], r["region"], r["city"])
            prev = latest.get(key)
            if prev is None or r["snapshot_date"] > prev["snapshot_date"]:
                latest[key] = r
        ingest_args = [
            IngestReviewsArgs(
                city_slug=_city_slug(r["country"], r["region"], r["city"], r["snapshot_date"]),
                reviews_url=r["reviews_url"],
                output_root=SHARED_REVIEWS,
            )
            for r in latest.values()
        ]
        n_cities = len(ingest_args)
        print(f"[s04] ingesting reviews for {n_cities} cities (latest snapshot each, "
              f"{len(passing_rows)} candidate tuples) ...", flush=True)
        t0 = time.time()
        with BudgetTracker("s04_reviews_ingest", n_inputs=n_cities, func_cpu=1) as bt:
            bt.set_workers(min(250, n_cities))
            results = remote_parallel_map(
                ingest_reviews_for_city, ingest_args,
                func_cpu=1, func_ram=4,
                max_parallelism=min(250, n_cities),
                grow=True, spinner=False,
            )
            n_ok = sum(1 for r in results if r.get("ok"))
            n_rows = sum(int(r.get("n_rows", 0)) for r in results if r.get("ok"))
            bt.set_succeeded(n_ok)
            bt.set_failed(n_cities - n_ok)
            bt.note(total_rows=n_rows)
        print(f"[s04]   ingested {n_rows:,} reviews from {n_ok}/{n_cities} cities in {time.time()-t0:.1f}s", flush=True)

        print(f"[s04] merging review parquets to {raw_path} ...", flush=True)
        [m] = remote_parallel_map(
            merge_reviews,
            [MergeReviewsArgs(shared_root=SHARED_REVIEWS, output_path=raw_path)],
            func_cpu=16, func_ram=64, max_parallelism=1, grow=True, spinner=False,
        )
        if not m.get("ok"):
            raise SystemExit(f"[s04] reviews merge failed: {m.get('error')}")
        print(f"[s04]   merged {m['n_rows']:,} reviews from {m['n_cities']} cities", flush=True)

    tier1_top_path = f"{SHARED_ROOT}/reviews_tier1_top_v2.parquet"
    rechunked_path = f"{SHARED_ROOT}/reviews_rechunked.parquet"
    if not args.skip_tier1:
        # Rewrite reviews_raw.parquet with row_group_size = batch size so each
        # tier-1 worker reads one row group instead of the full 50M-row file.
        print(f"[s04] rechunking reviews -> {rechunked_path} (row_group_size={REVIEW_TIER1_BATCH_SIZE}) ...", flush=True)
        [rc] = remote_parallel_map(
            rechunk_reviews_for_tier1,
            [RechunkReviewsArgs(input_path=raw_path,
                                 output_path=rechunked_path,
                                 row_group_size=REVIEW_TIER1_BATCH_SIZE)],
            func_cpu=8, func_ram=64, max_parallelism=1, grow=True, spinner=False,
        )
        if not rc.get("ok"):
            raise SystemExit(f"[s04] rechunk failed: {rc.get('error')}")
        n_total = int(rc["n_rows"])
        n_groups = int(rc["n_row_groups"])
        print(f"[s04]   rechunked {n_total:,} rows into {n_groups:,} row groups", flush=True)
        print(f"[s04] tier 1: heuristic-scoring {n_total:,} reviews ...", flush=True)

        batches: list[Tier1HeuristicArgs] = []
        for i, start in enumerate(range(0, n_total, REVIEW_TIER1_BATCH_SIZE)):
            end = min(start + REVIEW_TIER1_BATCH_SIZE, n_total)
            batches.append(Tier1HeuristicArgs(
                batch_id=i, reviews_path=rechunked_path,
                row_start=start, row_end=end,
                output_root=SHARED_REVIEWS_TIER1,
                keywords=list(REVIEW_HEURISTIC_KEYWORDS),
            ))
        n_workers = min(REVIEW_TIER1_MAX_PARALLELISM, len(batches))
        t0 = time.time()
        with BudgetTracker("s04_reviews_tier1", n_inputs=n_total, func_cpu=1) as bt:
            bt.set_workers(n_workers)
            results = remote_parallel_map(
                heuristic_score_batch, batches,
                func_cpu=1, func_ram=2,
                max_parallelism=n_workers,
                grow=True, spinner=False,
            )
            ok = sum(1 for r in results if r.get("shared_path"))
            bt.set_succeeded(ok)
            bt.set_failed(len(results) - ok)
        print(f"[s04]   tier 1 done in {time.time()-t0:.1f}s ({ok}/{len(batches)} batches)", flush=True)

        print(f"[s04]   reducing tier 1 -> top {REVIEW_TIER2_TOP_K:,} ...", flush=True)
        [t] = remote_parallel_map(
            merge_and_top_k_tier1,
            [TopKHeuristicArgs(shared_root=SHARED_REVIEWS_TIER1,
                               output_path=tier1_top_path,
                               top_k=REVIEW_TIER2_TOP_K)],
            func_cpu=16, func_ram=64, max_parallelism=1, grow=True, spinner=False,
        )
        if not t.get("ok"):
            raise SystemExit(f"[s04] tier1 merge failed: {t.get('error')}")
        print(f"[s04]   tier 1 top-{REVIEW_TIER2_TOP_K:,} written ({t['n_rows']:,} scored)", flush=True)

    tier3_input_path = f"{SHARED_ROOT}/reviews_tier3_input_v2.parquet"
    if not args.skip_tier2:
        [c2] = remote_parallel_map(
            count_parquet_rows, [CountParquetArgs(path=tier1_top_path)],
            func_cpu=2, func_ram=4, max_parallelism=1, grow=True, spinner=False,
        )
        n_top = int(c2["n"])
        print(f"[s04] tier 2: embedding {n_top:,} reviews ...", flush=True)

        embed_batch_size = 1000
        batches: list[EmbedTier2Args] = []
        for i, start in enumerate(range(0, n_top, embed_batch_size)):
            end = min(start + embed_batch_size, n_top)
            batches.append(EmbedTier2Args(
                batch_id=i,
                tier1_top_path=tier1_top_path,
                raw_reviews_path=raw_path,
                row_start=start, row_end=end,
                output_root=SHARED_REVIEWS_TIER2,
                model_name="sentence-transformers/all-MiniLM-L6-v2",
            ))
        n_workers = min(200, len(batches))
        t0 = time.time()
        with BudgetTracker("s04_reviews_tier2_embed", n_inputs=n_top, func_cpu=2) as bt:
            bt.set_workers(n_workers)
            results = remote_parallel_map(
                embed_reviews_batch, batches,
                func_cpu=2, func_ram=8,
                max_parallelism=n_workers,
                grow=True, spinner=False,
            )
            ok = sum(1 for r in results if r.get("shared_path"))
            bt.set_succeeded(ok)
            bt.set_failed(len(results) - ok)
        print(f"[s04]   embed done in {time.time()-t0:.1f}s ({ok}/{len(batches)} batches)", flush=True)

        print(f"[s04]   tier 2: clustering -> top {REVIEW_TIER3_TOP_K:,} for tier 3 ...", flush=True)
        [r] = remote_parallel_map(
            cluster_and_rerank_tier2,
            [ClusterTier2Args(
                shared_root=SHARED_REVIEWS_TIER2,
                output_path=tier3_input_path,
                n_clusters=REVIEW_TIER2_NUM_CLUSTERS,
                top_k_for_tier3=REVIEW_TIER3_TOP_K,
            )],
            func_cpu=16, func_ram=64, max_parallelism=1, grow=True, spinner=False,
        )
        if not r.get("ok"):
            raise SystemExit(f"[s04] tier2 cluster failed: {r.get('error')}")
        print(f"[s04]   tier 2 done: {r['n_rows']:,} rows -> top {r['n_top_k']:,}", flush=True)

    if not args.skip_tier3:
        [tdat] = remote_parallel_map(
            read_tier3_input,
            [ReadTier3InputArgs(path=tier3_input_path, cap=args.sample_tier3)],
            func_cpu=2, func_ram=4, max_parallelism=1, grow=True, spinner=False,
        )
        rows = tdat["rows"]
        n_top3 = len(rows)
        print(f"[s04] tier 3: Claude on {n_top3:,} reviews ...", flush=True)

        batches: list[ClaudeBatchArgs] = []
        for i, start in enumerate(range(0, n_top3, REVIEW_TIER3_BATCH_SIZE)):
            end = min(start + REVIEW_TIER3_BATCH_SIZE, n_top3)
            batches.append(ClaudeBatchArgs(
                batch_id=i,
                rows=rows[start:end],
                output_root=SHARED_REVIEWS_TIER3,
                anthropic_api_key=anthropic_key,
                model=ANTHROPIC_MODEL,
                max_tokens=ANTHROPIC_MAX_TOKENS,
                categories=list(REVIEW_HUMOR_CATEGORIES),
            ))
        n_workers = min(REVIEW_TIER3_MAX_PARALLELISM, len(batches))
        t0 = time.time()
        with BudgetTracker("s04_reviews_tier3_claude", n_inputs=n_top3, func_cpu=1) as bt:
            bt.set_workers(n_workers)
            results = remote_parallel_map(
                claude_score_batch, batches,
                func_cpu=1, func_ram=2,
                max_parallelism=n_workers,
                grow=True, spinner=False,
            )
            n_ok = sum(int(r.get("n_ok", 0)) for r in results)
            n_failed = sum(int(r.get("n_failed", 0)) for r in results)
            bt.set_succeeded(n_ok)
            bt.set_failed(n_failed)
        print(f"[s04]   tier 3 done in {time.time()-t0:.1f}s ({n_ok}/{n_top3} review categorizations)", flush=True)

    final_path = f"{SHARED_ROOT}/reviews_scored.parquet"
    print(f"[s04] merging tier 3 -> {final_path} ...", flush=True)
    [mc] = remote_parallel_map(
        merge_claude,
        [MergeClaudeArgs(
            shared_root=SHARED_REVIEWS_TIER3,
            tier3_input_path=tier3_input_path,
            raw_reviews_path=raw_path,
            output_path=final_path,
        )],
        func_cpu=16, func_ram=64, max_parallelism=1, grow=True, spinner=False,
    )
    if not mc.get("ok"):
        raise SystemExit(f"[s04] merge_claude failed: {mc.get('error')}")
    print(f"[s04]   final: {mc['n_rows']:,} rows ({mc['n_claude_rows']:,} with claude scores)", flush=True)

    ensure_dir(REVIEWS_SCORED_PATH.parent)
    manifest_path = REVIEWS_SCORED_PATH.with_suffix(".manifest.json")
    write_json(manifest_path, {
        "ok": True,
        "shared_path": final_path,
        "n_rows": mc["n_rows"],
        "n_claude_rows": mc["n_claude_rows"],
        "completed_at": time.time(),
    })
    print(f"[s04] DONE. Manifest at {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
