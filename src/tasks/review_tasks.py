"""Burla worker functions for the 3-tier review-scoring funnel.

Tier 1: heuristic scoring on every review (~50M rows). CPU, batched.
Tier 2: top 200k by tier 1 score, embed with sentence-transformers, KMeans cluster.
Tier 3: top 10k weirdest, score with Claude Haiku via the Anthropic API.

All worker functions are top-level, take a single ``@dataclass`` arg, and write
intermediate parquets to ``/workspace/shared`` so other Burla stages can pick
them up without re-uploading.
"""
from __future__ import annotations

import gzip
import io
import os
import re
import time
import traceback
from dataclasses import dataclass
from typing import List, Optional

import requests

# Hoist these so Burla pip-installs them on workers (see image_tasks.py).
import numpy as _np  # noqa: F401
import pandas as _pd  # noqa: F401
import pyarrow as _pa  # noqa: F401
import pyarrow.parquet as _pq  # noqa: F401
import sentence_transformers as _st  # noqa: F401
import sklearn  # noqa: F401
import sklearn.cluster  # noqa: F401
import anthropic as _anthropic  # noqa: F401


_REVIEW_HEAD = {"User-Agent": "Mozilla/5.0 (compatible; airbnb-burla/0.1)"}


@dataclass
class IngestReviewsArgs:
    city_slug: str
    reviews_url: str
    output_root: str  # /workspace/shared/airbnb/reviews_raw


def ingest_reviews_for_city(args: IngestReviewsArgs) -> dict:
    """Download a city's reviews.csv.gz, parse, write per-city parquet to shared FS."""
    out = {"city_slug": args.city_slug, "ok": False, "n_rows": 0,
           "shared_path": None, "elapsed_seconds": 0.0, "error": None}
    started = time.time()
    try:
        import pandas as pd
        r = requests.get(args.reviews_url, timeout=600, headers=_REVIEW_HEAD, stream=True)
        if r.status_code != 200:
            out["error"] = f"http_{r.status_code}"
            out["elapsed_seconds"] = time.time() - started
            return out
        with gzip.GzipFile(fileobj=io.BytesIO(r.content)) as gz:
            df = pd.read_csv(
                gz,
                usecols=["listing_id", "id", "date", "reviewer_id", "comments"],
                dtype={"comments": "string"},
                low_memory=False,
            )
        df = df.rename(columns={"id": "review_id"})
        df["city_slug"] = args.city_slug
        df["comments"] = df["comments"].fillna("").astype(str)
        df = df[df["comments"].str.len() > 0].reset_index(drop=True)

        os.makedirs(args.output_root, exist_ok=True)
        path = os.path.join(args.output_root, f"{args.city_slug}.parquet")
        df.to_parquet(path, compression="zstd", index=False)
        out.update({"ok": True, "n_rows": int(len(df)), "shared_path": path})
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["traceback"] = traceback.format_exc()[:1000]
    out["elapsed_seconds"] = time.time() - started
    return out


@dataclass
class MergeReviewsArgs:
    shared_root: str  # /workspace/shared/airbnb/reviews_raw
    output_path: str  # /workspace/shared/airbnb/reviews_raw.parquet


def merge_reviews(args: MergeReviewsArgs) -> dict:
    out = {"ok": False, "n_files": 0, "n_rows": 0, "n_cities": 0,
           "output_path": args.output_path, "error": None}
    try:
        import glob
        import pandas as pd
        files = sorted(glob.glob(os.path.join(args.shared_root, "*.parquet")))
        out["n_files"] = len(files)
        if not files:
            out["error"] = f"no parquets at {args.shared_root}"
            return out
        big = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        big = big.drop_duplicates(subset=["review_id"])
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        big.to_parquet(args.output_path, compression="zstd", index=False)
        out.update({
            "ok": True,
            "n_rows": int(len(big)),
            "n_cities": int(big["city_slug"].nunique()),
        })
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["traceback"] = traceback.format_exc()[:1000]
    return out


@dataclass
class RechunkReviewsArgs:
    input_path: str   # /workspace/shared/airbnb/reviews_raw.parquet
    output_path: str  # /workspace/shared/airbnb/reviews_rechunked.parquet
    row_group_size: int  # e.g. 5000 (= REVIEW_TIER1_BATCH_SIZE)


def rechunk_reviews_for_tier1(args: RechunkReviewsArgs) -> dict:
    """Rewrite the merged reviews parquet with row_group_size aligned to the
    tier-1 batch size so each tier-1 worker can read exactly one row group
    instead of streaming the full 50M-row file."""
    out = {"ok": False, "n_rows": 0, "n_row_groups": 0,
           "output_path": args.output_path, "error": None}
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        if os.path.exists(args.output_path):
            pf_existing = pq.ParquetFile(args.output_path)
            md = pf_existing.metadata
            if md.row_group(0).num_rows == args.row_group_size or md.num_rows == 0:
                out.update({"ok": True, "n_rows": int(md.num_rows),
                            "n_row_groups": int(md.num_row_groups),
                            "skipped": True})
                return out
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        pf = pq.ParquetFile(args.input_path)
        schema = pf.schema_arrow.remove_metadata()
        writer = pq.ParquetWriter(args.output_path, schema, compression="zstd")
        n = 0
        n_groups = 0
        try:
            for batch in pf.iter_batches(batch_size=args.row_group_size,
                                          columns=None):
                tbl = pa.Table.from_batches([batch])
                writer.write_table(tbl, row_group_size=args.row_group_size)
                n += tbl.num_rows
                n_groups += 1
        finally:
            writer.close()
        out.update({"ok": True, "n_rows": int(n), "n_row_groups": int(n_groups)})
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["traceback"] = traceback.format_exc()[:1000]
    return out


@dataclass
class Tier1HeuristicArgs:
    batch_id: int
    reviews_path: str  # parquet on shared FS (rechunked, one row group == one batch)
    row_start: int
    row_end: int
    output_root: str   # /workspace/shared/airbnb/reviews_tier1
    keywords: list


_BUT_RE = re.compile(r"\b(however|but)\b", re.IGNORECASE)


def heuristic_score(comment: str, keywords: list) -> dict:
    """Return a tier-1 score and a few features for one review.

    Score is a simple sum: keyword hits + 'however/but' marker + length penalty/bonus.
    """
    s = comment if isinstance(comment, str) else ""
    low = s.lower()
    n_keyword = sum(1 for k in keywords if k.lower() in low)
    n_marker = len(_BUT_RE.findall(s))
    length = len(s)
    score = n_keyword * 2.0 + n_marker * 1.5
    if 200 <= length <= 1500:
        score += 0.5
    if "would not stay" in low or "would not recommend" in low or "do not stay" in low:
        score += 1.0
    if "five stars" in low and any(w in low for w in ("but ", "however ", "scary", "weird", "strange")):
        score += 2.0
    return {
        "tier1_score": float(score),
        "n_keyword": int(n_keyword),
        "n_marker": int(n_marker),
        "length": int(length),
    }


def heuristic_score_batch(args: Tier1HeuristicArgs) -> dict:
    out = {"batch_id": args.batch_id, "n_inputs": 0, "shared_path": None,
           "elapsed_seconds": 0.0, "error": None}
    started = time.time()
    try:
        import pandas as pd
        import pyarrow.parquet as pq

        # The reviews parquet is rechunked so row group N == batch N. Reading
        # one row group only loads ~5000 rows instead of the full 50M-row file.
        pf = pq.ParquetFile(args.reviews_path)
        rg_idx = args.batch_id
        if rg_idx >= pf.metadata.num_row_groups:
            out["error"] = f"batch_id {rg_idx} >= num_row_groups {pf.metadata.num_row_groups}"
            out["elapsed_seconds"] = time.time() - started
            return out
        table = pf.read_row_group(
            rg_idx,
            columns=["review_id", "listing_id", "date", "comments", "city_slug"],
        )
        chunk = table.to_pandas().reset_index(drop=True)
        out["n_inputs"] = int(len(chunk))
        scores = chunk["comments"].apply(lambda c: heuristic_score(c, args.keywords))
        feats = pd.DataFrame(list(scores))
        # Include comments in tier-1 output so tier 2 (embed) doesn't have to
        # re-scan the 50M-row reviews_raw to fetch comment text per batch.
        result = pd.concat(
            [chunk[["review_id", "listing_id", "date", "city_slug", "comments"]],
             feats],
            axis=1,
        )
        os.makedirs(args.output_root, exist_ok=True)
        path = os.path.join(args.output_root, f"batch_{args.batch_id:06d}.parquet")
        result.to_parquet(path, compression="zstd", index=False)
        out["shared_path"] = path
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["traceback"] = traceback.format_exc()[:1000]
    out["elapsed_seconds"] = time.time() - started
    return out


@dataclass
class TopKHeuristicArgs:
    shared_root: str
    output_path: str
    top_k: int


def merge_and_top_k_tier1(args: TopKHeuristicArgs) -> dict:
    """Merge tier-1 batch parquets, write the top-K to ``output_path`` for tier 2."""
    out = {"ok": False, "n_files": 0, "n_rows": 0, "n_top_k": 0,
           "output_path": args.output_path, "error": None}
    try:
        import glob
        import pandas as pd
        files = sorted(glob.glob(os.path.join(args.shared_root, "batch_*.parquet")))
        out["n_files"] = len(files)
        if not files:
            out["error"] = f"no batches at {args.shared_root}"
            return out
        big = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        big = big.drop_duplicates(subset=["review_id"])
        out["n_rows"] = int(len(big))
        top = big.nlargest(int(args.top_k), "tier1_score").reset_index(drop=True)
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        top.to_parquet(args.output_path, compression="zstd", index=False)
        out["n_top_k"] = int(len(top))
        out["ok"] = True
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["traceback"] = traceback.format_exc()[:1000]
    return out


@dataclass
class EmbedTier2Args:
    batch_id: int
    tier1_top_path: str  # parquet on shared FS (top-K tier1 rows)
    raw_reviews_path: str  # parquet on shared FS (raw reviews to grab comment text)
    row_start: int
    row_end: int
    output_root: str
    model_name: str


_EMBED_STATE = {"model": None, "model_name": None}

# Pre-staged sentence-transformers weights on the Burla shared filesystem. We
# download these once via scripts/preload_st_weights.py to avoid hundreds of
# workers hammering HuggingFace at the same time.
_ST_SHARED_DIR = "/workspace/shared/airbnb/st_weights/all-MiniLM-L6-v2"
_ST_LOCAL_DIR = "/tmp/st_weights_all-MiniLM-L6-v2"
_ST_NODE_LOCK = "/tmp/st_weights.lock"


def _ensure_embedder(model_name: str):
    if _EMBED_STATE["model"] is not None and _EMBED_STATE["model_name"] == model_name:
        return _EMBED_STATE["model"]
    import fcntl
    import shutil
    from sentence_transformers import SentenceTransformer

    # Pin PyTorch threads (~80 workers per node otherwise oversubscribe CPUs).
    try:
        import torch
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:
        pass

    if os.path.isdir(_ST_SHARED_DIR) and os.path.exists(os.path.join(_ST_SHARED_DIR, "config.json")):
        with open(_ST_NODE_LOCK, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            if not (os.path.isdir(_ST_LOCAL_DIR)
                    and os.path.exists(os.path.join(_ST_LOCAL_DIR, "config.json"))):
                tmp = _ST_LOCAL_DIR + ".part"
                if os.path.exists(tmp):
                    shutil.rmtree(tmp)
                shutil.copytree(_ST_SHARED_DIR, tmp)
                if os.path.exists(_ST_LOCAL_DIR):
                    shutil.rmtree(_ST_LOCAL_DIR)
                os.replace(tmp, _ST_LOCAL_DIR)
            fcntl.flock(lock_file, fcntl.LOCK_UN)
        model = SentenceTransformer(_ST_LOCAL_DIR)
    else:
        # Fallback if pre-staging step was skipped.
        model = SentenceTransformer(model_name)
    _EMBED_STATE["model"] = model
    _EMBED_STATE["model_name"] = model_name
    return model


def embed_reviews_batch(args: EmbedTier2Args) -> dict:
    out = {"batch_id": args.batch_id, "n_inputs": 0, "shared_path": None,
           "elapsed_seconds": 0.0, "error": None}
    started = time.time()
    try:
        import numpy as np
        import pandas as pd
        # tier1_top now carries comments directly so we avoid scanning the
        # 50M-row reviews_raw parquet from hundreds of workers in parallel.
        top = pd.read_parquet(
            args.tier1_top_path,
            columns=["review_id", "tier1_score", "comments"],
        )
        chunk = top.iloc[args.row_start: args.row_end].reset_index(drop=True)
        out["n_inputs"] = int(len(chunk))
        if not len(chunk):
            out["shared_path"] = None
            return out
        merged = chunk.copy()
        merged["comments"] = merged["comments"].fillna("").astype(str)
        model = _ensure_embedder(args.model_name)
        emb = model.encode(
            merged["comments"].tolist(),
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        merged["embedding"] = list(emb.astype(np.float32))
        os.makedirs(args.output_root, exist_ok=True)
        path = os.path.join(args.output_root, f"batch_{args.batch_id:06d}.parquet")
        merged.to_parquet(path, compression="zstd", index=False)
        out["shared_path"] = path
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["traceback"] = traceback.format_exc()[:1000]
    out["elapsed_seconds"] = time.time() - started
    return out


@dataclass
class ClusterTier2Args:
    shared_root: str
    output_path: str
    n_clusters: int
    top_k_for_tier3: int


def cluster_and_rerank_tier2(args: ClusterTier2Args) -> dict:
    """Read all embedded batches, KMeans, identify weird clusters, write top-K for tier 3."""
    out = {"ok": False, "n_rows": 0, "n_clusters": 0, "n_top_k": 0,
           "output_path": args.output_path, "error": None}
    try:
        import glob
        import numpy as np
        import pandas as pd
        from sklearn.cluster import MiniBatchKMeans

        files = sorted(glob.glob(os.path.join(args.shared_root, "batch_*.parquet")))
        if not files:
            out["error"] = f"no embed batches at {args.shared_root}"
            return out
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        out["n_rows"] = int(len(df))
        if not len(df):
            out["error"] = "empty embed merge"
            return out

        emb = np.stack(df["embedding"].apply(np.asarray).to_list()).astype(np.float32)
        km = MiniBatchKMeans(
            n_clusters=int(args.n_clusters),
            n_init=10,
            random_state=42,
            batch_size=4096,
        )
        labels = km.fit_predict(emb)
        df["cluster"] = labels.astype(int)

        cluster_means = df.groupby("cluster")["tier1_score"].mean()
        cluster_rank = cluster_means.rank(ascending=False).astype(int)
        df["cluster_rank"] = df["cluster"].map(cluster_rank)
        df["combined_score"] = df["tier1_score"] - 0.1 * df["cluster_rank"].astype(float)

        top = df.nlargest(int(args.top_k_for_tier3), "combined_score").reset_index(drop=True)
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        keep_cols = ["review_id", "tier1_score", "comments", "cluster",
                     "cluster_rank", "combined_score"]
        top[keep_cols].to_parquet(args.output_path, compression="zstd", index=False)
        out.update({"ok": True, "n_clusters": int(args.n_clusters), "n_top_k": int(len(top))})
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["traceback"] = traceback.format_exc()[:1000]
    return out


@dataclass
class ClaudeBatchArgs:
    batch_id: int
    rows: list           # list of dicts: review_id, comments
    output_root: str
    anthropic_api_key: str
    model: str
    max_tokens: int
    categories: list


_CLAUDE_PROMPT_TEMPLATE = """You categorize Airbnb guest reviews for a viral-content roundup.

You MUST pick exactly one category from this fixed list, copying the label
character-for-character. Do not invent new labels.

{categories}

Definitions (resolve overlap by picking the one that best captures the WHY
the story is interesting, not just the surface keywords):

- "This escalated quickly": review starts mundane and ends with chaos.
- "Five stars but terrifying": a positive star rating despite a clearly
  unsettling story.
- "Passive aggressive poetry": carefully worded backhanded compliments,
  understatement, dry sarcasm.
- "Host said what now": odd, intrusive, or unhinged behavior from the host.
- "Pets and wildlife": cats, dogs, raccoons, snakes, geckos, monkeys,
  unexpected critters - whether welcome or not.
- "Bugs and pests": cockroaches, bedbugs, ants, rodents, infestations.
- "Plumbing and smells": toilets, sewage, leaks, mystery odors, sewage gas.
- "Noise complaint hall of fame": neighbors, parties, construction,
  rooftop bars, jackhammers at 4am.
- "Cleanliness mystery": dust, hair, stains, "deep clean" claims that
  clearly were not.
- "Weather and building drama": fires, blackouts, storms, leaks from above,
  elevator outages.
- "Lost in translation": Google Translate-flavored prose, language gaps,
  cross-cultural misunderstandings.
- "Photo did not match": the listing's photos misrepresented the reality
  in a meaningful way (size, view, condition).
- "Honest disaster": the stay was bad in a generic way that doesn't fit any
  category above. Use sparingly.
- "Not funny": the review is bland, generic, or simply positive with no
  story.

For each review, return JSON with these exact keys:
- review_id (echo back the integer)
- category (one of the labels above, copied verbatim)
- humor_score (integer 0-10; how funny is the SITUATION described, not the
  prose; reserve 8-10 for genuinely viral)
- one_line (a punchy headline, max 12 words, no emoji, no quotes)

Return ONLY a JSON array. No prose, no preamble. Reviews:
{block}"""


def claude_score_batch(args: ClaudeBatchArgs) -> dict:
    out = {"batch_id": args.batch_id, "n_inputs": len(args.rows),
           "n_ok": 0, "n_failed": 0, "shared_path": None,
           "elapsed_seconds": 0.0, "error": None}
    started = time.time()
    try:
        import json
        import anthropic
        client = anthropic.Anthropic(api_key=args.anthropic_api_key)
        block = "\n".join(
            f'{{"review_id": {r["review_id"]}, "comments": {json.dumps(str(r["comments"])[:1000])}}}'
            for r in args.rows
        )
        prompt = _CLAUDE_PROMPT_TEMPLATE.format(
            categories="\n".join(f"- {c}" for c in args.categories),
            block=block,
        )
        resp = client.messages.create(
            model=args.model,
            max_tokens=args.max_tokens * len(args.rows),
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        m = re.search(r"\[.*\]", text, re.DOTALL)
        parsed = json.loads(m.group(0)) if m else []
        rows = []
        for entry in parsed:
            try:
                rows.append({
                    "review_id": int(entry["review_id"]),
                    "claude_category": str(entry.get("category", "Not funny"))[:80],
                    "claude_humor_score": float(entry.get("humor_score", 0.0)),
                    "claude_one_line": str(entry.get("one_line", ""))[:160],
                })
                out["n_ok"] += 1
            except Exception:
                out["n_failed"] += 1
        if rows:
            import pandas as pd
            os.makedirs(args.output_root, exist_ok=True)
            path = os.path.join(args.output_root, f"batch_{args.batch_id:06d}.parquet")
            pd.DataFrame(rows).to_parquet(path, compression="zstd", index=False)
            out["shared_path"] = path
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["traceback"] = traceback.format_exc()[:1000]
    out["elapsed_seconds"] = time.time() - started
    return out


@dataclass
class MergeClaudeArgs:
    shared_root: str
    tier3_input_path: str  # /workspace/shared/airbnb/reviews_tier3_input.parquet (top 10k)
    raw_reviews_path: str  # /workspace/shared/airbnb/reviews_raw.parquet
    output_path: str       # /workspace/shared/airbnb/reviews_scored.parquet


def merge_claude(args: MergeClaudeArgs) -> dict:
    """Merge claude batch parquets with the tier-3 input + raw to produce reviews_scored.parquet."""
    out = {"ok": False, "n_files": 0, "n_rows": 0, "n_claude_rows": 0,
           "output_path": args.output_path, "error": None}
    try:
        import glob
        import pandas as pd
        tier3_in = pd.read_parquet(
            args.tier3_input_path,
            columns=["review_id", "tier1_score", "cluster", "combined_score"],
        )
        files = sorted(glob.glob(os.path.join(args.shared_root, "batch_*.parquet")))
        out["n_files"] = len(files)
        if files:
            claude = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            claude = claude.drop_duplicates(subset=["review_id"])
            out["n_claude_rows"] = int(len(claude))
            scored = tier3_in.merge(claude, on="review_id", how="left")
        else:
            scored = tier3_in.copy()
            for c in ("claude_category", "claude_humor_score", "claude_one_line"):
                scored[c] = None

        raw = pd.read_parquet(
            args.raw_reviews_path,
            columns=["review_id", "listing_id", "date", "comments", "city_slug"],
            filters=[("review_id", "in", scored["review_id"].tolist())],
        )
        scored = scored.merge(raw, on="review_id", how="left")
        scored["comments"] = scored["comments"].fillna("").astype(str)
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        scored.to_parquet(args.output_path, compression="zstd", index=False)
        out.update({"ok": True, "n_rows": int(len(scored))})
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["traceback"] = traceback.format_exc()[:1000]
    return out
