"""Burla worker functions for the WTF (absurd/weird) photo detector.

Two-stage funnel:

1. CPU+CLIP screen (already done in Stage 2): every image carries
   ``clip_wtf_absurd_object``, ``clip_wtf_unsettling_decor``,
   ``clip_wtf_unusual_scene``, ``clip_wtf_does_not_belong``. We pick the top-K
   by max of those four.

2. Claude Haiku rerank: each candidate photo goes to Haiku with a strict JSON
   schema asking ``is_absurd``, a normalized ``cluster`` label, ``one_line``
   caption, and a 0-10 score. We cluster by the ``cluster`` field (lower-cased,
   stripped) and surface the top-K per cluster on the site.

Workers are top-level, take a single ``@dataclass`` arg, and write parquet to
``/workspace/shared`` so the merge step can pick them up.
"""
from __future__ import annotations

import os
import re
import time
import traceback
from dataclasses import dataclass

# Hoist for cloudpickle bundling on Burla workers.
import numpy as _np  # noqa: F401
import pandas as _pd  # noqa: F401
import pyarrow as _pa  # noqa: F401
import pyarrow.parquet as _pq  # noqa: F401
import anthropic as _anthropic  # noqa: F401


@dataclass
class WtfHaikuBatchArgs:
    batch_id: int
    rows: list           # list of dicts: image_id, listing_id, image_idx,
                         #   image_url, clip_max
    output_root: str     # /workspace/shared/airbnb/wtf_haiku_v1
    anthropic_api_key: str
    model: str
    max_tokens: int


_WTF_PROMPT_TEMPLATE = """You are categorizing strange photos that hosts uploaded to Airbnb.

For each photo, decide if it is genuinely absurd, unsettling, or out-of-place
for a vacation rental. Examples of "yes": taxidermy in the living room, a
mannequin staring at the bed, a kitchen in the bathroom, a clown shrine, a
single chair in a 30 ft empty room, religious icons stacked on the toilet.

Be strict: a slightly unusual decor choice is NOT absurd. We want the photos
that make people stop and say "what."

Return a JSON array. For every input photo, include exactly one object:
{{
  "image_id": <echo back the image_id integer>,
  "is_absurd": true | false,
  "cluster": "<2-4 word lowercase tag, e.g. 'taxidermy', 'creepy doll',
              'kitchen in bedroom', 'religious shrine'>",
  "one_line": "<funny caption, max 12 words, no emoji>",
  "score": <0 to 10 integer>
}}

Return ONLY the JSON array, no prose. Photos:
{block}"""


def _norm_cluster(label: str) -> str:
    s = (label or "").strip().lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:60]


def wtf_haiku_score_batch(args: WtfHaikuBatchArgs) -> dict:
    """Send a batch of candidate WTF photos to Claude Haiku, parse JSON,
    write a per-batch parquet to shared FS."""
    out = {
        "batch_id": args.batch_id, "n_inputs": len(args.rows),
        "n_ok": 0, "n_failed": 0, "shared_path": None,
        "elapsed_seconds": 0.0, "error": None,
    }
    started = time.time()
    try:
        import json as _json
        import anthropic
        import pandas as pd

        client = anthropic.Anthropic(api_key=args.anthropic_api_key)
        block = "\n".join(
            f'{{"image_id": {int(r["image_id"])}, '
            f'"image_url": {_json.dumps(str(r["image_url"]))}}}'
            for r in args.rows
        )
        prompt = _WTF_PROMPT_TEMPLATE.format(block=block)

        # Build vision content blocks. Each photo is referenced by URL so
        # Anthropic fetches it directly; this is rate-limited and retries.
        content = [{"type": "text", "text": prompt}]
        for r in args.rows:
            content.append({
                "type": "image",
                "source": {"type": "url", "url": str(r["image_url"])},
            })

        resp = None
        last_err = None
        for attempt in range(4):
            try:
                resp = client.messages.create(
                    model=args.model,
                    max_tokens=args.max_tokens * len(args.rows),
                    messages=[{"role": "user", "content": content}],
                )
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(20.0, 2.0 * (2 ** attempt)))
        if resp is None:
            raise RuntimeError(f"haiku api failed after retries: {last_err}")

        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        m = re.search(r"\[.*\]", text, re.DOTALL)
        parsed = _json.loads(m.group(0)) if m else []

        rows: list[dict] = []
        url_by_id = {int(r["image_id"]): r for r in args.rows}
        for entry in parsed:
            try:
                iid = int(entry["image_id"])
                src = url_by_id.get(iid, {})
                rows.append({
                    "image_id": iid,
                    "listing_id": int(src.get("listing_id", 0)),
                    "image_idx": int(src.get("image_idx", -1)),
                    "image_url": str(src.get("image_url", "")),
                    "clip_max": float(src.get("clip_max", 0.0)),
                    "is_absurd": bool(entry.get("is_absurd", False)),
                    "cluster_raw": str(entry.get("cluster", ""))[:80],
                    "cluster": _norm_cluster(str(entry.get("cluster", ""))),
                    "one_line": str(entry.get("one_line", ""))[:160],
                    "haiku_score": float(entry.get("score", 0.0)),
                })
                out["n_ok"] += 1
            except Exception:
                out["n_failed"] += 1

        if rows:
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
class WtfMergeArgs:
    shared_root: str
    output_path: str
    min_cluster_size: int
    top_per_cluster: int


def merge_wtf_haiku(args: WtfMergeArgs) -> dict:
    """Merge per-batch Haiku parquets, filter to is_absurd=True, drop tiny
    clusters, keep top-K photos per cluster ranked by haiku_score."""
    out = {
        "ok": False, "n_files": 0, "n_rows": 0,
        "n_absurd": 0, "n_clusters": 0, "n_kept": 0,
        "output_path": args.output_path, "error": None,
    }
    try:
        import glob
        import pandas as pd
        files = sorted(glob.glob(os.path.join(args.shared_root, "batch_*.parquet")))
        out["n_files"] = len(files)
        if not files:
            out["error"] = f"no batches at {args.shared_root}"
            return out
        big = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        big = big.drop_duplicates(subset=["image_id"])
        out["n_rows"] = int(len(big))

        absurd = big[big["is_absurd"].astype(bool)].copy()
        out["n_absurd"] = int(len(absurd))
        if not len(absurd):
            os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
            absurd.to_parquet(args.output_path, compression="zstd", index=False)
            out.update({"ok": True})
            return out

        # Drop clusters smaller than min_cluster_size (likely Haiku noise).
        sizes = absurd["cluster"].value_counts()
        keep_clusters = set(sizes[sizes >= int(args.min_cluster_size)].index)
        absurd["kept_cluster"] = absurd["cluster"].where(
            absurd["cluster"].isin(keep_clusters), other="other"
        )
        out["n_clusters"] = int(absurd["kept_cluster"].nunique())

        absurd = absurd.sort_values(
            ["kept_cluster", "haiku_score"], ascending=[True, False]
        )
        top_each = absurd.groupby("kept_cluster", group_keys=False).head(
            int(args.top_per_cluster)
        )
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        top_each.to_parquet(args.output_path, compression="zstd", index=False)
        out["n_kept"] = int(len(top_each))
        out["ok"] = True
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["traceback"] = traceback.format_exc()[:1000]
    return out
