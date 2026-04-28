"""Top-level Burla worker functions for the photo manifest scrape (Stage 2a).

We pull `airbnb.com/rooms/<id>` HTML, extract photo URLs from the inline
``<script id="data-deferred-state-0">`` JSON blob (Airbnb's hydration payload),
and write a per-batch parquet to ``/workspace/shared/airbnb/photos/batches/``.

The HTML format is undocumented and Datadome will rate-limit aggressively.
We cap each worker at ~0.5 req/sec, retry on 429 with exponential backoff,
and capture the failure mode so the sample run can decide whether to halt.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import traceback
from dataclasses import dataclass, asdict
from typing import Optional

import requests

# Hoist these so Burla pip-installs them on workers (see image_tasks.py).
import numpy as _np  # noqa: F401
import pandas as _pd  # noqa: F401
import pyarrow as _pa  # noqa: F401
import pyarrow.parquet as _pq  # noqa: F401


_USER_AGENTS = [
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
]
_BASE_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

_DEFERRED_STATE_RE = re.compile(
    r'<script[^>]*id="data-deferred-state-0"[^>]*>(.*?)</script>',
    re.DOTALL,
)
_MUSCACHE_RE = re.compile(
    r'https?://[a-z0-9]+\.muscache\.com/(?:im/)?pictures/[^"\'\\\s]+',
    re.IGNORECASE,
)
_LISTING_PHOTO_BAD_TOKENS = (
    "AirbnbPlatformAssets",
    "/user/User-",
    "UserProfile",
    "search-bar-icons",
    "user-profile-pic",
    "icons-rebrand",
    "/airbnb/static/",
)
_BLOCK_MARKERS = (
    "captcha-delivery.com",
    "Please enable JS and disable any ad blocker",
    "/captcha-delivery/",
    "geo.captcha-delivery.com",
)
_NOT_FOUND_TITLE = "404 Page Not Found"


@dataclass
class FetchRoomArgs:
    listing_id: int
    listing_url: str = ""


@dataclass
class ScrapeBatchArgs:
    batch_id: int
    listing_ids: list[int]
    shared_root: str
    req_per_sec_per_worker: float = 0.5
    retry_limit: int = 2


def _build_room_url(listing_id: int, fallback_url: str = "") -> str:
    if fallback_url and "airbnb.com/rooms/" in fallback_url:
        return fallback_url.split("?")[0]
    return f"https://www.airbnb.com/rooms/{listing_id}"


def _classify_response(status: int, body: str) -> str:
    if status == 200 and any(m in body for m in _BLOCK_MARKERS):
        return "datadome_block"
    if status == 200 and _NOT_FOUND_TITLE in body and "data-deferred-state-0" not in body:
        return "not_found"
    if status == 200:
        return "ok"
    if status in (403, 451):
        return "forbidden"
    if status == 429:
        return "rate_limited"
    if status in (404, 410):
        return "not_found"
    if 500 <= status < 600:
        return "server_error"
    return f"http_{status}"


def fetch_room(args: FetchRoomArgs, *, retry_limit: int = 2) -> dict:
    """Fetch one Airbnb room page and return a dict with photo URLs + status."""
    url = _build_room_url(args.listing_id, args.listing_url)
    out = {
        "listing_id": args.listing_id,
        "url": url,
        "status": "unknown",
        "n_photos": 0,
        "photo_urls": [],
        "title": None,
        "error": None,
    }

    for attempt in range(retry_limit + 1):
        try:
            headers = {
                **_BASE_HEADERS,
                "User-Agent": random.choice(_USER_AGENTS),
            }
            r = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
            body = r.text or ""
            classification = _classify_response(r.status_code, body)

            if classification != "ok":
                out["status"] = classification
                if classification in ("rate_limited", "datadome_block") and attempt < retry_limit:
                    time.sleep(min(30.0, 2.0 * (2 ** attempt)) + random.uniform(0, 1.0))
                    continue
                return out

            photos = _extract_photos(body)
            title = _extract_title(body)
            out["status"] = "ok" if photos else "ok_empty"
            out["photo_urls"] = photos
            out["n_photos"] = len(photos)
            out["title"] = title
            return out
        except Exception as e:
            out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            if attempt < retry_limit:
                time.sleep(min(15.0, 1.0 * (2 ** attempt)) + random.uniform(0, 1.0))
                continue
            out["status"] = "exception"
            return out
    return out


def _extract_photos(html: str) -> list[str]:
    """Pull listing photo CDN URLs out of the page.

    Airbnb's HTML contains a mix of listing photos, host avatars, UI icons, and
    AI-generated review summaries, all served from the same muscache.com CDN.
    We exclude anything that looks like a UI asset or a user avatar via
    ``_LISTING_PHOTO_BAD_TOKENS``.
    """
    urls: list[str] = []
    seen = set()

    def _maybe_add(raw: str) -> None:
        clean = raw.split("?")[0]
        if any(tok in clean for tok in _LISTING_PHOTO_BAD_TOKENS):
            return
        if clean in seen:
            return
        seen.add(clean)
        urls.append(clean)

    m = _DEFERRED_STATE_RE.search(html)
    if m:
        try:
            payload = json.loads(_unescape_json(m.group(1)))
            for url in _walk_for_picture_urls(payload):
                _maybe_add(url)
        except (ValueError, TypeError):
            pass
    for m2 in _MUSCACHE_RE.finditer(html):
        _maybe_add(m2.group(0))
    return urls


def _unescape_json(s: str) -> str:
    return s.replace("&quot;", '"').replace("&amp;", "&").replace("&#x27;", "'").replace("\\u002F", "/")


def _walk_for_picture_urls(node, out: Optional[list] = None) -> list[str]:
    if out is None:
        out = []
    if isinstance(node, dict):
        for k, v in node.items():
            kl = str(k).lower()
            if kl in ("baseurl", "picture", "pictureurl", "url") and isinstance(v, str):
                if "muscache.com" in v and ("pictures/" in v or "/im/" in v):
                    out.append(v.split("?")[0])
            else:
                _walk_for_picture_urls(v, out)
    elif isinstance(node, list):
        for v in node:
            _walk_for_picture_urls(v, out)
    return out


def _extract_title(html: str) -> Optional[str]:
    m = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    title = m.group(1).strip()
    if " - Airbnb" in title:
        title = title.split(" - Airbnb")[0]
    return title[:200]


def scrape_batch(args: ScrapeBatchArgs) -> dict:
    """Scrape one batch of listings and write a parquet to shared FS.

    Each worker enforces ``req_per_sec_per_worker`` to stay polite. Returns a
    summary the orchestrator uses to compute success rate + halt thresholds.
    """
    out = {
        "batch_id": args.batch_id,
        "n_listings": len(args.listing_ids),
        "n_ok": 0,
        "n_empty": 0,
        "n_blocked": 0,
        "n_failed": 0,
        "n_total_photos": 0,
        "shared_path": None,
        "status_counts": {},
        "elapsed_seconds": 0.0,
        "error": None,
    }
    started = time.time()
    rows: list[dict] = []
    sleep_target = 1.0 / max(args.req_per_sec_per_worker, 0.001)

    try:
        os.makedirs(args.shared_root, exist_ok=True)

        shared_path = os.path.join(args.shared_root, f"batch_{args.batch_id:06d}.parquet")
        if os.path.exists(shared_path):
            try:
                import pandas as pd
                existing = pd.read_parquet(shared_path, columns=["listing_id"])
                out["n_listings"] = int(existing["listing_id"].nunique())
                out["n_ok"] = out["n_listings"]
                out["n_total_photos"] = int(len(existing))
                out["shared_path"] = shared_path
                out["resumed"] = True
                out["elapsed_seconds"] = time.time() - started
                return out
            except Exception:
                pass

        for lid in args.listing_ids:
            t = time.time()
            r = fetch_room(FetchRoomArgs(listing_id=int(lid)), retry_limit=args.retry_limit)
            status = r["status"]
            out["status_counts"][status] = out["status_counts"].get(status, 0) + 1
            if status == "ok":
                out["n_ok"] += 1
                out["n_total_photos"] += r["n_photos"]
                for idx, url in enumerate(r["photo_urls"]):
                    rows.append({
                        "listing_id": int(lid),
                        "image_idx": idx,
                        "image_url": url,
                        "title": r.get("title"),
                        "scraped_at": started,
                    })
            elif status == "ok_empty":
                out["n_empty"] += 1
            elif status in ("datadome_block", "rate_limited", "forbidden"):
                out["n_blocked"] += 1
            else:
                out["n_failed"] += 1
            elapsed = time.time() - t
            sleep_for = max(0.0, sleep_target - elapsed) + random.uniform(0, 0.4)
            time.sleep(sleep_for)

        if rows:
            import pandas as pd
            pd.DataFrame(rows).to_parquet(shared_path, compression="zstd", index=False)
            out["shared_path"] = shared_path
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["traceback"] = traceback.format_exc()[:1000]

    out["elapsed_seconds"] = time.time() - started
    return out


@dataclass
class ListListingIdsArgs:
    listings_parquet_path: str
    sample_n: int = 0
    seed: int = 42


def list_listing_ids(args: ListListingIdsArgs) -> dict:
    """Run on Burla. Reads listings_clean.parquet from shared FS, returns ids."""
    import pandas as pd
    df = pd.read_parquet(
        args.listings_parquet_path, columns=["listing_id", "picture_url", "listing_url"],
    )
    df = df.drop_duplicates(subset=["listing_id"])
    n_total_unique = int(len(df))
    if args.sample_n and args.sample_n < len(df):
        df = df.sample(n=args.sample_n, random_state=args.seed)
    return {
        "n_total": n_total_unique,
        "listing_ids": df["listing_id"].astype(int).tolist(),
    }


@dataclass
class MergePhotosArgs:
    shared_root: str
    output_path: str
    listings_history_path: str = ""  # optional: union picture_urls across snapshots


def merge_photo_batches(args: MergePhotosArgs) -> dict:
    """Reduce per-batch scrape parquets into one ``photo_manifest.parquet``.

    Steps:
    1. Concat every ``batch_*.parquet`` from the scrape stage.
    2. Drop intra-batch duplicates by ``(listing_id, image_idx)``.
    3. Optional: union with hero ``picture_url`` from
       ``listings_history.parquet`` so listings whose host later changed the
       hero photo still get every historical hero pulled. This is the
       cross-snapshot dedupe step.
    4. Final dedupe by ``image_url`` so the downloader/CLIP scorer never
       processes the same physical URL twice.
    """
    out = {
        "ok": False, "n_files": 0, "n_rows": 0, "n_listings": 0,
        "n_history_added": 0, "n_after_url_dedupe": 0,
        "output_path": args.output_path, "error": None,
    }
    try:
        import glob
        import pandas as pd
        files = sorted(glob.glob(os.path.join(args.shared_root, "batch_*.parquet")))
        out["n_files"] = len(files)
        if not files:
            out["error"] = f"no batch parquets at {args.shared_root}"
            return out
        dfs = [pd.read_parquet(f) for f in files]
        big = pd.concat(dfs, ignore_index=True)
        big = big.drop_duplicates(subset=["listing_id", "image_idx"])

        if args.listings_history_path and os.path.exists(args.listings_history_path):
            try:
                hist = pd.read_parquet(
                    args.listings_history_path,
                    columns=[c for c in (
                        "listing_id", "snapshot_date", "picture_url"
                    ) if c is not None],
                )
                hist = hist[hist["picture_url"].astype(str).str.startswith("http")]
                hist = hist.drop_duplicates(
                    subset=["listing_id", "picture_url"]
                )
                hist = hist.rename(columns={"picture_url": "image_url"})
                hist["image_idx"] = -1  # sentinel for "Inside Airbnb hero"
                if "scrape_ok" in big.columns:
                    hist["scrape_ok"] = True
                merge_cols = [c for c in big.columns if c in hist.columns]
                hist = hist[merge_cols + [c for c in hist.columns if c not in merge_cols]]
                before = len(big)
                big = pd.concat([big, hist], ignore_index=True)
                out["n_history_added"] = int(len(big) - before)
            except Exception as e:  # noqa: BLE001
                out["history_error"] = f"{type(e).__name__}: {str(e)[:200]}"

        if "image_url" in big.columns:
            big = big.drop_duplicates(subset=["image_url"])
        out["n_after_url_dedupe"] = int(len(big))

        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        big.to_parquet(args.output_path, compression="zstd", index=False)
        out["ok"] = True
        out["n_rows"] = int(len(big))
        out["n_listings"] = int(big["listing_id"].nunique())
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["traceback"] = traceback.format_exc()[:1000]
    return out
