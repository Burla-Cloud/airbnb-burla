"""Burla worker functions for the post-CLIP Haiku Vision categorisation pass.

Two parallel batches of work, both running on the same shared FS layout:

1. ``haiku_validate_pet_batch`` -- a small batch of CLIP top-pet candidates is
   sent to Claude Haiku Vision with a strict yes/no question
   ("is there clearly a real cat or dog visible in this photograph?"). Only
   YES rows survive into ``pets_validated.parquet``.

2. ``haiku_room_category_batch`` -- a small batch of CLIP top-messy candidates
   is sent to Claude Haiku Vision with a strict 4-way classifier
   ("ugly_bathroom" / "hectic_kitchen" / "drug_den_vibes" / "none"). Each kept
   row carries a one-line caption + score; we then surface the top-K per
   category in stage 6.

Workers take a single ``@dataclass`` arg, write per-batch parquet to shared FS,
and never raise; errors are returned in the dict so other batches keep going.
"""
from __future__ import annotations

import os
import re
import time
import traceback
from dataclasses import dataclass

import numpy as _np  # noqa: F401
import pandas as _pd  # noqa: F401
import pyarrow as _pa  # noqa: F401
import pyarrow.parquet as _pq  # noqa: F401
import anthropic as _anthropic  # noqa: F401


@dataclass
class HaikuPetBatchArgs:
    batch_id: int
    rows: list           # [{image_id, listing_id, image_idx, image_url, clip_max}]
    output_root: str
    anthropic_api_key: str
    model: str
    max_tokens: int


@dataclass
class HaikuRoomBatchArgs:
    batch_id: int
    rows: list           # [{image_id, listing_id, image_idx, image_url, clip_messy, brightness}]
    output_root: str
    anthropic_api_key: str
    model: str
    max_tokens: int


@dataclass
class HaikuTvBatchArgs:
    batch_id: int
    rows: list           # [{image_id, listing_id, image_idx, image_url, clip_tv}]
    output_root: str
    anthropic_api_key: str
    model: str
    max_tokens: int


_PET_PROMPT = """You are validating Airbnb photos that a CLIP model thought might contain a real cat or dog.

For each photo, decide if there is CLEARLY a real, living animal visible in the photograph.

STRICT RULES:
- Painting, mural, sculpture, plush toy, throw pillow with animal print = NOT a real animal.
- Pet only partially visible behind furniture is fine, count it as YES if you can clearly identify it.
- Animals other than cats and dogs (birds, rabbits, hamsters, etc.) count as YES with animal_type set accordingly.
- If you are not sure, choose NO.

Return a JSON array. For every input photo, include exactly one object:
{{
  "image_id": <echo back the image_id integer>,
  "is_real_animal": true | false,
  "animal_type": "cat" | "dog" | "bird" | "rabbit" | "fish" | "other" | "none",
  "one_line": "<funny caption, max 14 words, no emoji, mentions the animal>",
  "score": <0 to 10 integer of how clearly the animal is visible>
}}

Return ONLY the JSON array, no prose. Photos:
{block}"""


_ROOM_PROMPT = """You are categorising Airbnb photos that a CLIP model thought were messy or cluttered.

For each photo, choose ONE category:

- "ugly_bathroom": a bathroom that is clearly grimy, dated in a sad way, dirty, mouldy, broken, or has visible filth. Should make a normal traveler think twice. A merely small or beige bathroom is NOT ugly.

- "hectic_kitchen": a kitchen that is genuinely chaotic, cluttered, dirty dishes everywhere, hoarder vibes, food residue, gross stove, exposed wiring, mismatched appliances stacked. A merely small kitchen is NOT hectic.

- "drug_den_vibes": a room that gives genuine "did someone do drugs here last night" energy. Stained mattresses on the floor, dim lighting, sketchy ashtrays, broken blinds, peeling walls, exposed wiring, bare bulbs, no decor. Should feel unsettling, not just minimalist.

- "none": none of the above clearly applies.

Be strict. Use "none" liberally. We only want photos where the category is OBVIOUS.

Return a JSON array. For every input photo, include exactly one object:
{{
  "image_id": <echo back the image_id integer>,
  "category": "ugly_bathroom" | "hectic_kitchen" | "drug_den_vibes" | "none",
  "one_line": "<funny caption, max 14 words, no emoji>",
  "score": <0 to 10 integer of how strongly the category applies>
}}

Return ONLY the JSON array, no prose. Photos:
{block}"""


def _haiku_call_with_retry(client, model: str, max_tokens: int, content: list):
    last_err = None
    for attempt in range(4):
        try:
            return client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": content}],
            )
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(20.0, 2.0 * (2 ** attempt)))
    raise RuntimeError(f"haiku api failed after retries: {last_err}")


def _build_vision_content(prompt: str, rows: list) -> list:
    import json as _json
    block = "\n".join(
        f'{{"image_id": {int(r["image_id"])}, '
        f'"image_url": {_json.dumps(str(r["image_url"]))}}}'
        for r in rows
    )
    content = [{"type": "text", "text": prompt.format(block=block)}]
    for r in rows:
        content.append({
            "type": "image",
            "source": {"type": "url", "url": str(r["image_url"])},
        })
    return content


def _parse_json_array(text: str) -> list:
    import json as _json
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        return _json.loads(m.group(0))
    except Exception:
        return []


def haiku_validate_pet_batch(args: HaikuPetBatchArgs) -> dict:
    """Send candidate pet photos to Haiku Vision with a yes/no validator.
    Writes one parquet per batch with one row per input. Skips if already done."""
    out = {
        "batch_id": args.batch_id, "n_inputs": len(args.rows),
        "n_ok": 0, "n_failed": 0, "shared_path": None,
        "elapsed_seconds": 0.0, "error": None, "resumed": False,
    }
    started = time.time()
    shared_path = os.path.join(args.output_root, f"batch_{args.batch_id:06d}.parquet")
    if os.path.exists(shared_path):
        try:
            import pandas as pd
            existing = pd.read_parquet(shared_path, columns=["image_id"])
            out["n_ok"] = int(len(existing))
            out["shared_path"] = shared_path
            out["resumed"] = True
            out["elapsed_seconds"] = time.time() - started
            return out
        except Exception:
            pass
    try:
        import anthropic
        import pandas as pd
        client = anthropic.Anthropic(api_key=args.anthropic_api_key)
        content = _build_vision_content(_PET_PROMPT, args.rows)
        resp = _haiku_call_with_retry(
            client, args.model, args.max_tokens * len(args.rows), content,
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        parsed = _parse_json_array(text)
        url_by_id = {int(r["image_id"]): r for r in args.rows}
        rows: list[dict] = []
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
                    "is_real_animal": bool(entry.get("is_real_animal", False)),
                    "animal_type": str(entry.get("animal_type", "none"))[:20],
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


_TV_PROMPT = """You are validating Airbnb photos that a CLIP model thought might show a TV mounted in an awkward, high, or above-fireplace position.

For each photo, decide if there is genuinely a TV mounted high on the wall in a way that would make a viewer comment on it. Specifically:

- "above_fireplace": a TV mounted directly above a fireplace mantel.
- "unusually_high": a TV bolted way up near the ceiling, in a position where you would have to crane your neck to watch it.
- "normal": TV is at normal viewing height OR there is no TV in the photo. We do not want this category.
- "no_tv": no TV visible in the photo.

Be strict: a TV at a normal viewing height is "normal", even if mounted on the wall.

Return a JSON array. For every input photo, include exactly one object:
{{
  "image_id": <echo back the image_id integer>,
  "tv_placement": "above_fireplace" | "unusually_high" | "normal" | "no_tv",
  "one_line": "<short caption, max 14 words, describes the placement>",
  "score": <0 to 10 integer of how absurd the placement is, 0 if normal or no TV>
}}

Return ONLY the JSON array, no prose. Photos:
{block}"""


def haiku_validate_tv_batch(args: HaikuTvBatchArgs) -> dict:
    """Send candidate TV-above-fireplace photos to Haiku Vision for validation."""
    out = {
        "batch_id": args.batch_id, "n_inputs": len(args.rows),
        "n_ok": 0, "n_failed": 0, "shared_path": None,
        "elapsed_seconds": 0.0, "error": None, "resumed": False,
    }
    started = time.time()
    shared_path = os.path.join(args.output_root, f"batch_{args.batch_id:06d}.parquet")
    if os.path.exists(shared_path):
        try:
            import pandas as pd
            existing = pd.read_parquet(shared_path, columns=["image_id"])
            out["n_ok"] = int(len(existing))
            out["shared_path"] = shared_path
            out["resumed"] = True
            out["elapsed_seconds"] = time.time() - started
            return out
        except Exception:
            pass
    try:
        import anthropic
        import pandas as pd
        client = anthropic.Anthropic(api_key=args.anthropic_api_key)
        content = _build_vision_content(_TV_PROMPT, args.rows)
        resp = _haiku_call_with_retry(
            client, args.model, args.max_tokens * len(args.rows), content,
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        parsed = _parse_json_array(text)
        url_by_id = {int(r["image_id"]): r for r in args.rows}
        rows: list[dict] = []
        valid_p = {"above_fireplace", "unusually_high", "normal", "no_tv"}
        for entry in parsed:
            try:
                iid = int(entry["image_id"])
                src = url_by_id.get(iid, {})
                placement = str(entry.get("tv_placement", "normal")).strip().lower()
                if placement not in valid_p:
                    placement = "normal"
                rows.append({
                    "image_id": iid,
                    "listing_id": int(src.get("listing_id", 0)),
                    "image_idx": int(src.get("image_idx", -1)),
                    "image_url": str(src.get("image_url", "")),
                    "clip_tv": float(src.get("clip_tv", 0.0)),
                    "tv_placement": placement,
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


def haiku_room_category_batch(args: HaikuRoomBatchArgs) -> dict:
    """Send candidate messy-room photos to Haiku Vision with a 4-way classifier."""
    out = {
        "batch_id": args.batch_id, "n_inputs": len(args.rows),
        "n_ok": 0, "n_failed": 0, "shared_path": None,
        "elapsed_seconds": 0.0, "error": None, "resumed": False,
    }
    started = time.time()
    shared_path = os.path.join(args.output_root, f"batch_{args.batch_id:06d}.parquet")
    if os.path.exists(shared_path):
        try:
            import pandas as pd
            existing = pd.read_parquet(shared_path, columns=["image_id"])
            out["n_ok"] = int(len(existing))
            out["shared_path"] = shared_path
            out["resumed"] = True
            out["elapsed_seconds"] = time.time() - started
            return out
        except Exception:
            pass
    try:
        import anthropic
        import pandas as pd
        client = anthropic.Anthropic(api_key=args.anthropic_api_key)
        content = _build_vision_content(_ROOM_PROMPT, args.rows)
        resp = _haiku_call_with_retry(
            client, args.model, args.max_tokens * len(args.rows), content,
        )
        text = "".join(b.text for b in resp.content if hasattr(b, "text"))
        parsed = _parse_json_array(text)
        url_by_id = {int(r["image_id"]): r for r in args.rows}
        rows: list[dict] = []
        valid_cats = {"ugly_bathroom", "hectic_kitchen", "drug_den_vibes", "none"}
        for entry in parsed:
            try:
                iid = int(entry["image_id"])
                src = url_by_id.get(iid, {})
                cat = str(entry.get("category", "none")).strip().lower()
                if cat not in valid_cats:
                    cat = "none"
                rows.append({
                    "image_id": iid,
                    "listing_id": int(src.get("listing_id", 0)),
                    "image_idx": int(src.get("image_idx", -1)),
                    "image_url": str(src.get("image_url", "")),
                    "clip_messy": float(src.get("clip_messy", 0.0)),
                    "brightness": float(src.get("brightness", 0.0)),
                    "category": cat,
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
class CategoriesMergeArgs:
    pets_root: str
    rooms_root: str
    tv_root: str
    pets_output: str
    rooms_output: str
    tv_output: str


def merge_categories(args: CategoriesMergeArgs) -> dict:
    """Concat all per-batch parquets in pets_root, rooms_root, tv_root and
    write merged parquets. Filter pets to is_real_animal=True, rooms to
    category != 'none', tv to placement in {above_fireplace, unusually_high}."""
    out = {
        "ok": False,
        "n_pet_batches": 0, "n_pets_input": 0, "n_pets_real": 0,
        "n_room_batches": 0, "n_rooms_input": 0, "n_rooms_kept": 0,
        "n_tv_batches": 0, "n_tv_input": 0, "n_tv_kept": 0,
        "by_category": {}, "by_tv_placement": {}, "error": None,
    }
    try:
        import glob
        import pandas as pd

        pet_files = sorted(glob.glob(os.path.join(args.pets_root, "batch_*.parquet")))
        out["n_pet_batches"] = len(pet_files)
        if pet_files:
            pets = pd.concat([pd.read_parquet(f) for f in pet_files], ignore_index=True)
            pets = pets.drop_duplicates(subset=["image_id"])
            out["n_pets_input"] = int(len(pets))
            real = pets[pets["is_real_animal"].astype(bool)].copy()
            real = real.sort_values("haiku_score", ascending=False)
            os.makedirs(os.path.dirname(args.pets_output), exist_ok=True)
            real.to_parquet(args.pets_output, compression="zstd", index=False)
            out["n_pets_real"] = int(len(real))

        room_files = sorted(glob.glob(os.path.join(args.rooms_root, "batch_*.parquet")))
        out["n_room_batches"] = len(room_files)
        if room_files:
            rooms = pd.concat([pd.read_parquet(f) for f in room_files], ignore_index=True)
            rooms = rooms.drop_duplicates(subset=["image_id"])
            out["n_rooms_input"] = int(len(rooms))
            kept = rooms[rooms["category"] != "none"].copy()
            kept = kept.sort_values(["category", "haiku_score"], ascending=[True, False])
            os.makedirs(os.path.dirname(args.rooms_output), exist_ok=True)
            kept.to_parquet(args.rooms_output, compression="zstd", index=False)
            out["n_rooms_kept"] = int(len(kept))
            out["by_category"] = {
                str(k): int(v) for k, v in kept["category"].value_counts().items()
            }

        tv_files = sorted(glob.glob(os.path.join(args.tv_root, "batch_*.parquet")))
        out["n_tv_batches"] = len(tv_files)
        if tv_files:
            tv = pd.concat([pd.read_parquet(f) for f in tv_files], ignore_index=True)
            tv = tv.drop_duplicates(subset=["image_id"])
            out["n_tv_input"] = int(len(tv))
            keep_p = {"above_fireplace", "unusually_high"}
            tv_kept = tv[tv["tv_placement"].isin(keep_p)].copy()
            tv_kept = tv_kept.sort_values("haiku_score", ascending=False)
            os.makedirs(os.path.dirname(args.tv_output), exist_ok=True)
            tv_kept.to_parquet(args.tv_output, compression="zstd", index=False)
            out["n_tv_kept"] = int(len(tv_kept))
            out["by_tv_placement"] = {
                str(k): int(v) for k, v in tv_kept["tv_placement"].value_counts().items()
            }
        out["ok"] = True
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["traceback"] = traceback.format_exc()[:1000]
    return out
