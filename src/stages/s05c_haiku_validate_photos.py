"""Stage 5c: Haiku Vision validators for pets + room categories.

Two pools, both selected from images_cpu.parquet on the shared FS:

1. Pets: top ``CAT_PETS_TOP_K`` photos by max(clip_pet_dog, clip_pet_cat,
   clip_pet_on_furniture). Run through Haiku with a strict yes/no validator.
   Only YES rows survive into pets_validated.parquet.

2. Rooms: top ``CAT_ROOMS_TOP_K`` photos by clip_messy_room (deduped per
   listing). Run through Haiku with a 4-way classifier into
   ugly_bathroom / hectic_kitchen / drug_den_vibes / none. Anything not "none"
   is kept in room_categories.parquet for stage 6.

Stage 6 then surfaces top-K per section and writes the per-section JSON.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv

import numpy as _np  # noqa: F401
import pandas as _pd  # noqa: F401
import pyarrow as _pa  # noqa: F401
import pyarrow.parquet as _pq  # noqa: F401

from ..config import (
    ANTHROPIC_MAX_TOKENS, ANTHROPIC_MODEL, SHARED_ROOT,
)
from ..lib.budget import BudgetTracker
from ..lib.io import register_src_for_burla
from ..tasks.categories_tasks import (
    HaikuPetBatchArgs, HaikuRoomBatchArgs, HaikuTvBatchArgs, CategoriesMergeArgs,
    haiku_validate_pet_batch, haiku_room_category_batch,
    haiku_validate_tv_batch, merge_categories,
)


import pandas as pd
import traceback as _tb

CAT_PETS_TOP_K = 1500
CAT_ROOMS_TOP_K = 4000
CAT_TV_TOP_K = 2000
CAT_BATCH_SIZE = 6
CAT_MAX_PARALLELISM = 200

SHARED_HAIKU_PETS = f"{SHARED_ROOT}/haiku_pets_v1"
SHARED_HAIKU_ROOMS = f"{SHARED_ROOT}/haiku_rooms_v1"
SHARED_HAIKU_TV = f"{SHARED_ROOT}/haiku_tv_v1"
PETS_OUTPUT = f"{SHARED_ROOT}/pets_validated.parquet"
ROOMS_OUTPUT = f"{SHARED_ROOT}/room_categories.parquet"
TV_OUTPUT = f"{SHARED_ROOT}/tv_validated.parquet"


@dataclass
class SelectCategoryCandidatesArgs:
    images_cpu_path: str
    pets_top_k: int
    rooms_top_k: int
    tv_top_k: int


def select_category_candidates(args: SelectCategoryCandidatesArgs) -> dict:
    """Run on Burla. Read images_cpu.parquet once, return three top-K row lists."""
    out = {
        "ok": False, "pets": [], "rooms": [], "tv": [], "error": None,
        "n_total": 0, "n_pets": 0, "n_rooms": 0, "n_tv": 0,
    }
    try:
        cols = ["listing_id", "image_idx", "image_url", "download_ok",
                "brightness",
                "clip_messy_room", "clip_tv_above_fireplace",
                "clip_pet_dog", "clip_pet_cat", "clip_pet_on_furniture"]
        df = pd.read_parquet(args.images_cpu_path, columns=cols)
        df = df[df["download_ok"].astype(bool)]
        out["n_total"] = int(len(df))

        pet_df = df.copy()
        pet_df["clip_max"] = pet_df[
            ["clip_pet_dog", "clip_pet_cat", "clip_pet_on_furniture"]
        ].max(axis=1)
        pet_df = pet_df.sort_values("clip_max", ascending=False)
        pet_df = pet_df.drop_duplicates(subset=["listing_id"], keep="first")
        pet_df = pet_df.head(int(args.pets_top_k)).reset_index(drop=True)
        pet_df["image_id"] = pet_df.index.astype(int)
        out["pets"] = pet_df[
            ["image_id", "listing_id", "image_idx", "image_url", "clip_max"]
        ].to_dict("records")
        out["n_pets"] = len(out["pets"])

        room_df = df.copy()
        room_df = room_df.sort_values("clip_messy_room", ascending=False)
        room_df = room_df.drop_duplicates(subset=["listing_id"], keep="first")
        room_df = room_df.head(int(args.rooms_top_k)).reset_index(drop=True)
        room_df["image_id"] = (1_000_000_000 + room_df.index).astype(int)
        room_df = room_df.rename(columns={"clip_messy_room": "clip_messy"})
        out["rooms"] = room_df[
            ["image_id", "listing_id", "image_idx", "image_url",
             "clip_messy", "brightness"]
        ].to_dict("records")
        out["n_rooms"] = len(out["rooms"])

        tv_df = df.copy()
        tv_df = tv_df.sort_values("clip_tv_above_fireplace", ascending=False)
        tv_df = tv_df.drop_duplicates(subset=["listing_id"], keep="first")
        tv_df = tv_df.head(int(args.tv_top_k)).reset_index(drop=True)
        tv_df["image_id"] = (2_000_000_000 + tv_df.index).astype(int)
        tv_df = tv_df.rename(columns={"clip_tv_above_fireplace": "clip_tv"})
        out["tv"] = tv_df[
            ["image_id", "listing_id", "image_idx", "image_url", "clip_tv"]
        ].to_dict("records")
        out["n_tv"] = len(out["tv"])

        out["ok"] = True
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
        raise SystemExit("[s05c] ANTHROPIC_API_KEY not set")

    cpu_shared = f"{SHARED_ROOT}/images_cpu.parquet"
    print(
        f"[s05c] selecting top-{CAT_PETS_TOP_K} pet + top-{CAT_ROOMS_TOP_K} room "
        f"+ top-{CAT_TV_TOP_K} tv candidates from {cpu_shared} ...", flush=True,
    )
    [picked] = remote_parallel_map(
        select_category_candidates,
        [SelectCategoryCandidatesArgs(
            images_cpu_path=cpu_shared,
            pets_top_k=CAT_PETS_TOP_K,
            rooms_top_k=CAT_ROOMS_TOP_K,
            tv_top_k=CAT_TV_TOP_K,
        )],
        func_cpu=8, func_ram=64, max_parallelism=1, grow=True, spinner=False,
    )
    if not picked.get("ok"):
        print(f"[s05c] select failed: {picked.get('error')}", flush=True)
        if picked.get("traceback"):
            print(picked["traceback"], flush=True)
        raise SystemExit("[s05c] cannot continue")
    pet_rows = picked["pets"]
    room_rows = picked["rooms"]
    tv_rows = picked["tv"]
    print(
        f"[s05c]   selected {len(pet_rows):,} pet + "
        f"{len(room_rows):,} room + {len(tv_rows):,} tv candidates "
        f"of {picked['n_total']:,} total",
        flush=True,
    )

    pet_batches: list[HaikuPetBatchArgs] = []
    for i in range(0, len(pet_rows), CAT_BATCH_SIZE):
        pet_batches.append(HaikuPetBatchArgs(
            batch_id=i // CAT_BATCH_SIZE,
            rows=pet_rows[i: i + CAT_BATCH_SIZE],
            output_root=SHARED_HAIKU_PETS,
            anthropic_api_key=api_key,
            model=ANTHROPIC_MODEL,
            max_tokens=ANTHROPIC_MAX_TOKENS,
        ))
    room_batches: list[HaikuRoomBatchArgs] = []
    for i in range(0, len(room_rows), CAT_BATCH_SIZE):
        room_batches.append(HaikuRoomBatchArgs(
            batch_id=i // CAT_BATCH_SIZE,
            rows=room_rows[i: i + CAT_BATCH_SIZE],
            output_root=SHARED_HAIKU_ROOMS,
            anthropic_api_key=api_key,
            model=ANTHROPIC_MODEL,
            max_tokens=ANTHROPIC_MAX_TOKENS,
        ))
    tv_batches: list[HaikuTvBatchArgs] = []
    for i in range(0, len(tv_rows), CAT_BATCH_SIZE):
        tv_batches.append(HaikuTvBatchArgs(
            batch_id=i // CAT_BATCH_SIZE,
            rows=tv_rows[i: i + CAT_BATCH_SIZE],
            output_root=SHARED_HAIKU_TV,
            anthropic_api_key=api_key,
            model=ANTHROPIC_MODEL,
            max_tokens=ANTHROPIC_MAX_TOKENS,
        ))
    n_workers = min(
        CAT_MAX_PARALLELISM,
        max(1, len(pet_batches) + len(room_batches) + len(tv_batches)),
    )
    print(
        f"[s05c]   {len(pet_batches):,} pet + {len(room_batches):,} room + "
        f"{len(tv_batches):,} tv batches of {CAT_BATCH_SIZE}, "
        f"max {n_workers} Haiku workers",
        flush=True,
    )

    t0 = time.time()
    if pet_batches:
        with BudgetTracker("s05c_categories_pets", n_inputs=len(pet_rows), func_cpu=2) as bt:
            bt.set_workers(n_workers)
            results = remote_parallel_map(
                haiku_validate_pet_batch, pet_batches,
                func_cpu=2, func_ram=8,
                max_parallelism=n_workers, grow=True, spinner=False,
            )
            n_ok = sum(int(r.get("n_ok", 0)) for r in results)
            n_failed = sum(int(r.get("n_failed", 0)) for r in results)
            bt.set_succeeded(n_ok)
            bt.set_failed(n_failed)
            print(
                f"[s05c]   pet validator: {n_ok:,}/{len(pet_rows):,} scored "
                f"in {time.time()-t0:.1f}s",
                flush=True,
            )

    t1 = time.time()
    if room_batches:
        with BudgetTracker("s05c_categories_rooms", n_inputs=len(room_rows), func_cpu=2) as bt:
            bt.set_workers(n_workers)
            results = remote_parallel_map(
                haiku_room_category_batch, room_batches,
                func_cpu=2, func_ram=8,
                max_parallelism=n_workers, grow=True, spinner=False,
            )
            n_ok = sum(int(r.get("n_ok", 0)) for r in results)
            n_failed = sum(int(r.get("n_failed", 0)) for r in results)
            bt.set_succeeded(n_ok)
            bt.set_failed(n_failed)
            print(
                f"[s05c]   room classifier: {n_ok:,}/{len(room_rows):,} scored "
                f"in {time.time()-t1:.1f}s",
                flush=True,
            )

    t2 = time.time()
    if tv_batches:
        with BudgetTracker("s05c_categories_tv", n_inputs=len(tv_rows), func_cpu=2) as bt:
            bt.set_workers(n_workers)
            results = remote_parallel_map(
                haiku_validate_tv_batch, tv_batches,
                func_cpu=2, func_ram=8,
                max_parallelism=n_workers, grow=True, spinner=False,
            )
            n_ok = sum(int(r.get("n_ok", 0)) for r in results)
            n_failed = sum(int(r.get("n_failed", 0)) for r in results)
            bt.set_succeeded(n_ok)
            bt.set_failed(n_failed)
            print(
                f"[s05c]   tv validator: {n_ok:,}/{len(tv_rows):,} scored "
                f"in {time.time()-t2:.1f}s",
                flush=True,
            )

    [merge] = remote_parallel_map(
        merge_categories,
        [CategoriesMergeArgs(
            pets_root=SHARED_HAIKU_PETS,
            rooms_root=SHARED_HAIKU_ROOMS,
            tv_root=SHARED_HAIKU_TV,
            pets_output=PETS_OUTPUT,
            rooms_output=ROOMS_OUTPUT,
            tv_output=TV_OUTPUT,
        )],
        func_cpu=8, func_ram=64, max_parallelism=1, grow=True, spinner=False,
    )
    if not merge.get("ok"):
        raise SystemExit(f"[s05c] merge failed: {merge.get('error')}")
    print(
        f"[s05c]   merged: {merge['n_pets_real']:,} real pets / {merge['n_pets_input']:,}, "
        f"{merge['n_rooms_kept']:,} rooms / {merge['n_rooms_input']:,}, "
        f"{merge['n_tv_kept']:,} tvs / {merge['n_tv_input']:,}, "
        f"by_category: {merge.get('by_category', {})}, "
        f"by_tv_placement: {merge.get('by_tv_placement', {})}",
        flush=True,
    )
    print(
        f"[s05c] DONE. pets_validated.parquet, room_categories.parquet, "
        f"tv_validated.parquet on {SHARED_ROOT}",
        flush=True,
    )


if __name__ == "__main__":
    main()
