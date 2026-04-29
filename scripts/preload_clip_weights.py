"""Pre-download the CLIP openai weights and store them on the Burla shared
filesystem at /workspace/shared/airbnb/clip_weights/openai.bin so workers can
load them directly via open_clip's `pretrained=<path>` arg without the per-node
cache thrashing that blew up Stage 2b at 500-worker scale.

Run once before s02b. Idempotent: skips download if file already present.
"""
from __future__ import annotations
import sys
import time

import os
import requests
import traceback

sys.path.insert(0, ".")

from dataclasses import dataclass
from dotenv import load_dotenv

import numpy as _np  # noqa: F401
import pandas as _pd  # noqa: F401
import pyarrow as _pa  # noqa: F401
import pyarrow.parquet as _pq  # noqa: F401

from src.lib.io import register_src_for_burla


CLIP_BIN_URL = (
    "https://huggingface.co/timm/vit_base_patch32_clip_224.openai/"
    "resolve/main/open_clip_pytorch_model.bin"
)
CLIP_BIN_PATH = "/workspace/shared/airbnb/clip_weights/openai.bin"


@dataclass
class PreloadArgs:
    url: str
    dest: str


def preload(args: PreloadArgs) -> dict:

    out: dict = {"ok": False, "error": None}
    try:
        os.makedirs(os.path.dirname(args.dest), exist_ok=True)
        if os.path.exists(args.dest) and os.path.getsize(args.dest) > 100_000_000:
            out.update(ok=True, skipped=True, size_bytes=os.path.getsize(args.dest))
            return out

        tmp_path = args.dest + ".tmp"
        t0 = time.time()
        with requests.get(args.url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total = 0
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    total += len(chunk)
        os.replace(tmp_path, args.dest)
        out.update(
            ok=True, skipped=False, size_bytes=total,
            elapsed_sec=round(time.time() - t0, 2),
        )
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[:2000]
    return out


def main() -> None:
    load_dotenv()
    register_src_for_burla()
    from burla import remote_parallel_map

    [res] = remote_parallel_map(
        preload,
        [PreloadArgs(url=CLIP_BIN_URL, dest=CLIP_BIN_PATH)],
        func_cpu=2, func_ram=8, max_parallelism=1, grow=True, spinner=True,
    )
    import json
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
