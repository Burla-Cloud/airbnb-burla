"""Pre-download YOLOv8n weights to /workspace/shared so GPU workers don't race
on the GitHub release URL the first time they call _ensure_yolo()."""
from __future__ import annotations
import sys

import os
import requests
import traceback

sys.path.insert(0, ".")

import time
from dataclasses import dataclass
from dotenv import load_dotenv

from src.lib.io import register_src_for_burla


YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
YOLO_DEST = "/workspace/shared/airbnb/yolo_weights/yolov8n.pt"


@dataclass
class PreloadArgs:
    url: str
    dest: str


def preload(args: PreloadArgs) -> dict:

    out: dict = {"ok": False, "error": None}
    try:
        os.makedirs(os.path.dirname(args.dest), exist_ok=True)
        if os.path.exists(args.dest) and os.path.getsize(args.dest) > 1_000_000:
            out.update(ok=True, skipped=True, size_bytes=os.path.getsize(args.dest))
            return out
        tmp = args.dest + ".tmp"
        t0 = time.time()
        with requests.get(args.url, stream=True, timeout=120, allow_redirects=True) as r:
            r.raise_for_status()
            total = 0
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    total += len(chunk)
        os.replace(tmp, args.dest)
        out.update(
            ok=True, skipped=False, size_bytes=total,
            elapsed_sec=round(time.time() - t0, 2),
        )
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[:1500]
    return out


def main() -> None:
    load_dotenv()
    register_src_for_burla()
    from burla import remote_parallel_map

    [res] = remote_parallel_map(
        preload, [PreloadArgs(url=YOLO_URL, dest=YOLO_DEST)],
        func_cpu=2, func_ram=4, max_parallelism=1, grow=True, spinner=True,
    )
    import json
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
