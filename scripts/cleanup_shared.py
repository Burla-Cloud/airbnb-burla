"""Free space on /workspace/shared by deleting old intermediate batch parquets.

We keep:
- listings/*.parquet (Stage 1 output, needed by Stage 2b/4)
- photo_manifest.parquet (Stage 2a output)
- listings_clean.parquet (Stage 1 reduce output)

We delete:
- photos/batch_*.parquet (rolled into photo_manifest.parquet)
- photos_sample/* (sample run artifacts)
- images_cpu_*/ (intermediate before merge)
- images_cpu_debug/ (debug artifacts)
"""
from __future__ import annotations
import sys

import os
import glob
import shutil

sys.path.insert(0, ".")

from dataclasses import dataclass
from dotenv import load_dotenv

import pyarrow as _pa  # noqa: F401
import pyarrow.parquet as _pq  # noqa: F401

from src.lib.io import register_src_for_burla


@dataclass
class CleanupArgs:
    pass


def cleanup(_a: CleanupArgs) -> dict:

    out = {"deleted_files": 0, "deleted_dirs": [], "freed_bytes": 0, "errors": []}
    targets = [
        "/workspace/shared/airbnb/photos",
        "/workspace/shared/airbnb/photos_sample",
        "/workspace/shared/airbnb/images_cpu",
        "/workspace/shared/airbnb/images_cpu_sample",
        "/workspace/shared/airbnb/images_cpu_debug",
    ]
    for t in targets:
        if not os.path.exists(t):
            continue
        try:
            total = 0
            for root, _, files in os.walk(t):
                for f in files:
                    fp = os.path.join(root, f)
                    try:
                        total += os.path.getsize(fp)
                        out["deleted_files"] += 1
                    except Exception:
                        pass
            out["freed_bytes"] += total
            shutil.rmtree(t, ignore_errors=True)
            out["deleted_dirs"].append(t)
        except Exception as e:
            out["errors"].append(f"{t}: {type(e).__name__}: {e}")

    out["freed_gb"] = round(out["freed_bytes"] / 1e9, 3)
    return out


def main() -> None:
    load_dotenv()
    register_src_for_burla()
    from burla import remote_parallel_map

    [res] = remote_parallel_map(
        cleanup,
        [CleanupArgs()],
        func_cpu=4, func_ram=16, max_parallelism=1, grow=True, spinner=True,
    )
    import json
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
