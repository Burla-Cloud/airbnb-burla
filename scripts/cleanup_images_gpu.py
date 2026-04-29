"""Delete the failed Stage 3 batch parquets (all had 'No module named ultralytics')."""
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
class A:
    pass


def cleanup(_a: A) -> dict:

    out = {"deleted_files": 0, "freed_gb": 0.0}
    target = "/workspace/shared/airbnb/images_gpu"
    if not os.path.exists(target):
        out["status"] = "no-such-dir"
        return out
    total = 0
    for f in glob.glob(os.path.join(target, "batch_*.parquet")):
        try:
            total += os.path.getsize(f)
            os.remove(f)
            out["deleted_files"] += 1
        except Exception:
            pass
    out["freed_gb"] = round(total / 1e9, 3)
    out["status"] = "done"
    return out


def main() -> None:
    load_dotenv()
    register_src_for_burla()
    from burla import remote_parallel_map

    [res] = remote_parallel_map(
        cleanup, [A()], func_cpu=2, func_ram=4, max_parallelism=1, grow=True, spinner=True,
    )
    import json
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
