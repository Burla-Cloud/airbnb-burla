"""Delete the smoke-test debug batch parquet so the full run starts clean."""
from __future__ import annotations
import sys

import os
import shutil

sys.path.insert(0, ".")

from dataclasses import dataclass
from dotenv import load_dotenv

from src.lib.io import register_src_for_burla


@dataclass
class A:
    pass


def cleanup(_a: A) -> dict:
    out = {"removed": []}
    for path in [
        "/workspace/shared/airbnb/images_gpu_debug",
        "/workspace/shared/airbnb/images_gpu/batch_999999.parquet",
    ]:
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
                out["removed"].append(path)
            elif os.path.isfile(path):
                os.remove(path)
                out["removed"].append(path)
        except Exception as e:
            out.setdefault("errors", []).append(f"{path}: {e}")
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
