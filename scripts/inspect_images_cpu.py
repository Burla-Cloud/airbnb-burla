"""Inspect /workspace/shared/airbnb/images_cpu.parquet to see if it has clip_* cols."""
from __future__ import annotations
import sys

import pyarrow.parquet as pq

sys.path.insert(0, ".")

from dataclasses import dataclass
from dotenv import load_dotenv

import numpy as _np  # noqa: F401
import pandas as _pd  # noqa: F401
import pyarrow as _pa  # noqa: F401
import pyarrow.parquet as _pq  # noqa: F401

from src.config import SHARED_ROOT
from src.lib.io import register_src_for_burla


@dataclass
class A:
    path: str


def inspect(args: A) -> dict:
    sch = pq.read_schema(args.path)
    return {
        "n_rows": int(pq.read_metadata(args.path).num_rows),
        "columns": sch.names,
    }


def main() -> None:
    load_dotenv()
    register_src_for_burla()
    from burla import remote_parallel_map
    [res] = remote_parallel_map(
        inspect, [A(path=f"{SHARED_ROOT}/images_cpu.parquet")],
        func_cpu=2, func_ram=4, max_parallelism=1, grow=True, spinner=True,
    )
    import json
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
