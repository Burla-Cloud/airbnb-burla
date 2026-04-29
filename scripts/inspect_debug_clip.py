"""Verify the debug batch parquet has clip_* columns and reasonable values."""
from __future__ import annotations
import sys

import pandas as pd

sys.path.insert(0, ".")

from dataclasses import dataclass
from dotenv import load_dotenv

import numpy as _np  # noqa: F401
import pandas as _pd  # noqa: F401
import pyarrow as _pa  # noqa: F401
import pyarrow.parquet as _pq  # noqa: F401

from src.config import SHARED_IMAGES_CPU
from src.lib.io import register_src_for_burla


@dataclass
class A:
    path: str


def inspect(args: A) -> dict:
    df = pd.read_parquet(args.path)
    return {
        "columns": df.columns.tolist(),
        "n_rows": int(len(df)),
        "head_first_row": df.head(1).to_dict("records")[0] if len(df) else {},
    }


def main() -> None:
    load_dotenv()
    register_src_for_burla()
    from burla import remote_parallel_map

    [res] = remote_parallel_map(
        inspect,
        [A(path=f"{SHARED_IMAGES_CPU}_debug/batch_000000.parquet")],
        func_cpu=2, func_ram=4, max_parallelism=1, grow=True, spinner=True,
    )
    import json
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
