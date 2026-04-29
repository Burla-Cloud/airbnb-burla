"""Inspect the batch parquet from debug_cpu_batch to see per-row errors."""
from __future__ import annotations

import sys

import pandas as pd

sys.path.insert(0, ".")

from dotenv import load_dotenv

import numpy as _np  # noqa: F401
import pandas as _pd  # noqa: F401
import pyarrow as _pa  # noqa: F401
import pyarrow.parquet as _pq  # noqa: F401

from dataclasses import dataclass

from src.config import SHARED_ROOT, SHARED_IMAGES_CPU
from src.lib.io import register_src_for_burla


@dataclass
class InspectArgs:
    path: str


def inspect(args: InspectArgs) -> dict:
    df = pd.read_parquet(args.path)
    return {
        "n_rows": int(len(df)),
        "columns": df.columns.tolist(),
        "head": df.head(5).to_dict("records"),
        "error_value_counts": df["error"].value_counts(dropna=False).to_dict() if "error" in df.columns else {},
        "image_url_samples": df["image_url"].head(5).tolist(),
    }


def main() -> None:
    load_dotenv()
    register_src_for_burla()
    from burla import remote_parallel_map

    [res] = remote_parallel_map(
        inspect,
        [InspectArgs(path=f"{SHARED_IMAGES_CPU}_debug/batch_000000.parquet")],
        func_cpu=2, func_ram=4, max_parallelism=1, grow=True, spinner=True,
    )
    import json
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
