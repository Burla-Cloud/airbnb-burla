"""Inspect a single Stage 3 batch parquet to check whether YOLO actually ran."""
from __future__ import annotations
import sys

import os
import glob
import pandas as pd

sys.path.insert(0, ".")

from dataclasses import dataclass
from dotenv import load_dotenv

import numpy as _np  # noqa: F401
import pandas as _pd  # noqa: F401
import pyarrow as _pa  # noqa: F401
import pyarrow.parquet as _pq  # noqa: F401

from src.config import SHARED_IMAGES_GPU
from src.lib.io import register_src_for_burla


@dataclass
class A:
    shared_root: str


def inspect(args: A) -> dict:
    files = sorted(glob.glob(os.path.join(args.shared_root, "batch_*.parquet")))
    out = {"n_files": len(files), "samples": []}
    if not files:
        return out
    df = pd.read_parquet(files[0])
    out["columns"] = df.columns.tolist()
    out["n_rows_first_file"] = int(len(df))
    out["error_value_counts"] = (
        df["error"].value_counts(dropna=False).to_dict() if "error" in df.columns else {}
    )
    out["tv_detected_count"] = int(df["tv_detected"].sum()) if "tv_detected" in df.columns else None
    out["person_detected_count"] = int(df["person_detected"].sum()) if "person_detected" in df.columns else None
    out["head_first_3"] = df.head(3).to_dict("records")

    if len(files) > 1:
        df2 = pd.read_parquet(files[len(files) // 2])
        out["mid_file_n_rows"] = int(len(df2))
        out["mid_file_tv_count"] = int(df2["tv_detected"].sum()) if "tv_detected" in df2.columns else None
    return out


def main() -> None:
    load_dotenv()
    register_src_for_burla()
    from burla import remote_parallel_map

    [res] = remote_parallel_map(
        inspect, [A(shared_root=SHARED_IMAGES_GPU)],
        func_cpu=2, func_ram=4, max_parallelism=1, grow=True, spinner=True,
    )
    import json
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
