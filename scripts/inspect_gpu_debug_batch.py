"""Print the rows of the debug GPU batch parquet to see per-image errors."""
from __future__ import annotations
import sys

import pandas as pd

sys.path.insert(0, ".")

from dataclasses import dataclass
from dotenv import load_dotenv

import pyarrow as _pa  # noqa: F401
import pyarrow.parquet as _pq  # noqa: F401

from src.lib.io import register_src_for_burla


@dataclass
class A:
    pass


def inspect(_a: A) -> dict:
    df = pd.read_parquet("/workspace/shared/airbnb/images_gpu_debug/batch_999999.parquet")
    out = {
        "n_rows": int(len(df)),
        "columns": df.columns.tolist(),
        "error_value_counts": df["error"].value_counts(dropna=False).to_dict(),
        "head": df.head(5).to_dict("records"),
    }
    return out


def main() -> None:
    load_dotenv()
    register_src_for_burla()
    from burla import remote_parallel_map

    [res] = remote_parallel_map(
        inspect, [A()], func_cpu=2, func_ram=4, max_parallelism=1, grow=True, spinner=True,
    )
    import json
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
