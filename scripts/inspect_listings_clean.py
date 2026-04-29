"""Inspect listings_clean.parquet schema on shared FS."""
from __future__ import annotations
import sys

import pyarrow.parquet as pq

sys.path.insert(0, ".")
from dotenv import load_dotenv

import pyarrow as _pa  # noqa: F401
import pyarrow.parquet as _pq  # noqa: F401

from src.lib.io import register_src_for_burla


def _inspect(args) -> dict:
    out = {"ok": False, "error": None}
    try:
        for path in [
            "/workspace/shared/airbnb/listings_clean.parquet",
            "/workspace/shared/airbnb/images_cpu.parquet",
            "/workspace/shared/airbnb/images_gpu.parquet",
        ]:
            try:
                pf = pq.ParquetFile(path)
                schema = pf.schema_arrow
                cols = [(f.name, str(f.type)) for f in schema]
                out[path] = {"n_rows": pf.metadata.num_rows, "columns": cols}
            except Exception as e:
                out[path] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        out["ok"] = True
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return out


def main() -> None:
    load_dotenv()
    register_src_for_burla()
    from burla import remote_parallel_map
    [r] = remote_parallel_map(
        _inspect, [object()],
        func_cpu=2, func_ram=4, max_parallelism=1, grow=True, spinner=True,
    )
    import json
    print(json.dumps(r, indent=2, default=str))


if __name__ == "__main__":
    main()
