"""Delete old tier-1 batch parquets + tier1_top (without comments) so a
fresh tier-1 run can write the new schema with comments included."""
from __future__ import annotations
import os
import sys
import time

import glob

sys.path.insert(0, ".")

from dotenv import load_dotenv

from src.lib.io import register_src_for_burla


def _delete(args) -> dict:
    out = {"ok": False, "deleted": 0, "error": None, "elapsed_seconds": 0.0}
    started = time.time()
    try:
        n = 0
        roots = [
            "/workspace/shared/airbnb/reviews_tier1",
            "/workspace/shared/airbnb/reviews_tier2",
        ]
        for r in roots:
            if not os.path.isdir(r):
                continue
            for p in glob.glob(os.path.join(r, "batch_*.parquet")):
                try:
                    os.remove(p)
                    n += 1
                except FileNotFoundError:
                    pass
        for p in [
            "/workspace/shared/airbnb/reviews_tier1_top.parquet",
            "/workspace/shared/airbnb/reviews_tier3_input.parquet",
        ]:
            if os.path.exists(p):
                os.remove(p)
                n += 1
        out.update({"ok": True, "deleted": n})
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    out["elapsed_seconds"] = time.time() - started
    return out


def main() -> None:
    load_dotenv()
    register_src_for_burla()
    from burla import remote_parallel_map
    print("[cleanup_tier1] deleting old tier-1/tier-2 batches and tier1_top ...", flush=True)
    [r] = remote_parallel_map(
        _delete, [object()],
        func_cpu=2, func_ram=4, max_parallelism=1, grow=True, spinner=True,
    )
    if not r.get("ok"):
        print(f"[cleanup_tier1] failed: {r.get('error')}", flush=True)
        raise SystemExit(1)
    print(f"[cleanup_tier1] DONE: deleted {r['deleted']:,} files in {r['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
