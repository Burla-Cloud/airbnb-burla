"""See what disk space Burla workers have on different mounts."""
from __future__ import annotations
import sys

import os
import shutil

sys.path.insert(0, ".")

from dataclasses import dataclass
from dotenv import load_dotenv

import pyarrow as _pa  # noqa: F401

from src.lib.io import register_src_for_burla


@dataclass
class A:
    pass


def check(_a: A) -> dict:
    out = {"mounts": {}}
    for path in [
        "/", "/tmp", "/root", "/root/.cache", "/var", "/var/tmp",
        "/workspace", "/workspace/shared", "/workspace/shared/airbnb",
        "/dev/shm",
    ]:
        try:
            usage = shutil.disk_usage(path)
            out["mounts"][path] = {
                "total_gb": round(usage.total / 1e9, 2),
                "used_gb": round(usage.used / 1e9, 2),
                "free_gb": round(usage.free / 1e9, 2),
            }
        except Exception as e:
            out["mounts"][path] = {"error": f"{type(e).__name__}: {e}"}
    out["env_HF_HOME"] = os.environ.get("HF_HOME", "<unset>")
    out["env_XDG_CACHE_HOME"] = os.environ.get("XDG_CACHE_HOME", "<unset>")
    out["env_HOME"] = os.environ.get("HOME", "<unset>")
    return out


def main() -> None:
    load_dotenv()
    register_src_for_burla()
    from burla import remote_parallel_map
    [res] = remote_parallel_map(
        check, [A()],
        func_cpu=1, func_ram=2, max_parallelism=1, grow=True, spinner=True,
    )
    import json
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
