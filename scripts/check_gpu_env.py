"""Inspect the GPU worker container: torch version, CUDA, free disk."""
from __future__ import annotations
import sys

import torch
import shutil
import subprocess
import importlib

sys.path.insert(0, ".")

from dataclasses import dataclass
from dotenv import load_dotenv

from src.lib.io import register_src_for_burla


@dataclass
class A:
    pass


def inspect(_a: A) -> dict:
    out = {}
    try:
        out["torch_version"] = torch.__version__
        out["cuda_available"] = torch.cuda.is_available()
        out["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            out["device_name"] = torch.cuda.get_device_name(0)
            out["device_count"] = torch.cuda.device_count()
    except Exception as e:
        out["torch_error"] = f"{type(e).__name__}: {e}"

    try:
        free = shutil.disk_usage("/")
        out["root_free_gb"] = round(free.free / 1e9, 2)
        out["root_total_gb"] = round(free.total / 1e9, 2)
    except Exception:
        pass

    try:
        ver = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True)
        out["nvidia_smi_first_lines"] = "\n".join(ver.split("\n")[:6])
    except Exception as e:
        out["nvidia_smi_error"] = f"{type(e).__name__}: {e}"

    try:
        for mod in ["cv2", "ultralytics", "matplotlib", "seaborn", "pandas",
                    "psutil", "py_cpuinfo", "ultralytics_thop", "thop",
                    "pyparsing", "tqdm", "scipy"]:
            try:
                m = importlib.import_module(mod)
                out[f"has_{mod}"] = getattr(m, "__version__", "yes")
            except ImportError:
                out[f"has_{mod}"] = False
    except Exception:
        pass
    return out


def main() -> None:
    load_dotenv()
    register_src_for_burla()
    from burla import remote_parallel_map

    [res] = remote_parallel_map(
        inspect, [A()], func_cpu=4, func_ram=8, func_gpu="A100_40G",
        max_parallelism=1, grow=True, spinner=True,
    )
    import json
    print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
