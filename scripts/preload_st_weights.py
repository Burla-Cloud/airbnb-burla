"""Pre-download the sentence-transformers model weights to /workspace/shared
so Stage 4 tier 2 workers don't all race on a HuggingFace download.

Run this once before launching Stage 4 tier 2."""
from __future__ import annotations
import os
import sys
import shutil
import time

from sentence_transformers import SentenceTransformer
import traceback as _tb

sys.path.insert(0, ".")

from dotenv import load_dotenv

import sentence_transformers as _st  # noqa: F401

from src.lib.io import register_src_for_burla


SHARED_DIR = "/workspace/shared/airbnb/st_weights/all-MiniLM-L6-v2"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _preload(args) -> dict:
    out = {"ok": False, "shared_dir": SHARED_DIR, "model_name": MODEL_NAME,
           "n_files": 0, "elapsed_seconds": 0.0, "error": None}
    started = time.time()
    try:
        os.makedirs(os.path.dirname(SHARED_DIR), exist_ok=True)
        # If already populated, skip.
        marker = os.path.join(SHARED_DIR, "config.json")
        if os.path.exists(marker):
            files = sorted(os.listdir(SHARED_DIR))
            out.update({"ok": True, "n_files": len(files), "skipped": True})
            out["elapsed_seconds"] = time.time() - started
            return out
        # Download into a tmp local dir, then copy to shared.
        tmp_local = "/tmp/st_weights_tmp"
        if os.path.exists(tmp_local):
            shutil.rmtree(tmp_local)
        model = SentenceTransformer(MODEL_NAME, cache_folder="/tmp/hf_cache_for_st")
        model.save(tmp_local)
        os.makedirs(SHARED_DIR, exist_ok=True)
        for entry in os.listdir(tmp_local):
            src = os.path.join(tmp_local, entry)
            dst = os.path.join(SHARED_DIR, entry)
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        files = sorted(os.listdir(SHARED_DIR))
        out.update({"ok": True, "n_files": len(files)})
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {str(e)[:200]}"
        out["traceback"] = _tb.format_exc()[:1000]
    out["elapsed_seconds"] = time.time() - started
    return out


def main() -> None:
    load_dotenv()
    register_src_for_burla()
    from burla import remote_parallel_map

    print(f"[preload_st] downloading {MODEL_NAME} -> {SHARED_DIR} ...", flush=True)
    [r] = remote_parallel_map(
        _preload, [object()],
        func_cpu=4, func_ram=8, max_parallelism=1,
        grow=True, spinner=True,
    )
    if not r.get("ok"):
        print(f"[preload_st] failed: {r.get('error')}", flush=True)
        if r.get("traceback"):
            print(r["traceback"], flush=True)
        raise SystemExit(1)
    note = " (already present, skipped)" if r.get("skipped") else ""
    print(f"[preload_st] DONE{note} in {r['elapsed_seconds']:.1f}s "
          f"({r['n_files']} files in {SHARED_DIR})", flush=True)


if __name__ == "__main__":
    main()
