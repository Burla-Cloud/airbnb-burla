"""Cost + runtime tracker.

Stages call ``BudgetTracker.record_stage(...)`` at the end of their run.
The tracker writes a cumulative ledger to ``data/outputs/runtime_log.json``
and raises ``BudgetExceeded`` if any hard cap is breached.

Per-stage costs are *estimated* from worker count + wall time + Burla's
published per-CPU-minute pricing, plus Anthropic API costs from token counts.
The estimate is conservative on purpose so we halt before a real overage.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from ..config import (
    PIPELINE_HARD_CAP_HOURS, PIPELINE_HARD_CAP_SOFT_USD, PIPELINE_HARD_CAP_USD,
    RUNTIME_LOG_PATH, STAGE_BUDGETS,
)


BURLA_USD_PER_CPU_MIN = 0.001  # ~$0.06/hr per 1-CPU container, conservative
BURLA_USD_PER_A100_MIN = 0.05  # ~$3/hr per A100_40G, conservative
ANTHROPIC_HAIKU_USD_PER_INPUT_KTOK = 0.25
ANTHROPIC_HAIKU_USD_PER_OUTPUT_KTOK = 1.25


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class StageRecord:
    stage: str
    started_at: float
    ended_at: float
    wall_seconds: float
    n_workers: int
    func_cpu: int
    func_gpu: Optional[str]
    n_inputs: int
    n_succeeded: int
    n_failed: int
    notes: dict = field(default_factory=dict)
    estimated_usd: float = 0.0


def estimate_burla_cpu_usd(n_workers: int, wall_seconds: float, func_cpu: int = 1) -> float:
    """Cluster-wide CPU-minutes * dollars-per-CPU-min."""
    cpu_minutes = (n_workers * func_cpu) * (wall_seconds / 60.0)
    return cpu_minutes * BURLA_USD_PER_CPU_MIN


def estimate_burla_gpu_usd(n_workers: int, wall_seconds: float) -> float:
    return n_workers * (wall_seconds / 60.0) * BURLA_USD_PER_A100_MIN


def estimate_anthropic_usd(n_input_tokens: int, n_output_tokens: int) -> float:
    return (n_input_tokens / 1000.0) * ANTHROPIC_HAIKU_USD_PER_INPUT_KTOK + \
           (n_output_tokens / 1000.0) * ANTHROPIC_HAIKU_USD_PER_OUTPUT_KTOK


def _load_log() -> dict:
    if not RUNTIME_LOG_PATH.exists():
        return {"stages": [], "total_usd": 0.0, "total_hours": 0.0}
    return json.loads(RUNTIME_LOG_PATH.read_text())


def _save_log(log: dict) -> None:
    RUNTIME_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_LOG_PATH.write_text(json.dumps(log, indent=2))


class BudgetTracker:
    """Drop-in stage timer. Use as a context manager:

        with BudgetTracker("s00_validate", n_inputs=119, func_cpu=1) as bt:
            bt.set_workers(50)
            ... run remote_parallel_map ...
            bt.set_succeeded(118)
            bt.set_failed(1)
    """

    def __init__(self, stage: str, *, n_inputs: int, func_cpu: int = 1,
                 func_gpu: Optional[str] = None, anthropic_input_tokens: int = 0,
                 anthropic_output_tokens: int = 0):
        self.stage = stage
        self.n_inputs = n_inputs
        self.func_cpu = func_cpu
        self.func_gpu = func_gpu
        self.n_workers = 0
        self.n_succeeded = 0
        self.n_failed = 0
        self.notes: dict = {}
        self.anthropic_input_tokens = anthropic_input_tokens
        self.anthropic_output_tokens = anthropic_output_tokens
        self.started_at = 0.0
        self.ended_at = 0.0

    def __enter__(self) -> "BudgetTracker":
        self.started_at = time.time()
        return self

    def set_workers(self, n: int) -> None:
        self.n_workers = max(self.n_workers, int(n))

    def set_succeeded(self, n: int) -> None:
        self.n_succeeded = int(n)

    def set_failed(self, n: int) -> None:
        self.n_failed = int(n)

    def add_anthropic_tokens(self, in_tokens: int, out_tokens: int) -> None:
        self.anthropic_input_tokens += int(in_tokens)
        self.anthropic_output_tokens += int(out_tokens)

    def note(self, **kv) -> None:
        self.notes.update(kv)

    def __exit__(self, exc_type, exc, tb) -> None:
        self.ended_at = time.time()
        wall = self.ended_at - self.started_at
        if self.func_gpu:
            usd = estimate_burla_gpu_usd(self.n_workers, wall)
        else:
            usd = estimate_burla_cpu_usd(self.n_workers, wall, self.func_cpu)
        usd += estimate_anthropic_usd(self.anthropic_input_tokens, self.anthropic_output_tokens)

        rec = StageRecord(
            stage=self.stage,
            started_at=self.started_at,
            ended_at=self.ended_at,
            wall_seconds=wall,
            n_workers=self.n_workers,
            func_cpu=self.func_cpu,
            func_gpu=self.func_gpu,
            n_inputs=self.n_inputs,
            n_succeeded=self.n_succeeded,
            n_failed=self.n_failed,
            estimated_usd=round(usd, 4),
            notes=self.notes,
        )

        log = _load_log()
        log["stages"].append(asdict(rec))
        log["total_usd"] = round(sum(s["estimated_usd"] for s in log["stages"]), 4)
        log["total_hours"] = round(sum(s["wall_seconds"] for s in log["stages"]) / 3600.0, 4)
        _save_log(log)

        budget = STAGE_BUDGETS.get(self.stage, {})
        if budget:
            if usd > budget.get("usd", float("inf")) * 1.5:
                raise BudgetExceeded(
                    f"{self.stage}: estimated ${usd:.2f} exceeds soft cap "
                    f"${budget['usd']:.2f} by >50%"
                )
            if wall / 3600.0 > budget.get("hours", float("inf")) * 1.5:
                raise BudgetExceeded(
                    f"{self.stage}: wall {wall/3600:.2f}h exceeds soft cap "
                    f"{budget['hours']:.2f}h by >50%"
                )

        if log["total_usd"] > PIPELINE_HARD_CAP_USD:
            raise BudgetExceeded(
                f"Pipeline-wide hard cap ${PIPELINE_HARD_CAP_USD:.0f} exceeded "
                f"(actual ${log['total_usd']:.2f})."
            )
        if log["total_usd"] > PIPELINE_HARD_CAP_SOFT_USD and not log.get("soft_warned"):
            print(
                f"[budget] WARNING: pipeline at ${log['total_usd']:.2f}, "
                f"crossed soft cap ${PIPELINE_HARD_CAP_SOFT_USD:.0f}. "
                f"Hard cap is ${PIPELINE_HARD_CAP_USD:.0f}.",
                flush=True,
            )
            log["soft_warned"] = True
            _save_log(log)
        if log["total_hours"] > PIPELINE_HARD_CAP_HOURS:
            raise BudgetExceeded(
                f"Pipeline-wide hard cap {PIPELINE_HARD_CAP_HOURS:.0f}h exceeded "
                f"(actual {log['total_hours']:.2f}h)."
            )
