"""Auto-sizing formula for STT/TTS replica counts and per-replica threads.

This is the "1000-ready" core: unchanged code, run on a box with more cores,
must compute more replicas automatically. Every input has an env override so
the formula is never the only way to get a number -- see docs/
LOAD_SCALING_PLAN.md Phase 3 item 6 ("CPU replica/thread sweep") for how the
defaults below were chosen, and docs/CPU_PROFILING_REPORT.md for the
single-worker thread optimum (4) this box's numbers come from.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Mapping

from .hardware import CpuInfo, GpuInfo

# docs/CPU_PROFILING_REPORT.md: STT/TTS single-worker throughput saturates
# around 4 threads on this CPU class; going past 8 measurably regressed STT
# (hyperthreading contention). These are safe generic bounds, not a claim
# that 4 is optimal on every CPU -- re-derive via sweep_replicas.py (Phase 3
# item 6) on a materially different box.
DEFAULT_TARGET_THREADS = 4
MIN_THREADS_PER_REPLICA = 2
MAX_THREADS_PER_REPLICA = 8

# TTS costs roughly 25x STT's CPU per session (memory-bandwidth-bound decode
# vs. STT's comparatively cheap forward -- see CPU_PROFILING_REPORT.md), so
# the default split weights TTS over STT rather than an even 50:50.
DEFAULT_STT_TTS_SPLIT = "40:60"


@dataclasses.dataclass(frozen=True)
class ReplicaPlan:
    stt_replicas: int
    stt_threads: int
    tts_replicas: int
    tts_threads: int
    reserved_control_cores: int
    llm_cpu_reserve: int
    worker_cores: int
    physical_cores: int
    logical_cores: int
    notes: tuple[str, ...]

    def to_lines(self) -> list[str]:
        return [
            f"physical_cores={self.physical_cores} logical_cores={self.logical_cores}",
            f"reserved_control_cores={self.reserved_control_cores} llm_cpu_reserve={self.llm_cpu_reserve} "
            f"worker_cores={self.worker_cores}",
            f"stt: {self.stt_replicas} replica(s) x {self.stt_threads} threads",
            f"tts: {self.tts_replicas} replica(s) x {self.tts_threads} threads",
            *self.notes,
        ]


def _env_int_or_auto(env: Mapping[str, str], key: str) -> int | None:
    raw = env.get(key)
    if raw is None or not raw.strip() or raw.strip().lower() == "auto":
        return None
    return int(raw.strip())


def compute_replica_plan(cpu: CpuInfo, gpus: list[GpuInfo], env: Mapping[str, str]) -> ReplicaPlan:
    notes: list[str] = []

    reserved = _env_int_or_auto(env, "RESERVED_CONTROL_CORES")
    if reserved is None:
        reserved = max(2, math.ceil(cpu.physical_cores * 0.15))
        notes.append(f"RESERVED_CONTROL_CORES auto = max(2, ceil({cpu.physical_cores} * 0.15)) = {reserved}")

    llm_reserve = _env_int_or_auto(env, "LLM_CPU_RESERVE")
    if llm_reserve is None:
        llm_reserve = 1 if gpus else 0
        notes.append(f"LLM_CPU_RESERVE auto = {llm_reserve} ({'GPU present' if gpus else 'no GPU detected'})")

    worker_cores = max(2, cpu.physical_cores - reserved - llm_reserve)
    if worker_cores == 2 and cpu.physical_cores - reserved - llm_reserve < 2:
        notes.append("worker_cores floored at 2 (reserved+llm_reserve left too little budget)")

    split_raw = env.get("STT_TTS_CORE_SPLIT", DEFAULT_STT_TTS_SPLIT)
    try:
        stt_pct_s, tts_pct_s = split_raw.split(":")
        stt_pct, tts_pct = float(stt_pct_s), float(tts_pct_s)
        stt_share = stt_pct / (stt_pct + tts_pct)
    except (ValueError, ZeroDivisionError):
        stt_share = 0.4
        notes.append(f"STT_TTS_CORE_SPLIT={split_raw!r} unparseable; falling back to 40:60")
    stt_cores = max(1, round(worker_cores * stt_share))
    tts_cores = max(1, worker_cores - stt_cores)

    def _threads(key: str) -> int:
        val = _env_int_or_auto(env, key)
        if val is None:
            val = DEFAULT_TARGET_THREADS
        clamped = min(MAX_THREADS_PER_REPLICA, max(MIN_THREADS_PER_REPLICA, val))
        if clamped != val:
            notes.append(f"{key}={val} clamped to [{MIN_THREADS_PER_REPLICA},{MAX_THREADS_PER_REPLICA}] -> {clamped}")
        return clamped

    stt_threads = _threads("STT_THREADS_PER_REPLICA")
    tts_threads = _threads("TTS_THREADS_PER_REPLICA")

    stt_replicas = _env_int_or_auto(env, "STT_REPLICAS")
    if stt_replicas is None:
        stt_replicas = max(1, stt_cores // stt_threads)
        notes.append(f"STT_REPLICAS auto = max(1, {stt_cores} // {stt_threads}) = {stt_replicas}")

    tts_replicas = _env_int_or_auto(env, "TTS_REPLICAS")
    if tts_replicas is None:
        tts_replicas = max(1, tts_cores // tts_threads)
        notes.append(f"TTS_REPLICAS auto = max(1, {tts_cores} // {tts_threads}) = {tts_replicas}")

    return ReplicaPlan(
        stt_replicas=max(1, stt_replicas), stt_threads=stt_threads,
        tts_replicas=max(1, tts_replicas), tts_threads=tts_threads,
        reserved_control_cores=reserved, llm_cpu_reserve=llm_reserve,
        worker_cores=worker_cores, physical_cores=cpu.physical_cores,
        logical_cores=cpu.logical_cores, notes=tuple(notes),
    )


# Backward-compatible ports: replica 0 keeps today's single-instance
# defaults, so an unscaled run (STT_REPLICAS=1) is byte-identical to the
# pre-Phase-3 setup. Later replicas step by a stride wide enough to never
# collide with any other service's fixed port (orchestrator 50051, LLM
# 50053/50054-worker, LLM metrics 9106/9114).
STT_APP_PORT_BASE, STT_APP_PORT_STRIDE = 50052, 100
STT_METRICS_PORT_BASE, STT_METRICS_PORT_STRIDE = 9103, 20
TTS_APP_PORT_BASE, TTS_APP_PORT_STRIDE = 50054, 100
TTS_METRICS_PORT_BASE, TTS_METRICS_PORT_STRIDE = 9104, 20


def stt_ports(index: int) -> tuple[int, int]:
    return (STT_APP_PORT_BASE + index * STT_APP_PORT_STRIDE,
            STT_METRICS_PORT_BASE + index * STT_METRICS_PORT_STRIDE)


def tts_ports(index: int) -> tuple[int, int]:
    return (TTS_APP_PORT_BASE + index * TTS_APP_PORT_STRIDE,
            TTS_METRICS_PORT_BASE + index * TTS_METRICS_PORT_STRIDE)
