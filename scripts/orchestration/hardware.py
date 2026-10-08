"""Stdlib-only hardware detection for replica-count auto-sizing.

Deliberately independent of adaptive_llm/planning/hardware.py: that module
lives inside adaptive_llm's own venv (psutil, torch) and this one must run
under whichever bare `python3` launches the replica supervisor, which in turn
spawns processes across four separate per-service venvs. No pip dependency
here -- this is the one piece of Phase 3 that has to work before any venv is
even chosen.
"""
from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess


@dataclasses.dataclass(frozen=True)
class CpuInfo:
    physical_cores: int
    logical_cores: int


@dataclasses.dataclass(frozen=True)
class MemInfo:
    total_mib: int
    available_mib: int


@dataclasses.dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    total_vram_mib: int


def detect_cpu() -> CpuInfo:
    # sched_getaffinity respects a cgroup/taskset restriction (e.g. a
    # container CPU limit); cpu_count() does not -- this is what makes the
    # auto-sizing formula correctly shrink inside a constrained environment
    # and correctly grow on a bigger bare-metal box, unchanged.
    if hasattr(os, "sched_getaffinity"):
        logical = len(os.sched_getaffinity(0))
    else:
        logical = os.cpu_count() or 1
    physical = _physical_cores_from_proc_cpuinfo()
    if not physical:
        # SMT is the common case; assume 2 threads/core rather than treating
        # every logical CPU as physical (would double-count and oversubscribe).
        physical = max(1, logical // 2) or logical
    return CpuInfo(physical_cores=physical, logical_cores=logical)


def _physical_cores_from_proc_cpuinfo() -> int | None:
    try:
        with open("/proc/cpuinfo") as f:
            text = f.read()
    except OSError:
        return None
    cores: set[tuple[str, str]] = set()
    phys_id = "0"
    core_id: str | None = None
    for line in text.splitlines():
        if line.startswith("physical id"):
            phys_id = line.split(":", 1)[1].strip()
        elif line.startswith("core id"):
            core_id = line.split(":", 1)[1].strip()
        elif not line.strip():
            if core_id is not None:
                cores.add((phys_id, core_id))
            core_id = None
    if core_id is not None:
        cores.add((phys_id, core_id))
    return len(cores) or None


def detect_mem() -> MemInfo:
    total_mib = avail_mib = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_mib = int(line.split()[1]) // 1024
                elif line.startswith("MemAvailable:"):
                    avail_mib = int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return MemInfo(total_mib=total_mib, available_mib=avail_mib)


def detect_gpus() -> list[GpuInfo]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            gpus.append(GpuInfo(index=int(parts[0]), name=parts[1], total_vram_mib=int(float(parts[2]))))
        except ValueError:
            continue
    return gpus
