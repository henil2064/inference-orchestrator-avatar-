"""Static and dynamic hardware snapshots for planning."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import platform
import shutil
import subprocess
import time


MIB = 1024 * 1024


@dataclass(frozen=True)
class GpuStatic:
    index: int
    name: str
    total_vram_mib: int
    compute_capability: str | None = None
    uuid: str | None = None
    pci_bus_id: str | None = None
    pcie_link_gen: int | None = None
    pcie_link_width: int | None = None


@dataclass(frozen=True)
class GpuDynamic:
    index: int
    free_vram_mib: int
    used_vram_mib: int
    utilization_pct: float | None = None
    temperature_c: float | None = None


@dataclass(frozen=True)
class StaticSnapshot:
    os: str
    cpu: str
    logical_cores: int
    total_ram_mib: int
    cuda_available: bool
    gpus: list[GpuStatic] = field(default_factory=list)
    bandwidth: "BandwidthSnapshot" = field(default_factory=lambda: BandwidthSnapshot())
    # Real physical core count (see _physical_cores below), not a guess. Falls
    # back to logical_cores // 2 only when /proc/cpuinfo is unreadable
    # (non-Linux). Added because compatibility.py's llama.cpp thread sizing
    # used to divide logical_cores by 2 as a physical-core proxy -- correct
    # only by coincidence on a uniform 2x-SMT box, wrong on any box without
    # SMT (undercounts by half) or with a different SMT ratio.
    physical_cores: int = 1


@dataclass(frozen=True)
class DynamicSnapshot:
    available_ram_mib: int
    cpu_utilization_pct: float | None
    gpus: list[GpuDynamic] = field(default_factory=list)
    active_requests: int = 0
    loaded_models: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BandwidthSnapshot:
    """Optional locally measured bandwidths, in GB/s."""

    cpu_memory_gbps: float | None = None
    pcie_h2d_gbps: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class HardwareSnapshot:
    static: StaticSnapshot
    dynamic: DynamicSnapshot

    @property
    def has_cuda_gpu(self) -> bool:
        return self.static.cuda_available and bool(self.static.gpus)


def collect_snapshot(profile_bandwidth: bool = False) -> HardwareSnapshot:
    total_ram_mib, available_ram_mib, cpu_pct = _memory_and_cpu()
    static_gpus, dynamic_gpus = _nvidia_smi()
    bandwidth = _measure_bandwidth(static_gpus) if profile_bandwidth else BandwidthSnapshot()
    logical_cores = _logical_cores()
    return HardwareSnapshot(
        static=StaticSnapshot(
            os=f"{platform.system()} {platform.release()}",
            cpu=platform.processor() or "unknown",
            logical_cores=logical_cores,
            physical_cores=_physical_cores(logical_cores),
            total_ram_mib=total_ram_mib,
            cuda_available=bool(static_gpus),
            gpus=static_gpus,
            bandwidth=bandwidth,
        ),
        dynamic=DynamicSnapshot(
            available_ram_mib=available_ram_mib,
            cpu_utilization_pct=cpu_pct,
            gpus=dynamic_gpus,
        ),
    )


def _logical_cores() -> int:
    # Respects a cgroup/taskset restriction (e.g. a container CPU limit);
    # os.cpu_count() does not, and would overcount inside one.
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def _physical_cores(logical_cores: int) -> int:
    """Real physical core count via /proc/cpuinfo -- not a logical_cores/2
    guess, which is only right on a uniform 2x-SMT box and wrong (undercounts)
    on any box without SMT."""
    try:
        with open("/proc/cpuinfo") as f:
            text = f.read()
    except OSError:
        return max(1, logical_cores // 2) or logical_cores
    return _parse_physical_cores(text, logical_cores)


def _parse_physical_cores(cpuinfo_text: str, logical_cores: int) -> int:
    """Pure parser over /proc/cpuinfo's (physical id, core id) pairs -- split
    out from _physical_cores so it's testable without a real /proc/cpuinfo."""
    cores: set[tuple[str, str]] = set()
    phys_id = "0"
    core_id: str | None = None
    for line in cpuinfo_text.splitlines():
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
    return len(cores) or max(1, logical_cores // 2) or logical_cores


def collect_dynamic_snapshot() -> DynamicSnapshot:
    """Collect only values that may change while a worker is serving requests."""
    _, available_ram_mib, cpu_pct = _memory_and_cpu()
    _, dynamic_gpus = _nvidia_smi(include_pcie=False)
    return DynamicSnapshot(
        available_ram_mib=available_ram_mib,
        cpu_utilization_pct=cpu_pct,
        gpus=dynamic_gpus,
    )


def load_fixture(path: str) -> HardwareSnapshot:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    logical_cores = int(raw["static"].get("logical_cores", 1))
    return HardwareSnapshot(
        static=StaticSnapshot(
            os=raw["static"].get("os", "fixture"),
            cpu=raw["static"].get("cpu", "fixture"),
            logical_cores=logical_cores,
            # Fixtures predating this field fall back to the same
            # logical_cores // 2 heuristic the old code used everywhere, so
            # existing fixtures keep producing identical planner output.
            physical_cores=int(raw["static"].get("physical_cores", max(1, logical_cores // 2) or logical_cores)),
            total_ram_mib=int(raw["static"]["total_ram_mib"]),
            cuda_available=bool(raw["static"].get("cuda_available", False)),
            gpus=[GpuStatic(**gpu) for gpu in raw["static"].get("gpus", [])],
            bandwidth=BandwidthSnapshot(
                cpu_memory_gbps=raw["static"].get("bandwidth", {}).get("cpu_memory_gbps"),
                pcie_h2d_gbps={
                    int(index): float(value)
                    for index, value in raw["static"].get("bandwidth", {}).get("pcie_h2d_gbps", {}).items()
                },
            ),
        ),
        dynamic=DynamicSnapshot(
            available_ram_mib=int(raw["dynamic"]["available_ram_mib"]),
            cpu_utilization_pct=raw["dynamic"].get("cpu_utilization_pct"),
            gpus=[GpuDynamic(**gpu) for gpu in raw["dynamic"].get("gpus", [])],
            active_requests=int(raw["dynamic"].get("active_requests", 0)),
            loaded_models=list(raw["dynamic"].get("loaded_models", [])),
        ),
    )


def _memory_and_cpu() -> tuple[int, int, float | None]:
    try:
        import psutil

        mem = psutil.virtual_memory()
        return int(mem.total / MIB), int(mem.available / MIB), float(psutil.cpu_percent(interval=0.0))
    except Exception:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        total_mib = int((pages * page_size) / MIB)
        return total_mib, total_mib, None


def _nvidia_smi(include_pcie: bool = True) -> tuple[list[GpuStatic], list[GpuDynamic]]:
    if not shutil.which("nvidia-smi"):
        return [], []
    query = (
        "index,name,uuid,memory.total,memory.free,memory.used,"
        "utilization.gpu,temperature.gpu,compute_cap"
    )
    cmd = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return [], []

    pcie = _nvidia_pcie_info() if include_pcie else {}
    static: list[GpuStatic] = []
    dynamic: list[GpuDynamic] = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 9:
            continue
        index = int(parts[0])
        static.append(
            GpuStatic(
                index=index,
                name=parts[1],
                uuid=parts[2],
                total_vram_mib=int(float(parts[3])),
                compute_capability=parts[8] or None,
                pci_bus_id=pcie.get(index, {}).get("pci_bus_id"),
                pcie_link_gen=pcie.get(index, {}).get("pcie_link_gen"),
                pcie_link_width=pcie.get(index, {}).get("pcie_link_width"),
            )
        )
        dynamic.append(
            GpuDynamic(
                index=index,
                free_vram_mib=int(float(parts[4])),
                used_vram_mib=int(float(parts[5])),
                utilization_pct=_float_or_none(parts[6]),
                temperature_c=_float_or_none(parts[7]),
            )
        )
    return static, dynamic


def _nvidia_pcie_info() -> dict[int, dict[str, str | int | None]]:
    """Return PCIe topology when the installed nvidia-smi supports these fields."""
    query = "index,pci.bus_id,pci.link.gen.current,pci.link.width.current"
    cmd = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return {}
    info: dict[int, dict[str, str | int | None]] = {}
    for line in out.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        info[index] = {
            "pci_bus_id": parts[1] or None,
            "pcie_link_gen": _int_or_none(parts[2]),
            "pcie_link_width": _int_or_none(parts[3]),
        }
    return info


def _measure_bandwidth(gpus: list[GpuStatic]) -> BandwidthSnapshot:
    """Run optional short local benchmarks; unavailable values are left empty."""
    cpu_memory_gbps = _measure_cpu_memory_bandwidth_gbps()
    pcie_h2d_gbps = {
        gpu.index: bandwidth
        for gpu in gpus
        if (bandwidth := _measure_pcie_h2d_bandwidth_gbps(gpu.index)) is not None
    }
    return BandwidthSnapshot(cpu_memory_gbps=cpu_memory_gbps, pcie_h2d_gbps=pcie_h2d_gbps)


def _measure_cpu_memory_bandwidth_gbps(size_mib: int = 64, repeats: int = 3) -> float | None:
    try:
        size = size_mib * MIB
        source = bytearray(size)
        destination = bytearray(size)
        samples: list[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            destination[:] = source
            elapsed = time.perf_counter() - start
            if elapsed > 0:
                samples.append(size / elapsed / 1_000_000_000)
        return round(max(samples), 2) if samples else None
    except Exception:
        return None


def _measure_pcie_h2d_bandwidth_gbps(gpu_index: int, size_mib: int = 64, repeats: int = 3) -> float | None:
    try:
        import torch

        if not torch.cuda.is_available() or gpu_index >= torch.cuda.device_count():
            return None
        size = size_mib * MIB
        source = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        device = torch.device(f"cuda:{gpu_index}")
        samples: list[float] = []
        for _ in range(repeats):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            target = source.to(device, non_blocking=True)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            del target
            if elapsed > 0:
                samples.append(size / elapsed / 1_000_000_000)
        return round(max(samples), 2) if samples else None
    except Exception:
        return None


def _int_or_none(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None
