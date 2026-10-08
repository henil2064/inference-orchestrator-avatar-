"""Self-check for the replica auto-sizing formula. Stdlib only.
Run: python3 scripts/orchestration/test_plan.py (from repo root)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from orchestration.hardware import CpuInfo, GpuInfo
from orchestration.plan import compute_replica_plan, stt_ports, tts_ports


def test_dev_box_defaults_are_backward_compatible():
    """8 physical cores, one small GPU (this dev box's actual shape): the
    formula should land on exactly today's single-instance setup, and
    replica 0's ports must equal the pre-Phase-3 defaults."""
    cpu = CpuInfo(physical_cores=8, logical_cores=16)
    gpus = [GpuInfo(index=0, name="GTX 1650", total_vram_mib=4096)]
    plan = compute_replica_plan(cpu, gpus, {})
    assert plan.stt_replicas == 1 and plan.tts_replicas == 1
    assert plan.stt_threads == 4 and plan.tts_threads == 4
    assert stt_ports(0) == (50052, 9103)
    assert tts_ports(0) == (50054, 9104)


def test_bigger_box_yields_more_replicas_unchanged_code():
    """The whole point of Phase 3: the same formula, given more cores,
    produces more replicas without any code change."""
    cpu = CpuInfo(physical_cores=128, logical_cores=256)
    gpus = [GpuInfo(index=0, name="L40S", total_vram_mib=49152)]
    plan = compute_replica_plan(cpu, gpus, {})
    assert plan.stt_replicas > 1 and plan.tts_replicas > 1
    assert plan.tts_replicas >= plan.stt_replicas, \
        "TTS costs ~25x STT's CPU per session -- the default split must weight TTS at least as heavily"


def test_env_overrides_win_over_the_formula():
    cpu = CpuInfo(physical_cores=8, logical_cores=16)
    plan = compute_replica_plan(cpu, [], {
        "STT_REPLICAS": "2", "TTS_REPLICAS": "3",
        "STT_THREADS_PER_REPLICA": "2", "TTS_THREADS_PER_REPLICA": "6",
    })
    assert (plan.stt_replicas, plan.tts_replicas) == (2, 3)
    assert (plan.stt_threads, plan.tts_threads) == (2, 6)


def test_thread_override_is_clamped_to_the_measured_safe_range():
    """docs/CPU_PROFILING_REPORT.md: 16 threads measurably regressed STT on
    this CPU class (hyperthreading contention) -- never let an override pass
    that straight through unclamped."""
    cpu = CpuInfo(physical_cores=8, logical_cores=16)
    plan = compute_replica_plan(cpu, [], {"STT_THREADS_PER_REPLICA": "16"})
    assert plan.stt_threads == 8


def test_no_gpu_does_not_reserve_llm_cpu():
    cpu = CpuInfo(physical_cores=8, logical_cores=16)
    plan = compute_replica_plan(cpu, [], {})
    assert plan.llm_cpu_reserve == 0


def test_replica_ports_never_collide_across_a_wide_range():
    seen = set()
    for i in range(50):
        for port in (*stt_ports(i), *tts_ports(i)):
            assert port not in seen, f"port collision at replica {i}: {port}"
            seen.add(port)
