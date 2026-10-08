"""Execution planner for adaptive engine selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .compatibility import (
    ENGINE_LLAMACPP,
    ENGINE_SGLANG,
    ENGINE_VLLM,
    FULL_GPU,
    PARTIAL_GPU,
    Candidate,
    build_candidates,
    _planned_context_tokens,
)
from .hardware import HardwareSnapshot
from .registry import ModelSpec
from adaptive_llm.core.config import Config


@dataclass(frozen=True)
class ExecutionPlan:
    model_id: str
    engine: str
    execution_mode: str
    score: float
    context_tokens: int
    max_output_tokens: int
    n_gpu_layers: int | None
    cpu_threads: int | None
    gpu_index: int | None
    runtime_profile: str | None
    engine_env: dict[str, str]
    reasons: list[str]
    rejected: dict[str, list[str]]
    estimate: dict[str, int | str | list[str]] | None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def plan(model: ModelSpec, hardware: HardwareSnapshot, config: Config) -> ExecutionPlan:
    candidates = build_candidates(model, hardware, config)
    known_engines = {candidate.engine for candidate in candidates}
    if config.engine_override and config.engine_override not in known_engines:
        raise SystemExit(
            f"unsupported LLM_ENGINE={config.engine_override!r}; supported engines: "
            + ", ".join(sorted(known_engines))
        )
    scored = [_score_candidate(c, model, hardware) for c in candidates]
    allowed = [c for c in scored if c.compatible]
    if config.engine_override:
        allowed = [c for c in allowed if c.engine == config.engine_override]
    if not allowed:
        rejected = {c.engine: c.reasons or ["not selected"] for c in scored}
        raise SystemExit("no compatible LLM engine found:\n" + json.dumps(rejected, indent=2))

    selected = sorted(
        allowed,
        key=lambda c: (
            c.score,
            _mode_rank(c.execution_mode),
            c.estimate.total_mib if c.estimate else 0,
        ),
        reverse=True,
    )[0]
    rejected = {c.engine: c.reasons for c in scored if c.engine != selected.engine and c.reasons}
    context = _planned_context_tokens(model, config)
    runtime_profile, engine_env = _runtime_profile_env(
        model, hardware, selected.engine, selected.gpu_index
    )
    return ExecutionPlan(
        model_id=model.model_id,
        engine=selected.engine,
        execution_mode=selected.execution_mode or "",
        score=round(selected.score, 2),
        context_tokens=context,
        max_output_tokens=config.max_output_tokens or model.default_max_output_tokens,
        n_gpu_layers=selected.n_gpu_layers,
        cpu_threads=selected.cpu_threads,
        gpu_index=selected.gpu_index,
        runtime_profile=runtime_profile,
        engine_env=engine_env,
        reasons=selected.reasons,
        rejected=rejected,
        estimate=asdict(selected.estimate) if selected.estimate else None,
    )


def write_plan(plan_value: ExecutionPlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan_value.to_json() + "\n", encoding="utf-8")


def _score_candidate(candidate: Candidate, model: ModelSpec, hardware: HardwareSnapshot) -> Candidate:
    if not candidate.compatible:
        return candidate

    compatibility = 100
    hardware_fit = _hardware_fit(candidate, hardware)
    memory_fit = _memory_fit(candidate, hardware)
    performance = _performance_score(candidate, model, hardware)
    quantization = _quantization_score(candidate, model)
    context = 90
    load = max(40, 100 - hardware.dynamic.active_requests * 10)
    score = (
        compatibility * 0.25
        + hardware_fit * 0.20
        + memory_fit * 0.20
        + performance * 0.15
        + quantization * 0.10
        + context * 0.05
        + load * 0.05
    )
    return Candidate(**{**candidate.__dict__, "score": score})


def _hardware_fit(candidate: Candidate, hardware: HardwareSnapshot) -> int:
    if candidate.engine in {ENGINE_VLLM, ENGINE_SGLANG}:
        return 95 if hardware.has_cuda_gpu else 20
    if candidate.engine == ENGINE_LLAMACPP:
        if candidate.execution_mode == PARTIAL_GPU:
            return 80
        if candidate.execution_mode == FULL_GPU:
            return 86
        return 60
    return 50


def _memory_fit(candidate: Candidate, hardware: HardwareSnapshot) -> int:
    if not candidate.estimate:
        return 50
    if candidate.engine in {ENGINE_VLLM, ENGINE_SGLANG} and hardware.dynamic.gpus:
        free = max(g.free_vram_mib for g in hardware.dynamic.gpus)
        ratio = free / max(candidate.estimate.total_mib, 1)
        return min(100, int(ratio * 55))
    ratio = hardware.dynamic.available_ram_mib / max(candidate.estimate.total_mib, 1)
    return min(100, int(ratio * 45))


def _performance_score(candidate: Candidate, model: ModelSpec, hardware: HardwareSnapshot) -> int:
    if candidate.engine == ENGINE_VLLM:
        return 95
    if candidate.engine == ENGINE_SGLANG:
        # Radix caching makes it attractive for conversational profiles, while
        # the established vLLM path remains the generic default unless an
        # operator selects SGLang explicitly.
        return 93
    if candidate.engine == ENGINE_LLAMACPP:
        if candidate.execution_mode == FULL_GPU:
            return 75
        if candidate.execution_mode == PARTIAL_GPU:
            return 68
        return 45
    return 60


def _quantization_score(candidate: Candidate, model: ModelSpec) -> int:
    if candidate.engine == ENGINE_LLAMACPP and model.format == "gguf":
        return 100
    if candidate.engine == ENGINE_VLLM and model.format == "safetensors":
        return 90
    if candidate.engine == ENGINE_VLLM and model.format == "gguf":
        # vLLM serves GGUF with kernels ported from llama.cpp rather than its
        # own tuned ones, and falls back to dequantize-then-GEMM for quant
        # types with no MMQ kernel. Score it below both native pairings: on a
        # CUDA box it usually still wins overall, because continuous batching
        # is worth more than the kernel gap, but it is never the better
        # *quantization* fit.
        return 80
    if candidate.engine == ENGINE_SGLANG and model.format == "safetensors":
        return 92
    return 50


def _mode_rank(mode: str | None) -> int:
    return {FULL_GPU: 3, PARTIAL_GPU: 2}.get(mode or "", 1)


def _runtime_profile_env(
    model: ModelSpec,
    hardware: HardwareSnapshot,
    engine: str,
    gpu_index: int | None,
) -> tuple[str | None, dict[str, str]]:
    selected_gpu = next((gpu for gpu in hardware.static.gpus if gpu.index == gpu_index), None)
    if selected_gpu:
        total_vram_mib = selected_gpu.total_vram_mib
    elif not hardware.static.gpus:
        total_vram_mib = 0
    else:
        total_vram_mib = max(gpu.total_vram_mib for gpu in hardware.static.gpus)

    for profile in model.runtime_profiles:
        if profile.engine and profile.engine != engine:
            continue
        if profile.min_vram_mib is not None and total_vram_mib < profile.min_vram_mib:
            continue
        if profile.max_vram_mib is not None and total_vram_mib > profile.max_vram_mib:
            continue
        return profile.name, dict(profile.env)
    return None, {}
