"""Engine compatibility checks and hard gates."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import os

from .estimator import ResourceEstimate, estimate_llama_gpu_layers, estimate_llamacpp, estimate_vllm
from .hardware import HardwareSnapshot
from .registry import ModelSpec
from adaptive_llm.core.config import Config


ENGINE_VLLM = "vllm"
ENGINE_SGLANG = "sglang"
ENGINE_LLAMACPP = "llama.cpp"

# vLLM serves a GGUF checkpoint through its own single-file GGUF loader. The
# weights stay quantized in VRAM and go through kernels ported from llama.cpp,
# inside vLLM's normal paged-attention runtime. A GGUF split across shards is
# not supported by that loader, so only a single file qualifies (see
# _model_path_error below).
VLLM_FORMATS = ("safetensors", "gguf")

FULL_GPU = "FULL_GPU"
PARTIAL_GPU = "PARTIAL_GPU"
CPU_ONLY = "CPU_ONLY"


@dataclass(frozen=True)
class Candidate:
    engine: str
    compatible: bool
    execution_mode: str | None
    estimate: ResourceEstimate | None
    reasons: list[str] = field(default_factory=list)
    score: float = 0.0
    n_gpu_layers: int | None = None
    cpu_threads: int | None = None
    gpu_index: int | None = None


def build_candidates(model: ModelSpec, hardware: HardwareSnapshot, config: Config) -> list[Candidate]:
    context = _planned_context_tokens(model, config)
    safe_ram = compute_safe_ram_mib(hardware, config)
    safe_vram = compute_safe_vram_mib(hardware, config)
    return [
        _vllm_candidate(model, hardware, config, context, safe_vram),
        _sglang_candidate(model, hardware, config, context, safe_vram),
        _llamacpp_candidate(model, hardware, config, context, safe_vram, safe_ram),
    ]


def _planned_context_tokens(model: ModelSpec, config: Config) -> int:
    requested = config.max_context_tokens or model.default_context_tokens or min(model.max_context_tokens, 2048)
    return min(requested, model.max_context_tokens)


def _vllm_candidate(
    model: ModelSpec,
    hardware: HardwareSnapshot,
    config: Config,
    context_tokens: int,
    safe_vram_mib: int,
) -> Candidate:
    estimate = estimate_vllm(model, context_tokens)
    reasons: list[str] = []
    gpu_index = _selected_gpu_index(hardware)
    if model.format not in VLLM_FORMATS:
        reasons.append(
            "vLLM requires an HF safetensors directory or a single-file GGUF, got "
            f"{model.format}"
        )
    elif not config.dry_run:
        # vLLM's GGUFModelLoader takes one file and rejects a directory, which
        # is the mirror image of the safetensors checkpoint it otherwise wants.
        path_error = _model_path_error(model, require_file=model.format == "gguf")
        if path_error:
            reasons.append(path_error)
    if not hardware.has_cuda_gpu:
        reasons.append("CUDA GPU not available")
    if safe_vram_mib < estimate.total_mib:
        reasons.append(f"safe free VRAM {safe_vram_mib} MiB < estimated {estimate.total_mib} MiB")
    if not config.dry_run and importlib.util.find_spec("vllm") is None:
        reasons.append("vLLM is not installed in the adaptive LLM environment")
    return Candidate(
        engine=ENGINE_VLLM,
        compatible=not reasons,
        execution_mode=FULL_GPU if not reasons else None,
        estimate=estimate,
        reasons=reasons,
        gpu_index=gpu_index if not reasons else None,
    )


def _sglang_candidate(
    model: ModelSpec,
    hardware: HardwareSnapshot,
    config: Config,
    context_tokens: int,
    safe_vram_mib: int,
) -> Candidate:
    # SGLang serves the same HF/safetensors model family as vLLM, but keeps a
    # radix tree of token prefixes.  That is particularly valuable for our
    # request contract: every new voice turn contains the completed dialogue
    # followed by just one new user message.
    #
    # Keep the estimate deliberately equivalent to vLLM for now. Both engines
    # reserve model weights, a paged KV cache, and CUDA runtime scratch space;
    # an engine-specific estimator can be added once deployment measurements
    # are available.
    estimate = estimate_vllm(model, context_tokens)
    reasons: list[str] = []
    gpu_index = _selected_gpu_index(hardware)
    if model.format != "safetensors":
        reasons.append(f"SGLang requires an HF safetensors model, got {model.format}")
    elif not config.dry_run:
        path_error = _model_path_error(model, require_file=False)
        if path_error:
            reasons.append(path_error)
    if not hardware.has_cuda_gpu:
        reasons.append("CUDA GPU not available")
    if safe_vram_mib < estimate.total_mib:
        reasons.append(f"safe free VRAM {safe_vram_mib} MiB < estimated {estimate.total_mib} MiB")
    if not config.dry_run and not (
        os.environ.get("LLM_SGLANG_URL") or os.environ.get("LLM_SGLANG_PYTHON")
    ):
        reasons.append(
            "set LLM_SGLANG_PYTHON to an isolated SGLang environment, or LLM_SGLANG_URL to a managed server"
        )
    return Candidate(
        engine=ENGINE_SGLANG,
        compatible=not reasons,
        execution_mode=FULL_GPU if not reasons else None,
        estimate=estimate,
        reasons=reasons,
        gpu_index=gpu_index if not reasons else None,
    )


def _llamacpp_candidate(
    model: ModelSpec,
    hardware: HardwareSnapshot,
    config: Config,
    context_tokens: int,
    safe_vram_mib: int,
    safe_ram_mib: int,
) -> Candidate:
    estimate = estimate_llamacpp(model, context_tokens)
    reasons: list[str] = []
    gpu_index = _selected_gpu_index(hardware)
    if model.format != "gguf":
        reasons.append(f"llama.cpp requires GGUF, got {model.format}")
    elif not config.dry_run:
        path_error = _model_path_error(model, require_file=True)
        if path_error:
            reasons.append(path_error)
    if not config.dry_run and importlib.util.find_spec("llama_cpp") is None:
        reasons.append("llama-cpp-python is not installed in the adaptive LLM environment")

    layers = estimate_llama_gpu_layers(model, safe_vram_mib, context_tokens)
    mode = CPU_ONLY
    if hardware.has_cuda_gpu and layers > 0:
        mode = FULL_GPU if model.layers and layers >= model.layers else PARTIAL_GPU

    if safe_ram_mib < estimate.total_mib:
        reasons.append(f"safe free RAM {safe_ram_mib} MiB < estimated {estimate.total_mib} MiB")

    return Candidate(
        engine=ENGINE_LLAMACPP,
        compatible=not reasons,
        execution_mode=mode if not reasons else None,
        estimate=estimate,
        reasons=reasons,
        n_gpu_layers=layers if not reasons else None,
        # Real physical cores, not a logical_cores/2 guess (see hardware.py's
        # _physical_cores -- SMT hides real physical count from a plain
        # logical-core count, and the ratio isn't always 2x). Still capped at
        # 8: docs/CPU_PROFILING_REPORT.md measured this box's STT/TTS
        # single-worker throughput saturating around 4 threads and
        # regressing past 8 from hyperthreading contention -- a generic safe
        # ceiling pending a llama.cpp-specific measurement of its own.
        cpu_threads=max(1, min(8, hardware.static.physical_cores)) if not reasons else None,
        gpu_index=gpu_index if not reasons and mode != CPU_ONLY else None,
    )


def compute_safe_vram_mib(hardware: HardwareSnapshot, config: Config) -> int:
    if not hardware.dynamic.gpus:
        return 0
    gpu = max(hardware.dynamic.gpus, key=lambda g: g.free_vram_mib)
    static = next((s for s in hardware.static.gpus if s.index == gpu.index), None)
    total = static.total_vram_mib if static else gpu.free_vram_mib + gpu.used_vram_mib
    reserve = max(config.min_headroom_mib, int(total * config.vram_headroom_fraction))
    return max(0, gpu.free_vram_mib - reserve)


def _selected_gpu_index(hardware: HardwareSnapshot) -> int | None:
    if not hardware.has_cuda_gpu or not hardware.dynamic.gpus:
        return None
    return max(hardware.dynamic.gpus, key=lambda g: g.free_vram_mib).index


def _model_path_error(model: ModelSpec, *, require_file: bool) -> str | None:
    path = model.local_path
    if path is None:
        return f"model path does not exist: {model.model_path}"
    if require_file and not path.is_file():
        return f"GGUF model path must be a file: {model.model_path}"
    if not require_file and not path.is_dir():
        return f"vLLM model path must be a directory: {model.model_path}"
    return None


def compute_safe_ram_mib(hardware: HardwareSnapshot, config: Config) -> int:
    reserve = max(config.min_headroom_mib, int(hardware.static.total_ram_mib * config.ram_headroom_fraction))
    return max(0, hardware.dynamic.available_ram_mib - reserve)
