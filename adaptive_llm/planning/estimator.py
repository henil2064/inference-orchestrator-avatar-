"""Memory estimation for engine planning."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .hardware import MIB
from .registry import ModelSpec


@dataclass(frozen=True)
class ResourceEstimate:
    weight_mib: int
    kv_cache_mib: int
    runtime_mib: int
    total_mib: int
    confidence: str
    notes: list[str] = field(default_factory=list)


def estimate_weights_mib(model: ModelSpec, overhead: float = 1.20) -> int:
    if model.file_size_bytes:
        return math.ceil((model.file_size_bytes / MIB) * overhead)
    bytes_per_param = _bytes_per_parameter(model)
    if model.parameter_count_b <= 0:
        return 0
    return math.ceil(model.parameter_count_b * 1_000_000_000 * bytes_per_param * overhead / MIB)


def estimate_kv_cache_mib(model: ModelSpec, context_tokens: int, concurrent_sequences: int = 1) -> int:
    if not (model.layers and model.kv_heads and model.head_dim):
        return max(256, math.ceil(context_tokens * concurrent_sequences * 0.20))
    bytes_count = (
        model.layers
        * context_tokens
        * concurrent_sequences
        * 2
        * model.kv_heads
        * model.head_dim
        * model.kv_cache_dtype_bytes
    )
    return math.ceil(bytes_count / MIB)


def estimate_vllm(model: ModelSpec, context_tokens: int) -> ResourceEstimate:
    weight = estimate_weights_mib(model, overhead=1.10)
    kv = estimate_kv_cache_mib(model, context_tokens)
    runtime = max(768, math.ceil(weight * 0.05))
    return ResourceEstimate(
        weight_mib=weight,
        kv_cache_mib=kv,
        runtime_mib=runtime,
        total_mib=weight + kv + runtime,
        confidence="medium" if model.file_size_bytes else "low",
        notes=["weights + initial KV cache + vLLM runtime/scratch"],
    )


def estimate_llamacpp(model: ModelSpec, context_tokens: int) -> ResourceEstimate:
    weight = estimate_weights_mib(model, overhead=1.10)
    kv = estimate_kv_cache_mib(model, context_tokens)
    runtime = max(256, math.ceil(weight * 0.05))
    return ResourceEstimate(
        weight_mib=weight,
        kv_cache_mib=kv,
        runtime_mib=runtime,
        total_mib=weight + kv + runtime,
        confidence="medium" if model.file_size_bytes else "low",
        notes=["GGUF weights + KV cache + llama.cpp runtime"],
    )


def estimate_llama_gpu_layers(model: ModelSpec, safe_free_vram_mib: int, context_tokens: int) -> int:
    if not model.layers:
        return 0
    estimate = estimate_llamacpp(model, context_tokens)
    per_layer = max(1, estimate.weight_mib / model.layers)
    available_for_layers = max(0, safe_free_vram_mib - estimate.kv_cache_mib - estimate.runtime_mib)
    return max(0, min(model.layers, int(available_for_layers / per_layer)))


def _bytes_per_parameter(model: ModelSpec) -> float:
    quant = (model.quantization or "").upper()
    dtype = (model.weight_dtype or "").lower()
    if "Q4" in quant:
        return 0.55
    if "Q5" in quant:
        return 0.70
    if "Q8" in quant or dtype in {"int8", "uint8"}:
        return 1.0
    if dtype in {"bfloat16", "bf16", "float16", "fp16"}:
        return 2.0
    if dtype in {"float32", "fp32"}:
        return 4.0
    return 2.0
