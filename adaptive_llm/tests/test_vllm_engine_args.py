"""Pure vLLM tuning environment mapping; no GPU or vLLM import required."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from transformers import Gemma3TextConfig

from adaptive_llm.engines import common
from adaptive_llm.engines.common import (
    apply_triton_prefix_prefill_limit,
    disable_float16_safety_blocklist,
    engine_kwargs_from_env,
    ensure_gemma3_sliding_window_pattern,
    ensure_v1_default,
    ensure_v1_gpu_compatibility,
)


def test_gtx_tuning_values_map_to_vllm_arguments():
    kwargs = engine_kwargs_from_env(
        {
            "LLM_GPU_MEMORY_UTILIZATION": "0.70",
            "LLM_MAX_MODEL_LEN": "2048",
            "LLM_MAX_NUM_BATCHED_TOKENS": "512",
            "LLM_MAX_NUM_SEQS": "4",
            "LLM_ENABLE_PREFIX_CACHING": "1",
            "LLM_ENABLE_CHUNKED_PREFILL": "1",
            "LLM_ENFORCE_EAGER": "0",
            "LLM_KV_CACHE_DTYPE": "auto",
            "LLM_DISABLE_LOG_STATS": "0",
            "LLM_SWAP_SPACE_GIB": "0",
            "LLM_CPU_OFFLOAD_GB": "0",
        }
    )
    assert kwargs["gpu_memory_utilization"] == 0.70
    assert kwargs["max_model_len"] == 2048
    assert kwargs["max_num_batched_tokens"] == 512
    assert kwargs["max_num_seqs"] == 4
    assert kwargs["enable_prefix_caching"] is True
    assert kwargs["enable_chunked_prefill"] is True
    assert kwargs["enforce_eager"] is False
    assert kwargs["kv_cache_dtype"] == "auto"
    assert kwargs["disable_log_stats"] is False
    assert kwargs["swap_space"] == 0.0
    assert kwargs["cpu_offload_gb"] == 0.0


def test_empty_optional_values_do_not_override_vllm_defaults():
    kwargs = engine_kwargs_from_env({"LLM_MAX_NUM_SEQS": "", "LLM_SWAP_SPACE_GIB": ""})
    assert kwargs["max_num_seqs"] is None
    assert "swap_space" not in kwargs


def test_invalid_boolean_fails_with_actionable_error():
    try:
        engine_kwargs_from_env({"LLM_ENFORCE_EAGER": "sometimes"})
    except ValueError as exc:
        assert "LLM_ENFORCE_EAGER" in str(exc)
    else:
        raise AssertionError("expected invalid boolean to be rejected")


def test_v1_default_respects_explicit_opt_out():
    env: dict[str, str] = {}
    ensure_v1_default(env)
    assert env["VLLM_USE_V1"] == "1"
    env = {"VLLM_USE_V1": "0"}
    ensure_v1_default(env)
    assert env["VLLM_USE_V1"] == "0"


def test_pre_ampere_gpu_automatically_falls_back_to_vllm_v0():
    env: dict[str, str] = {}
    with patch.object(common, "cuda_compute_capability", return_value=(7, 5)):
        capability = ensure_v1_gpu_compatibility(env)

    assert capability == (7, 5)
    assert env["VLLM_USE_V1"] == "0"


def test_gemma3_legacy_sliding_window_pattern_alias_is_writable():
    ensure_gemma3_sliding_window_pattern()
    config = Gemma3TextConfig(sliding_window_pattern=6)

    assert config.sliding_window_pattern == 6
    assert config._sliding_window_pattern == 6


def test_unsafe_float16_guard_supports_newer_nested_vllm_layout():
    blocklist = {"gemma3"}

    def import_module(name):
        assert name == "vllm.config.model"
        return SimpleNamespace(_FLOAT16_NOT_SUPPORTED_MODELS=blocklist)

    with patch.object(common.importlib, "import_module", side_effect=import_module):
        assert disable_float16_safety_blocklist() is True
    assert not blocklist


def test_unsafe_float16_guard_supports_flat_vllm_layout():
    blocklist = {"gemma3"}

    def import_module(name):
        if name == "vllm.config.model":
            raise ModuleNotFoundError("not a package", name=name)
        assert name == "vllm.config"
        return SimpleNamespace(_FLOAT16_NOT_SUPPORTED_MODELS=blocklist)

    with patch.object(common.importlib, "import_module", side_effect=import_module):
        assert disable_float16_safety_blocklist() is True
    assert not blocklist


def test_pre_ampere_gpu_cannot_use_triton_prefix_prefill_at_head_size_256():
    kwargs = {
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "max_model_len": 2048,
        "max_num_batched_tokens": 256,
    }

    assert apply_triton_prefix_prefill_limit(kwargs, 256, (7, 5)) is True
    assert kwargs["enable_prefix_caching"] is False
    assert kwargs["enable_chunked_prefill"] is False
    # vLLM rejects a batch budget below the context length without chunking.
    assert kwargs["max_num_batched_tokens"] == 2048


def test_unset_context_length_drops_the_now_invalid_batch_budget():
    kwargs = {"enable_chunked_prefill": True, "max_model_len": None, "max_num_batched_tokens": 256}

    assert apply_triton_prefix_prefill_limit(kwargs, 256, (7, 5)) is True
    assert kwargs["max_num_batched_tokens"] is None


def test_triton_prefix_prefill_limit_leaves_smaller_heads_and_newer_gpus_alone():
    for head_size, capability in ((128, (7, 5)), (256, (8, 9)), (256, None)):
        kwargs = {"enable_prefix_caching": True, "enable_chunked_prefill": True}
        assert apply_triton_prefix_prefill_limit(kwargs, head_size, capability) is False
        assert kwargs["enable_prefix_caching"] is True
        assert kwargs["enable_chunked_prefill"] is True
