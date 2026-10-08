from dataclasses import replace
from pathlib import Path

from adaptive_llm.core.config import Config
import pytest

from adaptive_llm.planning.compatibility import (
    ENGINE_LLAMACPP, ENGINE_SGLANG, ENGINE_VLLM, FULL_GPU, PARTIAL_GPU,
)
from adaptive_llm.planning.hardware import GpuDynamic, GpuStatic, load_fixture
from adaptive_llm.planning.planner import plan
from adaptive_llm.planning.registry import load_registry, resolve_model


ROOT = Path(__file__).resolve().parents[2]


def _config(model_id: str) -> Config:
    return Config(
        model_id=model_id,
        model_profiles_path=ROOT / "adaptive_llm" / "model-profiles.json",
        model_ref=None,
        model_file=None,
        model_dir=ROOT / "adaptive_llm" / "models",
        model_profile=None,
        auto_download=True,
        public_addr="[::]:50053",
        worker_addr="127.0.0.1:50054",
        metrics_port=9106,
        engine_override=None,
        dry_run=True,
        auto_start_worker=False,
        worker_ready_timeout_s=1,
        vram_headroom_fraction=0.15,
        ram_headroom_fraction=0.10,
        min_headroom_mib=512,
        max_context_tokens=None,
        max_output_tokens=None,
        worker_python="python",
        max_inflight_requests=4,
        max_queued_requests=16,
        resource_metrics_interval_s=5,
        profile_bandwidth=False,
    )


def test_gtx1650_gguf_selects_llamacpp_partial_gpu():
    config = _config("medgemma_1_5_4b_q4")
    registry = load_registry(config.model_profiles_path)
    model = resolve_model(registry, config.model_id)
    hardware = load_fixture(str(ROOT / "adaptive_llm" / "tests" / "fixtures" / "gtx1650_4gb.json"))

    selected = plan(model, hardware, config)

    assert selected.engine == ENGINE_LLAMACPP
    assert selected.execution_mode == PARTIAL_GPU
    assert selected.n_gpu_layers and selected.n_gpu_layers > 0
    assert ENGINE_VLLM in selected.rejected


def test_rtx4090_safetensors_selects_vllm_full_gpu():
    config = _config("qwen3_8b_bf16")
    registry = load_registry(config.model_profiles_path)
    model = resolve_model(registry, config.model_id)
    hardware = load_fixture(str(ROOT / "adaptive_llm" / "tests" / "fixtures" / "rtx4090_24gb.json"))

    selected = plan(model, hardware, config)

    assert selected.engine == ENGINE_VLLM
    assert selected.execution_mode == "FULL_GPU"
    assert ENGINE_LLAMACPP in selected.rejected


def test_gtx1650_qwen_small_safetensors_selects_vllm_with_default_context():
    config = _config("qwen3_0_6b_bf16")
    registry = load_registry(config.model_profiles_path)
    model = resolve_model(registry, config.model_id)
    hardware = load_fixture(str(ROOT / "adaptive_llm" / "tests" / "fixtures" / "gtx1650_4gb.json"))

    selected = plan(model, hardware, config)

    assert selected.engine == ENGINE_VLLM
    assert selected.execution_mode == "FULL_GPU"
    assert selected.context_tokens == 2048
    assert selected.runtime_profile == "gtx_4gb_vllm_balanced"
    assert selected.engine_env["LLM_ENFORCE_EAGER"] == "0"
    # Phase 2 un-throttle candidate for this GPU class: measured headroom on
    # this exact box (see gtx1650_4gb.json) showed ~6x concurrency already
    # fit at 0.70 utilization, so 8 seqs / 0.75 util is a conservative step,
    # not the L40S-scale numbers the plan's Phase 2 doc names for that box.
    assert selected.engine_env["LLM_GPU_MEMORY_UTILIZATION"] == "0.75"
    assert selected.engine_env["LLM_MAX_NUM_SEQS"] == "8"
    assert selected.engine_env["LLM_ENABLE_PREFIX_CACHING"] == "1"


def test_sglang_override_selects_full_gpu_for_safetensors():
    config = replace(_config("qwen3_0_6b_bf16"), engine_override=ENGINE_SGLANG)
    registry = load_registry(config.model_profiles_path)
    model = resolve_model(registry, config.model_id)
    hardware = load_fixture(str(ROOT / "adaptive_llm" / "tests" / "fixtures" / "gtx1650_4gb.json"))

    selected = plan(model, hardware, config)

    assert selected.engine == ENGINE_SGLANG
    assert selected.execution_mode == "FULL_GPU"


def test_selected_gpu_is_carried_into_plan_and_used_for_runtime_profile():
    config = _config("qwen3_0_6b_bf16")
    registry = load_registry(config.model_profiles_path)
    model = resolve_model(registry, config.model_id)
    base = load_fixture(str(ROOT / "adaptive_llm" / "tests" / "fixtures" / "gtx1650_4gb.json"))
    hardware = replace(
        base,
        static=replace(
            base.static,
            gpus=base.static.gpus
            + [GpuStatic(index=1, name="RTX fixture", total_vram_mib=24576)],
        ),
        dynamic=replace(
            base.dynamic,
            gpus=base.dynamic.gpus
            + [GpuDynamic(index=1, free_vram_mib=22000, used_vram_mib=2576)],
        ),
    )

    selected = plan(model, hardware, config)

    assert selected.gpu_index == 1
    assert selected.runtime_profile is None


def test_missing_model_path_is_rejected_before_worker_startup():
    config = replace(_config("medgemma_1_5_4b_q4"), dry_run=False)
    registry = load_registry(config.model_profiles_path)
    model = replace(resolve_model(registry, config.model_id), model_path="/missing/model.gguf")
    hardware = load_fixture(str(ROOT / "adaptive_llm" / "tests" / "fixtures" / "gtx1650_4gb.json"))

    try:
        plan(model, hardware, config)
    except SystemExit as exc:
        assert "model path does not exist: /missing/model.gguf" in str(exc)
    else:
        raise AssertionError("expected missing model path to be rejected")


def test_single_file_gguf_can_be_served_by_vllm(tmp_path: Path):
    """vLLM loads a GGUF through its own loader, so a GGUF must not be
    rejected out of hand -- only a *sharded* one it cannot open."""
    config = replace(_config("medgemma_1_5_4b_q4"), dry_run=False, engine_override=ENGINE_VLLM)
    registry = load_registry(config.model_profiles_path)
    checkpoint = tmp_path / "medgemma-Q4_K_M.gguf"
    checkpoint.write_bytes(b"GGUF")
    model = replace(resolve_model(registry, config.model_id), model_path=str(checkpoint))
    hardware = load_fixture(str(ROOT / "adaptive_llm" / "tests" / "fixtures" / "rtx4090_24gb.json"))

    selected = plan(model, hardware, config)

    assert selected.engine == ENGINE_VLLM
    assert selected.execution_mode == FULL_GPU


def test_vllm_rejects_a_gguf_path_that_is_not_a_single_file(tmp_path: Path):
    config = replace(_config("medgemma_1_5_4b_q4"), dry_run=False, engine_override=ENGINE_VLLM)
    registry = load_registry(config.model_profiles_path)
    sharded = tmp_path / "medgemma-Q4_K_M-split.gguf"
    sharded.mkdir()
    model = replace(resolve_model(registry, config.model_id), model_path=str(sharded))
    hardware = load_fixture(str(ROOT / "adaptive_llm" / "tests" / "fixtures" / "rtx4090_24gb.json"))

    with pytest.raises(SystemExit) as exc:
        plan(model, hardware, config)

    assert "GGUF model path must be a file" in str(exc.value)
