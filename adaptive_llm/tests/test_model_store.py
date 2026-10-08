"""Offline checks for Hugging Face cache resolution and JSON profiles."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import httpx
import huggingface_hub.utils as hf_errors
import pytest

from adaptive_llm.core.config import Config
from adaptive_llm.planning import model_store
from adaptive_llm.planning.hardware import load_fixture
from adaptive_llm.planning.model_store import resolve_huggingface_model
from adaptive_llm.planning.registry import load_registry


ROOT = Path(__file__).resolve().parents[2]
# All resolve_huggingface_model() calls below use dry_run=True configs (see
# _config's defaults), so _reject_if_remote_size_cannot_fit's pre-download
# size check short-circuits immediately -- this fixture's exact values don't
# matter to these tests, just that a valid HardwareSnapshot is passed.
_HARDWARE = load_fixture(str(ROOT / "adaptive_llm" / "tests" / "fixtures" / "gtx1650_4gb.json"))


def _config(cache_root: Path, model_ref: str, **changes) -> Config:
    values = dict(
        model_id="medgemma_1_5_4b_q4",
        model_profiles_path=ROOT / "adaptive_llm" / "model-profiles.json",
        model_ref=model_ref,
        model_file=None,
        model_dir=cache_root,
        model_profile=None,
        auto_download=False,
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
        max_inflight_requests=1,
        max_queued_requests=0,
        resource_metrics_interval_s=5,
        profile_bandwidth=False,
    )
    values.update(changes)
    return Config(**values)


def test_cached_safetensors_model_is_inferred_and_matched_to_profile(tmp_path: Path):
    model_dir = tmp_path / "huggingface" / "Qwen--Qwen3-0.6B"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "torch_dtype": "bfloat16",
                "max_position_embeddings": 32768,
                "num_hidden_layers": 28,
                "hidden_size": 1024,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
            }
        ),
        encoding="utf-8",
    )
    (model_dir / "model.safetensors").write_bytes(b"weights")

    config = _config(tmp_path, "Qwen/Qwen3-0.6B")
    model = resolve_huggingface_model(config, load_registry(config.model_profiles_path), _HARDWARE)

    assert model.format == "safetensors"
    assert model.model_path == str(model_dir)
    assert model.architecture == "qwen3"
    assert model.runtime_profiles[0].name == "gtx_4gb_vllm_balanced"
    # A matched profile may deliberately supply a more conservative value than
    # the generic config inference.
    assert model.head_dim == 128
    assert model.file_size_bytes == len(b"weights") + len((model_dir / "config.json").read_bytes())


def test_missing_cache_does_not_download_when_disabled(tmp_path: Path):
    config = _config(tmp_path, "Qwen/Qwen3-0.6B")
    try:
        resolve_huggingface_model(config, load_registry(config.model_profiles_path), _HARDWARE)
    except SystemExit as exc:
        assert "LLM_AUTO_DOWNLOAD=1" in str(exc)
    else:
        raise AssertionError("expected missing cache to fail without downloading")


def test_cached_single_gguf_is_selected_without_an_explicit_filename(tmp_path: Path):
    model_dir = tmp_path / "huggingface" / "owner--model"
    model_dir.mkdir(parents=True)
    gguf = model_dir / "model-Q4_K_M.gguf"
    gguf.write_bytes(b"weights")

    config = _config(tmp_path, "owner/model")
    model = resolve_huggingface_model(config, {}, _HARDWARE)

    assert model.format == "gguf"
    assert model.model_path == str(gguf)


def test_profile_artifact_inside_model_dir_is_reused(tmp_path: Path):
    gguf = tmp_path / "existing" / "model-Q4_K_M.gguf"
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"weights")
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(
        json.dumps(
            {
                "models": {
                    "known": {
                        "model_ref": "owner/model",
                        "model_path": str(gguf),
                        "format": "gguf"
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    config = _config(tmp_path, "owner/model", model_profiles_path=profile_path)
    model = resolve_huggingface_model(config, load_registry(profile_path), _HARDWARE)

    assert model.model_path == str(gguf)


def _http_error(cls, status_code: int, message: str = "boom"):
    response = httpx.Response(status_code, request=httpx.Request("GET", "https://huggingface.co/x"))
    return cls(message, response=response)


# ---------------------------------------------------------------------------
# Sharded-download completeness: an interrupted snapshot_download can leave a
# sharded safetensors model with some shards missing while still satisfying a
# naive "*.safetensors exists" glob check.
# ---------------------------------------------------------------------------

def _write_sharded_config(model_dir: Path, shard_names: list[str]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps({"model_type": "qwen3"}), encoding="utf-8")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {f"layer.{i}": name for i, name in enumerate(shard_names)}}),
        encoding="utf-8",
    )


def test_sharded_snapshot_with_a_missing_shard_is_not_complete(tmp_path: Path):
    model_dir = tmp_path / "model"
    _write_sharded_config(model_dir, ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"weights")
    # model-00002-of-00002.safetensors is missing -- an interrupted download.

    assert model_store._snapshot_is_complete(model_dir) is False


def test_sharded_snapshot_with_every_shard_present_is_complete(tmp_path: Path):
    model_dir = tmp_path / "model"
    _write_sharded_config(model_dir, ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"weights")
    (model_dir / "model-00002-of-00002.safetensors").write_bytes(b"weights")

    assert model_store._snapshot_is_complete(model_dir) is True


# ---------------------------------------------------------------------------
# Error classification: a gated/missing/not-found repo should never be
# retried, and its message should tell the operator what to actually do.
# ---------------------------------------------------------------------------

def test_gated_repo_is_not_retryable_and_names_hf_token():
    reason, retryable = model_store._classify_hf_error(_http_error(hf_errors.GatedRepoError, 403))

    assert retryable is False
    assert "HF_TOKEN" in reason


def test_repository_not_found_is_not_retryable():
    reason, retryable = model_store._classify_hf_error(_http_error(hf_errors.RepositoryNotFoundError, 404))

    assert retryable is False
    assert "no such" in reason.lower()


def test_rate_limit_is_retryable():
    _, retryable = model_store._classify_hf_error(_http_error(hf_errors.HfHubHTTPError, 429))

    assert retryable is True


def test_network_transport_error_is_retryable():
    _, retryable = model_store._classify_hf_error(httpx.ConnectError("connection refused"))

    assert retryable is True


# ---------------------------------------------------------------------------
# Retry policy: retryable failures get a bounded number of attempts with
# backoff; non-retryable failures fail on the first attempt.
# ---------------------------------------------------------------------------

def test_non_retryable_failure_is_not_retried(monkeypatch):
    monkeypatch.setattr(model_store.time, "sleep", lambda _s: None)
    calls = []

    def attempt():
        calls.append(1)
        raise _http_error(hf_errors.RepositoryNotFoundError, 404)

    with pytest.raises(SystemExit) as exc_info:
        model_store._download_with_retry("owner/model", "download snapshot", attempt)

    assert len(calls) == 1
    assert "no such" in str(exc_info.value).lower()


def test_retryable_failure_is_retried_a_bounded_number_of_times(monkeypatch):
    monkeypatch.setattr(model_store.time, "sleep", lambda _s: None)
    calls = []

    def attempt():
        calls.append(1)
        raise httpx.ConnectError("connection refused")

    with pytest.raises(SystemExit) as exc_info:
        model_store._download_with_retry("owner/model", "download snapshot", attempt)

    assert len(calls) == 1 + len(model_store._RETRY_DELAYS_S)
    assert "network error" in str(exc_info.value).lower()


def test_retryable_failure_succeeds_after_a_transient_hiccup(monkeypatch):
    monkeypatch.setattr(model_store.time, "sleep", lambda _s: None)
    calls = []

    def attempt():
        calls.append(1)
        if len(calls) < 2:
            raise httpx.ConnectError("connection refused")
        return "ok"

    assert model_store._download_with_retry("owner/model", "download snapshot", attempt) == "ok"
    assert len(calls) == 2


def test_resolve_huggingface_model_reports_a_gated_repo_actionably(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(model_store.time, "sleep", lambda _s: None)

    def fake_snapshot_download(*, repo_id, local_dir, token):
        raise _http_error(hf_errors.GatedRepoError, 403)

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    config = _config(tmp_path, "owner/gated-model", auto_download=True)

    with pytest.raises(SystemExit) as exc_info:
        resolve_huggingface_model(config, load_registry(config.model_profiles_path), _HARDWARE)

    message = str(exc_info.value)
    assert "gated" in message.lower()
    assert "HF_TOKEN" in message
    assert "owner/gated-model" in message


# ---------------------------------------------------------------------------
# Pre-download size check: reject a model that obviously can't fit before
# downloading any weight bytes, using Hugging Face's file-size metadata.
# ---------------------------------------------------------------------------

class _FakeSibling:
    def __init__(self, rfilename: str, size: int):
        self.rfilename = rfilename
        self.size = size


class _FakeModelInfo:
    def __init__(self, siblings):
        self.siblings = siblings


def _patch_hf_api(monkeypatch, siblings):
    class _FakeHfApi:
        def model_info(self, repo_id, files_metadata=True, token=None):
            return _FakeModelInfo(siblings)

    monkeypatch.setattr("huggingface_hub.HfApi", _FakeHfApi)


def test_oversized_safetensors_repo_is_rejected_before_downloading(tmp_path: Path, monkeypatch):
    _patch_hf_api(monkeypatch, [_FakeSibling("model.safetensors", 8_000 * 1024 * 1024)])  # 8 GiB, won't fit 4 GiB

    def fail_if_called(*args, **kwargs):
        raise AssertionError("must not download a model already known to be too big")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fail_if_called)
    config = _config(tmp_path, "owner/huge-model", auto_download=True, dry_run=False)

    with pytest.raises(SystemExit) as exc_info:
        model_store._ensure_hf_snapshot(config, tmp_path / "cache", _HARDWARE)

    message = str(exc_info.value)
    assert "owner/huge-model" in message
    assert "MiB" in message


def test_small_safetensors_repo_proceeds_to_download(tmp_path: Path, monkeypatch):
    _patch_hf_api(monkeypatch, [_FakeSibling("model.safetensors", 500 * 1024 * 1024)])  # 500 MiB, fits easily
    called = []
    monkeypatch.setattr("huggingface_hub.snapshot_download",
                        lambda *, repo_id, local_dir, token: called.append(repo_id))
    config = _config(tmp_path, "owner/small-model", auto_download=True, dry_run=False)

    model_store._ensure_hf_snapshot(config, tmp_path / "cache", _HARDWARE)

    assert called == ["owner/small-model"]


def test_no_gpu_rejects_any_safetensors_repo_regardless_of_size(tmp_path: Path, monkeypatch):
    _patch_hf_api(monkeypatch, [_FakeSibling("model.safetensors", 10 * 1024 * 1024)])  # tiny, but no GPU at all

    def fail_if_called(*args, **kwargs):
        raise AssertionError("must not download a GPU-only format with no GPU present")

    monkeypatch.setattr("huggingface_hub.snapshot_download", fail_if_called)
    no_gpu_hardware = load_fixture(str(ROOT / "adaptive_llm" / "tests" / "fixtures" / "gtx1650_4gb.json"))
    no_gpu_hardware = replace(no_gpu_hardware, static=replace(no_gpu_hardware.static, cuda_available=False, gpus=[]))
    config = _config(tmp_path, "owner/tiny-model", auto_download=True, dry_run=False)

    with pytest.raises(SystemExit) as exc_info:
        model_store._ensure_hf_snapshot(config, tmp_path / "cache", no_gpu_hardware)

    assert "no cuda gpu" in str(exc_info.value).lower() or "no CUDA GPU" in str(exc_info.value)


def test_hf_metadata_failure_does_not_block_the_download_attempt(tmp_path: Path, monkeypatch):
    """Best-effort: if the size pre-check itself can't reach Hugging Face,
    fall through to the real download attempt (which has its own, already-
    good error classification) instead of blocking on an optimization."""
    class _BrokenHfApi:
        def model_info(self, repo_id, files_metadata=True, token=None):
            raise RuntimeError("network unreachable")

    monkeypatch.setattr("huggingface_hub.HfApi", _BrokenHfApi)
    called = []
    monkeypatch.setattr("huggingface_hub.snapshot_download",
                        lambda *, repo_id, local_dir, token: called.append(repo_id))
    config = _config(tmp_path, "owner/model", auto_download=True, dry_run=False)

    model_store._ensure_hf_snapshot(config, tmp_path / "cache", _HARDWARE)

    assert called == ["owner/model"]


def test_dry_run_skips_the_pre_download_check_entirely(tmp_path: Path, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("dry_run must never touch the network")

    monkeypatch.setattr("huggingface_hub.HfApi", fail_if_called)
    config = _config(tmp_path, "owner/model", auto_download=True, dry_run=True)

    model_store._reject_if_remote_size_cannot_fit(config, _HARDWARE, filename=None)  # must not raise/call HfApi


def test_oversized_gguf_file_is_rejected_before_downloading(tmp_path: Path, monkeypatch):
    _patch_hf_api(monkeypatch, [_FakeSibling("model-Q4.gguf", 8_000 * 1024 * 1024)])

    def fail_if_called(*args, **kwargs):
        raise AssertionError("must not download a GGUF file already known to be too big")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fail_if_called)
    config = _config(tmp_path, "owner/huge-gguf-model", auto_download=True, dry_run=False)

    with pytest.raises(SystemExit) as exc_info:
        model_store._ensure_gguf(config, tmp_path / "cache", "model-Q4.gguf", _HARDWARE)

    assert "owner/huge-gguf-model" in str(exc_info.value)


def _write_gguf(path: Path, architecture: str = "gemma3") -> Path:
    """Write the smallest valid GGUF that still carries a full header."""
    gguf_py = pytest.importorskip("gguf")
    numpy = pytest.importorskip("numpy")

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf_py.GGUFWriter(str(path), architecture)
    writer.add_context_length(131072)
    writer.add_block_count(62)
    writer.add_embedding_length(5376)
    writer.add_head_count(32)
    writer.add_head_count_kv(16)
    writer.add_key_length(128)
    writer.add_tensor("token_embd.weight", numpy.zeros((2, 4), dtype=numpy.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return path


def test_gguf_shape_and_context_come_from_the_files_own_header(tmp_path: Path):
    """A GGUF has no config.json, so without reading its header the planner
    sizes the KV cache from a guess and clamps every context to 2048."""
    checkpoint = _write_gguf(tmp_path / "huggingface" / "owner--model" / "model-Q4_K_M.gguf")

    config = _config(tmp_path, "owner/model")
    model = resolve_huggingface_model(config, {}, _HARDWARE)

    assert model.model_path == str(checkpoint)
    assert model.architecture == "gemma3"
    assert model.max_context_tokens == 131072
    assert (model.layers, model.hidden_size, model.kv_heads, model.head_dim) == (62, 5376, 16, 128)
    # The served default stays conservative; only the ceiling is raised.
    assert model.default_context_tokens == 2048


def test_an_unreadable_gguf_header_falls_back_to_the_previous_defaults(tmp_path: Path):
    checkpoint = tmp_path / "huggingface" / "owner--model" / "model-Q4_K_M.gguf"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"not actually a GGUF file")

    config = _config(tmp_path, "owner/model")
    model = resolve_huggingface_model(config, {}, _HARDWARE)

    assert model.architecture == "unknown"
    assert model.max_context_tokens == 2048
    assert model.layers is None


def test_a_profile_overrides_metadata_read_from_the_gguf(tmp_path: Path):
    """model-profiles.json stays authoritative for planning/tuning metadata,
    and is the only place a GGUF's tokenizer repository can come from."""
    _write_gguf(tmp_path / "huggingface" / "owner--model" / "model-Q4_K_M.gguf")
    profiles = tmp_path / "profiles.json"
    profiles.write_text(json.dumps({"models": {"tuned": {
        "model_ref": "owner/model", "format": "gguf", "architecture": "gemma3",
        "parameter_count_b": 27.0, "tokenizer": "google/medgemma-27b-it",
        "max_context_tokens": 8192, "layers": 62,
    }}}), encoding="utf-8")

    config = _config(tmp_path, "owner/model", model_profiles_path=profiles)
    model = resolve_huggingface_model(config, load_registry(profiles), _HARDWARE)

    assert model.tokenizer == "google/medgemma-27b-it"
    assert model.max_context_tokens == 8192
    assert model.format == "gguf"
