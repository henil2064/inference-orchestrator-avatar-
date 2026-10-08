"""Offline checks for the vLLM-GGUF compatibility shims.

These cover the four places where vLLM 0.8.3 cannot correctly load and serve a
Gemma 3 GGUF unassisted. Each shim is exercised against a stand-in for the
vLLM internals it patches, so the suite stays fast and needs no CUDA, no vLLM
import, and no model file.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from adaptive_llm.engines.common import (
    attention_head_size,
    ensure_gguf_gemma3_arch_alias,
    ensure_gguf_gemma3_quantized_embedding,
    ensure_gguf_gemma3_sliding_window,
    ensure_v0_for_gguf,
    gguf_source_repository,
    remove_gemma_norm_offset,
    require_gguf_runtime,
)


GGUF = "/models/medgemma-27b-it-Q4_K_M.gguf"


def test_gguf_forces_the_v0_engine_when_v1_was_not_requested():
    env: dict[str, str] = {}

    assert ensure_v0_for_gguf(env, GGUF) is True
    assert env["VLLM_USE_V1"] == "0"


def test_an_explicit_v1_choice_is_left_alone_for_gguf():
    # vLLM raises rather than downgrading when V1 was asked for explicitly.
    # That error belongs to the operator who set the variable, not to us.
    env = {"VLLM_USE_V1": "1"}

    assert ensure_v0_for_gguf(env, GGUF) is False
    assert env["VLLM_USE_V1"] == "1"


@pytest.mark.parametrize("model_path", [None, "", "/models/medgemma-27b-it"])
def test_a_safetensors_model_keeps_vllms_own_engine_selection(model_path):
    env: dict[str, str] = {}

    assert ensure_v0_for_gguf(env, model_path) is False
    assert "VLLM_USE_V1" not in env


def test_attention_head_size_reads_a_gguf_files_own_metadata(monkeypatch):
    """A bare .gguf path is not an HF config directory, so from_pretrained()
    must be pointed at it with gguf_file= or apply_triton_prefix_prefill_limit()
    silently never fires for any GGUF model -- see the head_size 256 pre-Ampere
    crash this module's docstring and VLLM_TUNING.md describe."""
    from transformers import AutoConfig

    seen = {}

    def fake_from_pretrained(model_path, **kwargs):
        seen["model_path"] = model_path
        seen["gguf_file"] = kwargs.get("gguf_file")
        return SimpleNamespace(head_dim=256)

    monkeypatch.setattr(AutoConfig, "from_pretrained", fake_from_pretrained)

    assert attention_head_size("/models/medgemma-1.5-4b-it-Q4_K_S.gguf") == 256
    assert seen == {
        "model_path": "/models",
        "gguf_file": "medgemma-1.5-4b-it-Q4_K_S.gguf",
    }


def test_attention_head_size_still_uses_a_plain_path_for_a_non_gguf_checkpoint(monkeypatch):
    from transformers import AutoConfig

    seen = {}

    def fake_from_pretrained(model_path, **kwargs):
        seen["model_path"] = model_path
        seen["gguf_file"] = kwargs.get("gguf_file")
        return SimpleNamespace(head_dim=128)

    monkeypatch.setattr(AutoConfig, "from_pretrained", fake_from_pretrained)

    assert attention_head_size("/models/medgemma-27b-it") == 128
    assert seen == {"model_path": "/models/medgemma-27b-it", "gguf_file": None}


@pytest.fixture
def gguf_arch_names():
    """Restore the gguf package's architecture table after a test mutates it."""
    gguf = pytest.importorskip("gguf")
    original = dict(gguf.MODEL_ARCH_NAMES)
    yield gguf
    gguf.MODEL_ARCH_NAMES.clear()
    gguf.MODEL_ARCH_NAMES.update(original)


def test_gemma3_arch_alias_matches_the_config_spelling(gguf_arch_names):
    gguf = gguf_arch_names

    assert ensure_gguf_gemma3_arch_alias() is True
    # vLLM's GGUF loader searches this table for the config's model_type,
    # which Transformers reports as gemma3_text for a text-only Gemma 3 GGUF.
    assert gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.GEMMA3] == "gemma3_text"


def test_gemma3_arch_alias_is_idempotent(gguf_arch_names):
    assert ensure_gguf_gemma3_arch_alias() is True
    assert ensure_gguf_gemma3_arch_alias() is False


class _FakeConfig:
    """Enough of a PretrainedConfig for the sliding-window shim."""


class _FakeModelConfig:
    def __init__(self, config):
        self.hf_config = config
        self.hf_text_config = config


def _install_fake_loader(monkeypatch, recorder):
    class GGUFModelLoader:
        def _get_gguf_weights_map(self, model_config):
            recorder.append(getattr(model_config.hf_config, "sliding_window", "absent"))
            return {"blk.0.attn_q.weight": "layers.0.self_attn.q_proj.weight"}

    module = types.ModuleType("vllm.model_executor.model_loader.loader")
    module.GGUFModelLoader = GGUFModelLoader
    for name in ("vllm", "vllm.model_executor", "vllm.model_executor.model_loader"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.model_loader.loader", module)
    return GGUFModelLoader


def test_sliding_window_is_restored_only_while_the_weight_map_is_built(monkeypatch):
    seen: list[object] = []
    loader_cls = _install_fake_loader(monkeypatch, seen)

    assert ensure_gguf_gemma3_sliding_window() is True

    config = _FakeConfig()
    # vLLM moves the value aside before any loader runs.
    config.interleaved_sliding_window = 1024
    loader_cls()._get_gguf_weights_map(_FakeModelConfig(config))

    assert seen == [1024], "Transformers' Gemma3Attention reads config.sliding_window"
    assert not hasattr(config, "sliding_window"), "vLLM's own model expects it gone again"


def test_sliding_window_shim_leaves_an_unaffected_config_untouched(monkeypatch):
    seen: list[object] = []
    loader_cls = _install_fake_loader(monkeypatch, seen)
    ensure_gguf_gemma3_sliding_window()

    config = _FakeConfig()
    config.sliding_window = 4096
    loader_cls()._get_gguf_weights_map(_FakeModelConfig(config))

    assert seen == [4096]
    assert config.sliding_window == 4096


def test_sliding_window_shim_is_idempotent(monkeypatch):
    _install_fake_loader(monkeypatch, [])

    assert ensure_gguf_gemma3_sliding_window() is True
    assert ensure_gguf_gemma3_sliding_window() is False


class _FakeQuantConfig:
    def __init__(self, name):
        self._name = name

    def get_name(self):
        return self._name


class _FakeVllmConfig:
    def __init__(self, quant_config):
        self.quant_config = quant_config


def _install_fake_gemma3(monkeypatch, recorder):
    """Stand in for vllm.model_executor.models.gemma3.

    Gemma3Model's real __init__ reads the module-global VocabParallelEmbedding;
    this one records the quant_config that symbol was called with.
    """

    def VocabParallelEmbedding(*args, quant_config=None, **kwargs):
        recorder.append(quant_config)
        return object()

    class Gemma3Model:
        def __init__(self, *, vllm_config, prefix=""):
            self.embed_tokens = module.VocabParallelEmbedding(262208, 5376, prefix=prefix)

    module = types.ModuleType("vllm.model_executor.models.gemma3")
    module.VocabParallelEmbedding = VocabParallelEmbedding
    module.Gemma3Model = Gemma3Model
    models = types.ModuleType("vllm.model_executor.models")
    models.gemma3 = module
    for name in ("vllm", "vllm.model_executor"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models", models)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.models.gemma3", module)
    return module


def test_gguf_embedding_receives_the_quantization_config(monkeypatch):
    seen: list[object] = []
    module = _install_fake_gemma3(monkeypatch, seen)
    quant_config = _FakeQuantConfig("gguf")

    assert ensure_gguf_gemma3_quantized_embedding() is True
    module.Gemma3Model(vllm_config=_FakeVllmConfig(quant_config))

    assert seen == [quant_config], "a GGUF token_embd needs its qweight_type parameter"
    # The binding must not outlive the one model it was installed for.
    assert module.VocabParallelEmbedding.__name__ == "VocabParallelEmbedding"


@pytest.mark.parametrize("quant_config", [None, _FakeQuantConfig("awq")])
def test_a_non_gguf_checkpoint_keeps_the_unquantized_embedding(monkeypatch, quant_config):
    seen: list[object] = []
    module = _install_fake_gemma3(monkeypatch, seen)
    ensure_gguf_gemma3_quantized_embedding()

    module.Gemma3Model(vllm_config=_FakeVllmConfig(quant_config))

    assert seen == [None]


def test_gguf_embedding_shim_is_idempotent(monkeypatch):
    _install_fake_gemma3(monkeypatch, [])

    assert ensure_gguf_gemma3_quantized_embedding() is True
    assert ensure_gguf_gemma3_quantized_embedding() is False


def test_gemma_norm_weights_lose_llama_cpps_baked_in_offset():
    """llama.cpp writes Gemma RMSNorm weights as w+1; vLLM's GemmaRMSNorm
    applies the (1+w) form itself, so the offset has to come back off."""
    torch = pytest.importorskip("torch")
    weights = [
        ("layers.0.input_layernorm.weight", torch.tensor([2.21, 2.96])),
        ("layers.0.self_attn.q_norm.weight", torch.tensor([2.94, -0.62])),
        ("norm.weight", torch.tensor([5.31, 9.5])),
    ]

    corrected = dict(remove_gemma_norm_offset(iter(weights)))

    assert torch.allclose(corrected["layers.0.input_layernorm.weight"], torch.tensor([1.21, 1.96]))
    assert torch.allclose(corrected["layers.0.self_attn.q_norm.weight"], torch.tensor([1.94, -1.62]))
    assert torch.allclose(corrected["norm.weight"], torch.tensor([4.31, 8.5]))


def test_projection_weights_and_quant_markers_are_left_alone():
    torch = pytest.importorskip("torch")
    qweight = torch.tensor([1, 2], dtype=torch.uint8)
    weights = [
        ("layers.0.self_attn.q_proj.qweight", qweight),
        # An integer marker whose name still ends in the norm parameter it
        # describes must never be treated as a norm weight.
        ("layers.0.input_layernorm.qweight_type", torch.tensor(12)),
        ("layers.0.mlp.gate_proj.weight", torch.tensor([1.0, 2.0])),
    ]

    passed_through = dict(remove_gemma_norm_offset(iter(weights)))

    assert torch.equal(passed_through["layers.0.self_attn.q_proj.qweight"], qweight)
    assert torch.equal(passed_through["layers.0.input_layernorm.qweight_type"], torch.tensor(12))
    assert torch.allclose(passed_through["layers.0.mlp.gate_proj.weight"], torch.tensor([1.0, 2.0]))


def test_a_usable_gguf_package_reports_its_version(monkeypatch):
    monkeypatch.setattr("importlib.metadata.version", lambda name: "0.17.1")

    assert require_gguf_runtime() == "0.17.1"


def test_vllms_own_gguf_pin_is_rejected_with_the_install_command(monkeypatch):
    """vLLM pins gguf==0.10.0, which cannot open a GGUF under NumPy 2 at all.
    The override that fixes it cannot live in requirements.txt, so it is easy
    to miss -- say so here rather than deep inside vLLM."""
    monkeypatch.setattr("importlib.metadata.version", lambda name: "0.10.0")

    with pytest.raises(SystemExit) as exc:
        require_gguf_runtime()

    assert "requirements-overrides.txt" in str(exc.value)


def test_a_missing_gguf_package_is_reported_the_same_way(monkeypatch):
    from importlib.metadata import PackageNotFoundError

    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr("importlib.metadata.version", missing)

    with pytest.raises(SystemExit) as exc:
        require_gguf_runtime()

    assert "requirements-overrides.txt" in str(exc.value)


class _FakeField:
    def __init__(self, value):
        self._value = value

    def contents(self):
        return self._value


def _fake_gguf_fields(monkeypatch, fields):
    gguf = pytest.importorskip("gguf")

    class _FakeReader:
        def __init__(self, path, mode="r"):
            self.fields = fields

    monkeypatch.setattr(gguf, "GGUFReader", _FakeReader)


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://huggingface.co/google/medgemma-27b-text-it", "google/medgemma-27b-text-it"),
        ("https://huggingface.co/google/medgemma-27b-text-it/", "google/medgemma-27b-text-it"),
        # An uploader's own repo_url is the organization page, not a model.
        ("https://huggingface.co/unsloth", None),
        ("https://huggingface.co/owner/model/tree/main", None),
        ("https://example.com/owner/model", None),
        ("", None),
    ],
)
def test_the_source_repository_is_read_strictly(monkeypatch, url, expected):
    _fake_gguf_fields(monkeypatch, {"general.base_model.0.repo_url": _FakeField(url)})

    assert gguf_source_repository("/models/x.gguf") == expected


def test_a_gguf_recording_no_base_model_yields_nothing(monkeypatch):
    """The caller then demands an explicit tokenizer rather than guessing."""
    _fake_gguf_fields(monkeypatch, {})

    assert gguf_source_repository("/models/x.gguf") is None


def test_an_unreadable_gguf_yields_nothing(monkeypatch):
    gguf = pytest.importorskip("gguf")

    def explode(path, mode="r"):
        raise OSError("not a GGUF")

    monkeypatch.setattr(gguf, "GGUFReader", explode)

    assert gguf_source_repository("/models/x.gguf") is None
