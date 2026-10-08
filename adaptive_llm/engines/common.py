"""Shared engine-worker helpers."""

from __future__ import annotations

import importlib
import os
import re
import socket
import time
from typing import Mapping, MutableMapping


def load_dotenv() -> None:
    path = os.environ.get("LLM_ENV_FILE")
    if not path or not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            # ``#`` is valid in values (including Hugging Face tokens).  Treat
            # it as an inline comment only when separated from an unquoted
            # value by whitespace; whole-line comments are ignored below.
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip()
            if value[:1] in {"'", '"'} and value[-1:] == value[:1]:
                value = value[1:-1]
            else:
                value = re.split(r"\s+#", value, maxsplit=1)[0].rstrip()
            os.environ.setdefault(key.strip(), value)


def now_ms() -> int:
    return int(time.time() * 1000)


def telemetry_attributes(*, gpu_index: int | str | None = None) -> dict[str, str]:
    """Stable worker identity sent with every inference profile.

    Values are deliberately configured, rather than inferred from a visible
    CUDA ordinal: containers can remap CUDA_VISIBLE_DEVICES and make ordinal 0
    ambiguous. Set TELEMETRY_GPU_UUID/TELEMETRY_GPU_INDEX at deployment time.
    """
    visible_device = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0].strip()
    configured_gpu = os.environ.get("TELEMETRY_GPU_INDEX") or gpu_index or visible_device
    configured_gpu = str(configured_gpu) if configured_gpu not in (None, "") else None
    # CUDA_VISIBLE_DEVICES may carry a UUID instead of an ordinal. Preserve it
    # as an identity and never try to store it in the numeric gpu_index column.
    visible_uuid = configured_gpu if configured_gpu and configured_gpu.startswith("GPU-") else None
    values = {
        "deployment_id": os.environ.get("TELEMETRY_DEPLOYMENT_ID"),
        "host_id": os.environ.get("TELEMETRY_HOST_ID") or socket.gethostname(),
        "service_instance_id": os.environ.get("TELEMETRY_SERVICE_INSTANCE_ID"),
        "model_revision": os.environ.get("LLM_MODEL_REVISION"),
        "gpu_uuid": os.environ.get("TELEMETRY_GPU_UUID") or visible_uuid,
        "gpu_index": None if visible_uuid else configured_gpu,
    }
    return {key: value for key, value in values.items() if value not in (None, "")}


def _optional_int(env: Mapping[str, str], key: str) -> int | None:
    """Read a positive integer vLLM option, leaving an empty value unset."""
    raw = env.get(key)
    if raw is None or not raw.strip():
        return None
    value = int(raw)
    if value < 1:
        raise ValueError(f"{key} must be at least 1, got {raw!r}")
    return value


def _optional_float(env: Mapping[str, str], key: str) -> float | None:
    """Read a non-negative floating-point vLLM option, leaving it unset if empty."""
    raw = env.get(key)
    if raw is None or not raw.strip():
        return None
    value = float(raw)
    if value < 0:
        raise ValueError(f"{key} must be non-negative, got {raw!r}")
    return value


def _env_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be a boolean (0/1, true/false), got {raw!r}")


def engine_kwargs_from_env(env: Mapping[str, str]) -> dict[str, object]:
    """Map documented ``LLM_*`` environment variables to ``AsyncEngineArgs``.

    Model identity and text-only mode stay in the worker because they are part
    of the selected execution plan. An unset optional parameter is omitted so
    the installed vLLM version retains ownership of its default.
    """
    gpu_memory_utilization = float(env.get("LLM_GPU_MEMORY_UTILIZATION", "0.85"))
    if not 0 < gpu_memory_utilization <= 1:
        raise ValueError(
            "LLM_GPU_MEMORY_UTILIZATION must be greater than 0 and at most 1"
        )

    kwargs: dict[str, object] = {
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": _optional_int(env, "LLM_MAX_MODEL_LEN"),
        "max_num_batched_tokens": _optional_int(env, "LLM_MAX_NUM_BATCHED_TOKENS"),
        "max_num_seqs": _optional_int(env, "LLM_MAX_NUM_SEQS"),
        "enforce_eager": _env_bool(env, "LLM_ENFORCE_EAGER", False),
        "enable_prefix_caching": _env_bool(env, "LLM_ENABLE_PREFIX_CACHING", True),
        "disable_log_stats": _env_bool(env, "LLM_DISABLE_LOG_STATS", False),
        "dtype": env.get("LLM_DTYPE", "auto") or "auto",
        "quantization": env.get("LLM_QUANTIZATION") or None,
        "kv_cache_dtype": env.get("LLM_KV_CACHE_DTYPE", "auto") or "auto",
        "load_format": env.get("LLM_LOAD_FORMAT", "auto") or "auto",
    }
    chunked_prefill = env.get("LLM_ENABLE_CHUNKED_PREFILL")
    if chunked_prefill is not None and chunked_prefill.strip():
        kwargs["enable_chunked_prefill"] = _env_bool(env, "LLM_ENABLE_CHUNKED_PREFILL", True)
    swap_space = _optional_float(env, "LLM_SWAP_SPACE_GIB")
    if swap_space is not None:
        kwargs["swap_space"] = swap_space
    cpu_offload = _optional_float(env, "LLM_CPU_OFFLOAD_GB")
    if cpu_offload is not None:
        kwargs["cpu_offload_gb"] = cpu_offload
    return kwargs


def ensure_v1_default(env: MutableMapping[str, str]) -> None:
    """Prefer vLLM V1 unless an operator explicitly selected another engine."""
    env.setdefault("VLLM_USE_V1", "1")


def cuda_compute_capability() -> tuple[int, int] | None:
    """Return the active CUDA device capability, or ``None`` if unavailable.

    This must be usable before vLLM is imported, so it deliberately touches
    only torch.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_capability()
    except (ImportError, RuntimeError):
        # Let vLLM report its normal platform error when CUDA cannot be
        # queried. These compatibility shims must not mask it.
        return None


def ensure_v0_for_gguf(env: MutableMapping[str, str], model_path: str | None) -> bool:
    """Keep a GGUF checkpoint on the vLLM V0 engine.

    vLLM 0.8.x lists ``gguf`` in ``V1_UNSUPPORTED_QUANT``. When V1 was merely
    inferred it downgrades itself with a log line, but when ``VLLM_USE_V1=1``
    is *set* it raises instead -- and ensure_v1_default() sets exactly that on
    any Ampere-or-newer GPU, so an unconfigured A100 would turn a supported
    GGUF model into a hard startup failure. An operator who set the variable
    by hand keeps their choice, including the resulting error.
    """
    if not model_path or not model_path.lower().endswith(".gguf"):
        return False
    if "VLLM_USE_V1" in env:
        return False
    env["VLLM_USE_V1"] = "0"
    return True


# vLLM 0.8.3 pins gguf==0.10.0. That version cannot open a GGUF at all under
# NumPy 2 (its GGUFReader calls ndarray.newbyteorder, removed in NumPy 2) and
# predates Gemma 3, so requirements-overrides.txt installs a newer one.
GGUF_MIN_VERSION = "0.17.1"


def gguf_source_repository(model_path: str) -> str | None:
    """Return the Hugging Face repo a GGUF says it was converted from.

    Conversion tools record the original model as
    ``general.base_model.0.repo_url``. That repo is exactly what a GGUF served
    by vLLM still needs -- the tokenizer and chat template, and the config
    fields the conversion drops -- so reading it here means swapping to another
    GGUF of a known architecture does not also require hand-writing a profile
    entry just to name a tokenizer.

    Deliberately strict: only an ``https://huggingface.co/<owner>/<name>`` URL
    is accepted. ``general.repo_url`` is *not* consulted, because uploaders set
    it to the organization page (``https://huggingface.co/unsloth``), which
    would silently resolve to the wrong thing. Returns None when the file
    records nothing usable, leaving the caller to demand an explicit tokenizer.
    """
    from gguf import GGUFReader

    try:
        field = GGUFReader(model_path, "r").fields.get("general.base_model.0.repo_url")
        url = field.contents() if field is not None else None
    except Exception:  # noqa: BLE001 - the engine reports a real file problem itself
        return None
    if not isinstance(url, str):
        return None
    prefix = "https://huggingface.co/"
    if not url.startswith(prefix):
        return None
    parts = url[len(prefix):].strip("/").split("/")
    return "/".join(parts) if len(parts) == 2 and all(parts) else None


def require_gguf_runtime() -> str:
    """Fail early and clearly when the installed ``gguf`` cannot serve a GGUF.

    The override in requirements-overrides.txt is easy to miss: it cannot live
    in requirements.txt (pip refuses to resolve it against vLLM's pin), so an
    environment built by a plain ``pip install -r requirements.txt`` keeps the
    unusable 0.10.0. Without this check that surfaces much later, deep inside
    vLLM, as an AttributeError about NumPy byte order that says nothing about
    the real cause. Returns the installed version.
    """
    from importlib.metadata import PackageNotFoundError, version

    from packaging.version import Version

    remedy = (
        "install it with: adaptive_llm/.venv/bin/python -m pip install --no-deps "
        "--upgrade -r adaptive_llm/requirements-overrides.txt"
    )
    try:
        installed = version("gguf")
    except PackageNotFoundError:
        raise SystemExit(f"serving a GGUF model needs the `gguf` package; {remedy}") from None
    if Version(installed) < Version(GGUF_MIN_VERSION):
        raise SystemExit(
            f"`gguf` {installed} cannot serve a GGUF model here: it needs to be at least "
            f"{GGUF_MIN_VERSION} to read a GGUF under NumPy 2 and to know the Gemma 3 "
            f"architecture. vLLM pins the older version, so {remedy}"
        )
    return installed


def ensure_gguf_gemma3_arch_alias() -> bool:
    """Let vLLM's GGUF loader find Gemma 3's tensor-name map.

    A Gemma 3 GGUF holds only the language model (the vision tower ships as a
    separate mmproj file), so Transformers reports its config as model_type
    ``gemma3_text``. vLLM's GGUF loader then searches ``gguf.MODEL_ARCH_NAMES``
    for that exact string to pick the architecture whose tensor names it must
    translate -- but the gguf package spells that architecture ``gemma3``, and
    the lookup dies with "Unknown gguf model_type: gemma3_text".

    Renaming the entry is safe because only this lookup reads the value: every
    tensor map is keyed by the ``MODEL_ARCH`` member itself, and a GGUF file's
    own architecture string is read from its header, not from this table.
    Returns whether the alias was installed.
    """
    import gguf

    architecture = getattr(gguf.MODEL_ARCH, "GEMMA3", None)
    if architecture is None:
        # Too old a gguf package to know Gemma 3 at all; vLLM will report the
        # unknown architecture itself, which is the clearer error here.
        return False
    names = gguf.MODEL_ARCH_NAMES
    if "gemma3_text" in names.values():
        return False
    names[architecture] = "gemma3_text"
    return True


def ensure_gguf_gemma3_sliding_window() -> bool:
    """Restore the Gemma 3 config field vLLM hides, for the one call that needs it.

    vLLM treats Gemma 3's interleaved attention as full attention everywhere
    except the attention layer itself, and marks that by moving the config's
    ``sliding_window`` to ``interleaved_sliding_window`` -- ``delattr``-ing the
    original. Every other loader hands that config only to vLLM's own model
    classes, which expect the rename.

    The GGUF loader is the exception. It has no HF checkpoint to enumerate, so
    it instantiates a *Transformers* ``Gemma3ForCausalLM`` on the meta device
    purely to read back parameter names, and that constructor reads
    ``config.sliding_window`` -- which by then no longer exists, so loading
    dies with an AttributeError before a single weight is read.

    Put the value back for the duration of that one call and take it away
    again immediately, so vLLM's own model still sees the config it expects.
    Must be called after vLLM is imported. Returns whether the patch was
    installed (it is idempotent).
    """
    from vllm.model_executor.model_loader.loader import GGUFModelLoader

    original = GGUFModelLoader._get_gguf_weights_map
    if getattr(original, "_adaptive_llm_restores_sliding_window", False):
        return False

    def _get_gguf_weights_map(self, model_config):
        configs = [model_config.hf_config, getattr(model_config, "hf_text_config", None)]
        restored = []
        for config in configs:
            if config is None or hasattr(config, "sliding_window"):
                continue
            window = getattr(config, "interleaved_sliding_window", None)
            if window is not None:
                config.sliding_window = window
                restored.append(config)
        try:
            return original(self, model_config)
        finally:
            for config in restored:
                delattr(config, "sliding_window")

    _get_gguf_weights_map._adaptive_llm_restores_sliding_window = True
    GGUFModelLoader._get_gguf_weights_map = _get_gguf_weights_map
    return True


def ensure_gguf_gemma3_quantized_embedding() -> bool:
    """Give Gemma 3's token embedding the GGUF quantization method it needs.

    Every other layer vLLM builds for Gemma 3 receives the engine's
    ``quant_config``, but vLLM 0.8.3 constructs the model's
    ``VocabParallelEmbedding`` without one -- unlike its Llama implementation,
    which passes it. That is invisible for an unquantized checkpoint and fatal
    for a GGUF one: a Q4_K_M export stores ``token_embd`` quantized (Q6_K
    here), so vLLM's GGUF weight iterator emits an ``embed_tokens.qweight_type``
    entry alongside the weight, the plain embedding has no such parameter, and
    loading dies with ``KeyError: 'embed_tokens.qweight_type'``.

    Bind the active config to the embedding class for the duration of the
    model's construction, which is the same thing passing the argument would
    do, and leaves every other checkpoint's behavior untouched. Gemma 3 ties
    its output projection to this embedding, so this also covers the logits
    path -- GGUFEmbeddingMethod extends GGUFLinearMethod and implements both.

    Must be called after vLLM is imported. Idempotent; returns whether the
    patch was installed.
    """
    import functools

    from vllm.model_executor.models import gemma3

    original_init = gemma3.Gemma3Model.__init__
    if getattr(original_init, "_adaptive_llm_quantizes_embedding", False):
        return False

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        vllm_config = kwargs.get("vllm_config")
        quant_config = getattr(vllm_config, "quant_config", None)
        if quant_config is None or quant_config.get_name() != "gguf":
            return original_init(self, *args, **kwargs)
        embedding_cls = gemma3.VocabParallelEmbedding
        gemma3.VocabParallelEmbedding = functools.partial(
            embedding_cls, quant_config=quant_config
        )
        try:
            return original_init(self, *args, **kwargs)
        finally:
            gemma3.VocabParallelEmbedding = embedding_cls

    __init__._adaptive_llm_quantizes_embedding = True
    gemma3.Gemma3Model.__init__ = __init__
    return True


# llama.cpp's HF->GGUF conversion adds 1 to every Gemma RMSNorm weight, for
# every Gemma generation, because its own kernel computes ``x * w`` while
# Gemma's definition is ``x * (1 + w)``.
GEMMA_GGUF_ARCHITECTURES = frozenset({"gemma", "gemma2", "gemma3"})


def gguf_architecture(model_path: str) -> str | None:
    """Return a GGUF file's own ``general.architecture``, or None if unreadable."""
    try:
        from gguf import GGUFReader

        field = GGUFReader(model_path, "r").fields.get("general.architecture")
    except Exception:  # noqa: BLE001 - the engine reports a real file problem itself
        return None
    if field is None:
        return None
    try:
        value = field.contents()
    except Exception:  # noqa: BLE001 - an unexpected field layout is not fatal here
        return None
    return value if isinstance(value, str) else None


def ensure_gguf_gemma_norm_offset() -> bool:
    """Take back the +1 llama.cpp bakes into a Gemma GGUF's RMSNorm weights.

    Transformers undoes this when it reads a GGUF (see its
    ``Gemma2TensorProcessor``). vLLM's GGUF loader streams tensors verbatim and
    has no per-architecture equivalent, while its ``GemmaRMSNorm`` applies the
    ``(1 + w)`` form -- so every norm ends up scaled by ``(2 + w_original)``.

    This is the failure mode worth naming: nothing errors. The model loads, the
    engine reports itself healthy, and generation returns fluent-looking
    garbage that runs to the token limit because it never produces an
    end-of-turn token. Verified against the bf16 checkpoint: each norm weight
    in this Q4_K_M export is exactly its safetensors value plus one.

    Only Gemma checkpoints are touched, decided from the GGUF's own recorded
    architecture. Must be called after vLLM is imported; returns whether the
    patch was installed.
    """
    from vllm.model_executor.model_loader.loader import GGUFModelLoader

    original = GGUFModelLoader._get_weights_iterator
    if getattr(original, "_adaptive_llm_removes_gemma_norm_offset", False):
        return False

    def _get_weights_iterator(self, model_name_or_path, gguf_to_hf_name_map):
        weights = original(self, model_name_or_path, gguf_to_hf_name_map)
        if gguf_architecture(model_name_or_path) not in GEMMA_GGUF_ARCHITECTURES:
            return weights
        return remove_gemma_norm_offset(weights)

    _get_weights_iterator._adaptive_llm_removes_gemma_norm_offset = True
    GGUFModelLoader._get_weights_iterator = _get_weights_iterator
    return True


def remove_gemma_norm_offset(weights):
    """Subtract llama.cpp's baked-in 1 from each Gemma RMSNorm weight."""
    for name, weight in weights:
        # A GGUF stores norms unquantized, so vLLM's iterator leaves them named
        # ``.weight``; a quantized tensor arrives as ``.qweight`` with a
        # separate integer ``.qweight_type`` marker. Check the dtype as well,
        # so an unexpected layout can never turn into silent corruption.
        if name.endswith("norm.weight") and weight.is_floating_point():
            weight = weight - 1.0
        yield name, weight


# Fields a GGUF header cannot express, or that llama.cpp writes but
# Transformers' GGUF reader does not map back. Left at the Transformers default
# they are silently wrong rather than missing, which is the dangerous case:
# MedGemma/Gemma 3 27B scales queries by 168**-0.5 while the default is
# 256**-0.5, and its global-attention layers expect linear RoPE scaling by 8.
# Neither mistake raises anything -- they just quietly make a worse model.
GGUF_LOSSY_CONFIG_FIELDS = (
    "query_pre_attn_scalar",
    "rope_scaling",
    "rope_local_base_freq",
    "attn_logit_softcapping",
    "final_logit_softcapping",
)


def gguf_config_overrides(source_ref: str) -> dict[str, object]:
    """Recover config fields from the repository a GGUF was converted from.

    ``source_ref`` is the same Hugging Face repository the tokenizer comes
    from, which a GGUF model must name anyway because the file carries no chat
    template. Only the fields above are taken: everything else either survives
    the conversion intact or is the engine's business.

    Failures are raised, not swallowed. This runs at startup, the tokenizer is
    fetched from the same repository moments later, and the alternative to a
    clear error is a model that serves slightly wrong attention forever.
    """
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(source_ref)
    text_config = getattr(config, "text_config", config)
    return {
        field: getattr(text_config, field)
        for field in GGUF_LOSSY_CONFIG_FIELDS
        if getattr(text_config, field, None) is not None
    }


def ensure_v1_gpu_compatibility(env: MutableMapping[str, str]) -> tuple[int, int] | None:
    """Fall back to V0 when vLLM V1 cannot run on the active CUDA GPU.

    vLLM 0.8.x requires compute capability 8.0 (Ampere) or newer for V1.
    The check intentionally runs before importing vLLM, because it reads this
    environment variable during its startup configuration.
    """
    ensure_v1_default(env)
    if env.get("VLLM_USE_V1") != "1":
        return None

    capability = cuda_compute_capability()
    if capability is not None and capability[0] < 8:
        env["VLLM_USE_V1"] = "0"
        return capability
    return None


# vLLM's Triton prefix-prefill kernel (``vllm/attention/ops/prefix_prefill.py``)
# stages ``BLOCK_M x head_size`` query/key/value tiles in shared memory. At
# head_size 256 -- Gemma 3 and MedGemma -- it requests 72 KiB even with the
# reduced pre-Ampere block size, while a pre-Ampere SM offers only 64 KiB per
# block, so the launch dies with ``triton.runtime.errors.OutOfResources``.
# head_size 128 (Qwen3) asks for 36 KiB and is fine.
TRITON_PREFIX_PREFILL_MAX_HEAD_SIZE_PRE_AMPERE = 128


def apply_triton_prefix_prefill_limit(
    kwargs: MutableMapping[str, object],
    head_size: int | None,
    capability: tuple[int, int] | None,
) -> bool:
    """Avoid the vLLM code paths that reach the Triton prefix-prefill kernel.

    Prefix caching and chunked prefill both give a prompt a non-empty block
    table, which routes it through that kernel instead of the xFormers prefill
    path. Returns whether the requested plan had to be changed.
    """
    if head_size is None or capability is None or capability[0] >= 8:
        return False
    if head_size <= TRITON_PREFIX_PREFILL_MAX_HEAD_SIZE_PRE_AMPERE:
        return False
    if not (kwargs.get("enable_prefix_caching") or kwargs.get("enable_chunked_prefill")):
        return False

    kwargs["enable_prefix_caching"] = False
    kwargs["enable_chunked_prefill"] = False
    # Without chunked prefill vLLM requires a whole prompt to fit into one
    # scheduler batch, so an explicit smaller batch budget is now invalid.
    max_model_len = kwargs.get("max_model_len")
    batched_tokens = kwargs.get("max_num_batched_tokens")
    if isinstance(max_model_len, int):
        if not isinstance(batched_tokens, int) or batched_tokens < max_model_len:
            kwargs["max_num_batched_tokens"] = max_model_len
    elif batched_tokens is not None:
        # The context length comes from the checkpoint, so let vLLM derive a
        # consistent batch budget rather than guessing one here.
        kwargs["max_num_batched_tokens"] = None
    return True


def disable_float16_safety_blocklist() -> bool:
    """Disable vLLM's optional float16 guard across supported module layouts.

    Returns whether a guard was found and cleared.  vLLM 0.8.3 does not have
    this guard, so it is intentionally a no-op on that release.
    """
    for module_name in ("vllm.config.model", "vllm.config"):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            # Do not hide a missing dependency while importing a module that
            # otherwise exists; only the alternate vLLM layout is optional.
            if exc.name != module_name:
                raise
            continue
        blocklist = getattr(module, "_FLOAT16_NOT_SUPPORTED_MODELS", None)
        if blocklist is not None:
            blocklist.clear()
            return True
    return False


def ensure_gemma3_sliding_window_pattern() -> None:
    """Bridge the Gemma 3 config field expected by older vLLM releases.

    Transformers 4.57 stores the value privately as
    ``_sliding_window_pattern``.  Some model configs (including Gemma 3 1B)
    still provide the public legacy key while others (including newer
    MedGemma exports) provide the private key.  The alias must therefore be
    writable while Transformers constructs a config from either form.
    """
    from transformers import Gemma3TextConfig

    if not hasattr(Gemma3TextConfig, "sliding_window_pattern"):
        Gemma3TextConfig.sliding_window_pattern = property(  # type: ignore[attr-defined]
            lambda config: config._sliding_window_pattern,
            lambda config, value: setattr(config, "_sliding_window_pattern", value),
        )


def attention_head_size(model_path: str) -> int | None:
    """Read the attention head size from a checkpoint's Transformers config.

    Returns ``None`` when the value cannot be determined -- so callers keep
    vLLM's own behavior instead of acting on a guess.
    """
    try:
        from transformers import AutoConfig

        if model_path.lower().endswith(".gguf"):
            # A single GGUF file is not an HF config directory: passing it
            # straight to from_pretrained() always raises, which used to make
            # this function return None for every GGUF model -- silently
            # disabling apply_triton_prefix_prefill_limit() below for exactly
            # the checkpoints (head_size 256 Gemma 3 / MedGemma GGUFs) its
            # pre-Ampere safety check exists for. Transformers can read a
            # GGUF's own metadata directly via gguf_file=.
            directory, filename = os.path.split(model_path)
            config = AutoConfig.from_pretrained(directory or ".", gguf_file=filename)
        else:
            config = AutoConfig.from_pretrained(model_path)
    except Exception:  # noqa: BLE001 - vLLM reports the real load error later
        return None
    text_config = getattr(config, "text_config", config)
    head_dim = getattr(text_config, "head_dim", None)
    if head_dim:
        return int(head_dim)
    hidden_size = getattr(text_config, "hidden_size", None)
    num_heads = getattr(text_config, "num_attention_heads", None)
    if hidden_size and num_heads:
        return int(hidden_size) // int(num_heads)
    return None
