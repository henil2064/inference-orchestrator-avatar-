"""Resolve Hugging Face models into the service-owned local model cache."""

from __future__ import annotations

from dataclasses import replace
import errno
import json
import logging
import os
import time
from pathlib import Path

from .compatibility import compute_safe_ram_mib, compute_safe_vram_mib
from .estimator import estimate_llamacpp, estimate_vllm
from .hardware import HardwareSnapshot
from .registry import ModelSpec, find_profile
from adaptive_llm.core.config import Config

log = logging.getLogger(__name__)

# Only network/server-side hiccups are worth retrying. A gated repo, a typo'd
# ref, or a missing file will never succeed on attempt 2 -- retrying those
# just delays a failure the operator could have fixed on the spot.
_RETRY_DELAYS_S = (2, 5, 10)


def resolve_huggingface_model(
    config: Config, registry: dict[str, ModelSpec], hardware: HardwareSnapshot
) -> ModelSpec:
    """Fetch only when needed, then derive a model spec from local model metadata.

    This layer is deliberately engine-neutral. Engines receive a resolved local
    path and the planner remains responsible for deciding whether they can run it.
    ``hardware`` is used only for the pre-download size check below (see
    _reject_if_remote_size_cannot_fit) -- resolution itself stays engine-neutral.
    """
    assert config.model_ref
    profile = find_profile(registry, config.model_ref, config.model_profile)
    cache_dir = config.model_dir / "huggingface" / _safe_repo_name(config.model_ref)
    requested_file = config.model_file

    if requested_file:
        if not requested_file.lower().endswith(".gguf"):
            raise SystemExit("LLM_MODEL_FILE currently selects a GGUF file and must end in .gguf")
        model_path = _ensure_gguf(config, cache_dir, requested_file, hardware)
        inferred = _gguf_spec(config.model_ref, model_path)
    elif (local_profile_path := _profile_artifact_in_model_dir(profile, config.model_dir)) is not None:
        inferred = (
            _gguf_spec(config.model_ref, local_profile_path)
            if local_profile_path.is_file() and local_profile_path.suffix.lower() == ".gguf"
            else _hf_directory_spec(config.model_ref, local_profile_path)
        )
    else:
        existing_ggufs = sorted(cache_dir.glob("*.gguf")) if cache_dir.is_dir() else []
        if len(existing_ggufs) == 1:
            inferred = _gguf_spec(config.model_ref, existing_ggufs[0])
        elif len(existing_ggufs) > 1:
            raise SystemExit(
                "multiple cached GGUF files found; set LLM_MODEL_FILE to select one: "
                + ", ".join(path.name for path in existing_ggufs)
            )
        else:
            _ensure_hf_snapshot(config, cache_dir, hardware)
            downloaded_ggufs = sorted(cache_dir.glob("*.gguf"))
            if len(downloaded_ggufs) == 1:
                inferred = _gguf_spec(config.model_ref, downloaded_ggufs[0])
            elif len(downloaded_ggufs) > 1:
                raise SystemExit(
                    "Hugging Face repository contains multiple GGUF files; set LLM_MODEL_FILE to select one: "
                    + ", ".join(path.name for path in downloaded_ggufs)
                )
            else:
                inferred = _hf_directory_spec(config.model_ref, cache_dir)

    if profile is None:
        return inferred
    # Profiles intentionally override only planning/tuning metadata. The actual
    # path and format always come from the resolved local artifact.
    return replace(
        profile,
        model_path=inferred.model_path,
        model_ref=config.model_ref,
        format=inferred.format,
        file_size_bytes=inferred.file_size_bytes,
        architecture=profile.architecture if profile.architecture != "unknown" else inferred.architecture,
        tokenizer=profile.tokenizer or inferred.tokenizer,
        layers=profile.layers or inferred.layers,
        hidden_size=profile.hidden_size or inferred.hidden_size,
        kv_heads=profile.kv_heads or inferred.kv_heads,
        head_dim=profile.head_dim or inferred.head_dim,
    )


def _ensure_gguf(config: Config, cache_dir: Path, filename: str, hardware: HardwareSnapshot) -> Path:
    target = cache_dir / filename
    if target.is_file():
        return target
    if not config.auto_download:
        raise SystemExit(f"model is not cached at {target}; set LLM_AUTO_DOWNLOAD=1 to fetch it")
    _reject_if_remote_size_cannot_fit(config, hardware, filename=filename)
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)

    def attempt() -> Path:
        return Path(hf_hub_download(
            repo_id=config.model_ref, filename=filename, local_dir=str(cache_dir), token=_hf_token()))

    return _download_with_retry(config.model_ref, f"download {filename!r}", attempt)


def _ensure_hf_snapshot(config: Config, cache_dir: Path, hardware: HardwareSnapshot) -> None:
    if _snapshot_is_complete(cache_dir):
        return
    if not config.auto_download:
        raise SystemExit(f"model is not cached at {cache_dir}; set LLM_AUTO_DOWNLOAD=1 to fetch it")
    _reject_if_remote_size_cannot_fit(config, hardware, filename=None)
    from huggingface_hub import snapshot_download

    def attempt() -> None:
        snapshot_download(repo_id=config.model_ref, local_dir=str(cache_dir), token=_hf_token())

    _download_with_retry(config.model_ref, "download snapshot", attempt)


def _reject_if_remote_size_cannot_fit(
    config: Config, hardware: HardwareSnapshot, *, filename: str | None
) -> None:
    """Query Hugging Face's file-size metadata (no download) and reject
    obviously-too-big models before pulling any weight bytes.

    Without this, LLM_MODEL_REF pointed at a model too big for this GPU/box
    downloaded the *entire* thing -- often many GiB, on a box that may
    already be low on disk -- only to have the planner reject it afterward
    for insufficient VRAM/RAM. This is a coarse, conservative pre-flight
    gate (weights only, no KV/runtime overhead beyond estimate_vllm/
    estimate_llamacpp's own margins, and a config.max_context_tokens-or-2048
    proxy context since the real one isn't known before config.json is
    fetched) -- the accurate, authoritative check still happens in the
    planner once the model is actually resolved. Best-effort: any failure to
    reach the Hugging Face metadata API (network, auth, gated repo) is not
    itself an error here -- the download attempt right after this will
    surface that with its own, already-good error classification.
    """
    if config.dry_run:
        return
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(config.model_ref, files_metadata=True, token=_hf_token())
        siblings = info.siblings or []
    except Exception:
        log.debug("could not pre-check remote size for %r; proceeding to download", config.model_ref, exc_info=True)
        return

    context_tokens = config.max_context_tokens or 2048
    if filename is not None:
        sizes = [s.size for s in siblings if s.rfilename == filename and s.size]
        if not sizes:
            return  # couldn't find/size the specific file; let the real download surface any error
        proxy = ModelSpec(model_id="_size_probe", display_name="", format="gguf", architecture="unknown",
                          parameter_count_b=0, model_path="", file_size_bytes=sizes[0])
        estimate = estimate_llamacpp(proxy, context_tokens)
        safe_mib, budget_kind = compute_safe_ram_mib(hardware, config), "RAM"
    else:
        weight_bytes = sum(s.size for s in siblings if s.rfilename.endswith(".safetensors") and s.size)
        if not weight_bytes:
            return  # metadata without sizes (older hub) or no safetensors listed yet; don't block on a guess
        if not hardware.has_cuda_gpu:
            raise SystemExit(
                f"{config.model_ref} is an HF safetensors model (vLLM/SGLang only, both GPU-only), but no "
                "CUDA GPU was detected on this box -- not downloading it."
            )
        proxy = ModelSpec(model_id="_size_probe", display_name="", format="safetensors", architecture="unknown",
                          parameter_count_b=0, model_path="", file_size_bytes=weight_bytes)
        estimate = estimate_vllm(proxy, context_tokens)
        safe_mib, budget_kind = compute_safe_vram_mib(hardware, config), "VRAM"

    if estimate.total_mib > safe_mib:
        raise SystemExit(
            f"{config.model_ref} weighs an estimated {estimate.weight_mib} MiB on Hugging Face "
            f"(~{estimate.total_mib} MiB with KV cache/runtime overhead), but only {safe_mib} MiB of safe "
            f"free {budget_kind} is available on this box -- not downloading it. Pick a smaller model, free "
            f"up {budget_kind}, or lower LLM_MAX_MODEL_LEN if the shortfall is close."
        )


def _snapshot_is_complete(cache_dir: Path) -> bool:
    """A single ``*.safetensors`` match is not enough evidence for a sharded
    model: an interrupted ``snapshot_download`` can leave some shards missing
    while still satisfying a plain glob check. That stale partial cache would
    then be silently reused on the next start and only fail later, confusingly,
    inside the engine at model-load time -- so check every shard the model's
    own index says it needs."""
    if not (cache_dir / "config.json").is_file():
        return False
    index_path = cache_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return (cache_dir / "model.safetensors").is_file()
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    shard_names = set(index.get("weight_map", {}).values())
    return bool(shard_names) and all((cache_dir / name).is_file() for name in shard_names)


def _download_with_retry(model_ref: str, description: str, attempt):
    """Run one Hugging Face download call, retrying only failures classified
    as transient. A gated/missing/not-found repo fails immediately with an
    actionable reason instead of retrying a request that can never succeed."""
    last_reason = "unknown error"
    last_exc: Exception | None = None
    for delay_s in (0, *_RETRY_DELAYS_S):
        if delay_s:
            log.warning("retrying %s for %r in %ds (after: %s)", description, model_ref, delay_s, last_reason)
            time.sleep(delay_s)
        try:
            return attempt()
        except Exception as exc:
            reason, retryable = _classify_hf_error(exc)
            last_reason = reason
            if not retryable:
                raise SystemExit(_hf_failure_message(model_ref, description, reason)) from exc
            last_exc = exc
    raise SystemExit(_hf_failure_message(model_ref, description, last_reason)) from last_exc


def _hf_directory_spec(model_ref: str, path: Path) -> ModelSpec:
    config_path = path / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"Hugging Face model cache is missing config.json: {path}")
    if not list(path.rglob("*.safetensors")):
        raise SystemExit(
            f"Hugging Face model at {path} has no safetensors weights; this runtime currently supports "
            "HF safetensors directories or explicitly selected GGUF files"
        )
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid Hugging Face config.json in {path}: {exc}") from exc
    hidden_size = _as_int(raw.get("hidden_size") or raw.get("n_embd"))
    heads = _as_int(raw.get("num_attention_heads") or raw.get("n_head"))
    return ModelSpec(
        model_id=model_ref,
        display_name=model_ref,
        format="safetensors",
        architecture=str(raw.get("model_type", "unknown")).lower(),
        parameter_count_b=0,
        model_path=str(path),
        model_ref=model_ref,
        file_size_bytes=_path_size(path),
        weight_dtype=str(raw.get("torch_dtype", "")).lower() or None,
        max_context_tokens=_context_limit(raw),
        default_context_tokens=min(_context_limit(raw), 2048),
        layers=_as_int(raw.get("num_hidden_layers") or raw.get("num_layers") or raw.get("n_layer")),
        hidden_size=hidden_size,
        kv_heads=_as_int(raw.get("num_key_value_heads") or raw.get("n_head_kv")) or heads,
        head_dim=_as_int(raw.get("head_dim")) or (hidden_size // heads if hidden_size and heads else None),
    )


def _gguf_spec(model_ref: str, path: Path) -> ModelSpec:
    metadata = _gguf_metadata(path)
    context_limit = metadata.get("context_length") or 2048
    return ModelSpec(
        model_id=model_ref,
        display_name=f"{model_ref} ({path.name})",
        format="gguf",
        architecture=str(metadata.get("architecture") or "unknown").lower(),
        parameter_count_b=0,
        model_path=str(path),
        model_ref=model_ref,
        file_size_bytes=_path_size(path),
        quantization=_quantization_from_name(path.name),
        max_context_tokens=context_limit,
        default_context_tokens=min(context_limit, 2048),
        layers=metadata.get("layers"),
        hidden_size=metadata.get("hidden_size"),
        kv_heads=metadata.get("kv_heads"),
        head_dim=metadata.get("head_dim"),
    )


def _gguf_metadata(path: Path) -> dict[str, int | str]:
    """Read the shape and context length a GGUF file records about itself.

    A GGUF carries no config.json, so without this every GGUF model reached the
    planner as "unknown" architecture with no layer/head counts and a flat
    2048-token context. Two things went wrong as a result: the KV-cache
    estimate fell back to a crude per-token guess, and the served context was
    silently clamped to 2048 no matter what the checkpoint (or
    LLM_MAX_MODEL_LEN) allowed, because the planner takes the minimum of the
    request and the model's stated limit.

    Best-effort by design: a header this cannot parse leaves the caller with
    its previous defaults instead of failing a service that would otherwise
    start. The engine reads the same file itself and reports any real problem
    with it at load time.
    """
    try:
        from gguf import GGUFReader

        reader = GGUFReader(str(path), "r")
    except Exception:  # noqa: BLE001 - metadata is an optimization, never a gate
        log.debug("could not open %s for GGUF metadata", path, exc_info=True)
        return {}

    def field(key: str) -> object | None:
        entry = reader.fields.get(key)
        if entry is None:
            return None
        try:
            return entry.contents()
        except Exception:  # noqa: BLE001 - an unexpected field layout is not fatal
            return None

    architecture = field("general.architecture")
    if not isinstance(architecture, str) or not architecture:
        return {}

    values: dict[str, int | str] = {"architecture": architecture}
    for name, key in (
        ("context_length", f"{architecture}.context_length"),
        ("layers", f"{architecture}.block_count"),
        ("hidden_size", f"{architecture}.embedding_length"),
        ("kv_heads", f"{architecture}.attention.head_count_kv"),
        ("head_dim", f"{architecture}.attention.key_length"),
    ):
        number = _as_int(field(key))
        if number and number > 0:
            values[name] = number
    if "head_dim" not in values:
        heads = _as_int(field(f"{architecture}.attention.head_count"))
        if heads and values.get("hidden_size"):
            values["head_dim"] = int(values["hidden_size"]) // heads
    return values


def _safe_repo_name(model_ref: str) -> str:
    return model_ref.replace("/", "--").replace("\\", "--")


def _profile_artifact_in_model_dir(profile: ModelSpec | None, model_dir: Path) -> Path | None:
    """Reuse an older profile-managed artifact, but never reach outside the cache."""
    if profile is None:
        return None
    path = profile.local_path
    if path is None:
        return None
    try:
        path.resolve().relative_to(model_dir.resolve())
    except ValueError:
        return None
    return path


def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _classify_hf_error(exc: Exception) -> tuple[str, bool]:
    """Return (actionable reason, retryable) for a Hugging Face download failure."""
    from huggingface_hub.utils import (
        EntryNotFoundError, GatedRepoError, HfHubHTTPError,
        LocalEntryNotFoundError, RepositoryNotFoundError, RevisionNotFoundError,
    )

    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return "no space left on the model cache device", False
    # Ordered most- to least-specific: several of these subclass HfHubHTTPError,
    # so a generic status-code check alone would blur them together.
    if isinstance(exc, GatedRepoError):
        return ("this repository is gated -- accept its license at huggingface.co and set "
                "HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) to a token with access to it"), False
    if isinstance(exc, RepositoryNotFoundError):
        return "no such Hugging Face repository -- check the model ref for typos", False
    if isinstance(exc, RevisionNotFoundError):
        return "the requested revision/branch does not exist in this repository", False
    if isinstance(exc, EntryNotFoundError):
        return "the requested file does not exist in this repository", False
    if isinstance(exc, LocalEntryNotFoundError):
        return ("could not reach Hugging Face and no local copy is cached -- "
                "check network access, or pre-populate the cache offline"), True
    if isinstance(exc, HfHubHTTPError):
        status = getattr(exc.response, "status_code", None)
        if status in (401, 403):
            return ("access denied -- this repository may be gated or private; accept its "
                     "license and/or set HF_TOKEN to a token with access to it"), False
        if status == 429:
            return "Hugging Face rate-limited this download", True
        if isinstance(status, int) and status >= 500:
            return f"Hugging Face returned a server error (HTTP {status})", True
        return f"Hugging Face request failed (HTTP {status})", False
    try:
        import httpx
        if isinstance(exc, httpx.TransportError):
            return f"network error reaching Hugging Face ({exc})", True
    except ImportError:
        pass
    return str(exc), False


def _hf_failure_message(model_ref: str, description: str, reason: str) -> str:
    token = _hf_token()
    message = reason.replace(token, "[redacted]") if token else reason
    return f"could not {description} for Hugging Face model {model_ref!r}: {message}"


def _as_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _context_limit(raw: dict) -> int:
    values = [_as_int(raw.get(key)) for key in ("max_position_embeddings", "n_positions", "max_sequence_length")]
    return max(value for value in values if value is not None) if any(values) else 2048


def _quantization_from_name(name: str) -> str | None:
    stem = Path(name).stem.upper()
    return next((item for item in ("Q2", "Q3", "Q4", "Q5", "Q6", "Q8") if item in stem), None)


def _path_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
