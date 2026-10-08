"""Model registry parsing and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import os


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    engine: str | None = None
    max_vram_mib: int | None = None
    min_vram_mib: int | None = None
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    format: str
    architecture: str
    parameter_count_b: float
    model_path: str
    model_ref: str | None = None
    tokenizer: str | None = None
    quantization: str | None = None
    weight_dtype: str | None = None
    max_context_tokens: int = 2048
    default_context_tokens: int | None = None
    default_max_output_tokens: int = 256
    layers: int | None = None
    hidden_size: int | None = None
    kv_heads: int | None = None
    head_dim: int | None = None
    kv_cache_dtype_bytes: int = 2
    file_size_bytes: int | None = None
    runtime_profiles: tuple[RuntimeProfile, ...] = ()

    @property
    def local_path(self) -> Path | None:
        path = Path(self.model_path)
        if not path.is_absolute():
            path = ROOT / path
        if path.exists():
            return path
        return None


def load_registry(path: Path) -> dict[str, ModelSpec]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    models = raw.get("models", {})
    registry: dict[str, ModelSpec] = {}
    for model_id, data in models.items():
        file_size = _file_size(data.get("model_path"))
        registry[model_id] = ModelSpec(
            model_id=model_id,
            display_name=data.get("display_name", model_id),
            format=str(data.get("format", "safetensors")).lower(),
            architecture=str(data.get("architecture", "unknown")).lower(),
            parameter_count_b=float(data.get("parameter_count_b", 0)),
            quantization=data.get("quantization"),
            weight_dtype=data.get("weight_dtype"),
            model_path=str(data.get("model_path", "")),
            model_ref=data.get("model_ref"),
            tokenizer=data.get("tokenizer"),
            max_context_tokens=int(data.get("max_context_tokens", 2048)),
            default_context_tokens=_optional_int(data.get("default_context_tokens")),
            default_max_output_tokens=int(data.get("default_max_output_tokens", 256)),
            layers=_optional_int(data.get("layers")),
            hidden_size=_optional_int(data.get("hidden_size")),
            kv_heads=_optional_int(data.get("kv_heads")),
            head_dim=_optional_int(data.get("head_dim")),
            kv_cache_dtype_bytes=int(data.get("kv_cache_dtype_bytes", 2)),
            file_size_bytes=file_size,
            runtime_profiles=_runtime_profiles(data.get("runtime_profiles", [])),
        )
    return registry


def resolve_model(registry: dict[str, ModelSpec], model_id: str) -> ModelSpec:
    if os.environ.get("LLM_MODEL_PATH"):
        model_path = os.environ["LLM_MODEL_PATH"]
        fmt = "gguf" if model_path.lower().endswith(".gguf") else "safetensors"
        return ModelSpec(
            model_id="env_model_path",
            display_name="Model from LLM_MODEL_PATH",
            format=fmt,
            architecture=os.environ.get("LLM_ARCHITECTURE", "unknown").lower(),
            parameter_count_b=float(os.environ.get("LLM_PARAMETER_COUNT_B", "0")),
            quantization=os.environ.get("LLM_QUANTIZATION") or None,
            weight_dtype=os.environ.get("LLM_WEIGHT_DTYPE") or None,
            model_path=model_path,
            model_ref=None,
            tokenizer=os.environ.get("LLM_TOKENIZER") or None,
            max_context_tokens=int(os.environ.get("LLM_MAX_MODEL_LEN", "2048")),
            default_context_tokens=int(os.environ.get("LLM_MAX_MODEL_LEN", "2048")),
            default_max_output_tokens=int(os.environ.get("LLM_MAX_TOKENS", "256")),
            file_size_bytes=_file_size(model_path),
        )

    try:
        return registry[model_id]
    except KeyError as exc:
        known = ", ".join(sorted(registry))
        raise SystemExit(f"unknown LLM_MODEL_ID={model_id!r}; known models: {known}") from exc


def find_profile(
    registry: dict[str, ModelSpec], model_ref: str, requested_profile: str | None
) -> ModelSpec | None:
    """Return an explicitly selected profile or one matching a HF model reference."""
    if requested_profile:
        try:
            profile = registry[requested_profile]
        except KeyError as exc:
            known = ", ".join(sorted(registry)) or "none"
            raise SystemExit(
                f"unknown LLM_MODEL_PROFILE={requested_profile!r}; known profiles: {known}"
            ) from exc
        if profile.model_ref and profile.model_ref != model_ref:
            raise SystemExit(
                f"model profile {requested_profile!r} is for {profile.model_ref!r}, not {model_ref!r}"
            )
        return profile
    return next((profile for profile in registry.values() if profile.model_ref == model_ref), None)


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


def _runtime_profiles(raw_profiles: object) -> tuple[RuntimeProfile, ...]:
    profiles: list[RuntimeProfile] = []
    if not isinstance(raw_profiles, list):
        return ()
    for raw in raw_profiles:
        if not isinstance(raw, dict):
            continue
        env = {str(k): str(v) for k, v in raw.get("env", {}).items()}
        profiles.append(
            RuntimeProfile(
                name=str(raw["name"]),
                engine=str(raw["engine"]) if raw.get("engine") else None,
                max_vram_mib=_optional_int(raw.get("max_vram_mib")),
                min_vram_mib=_optional_int(raw.get("min_vram_mib")),
                env=env,
            )
        )
    return tuple(profiles)


def _file_size(path_value: str | None) -> int | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return None
