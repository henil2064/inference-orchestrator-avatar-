"""Configuration loading for the adaptive LLM facade."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Config:
    model_id: str
    model_profiles_path: Path
    model_ref: str | None
    model_file: str | None
    model_dir: Path
    model_profile: str | None
    auto_download: bool
    public_addr: str
    worker_addr: str
    metrics_port: int
    engine_override: str | None
    dry_run: bool
    auto_start_worker: bool
    worker_ready_timeout_s: float
    vram_headroom_fraction: float
    ram_headroom_fraction: float
    min_headroom_mib: int
    max_context_tokens: int | None
    max_output_tokens: int | None
    worker_python: str
    max_inflight_requests: int
    max_queued_requests: int
    resource_metrics_interval_s: float
    profile_bandwidth: bool


def load_config() -> Config:
    # ADAPTIVE_LLM_REGISTRY is retained as a short-lived compatibility alias.
    profiles_value = os.environ.get(
        "ADAPTIVE_LLM_MODEL_PROFILES",
        os.environ.get("ADAPTIVE_LLM_REGISTRY", str(ROOT / "adaptive_llm" / "model-profiles.json")),
    )
    model_profiles_path = Path(profiles_value)
    if not model_profiles_path.is_absolute():
        model_profiles_path = ROOT / model_profiles_path

    model_dir = Path(os.environ.get("LLM_MODEL_DIR", ROOT / "adaptive_llm" / "models"))
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir

    return Config(
        model_id=os.environ.get("LLM_MODEL_ID", "medgemma_1_5_4b_q4"),
        model_profiles_path=model_profiles_path,
        model_ref=os.environ.get("LLM_MODEL_REF") or None,
        model_file=os.environ.get("LLM_MODEL_FILE") or None,
        model_dir=model_dir,
        model_profile=os.environ.get("LLM_MODEL_PROFILE") or None,
        auto_download=_env_bool("LLM_AUTO_DOWNLOAD", True),
        public_addr=os.environ.get("LLM_ADDR", "[::]:50053"),
        # Phase 3 fix: 50054 collides with TTS's default gRPC port
        # (TTS_ADDR), so running TTS alongside the local worker on the same
        # box (the normal case) put two servers on the same port -- whichever
        # bound second failed silently from the operator's perspective. 50055
        # is outside both the STT block (50052 + 100*i) and TTS block
        # (50054 + 100*i) for every replica index, so it never collides.
        worker_addr=os.environ.get("ADAPTIVE_LLM_WORKER_ADDR", "127.0.0.1:50055"),
        metrics_port=int(os.environ.get("LLM_METRICS_PORT", "9106")),
        engine_override=os.environ.get("LLM_ENGINE") or None,
        dry_run=_env_bool("ADAPTIVE_LLM_DRY_RUN", False),
        auto_start_worker=_env_bool("ADAPTIVE_LLM_AUTO_START_WORKER", True),
        worker_ready_timeout_s=float(os.environ.get("ADAPTIVE_LLM_WORKER_READY_TIMEOUT_S", "900")),
        vram_headroom_fraction=float(os.environ.get("ADAPTIVE_LLM_VRAM_HEADROOM", "0.15")),
        ram_headroom_fraction=float(os.environ.get("ADAPTIVE_LLM_RAM_HEADROOM", "0.10")),
        min_headroom_mib=int(os.environ.get("ADAPTIVE_LLM_MIN_HEADROOM_MIB", "512")),
        max_context_tokens=(
            int(os.environ["LLM_MAX_MODEL_LEN"]) if os.environ.get("LLM_MAX_MODEL_LEN") else None
        ),
        max_output_tokens=(
            int(os.environ["LLM_MAX_TOKENS"]) if os.environ.get("LLM_MAX_TOKENS") else None
        ),
        worker_python=os.environ.get("ADAPTIVE_LLM_PYTHON", sys.executable),
        max_inflight_requests=max(1, int(os.environ.get("ADAPTIVE_LLM_MAX_INFLIGHT_REQUESTS", "4"))),
        max_queued_requests=max(0, int(os.environ.get("ADAPTIVE_LLM_MAX_QUEUED_REQUESTS", "16"))),
        resource_metrics_interval_s=max(
            0.5, float(os.environ.get("ADAPTIVE_LLM_RESOURCE_METRICS_INTERVAL_S", "5"))
        ),
        profile_bandwidth=_env_bool("ADAPTIVE_LLM_PROFILE_BANDWIDTH", False),
    )
