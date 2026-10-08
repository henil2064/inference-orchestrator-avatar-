# vLLM Utilization & Concurrency (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing vLLM worker use continuous batching, prefix caching, FP8, and CUDA graphs correctly, and stop the facade from throttling concurrency — so MedGemma-27B on one L40S serves ~50 concurrent voice sessions at full GPU utilization.

**Architecture:** Keep the current shape (`main.py` facade → gRPC → `vllm_worker.py` → vLLM `AsyncLLMEngine`). All new engine tuning is env-driven and assembled by one pure helper in `engines/common.py`, so it is unit-testable without a GPU or vLLM installed. The production knobs are injected through a new `model-profiles.json` runtime profile (no planner code change — the profile's `env` dict already flows `plan.engine_env` → `runtime_manager.start_worker` → worker `os.environ`). The facade's concurrency cap is raised from a batch-throttle to a real admission limit.

**Tech Stack:** Python 3.11 asyncio, vLLM (modern pin), gRPC (`grpc.aio`), prometheus_client, transformers tokenizer. Tests are stdlib `asyncio.run` + `assert` with `if __name__ == "__main__"` runners (existing repo convention), also runnable under pytest.

**Spec:** Design agreed in-session 2026-08-31 (not committed, per request). Summary reproduced in Global Constraints below; the [Anatomy of vLLM](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm), [V1 alpha](https://vllm.ai/blog/2025-01-27-v1-alpha-release), and [FP8 KV cache](https://vllm.ai/blog/2026-04-22-fp8-kvcache) posts are the source references.

## Global Constraints

- **Target hardware/model:** MedGemma-27B on a single L40S (Ada, 48 GB, sm_89), ~50 concurrent sessions. One replica per GPU; **no tensor parallelism** (no NVLink).
- **Weights: FP8 unconditional** — `quantization="fp8"` (on-the-fly dynamic FP8 of the bf16 `google/medgemma-27b-it` checkpoint). Needed to fit 27B + 50-way KV in 48 GB.
- **KV cache FP8: gated** — profile ships `LLM_KV_CACHE_DTYPE=auto`; flip to `fp8` only after Task 7's L40S benchmark passes (FP8-KV is validated on H100/B200, not Ada).
- **Tuning goal: balanced** — a latency SLO (TTFT/ITL for the voice loop) with throughput maximized under it. Concretely: `max_num_batched_tokens=2048`, `max_num_seqs=64`.
- **vLLM V1 engine** — enabled via `VLLM_USE_V1=1`; turns prefix caching + chunked prefill + piecewise CUDA graphs on by default. We still set `enable_prefix_caching` explicitly.
- **No gRPC contract change**, **no planner code change** (profiles-only), **llama.cpp worker path must keep installing and passing tests** in the shared environment.
- **vLLM pin:** bump `vllm>=0.6.0` → a modern release with V1 default and FP8-KV skip-layers support (Task 5).
- Run the full suite with: `python -m pytest adaptive_llm/tests -q` (from repo root `C:\mondee\projects\inference-orchestrator`).

---

### Task 1: Pure engine-args builder in `engines/common.py`

The one piece of real logic worth testing: turn `LLM_*` env vars into the kwargs dict for `AsyncEngineArgs`, plus a `VLLM_USE_V1` defaulter. Lives in `common.py`, which imports **no vLLM**, so it is fully testable on any machine.

**Files:**
- Modify: `adaptive_llm/engines/common.py`
- Test: `adaptive_llm/tests/test_vllm_engine_args.py` (create)

**Interfaces:**
- Produces: `engine_kwargs_from_env(env: Mapping[str, str]) -> dict[str, object]` — returns AsyncEngineArgs kwargs (everything **except** `model` and `tokenizer`, which the worker supplies).
- Produces: `ensure_v1_default(env: MutableMapping[str, str]) -> None` — sets `env["VLLM_USE_V1"] = "1"` only if unset.

- [ ] **Step 1: Write the failing test**

```python
# adaptive_llm/tests/test_vllm_engine_args.py
"""Pure env -> AsyncEngineArgs kwargs mapping (no vLLM import needed)."""

from __future__ import annotations

from adaptive_llm.engines.common import engine_kwargs_from_env, ensure_v1_default


def test_defaults_are_production_safe():
    kw = engine_kwargs_from_env({})
    assert kw["gpu_memory_utilization"] == 0.85
    assert kw["enforce_eager"] is False          # CUDA graphs on by default
    assert kw["enable_prefix_caching"] is True
    assert kw["disable_log_stats"] is False       # vLLM Prometheus series exposed
    assert kw["dtype"] == "auto"
    assert kw["quantization"] is None
    assert kw["kv_cache_dtype"] == "auto"
    assert kw["max_num_seqs"] == 64
    # unset numeric knobs stay None so vLLM picks its own default
    assert kw["max_model_len"] is None
    assert kw["max_num_batched_tokens"] is None


def test_env_overrides_map_to_engine_args():
    env = {
        "LLM_GPU_MEMORY_UTILIZATION": "0.90",
        "LLM_MAX_MODEL_LEN": "8192",
        "LLM_MAX_NUM_BATCHED_TOKENS": "2048",
        "LLM_MAX_NUM_SEQS": "48",
        "LLM_ENFORCE_EAGER": "1",
        "LLM_QUANTIZATION": "fp8",
        "LLM_KV_CACHE_DTYPE": "fp8",
        "LLM_ENABLE_PREFIX_CACHING": "0",
    }
    kw = engine_kwargs_from_env(env)
    assert kw["gpu_memory_utilization"] == 0.90
    assert kw["max_model_len"] == 8192
    assert kw["max_num_batched_tokens"] == 2048
    assert kw["max_num_seqs"] == 48
    assert kw["enforce_eager"] is True
    assert kw["quantization"] == "fp8"
    assert kw["kv_cache_dtype"] == "fp8"
    assert kw["enable_prefix_caching"] is False


def test_ensure_v1_default_sets_only_when_unset():
    env: dict[str, str] = {}
    ensure_v1_default(env)
    assert env["VLLM_USE_V1"] == "1"
    env2 = {"VLLM_USE_V1": "0"}
    ensure_v1_default(env2)
    assert env2["VLLM_USE_V1"] == "0"   # respect an explicit opt-out


if __name__ == "__main__":
    test_defaults_are_production_safe()
    test_env_overrides_map_to_engine_args()
    test_ensure_v1_default_sets_only_when_unset()
    print("ok vllm engine args")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest adaptive_llm/tests/test_vllm_engine_args.py -q`
Expected: FAIL with `ImportError: cannot import name 'engine_kwargs_from_env'`.

- [ ] **Step 3: Write minimal implementation**

Append to `adaptive_llm/engines/common.py` (keep the existing `from __future__ import annotations` at top; add `from typing import Mapping, MutableMapping` to the imports):

```python
def _opt_int(env: Mapping[str, str], key: str) -> int | None:
    raw = env.get(key)
    return int(raw) if raw not in (None, "") else None


def _bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def engine_kwargs_from_env(env: Mapping[str, str]) -> dict[str, object]:
    """Map LLM_* env vars to vLLM AsyncEngineArgs kwargs (model/tokenizer excluded).

    Defaults are production-safe for a single-tenant L40S: CUDA graphs on,
    prefix caching on, stats exposed. Numeric knobs left unset stay None so
    vLLM chooses its own default. See docs/superpowers/plans/2026-08-31-vllm-utilization.md.
    """
    return {
        "gpu_memory_utilization": float(env.get("LLM_GPU_MEMORY_UTILIZATION", "0.85")),
        "max_model_len": _opt_int(env, "LLM_MAX_MODEL_LEN"),
        "max_num_batched_tokens": _opt_int(env, "LLM_MAX_NUM_BATCHED_TOKENS"),
        "max_num_seqs": int(env.get("LLM_MAX_NUM_SEQS", "64")),
        "enforce_eager": _bool(env, "LLM_ENFORCE_EAGER", False),
        "quantization": (env.get("LLM_QUANTIZATION") or None),
        "kv_cache_dtype": env.get("LLM_KV_CACHE_DTYPE", "auto"),
        "enable_prefix_caching": _bool(env, "LLM_ENABLE_PREFIX_CACHING", True),
        "disable_log_stats": _bool(env, "LLM_DISABLE_LOG_STATS", False),
        "dtype": env.get("LLM_DTYPE", "auto"),
    }


def ensure_v1_default(env: MutableMapping[str, str]) -> None:
    """Opt into the vLLM V1 engine unless the operator explicitly set the flag."""
    env.setdefault("VLLM_USE_V1", "1")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest adaptive_llm/tests/test_vllm_engine_args.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add adaptive_llm/engines/common.py adaptive_llm/tests/test_vllm_engine_args.py
git commit -m "feat(vllm): pure env->AsyncEngineArgs builder with prod-safe defaults"
```

---

### Task 2: Wire the worker to the builder, enable V1, lazy-import vLLM

Replace the inline `AsyncEngineArgs(...)` block with the Task 1 helper, opt into V1 before the engine is built, and move the heavy `vllm` import inside `load_engine()` so the module imports without vLLM installed (lets Task 2's test run anywhere).

**Files:**
- Modify: `adaptive_llm/engines/vllm_worker.py` (imports at top; `load_engine` at lines ~79-101)
- Test: `adaptive_llm/tests/test_vllm_worker_load.py` (create)

**Interfaces:**
- Consumes: `engine_kwargs_from_env`, `ensure_v1_default` from Task 1.
- Produces: unchanged public behavior of `load_engine()`; internally it now calls the helper.

- [ ] **Step 1: Write the failing test**

```python
# adaptive_llm/tests/test_vllm_worker_load.py
"""load_engine assembles AsyncEngineArgs from env and opts into V1, with vLLM faked."""

from __future__ import annotations

import sys
import types

import pytest


def _install_fake_vllm(captured):
    mod = types.ModuleType("vllm")

    class FakeAsyncEngineArgs:
        def __init__(self, **kwargs):
            captured["args"] = kwargs

    class FakeSamplingParams:  # imported at module top in the worker
        def __init__(self, **kwargs):
            pass

    mod.AsyncEngineArgs = FakeAsyncEngineArgs
    mod.SamplingParams = FakeSamplingParams
    engine_mod = types.ModuleType("vllm.engine")
    async_mod = types.ModuleType("vllm.engine.async_llm_engine")

    class FakeAsyncLLMEngine:
        @classmethod
        def from_engine_args(cls, args):
            return ("engine", args)

    async_mod.AsyncLLMEngine = FakeAsyncLLMEngine
    sys.modules["vllm"] = mod
    sys.modules["vllm.engine"] = engine_mod
    sys.modules["vllm.engine.async_llm_engine"] = async_mod


def test_load_engine_uses_builder_and_sets_v1(monkeypatch):
    captured: dict = {}
    _install_fake_vllm(captured)
    # Fake the tokenizer load so no network/model is needed.
    import transformers
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", classmethod(lambda cls, *_a, **_k: object())
    )
    monkeypatch.setenv("LLM_MODEL_PATH", "/models/medgemma-27b-it")
    monkeypatch.setenv("LLM_GPU_MEMORY_UTILIZATION", "0.90")
    monkeypatch.delenv("VLLM_USE_V1", raising=False)

    sys.modules.pop("adaptive_llm.engines.vllm_worker", None)
    import adaptive_llm.engines.vllm_worker as worker

    engine, tokenizer = worker.load_engine()
    assert engine == "engine"
    assert captured["args"]["gpu_memory_utilization"] == 0.90
    assert captured["args"]["enable_prefix_caching"] is True
    assert captured["args"]["model"] == "/models/medgemma-27b-it"
    import os
    assert os.environ["VLLM_USE_V1"] == "1"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest adaptive_llm/tests/test_vllm_worker_load.py -q`
Expected: FAIL — currently `vllm_worker` imports `vllm` at module top (real import error or, once faked, `load_engine` does not call the builder / does not set `VLLM_USE_V1`).

- [ ] **Step 3: Write minimal implementation**

In `adaptive_llm/engines/vllm_worker.py`:

1. Remove the two top-level vLLM imports:
```python
from vllm import AsyncEngineArgs, SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine
```
2. Add the helpers to the existing common import:
```python
from adaptive_llm.engines.common import (
    engine_kwargs_from_env,
    ensure_v1_default,
    load_dotenv,
    now_ms,
    telemetry_attributes,
)
```
3. Replace the body of `load_engine()` (the `engine_args = AsyncEngineArgs(...)` block and the return) with:
```python
def load_engine():
    if not MODEL_PATH:
        raise SystemExit("set LLM_MODEL_PATH for vLLM worker")
    ensure_v1_default(os.environ)
    from vllm import AsyncEngineArgs  # heavy; import only when actually serving
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    tokenizer_id = TOKENIZER or MODEL_PATH
    log.info(
        "loading %s with vLLM V1 (tokenizer=%s, prompt_id=%s, text_only=%s) ...",
        MODEL_PATH, tokenizer_id, PROMPT_ID, LANGUAGE_MODEL_ONLY,
    )
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        tokenizer=tokenizer_id,
        language_model_only=LANGUAGE_MODEL_ONLY,
        **engine_kwargs_from_env(os.environ),
    )
    return AsyncLLMEngine.from_engine_args(engine_args), AutoTokenizer.from_pretrained(tokenizer_id)
```
4. Add `from vllm import SamplingParams` **inside** each of `Generate` and `GenerateStream` at the point `SamplingParams(...)` is constructed (or add a single module-level lazy accessor). Minimal: put `from vllm import SamplingParams` at the top of both methods before use.

> Note: the module-level constants (`GPU_MEMORY_UTILIZATION`, `MAX_NUM_BATCHED_TOKENS`, `QUANTIZATION`, `ENFORCE_EAGER`, etc., lines ~36-39) are now dead — the builder reads `os.environ` directly. Delete the ones only used by the old `AsyncEngineArgs` block; keep `MAX_TOKENS`, `TEMPERATURE`, `MODEL_PATH`, `TOKENIZER`, `MAX_MODEL_LEN`, `CONTEXT_SAFETY_TOKENS`, `ENABLE_THINKING`, `LANGUAGE_MODEL_ONLY`, `QUANTIZATION` (still used in the profile/`InferenceProfile`).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest adaptive_llm/tests/test_vllm_worker_load.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add adaptive_llm/engines/vllm_worker.py adaptive_llm/tests/test_vllm_worker_load.py
git commit -m "feat(vllm): worker builds engine args via helper, opts into V1, lazy-imports vllm"
```

---

### Task 3: Turn the facade concurrency cap into an admission limit

`Semaphore(max_inflight_requests)` currently caps vLLM's running batch at the default **4**, defeating continuous batching. Raise the default so vLLM's own scheduler batches ~50 sessions; keep `max_queued_requests` as backpressure. This is a config default + doc change; the mechanism is unchanged.

**Files:**
- Modify: `adaptive_llm/core/config.py:88` (default for `ADAPTIVE_LLM_MAX_INFLIGHT_REQUESTS`)
- Modify: `adaptive_llm/main.py:98-116` (docstring/comment clarifying admission semantics)
- Test: `adaptive_llm/tests/test_config_admission.py` (create)

**Interfaces:**
- Consumes: nothing new.
- Produces: `Config.max_inflight_requests` default now 48.

- [ ] **Step 1: Write the failing test**

```python
# adaptive_llm/tests/test_config_admission.py
"""max_inflight_requests must default high enough to feed vLLM continuous batching."""

from __future__ import annotations

from adaptive_llm.core.config import load_config


def test_inflight_default_feeds_continuous_batching(monkeypatch):
    monkeypatch.delenv("ADAPTIVE_LLM_MAX_INFLIGHT_REQUESTS", raising=False)
    cfg = load_config()
    assert cfg.max_inflight_requests == 48


def test_inflight_still_overridable(monkeypatch):
    monkeypatch.setenv("ADAPTIVE_LLM_MAX_INFLIGHT_REQUESTS", "8")
    assert load_config().max_inflight_requests == 8


if __name__ == "__main__":
    import types
    fake = types.SimpleNamespace(setenv=lambda *_: None, delenv=lambda *a, **k: None)
    test_inflight_default_feeds_continuous_batching(fake)
    print("ok admission default (override test needs pytest monkeypatch)")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest adaptive_llm/tests/test_config_admission.py -q`
Expected: FAIL — default is currently 4, not 48.

- [ ] **Step 3: Write minimal implementation**

In `adaptive_llm/core/config.py`, change line 88 from `"4"` to `"48"`:
```python
        max_inflight_requests=max(1, int(os.environ.get("ADAPTIVE_LLM_MAX_INFLIGHT_REQUESTS", "48"))),
```
In `adaptive_llm/main.py`, add a clarifying comment above `self._slots = asyncio.Semaphore(max_inflight_requests)` (line ~111):
```python
        # Admission limit, NOT a batch throttle: vLLM's own scheduler does
        # continuous batching, so this should track the engine's max_num_seqs
        # (~concurrency target), not a small number. Anything beyond this plus
        # max_queued_requests is rejected with RESOURCE_EXHAUSTED.
        self._slots = asyncio.Semaphore(max_inflight_requests)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest adaptive_llm/tests/test_config_admission.py adaptive_llm/tests/test_facade.py -q`
Expected: PASS (new tests pass; existing `test_facade` admission test still passes — it constructs the servicer with explicit `max_inflight_requests=1`, so it is unaffected).

- [ ] **Step 5: Commit**

```bash
git add adaptive_llm/core/config.py adaptive_llm/main.py adaptive_llm/tests/test_config_admission.py
git commit -m "feat(facade): raise inflight default to 48 so vLLM continuous batching engages"
```

---

### Task 4: Add the MedGemma-27B FP8 model + `l40s_medgemma27b` runtime profile

Register the production model and the runtime profile that injects every Phase-1 knob. Pure `model-profiles.json` change — the planner already selects a profile by engine + VRAM window and passes its `env` through to the worker.

**Files:**
- Modify: `adaptive_llm/model-profiles.json` (add one model entry under `"models"`)
- Test: `adaptive_llm/tests/test_prod_profile.py` (create)

**Interfaces:**
- Consumes: `load_registry`, `_runtime_profile_env` (existing).
- Produces: registry key `medgemma_27b`; profile name `l40s_medgemma27b`.

- [ ] **Step 1: Write the failing test**

```python
# adaptive_llm/tests/test_prod_profile.py
"""The L40S MedGemma-27B profile injects the Phase-1 vLLM knobs."""

from __future__ import annotations

from pathlib import Path

from adaptive_llm.planning.registry import load_registry
from adaptive_llm.planning.planner import _runtime_profile_env
from adaptive_llm.planning.hardware import HardwareSnapshot, StaticSnapshot, DynamicSnapshot, GpuStatic

ROOT = Path(__file__).resolve().parents[2]


def _l40s_hardware() -> HardwareSnapshot:
    gpu = GpuStatic(index=0, name="NVIDIA L40S", total_vram_mib=46068, compute_capability="8.9")
    return HardwareSnapshot(
        static=StaticSnapshot(os="Linux", cpu="x", logical_cores=60, total_ram_mib=225000,
                              cuda_available=True, gpus=[gpu]),
        dynamic=DynamicSnapshot(available_ram_mib=200000, cpu_utilization_pct=1.0, gpus=[]),
    )


def test_prod_profile_registered_and_selected():
    registry = load_registry(ROOT / "adaptive_llm" / "model-profiles.json")
    assert "medgemma_27b" in registry
    model = registry["medgemma_27b"]
    assert model.quantization == "fp8"
    assert model.format == "safetensors"

    name, env = _runtime_profile_env(model, _l40s_hardware(), "vllm", 0)
    assert name == "l40s_medgemma27b"
    assert env["LLM_GPU_MEMORY_UTILIZATION"] == "0.90"
    assert env["LLM_MAX_NUM_SEQS"] == "64"
    assert env["LLM_MAX_NUM_BATCHED_TOKENS"] == "2048"
    assert env["LLM_ENABLE_PREFIX_CACHING"] == "1"
    assert env["LLM_ENFORCE_EAGER"] == "0"
    assert env["VLLM_USE_V1"] == "1"
    # FP8 KV is gated on the L40S benchmark (Task 7); ships as auto.
    assert env["LLM_KV_CACHE_DTYPE"] == "auto"


if __name__ == "__main__":
    test_prod_profile_registered_and_selected()
    print("ok prod profile")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest adaptive_llm/tests/test_prod_profile.py -q`
Expected: FAIL — `medgemma_27b` not in registry.

- [ ] **Step 3: Write minimal implementation**

Add this entry to `adaptive_llm/model-profiles.json` inside the `"models"` object (add a comma after the previous entry). KV-estimate fields (`layers`/`hidden_size`/`kv_heads`/`head_dim`) match Gemma-3-27B and feed the planner's memory estimate only:

```json
    "medgemma_27b": {
      "display_name": "MedGemma 27B IT (FP8)",
      "model_ref": "google/medgemma-27b-it",
      "format": "safetensors",
      "architecture": "gemma3",
      "parameter_count_b": 27,
      "quantization": "fp8",
      "weight_dtype": "fp8",
      "model_path": "adaptive_llm/models/google/medgemma-27b-it",
      "tokenizer": "google/medgemma-27b-it",
      "max_context_tokens": 32768,
      "default_context_tokens": 8192,
      "default_max_output_tokens": 512,
      "layers": 62,
      "hidden_size": 5376,
      "kv_heads": 16,
      "head_dim": 128,
      "kv_cache_dtype_bytes": 2,
      "runtime_profiles": [
        {
          "name": "l40s_medgemma27b",
          "engine": "vllm",
          "min_vram_mib": 40000,
          "env": {
            "LLM_GPU_MEMORY_UTILIZATION": "0.90",
            "LLM_MAX_NUM_SEQS": "64",
            "LLM_MAX_NUM_BATCHED_TOKENS": "2048",
            "LLM_ENABLE_PREFIX_CACHING": "1",
            "LLM_KV_CACHE_DTYPE": "auto",
            "LLM_ENFORCE_EAGER": "0",
            "VLLM_USE_V1": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"
          }
        }
      ]
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest adaptive_llm/tests/test_prod_profile.py -q`
Expected: PASS. Also run `python -c "import json; json.load(open('adaptive_llm/model-profiles.json'))"` to confirm valid JSON.

- [ ] **Step 5: Commit**

```bash
git add adaptive_llm/model-profiles.json adaptive_llm/tests/test_prod_profile.py
git commit -m "feat(profiles): add MedGemma-27B FP8 model + l40s_medgemma27b runtime profile"
```

---

### Task 5: Bump the vLLM pin and verify the shared environment

Move `vllm>=0.6.0` to a modern release (V1 default, FP8-KV `--kv-cache-dtype-skip-layers`, offload connector) and confirm the co-located llama.cpp worker still installs and all tests pass.

**Files:**
- Modify: `adaptive_llm/requirements.txt`
- Modify: `adaptive_llm/docs/DOCUMENTATION.md` (env-var table — see Step 4)

**Interfaces:** none (dependency + docs).

- [ ] **Step 1: Bump the pin**

In `adaptive_llm/requirements.txt` change:
```
vllm>=0.6.0
```
to:
```
vllm>=0.14.0
```
(0.14.x is the floor that has V1-default, FP8-KV skip-layers, and the `native` KV-offload backend used in Phase 2. If a newer release is already validated in the target image, pin that instead.)

- [ ] **Step 2: Verify the shared environment installs**

Run (in the deployment venv / target image):
```bash
pip install -r adaptive_llm/requirements.txt
python -c "import vllm, llama_cpp; print('vllm', vllm.__version__)"
```
Expected: both import; no resolver conflict between vLLM's torch pin and `llama-cpp-python`. If torch conflicts, record the resolved torch version and, if needed, split the two workers into separate requirements files (note it and stop for review — that is a scope change).

- [ ] **Step 3: Full test suite still green**

Run: `python -m pytest adaptive_llm/tests -q`
Expected: PASS (the vLLM upgrade must not break the pure-Python tests; none import vLLM).

- [ ] **Step 4: Document the new env vars**

In `adaptive_llm/docs/DOCUMENTATION.md`, add a row per new knob to the worker env table: `LLM_MAX_NUM_SEQS`, `LLM_KV_CACHE_DTYPE`, `LLM_ENABLE_PREFIX_CACHING`, `LLM_DISABLE_LOG_STATS`, `VLLM_USE_V1` — each with its default and one-line meaning from Task 1. Note that `vllm:` Prometheus series (e.g. `vllm:num_requests_running`, `vllm:gpu_cache_usage_perc`, `vllm:prefix_cache_hit_rate`) now appear on the worker metrics port (`LLM_WORKER_METRICS_PORT`, default 9114) because `disable_log_stats=False`.

- [ ] **Step 5: Commit**

```bash
git add adaptive_llm/requirements.txt adaptive_llm/docs/DOCUMENTATION.md
git commit -m "chore(vllm): pin modern vLLM (V1/FP8-KV); document new worker knobs"
```

---

### Task 6: Benchmark on L40S and decide FP8 KV cache (gate)

Prove the throttle is gone and the batch engages, then make the FP8-KV go/no-go. This task is a manual measurement with concrete commands and acceptance thresholds — the only non-TDD task, because it needs the real GPU + model.

**Files:**
- Create: `adaptive_llm/tests/loadgen_grpc.py` (concurrent gRPC load client — real, not a unit test)
- Modify (conditionally): `adaptive_llm/model-profiles.json` (flip `LLM_KV_CACHE_DTYPE` to `fp8` **iff** the gate passes)

**Interfaces:**
- Consumes: the running facade at `LLM_ADDR`, the `medgemma_27b` profile, the `InferenceProfile` already returned in each `GenerateResponse` (has `ttft_ms`, `decode_tokens_per_second`, `queue_ms`).

- [ ] **Step 1: Write the load client**

```python
# adaptive_llm/tests/loadgen_grpc.py
"""Fire N concurrent Generate calls at the facade; report TTFT/latency percentiles.

Usage: python -m adaptive_llm.tests.loadgen_grpc --addr 127.0.0.1:50053 --concurrency 50 --requests 500
Reads timing from the InferenceProfile the facade already returns.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import grpc

from adaptive_llm.proto import llm_pb2, llm_pb2_grpc

PROMPT = "A 54-year-old patient reports intermittent chest tightness on exertion. Summarize the differential."


async def _one(stub, sem, results):
    async with sem:
        req = llm_pb2.GenerateRequest(session_id="load", text=PROMPT)
        t0 = time.perf_counter()
        reply = await stub.Generate(req)
        results.append((
            (time.perf_counter() - t0) * 1000.0,
            reply.profile.ttft_ms,
            reply.profile.queue_ms,
            reply.profile.decode_tokens_per_second,
        ))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--addr", default="127.0.0.1:50053")
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--requests", type=int, default=500)
    args = ap.parse_args()

    async with grpc.aio.insecure_channel(args.addr) as channel:
        stub = llm_pb2_grpc.LlmStub(channel)
        sem = asyncio.Semaphore(args.concurrency)
        results: list[tuple[float, float, float, float]] = []
        t0 = time.perf_counter()
        await asyncio.gather(*[_one(stub, sem, results) for _ in range(args.requests)])
        wall = time.perf_counter() - t0

    def pct(xs, p):
        return round(statistics.quantiles(sorted(xs), n=100)[p - 1], 1)

    e2e = [r[0] for r in results]
    ttft = [r[1] for r in results]
    queue = [r[2] for r in results]
    tps = [r[3] for r in results if r[3] > 0]
    print(f"requests={len(results)} concurrency={args.concurrency} wall={wall:.1f}s "
          f"throughput={len(results) / wall:.1f} req/s")
    print(f"TTFT  p50={pct(ttft,50)}ms p95={pct(ttft,95)}ms")
    print(f"E2E   p50={pct(e2e,50)}ms p95={pct(e2e,95)}ms")
    print(f"QUEUE p50={pct(queue,50)}ms p95={pct(queue,95)}ms  (should be ~0 if not throttled)")
    print(f"decode tok/s median={statistics.median(tps):.1f}")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: Baseline run (KV=auto, the shipped profile)**

Start the facade with `LLM_MODEL_ID=medgemma_27b` on the L40S box, then:
```bash
python -m adaptive_llm.tests.loadgen_grpc --concurrency 50 --requests 500
```
Record TTFT/E2E/queue/throughput. **Verify the throttle is gone:** `QUEUE p95` should be near zero and `vllm:num_requests_running` (on :9114/metrics) should climb well above 4 under load. If queue time is high or running-batch stays ≤4, the admission cap or `max_num_seqs` is misconfigured — fix before proceeding.

- [ ] **Step 3: FP8-KV run and gate decision**

Restart the facade with `LLM_KV_CACHE_DTYPE=fp8` (env override; if MedGemma's sliding-window layers regress, also try the worker/vLLM `--kv-cache-dtype-skip-layers sliding_window` equivalent). Re-run the load client, and run an accuracy spot-check on a small held-out medical Q&A set comparing FP8-KV vs auto outputs.

**Gate — enable FP8 KV in the profile only if all hold:**
- Accuracy recovery within tolerance (≈≥97% agreement / no material regression on the spot-check).
- Throughput and/or KV headroom (`vllm:gpu_cache_usage_perc`) improve, or concurrency ceiling rises, with no TTFT regression at your typical (<7k-token) turn lengths.

If the gate passes, set `"LLM_KV_CACHE_DTYPE": "fp8"` in the `l40s_medgemma27b` profile. If it fails on Ada, leave `auto` and record the finding in `DOCUMENTATION.md`.

- [ ] **Step 4: Commit**

```bash
git add adaptive_llm/tests/loadgen_grpc.py adaptive_llm/model-profiles.json adaptive_llm/docs/DOCUMENTATION.md
git commit -m "test(vllm): gRPC load client; record L40S FP8-KV benchmark + gate decision"
```

---

## Self-Review

**Spec coverage:**
- Phase 1a engine knobs (V1, FP8 weights, prefix cache, chunked-prefill budget, max_num_seqs, gpu_mem, CUDA graphs, kv_cache_dtype) → Tasks 1, 2, 4. ✓
- Phase 1b facade concurrency fix → Task 3. ✓
- Phase 1c prod profile + version pin → Tasks 4, 5. ✓
- Phase 1d metrics (vLLM native Prometheus on existing endpoint via `disable_log_stats=False`) → Task 1 default + documented in Task 5; verified in Task 6 Step 2. ✓
- Phase 1e L40S benchmark + FP8-KV gate + sliding-window caveat → Task 6. ✓
- Constraints (no gRPC/planner change, llama.cpp keeps working, balanced tuning, FP8 weights unconditional) → honored across tasks; llama.cpp install verified in Task 5 Step 2. ✓
- Phase 2/3 (KV-offload, replicas, disaggregation, Mooncake) → intentionally **out of scope** for this plan.

**Placeholder scan:** No TBD/TODO; every code step has real content and thresholds. Task 6 is inherently manual but carries concrete commands and pass/fail criteria, not "benchmark it".

**Type consistency:** `engine_kwargs_from_env` / `ensure_v1_default` names identical across Tasks 1 and 2. Env-var names (`LLM_MAX_NUM_SEQS`, `LLM_KV_CACHE_DTYPE`, `LLM_ENABLE_PREFIX_CACHING`) identical across Tasks 1, 4, 5. Profile name `l40s_medgemma27b` and key `medgemma_27b` identical across Tasks 4 and 6.

**Note for the executor:** Tasks 1-5 are machine-independent (no GPU, no vLLM install needed for their tests). Task 6 must run on the L40S target with the model present.
