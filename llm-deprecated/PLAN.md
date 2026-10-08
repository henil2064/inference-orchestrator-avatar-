# Adaptive LLM Runtime Plan

## Goal

Replace the two manually selected LLM service scripts with one stable public
gRPC service that selects the best locally available inference engine for the
requested model and the machine it is starting on. The public contract remains
exactly `llm.v1.Llm/Generate` on `LLM_ADDR`; the orchestrator continues to use
`LLM_TARGET=localhost:50053` and requires no code or protobuf change.

The selector must choose from `vLLM`, `SGLang`, and `llama.cpp`, plan an
execution mode (`FULL_GPU`, `PARTIAL_GPU`, `CPU_ONLY`, `MULTI_GPU`, or
`DISTRIBUTED`), and expose its chosen plan in logs, health/status, and
Prometheus metrics. Initial support should implement `vLLM` and `llama.cpp`.
`SGLang`, multi-GPU, and distributed operation are registry/planner-ready but
remain explicitly unavailable until their adapters are implemented and tested.

## Current state and constraints

- `server.py` runs vLLM and `server_unsloth.py` runs llama.cpp. Both already
  implement the same unary gRPC RPC defined in `../proto/llm.proto`.
- vLLM accepts concurrent requests through its native continuous batching.
  llama.cpp is intentionally serialized by a one-worker executor.
- The two engines have separate Python dependency stacks and CUDA build
  requirements. The adaptive design must not require importing vLLM,
  llama-cpp-python, and future engine packages into one process or one virtual
  environment.
- `Generate` is deliberately single-turn today. This plan must preserve that
  behavior and must not add history, streaming, or RAG to the RPC.
- A model selection is a *runtime lifecycle* decision, not a per-request
  decision. Reloading weights for every RPC would be unsafe and slow. Dynamic
  telemetry informs startup, admission, and a safe future re-plan only when
  there are no active requests.

## Target architecture

```text
orchestrator -- unchanged llm.v1/Llm.Generate --> public LLM facade (:50053)
                                                   |
                                                   +-- Model Registry
                                                   +-- Architecture Detector
                                                   +-- Hardware Profiler
                                                   +-- Compatibility + Resource Estimator
                                                   +-- Execution Planner
                                                   +-- Runtime Manager
                                                           |
                                      private Unix socket / loopback gRPC worker
                                                           |
                         vLLM worker | llama.cpp worker | future SGLang worker
```

The facade is a lightweight Python process with only gRPC, Prometheus, and
hardware-probing dependencies. At startup it resolves `LLM_MODEL_ID` in the
registry, captures a hardware/resource snapshot, produces a persisted
`ExecutionPlan`, and asks the runtime manager to start exactly one worker in
that engine's dedicated virtual environment. The worker is the only process
that imports the engine library and owns model memory. The facade proxies
`Generate` to it and returns the existing response untouched.

Use a Unix-domain socket by default for the private facade-to-worker RPC;
allow a loopback TCP address only for development. The worker must never expose
its port outside localhost. Worker restarts, crash-loop limits, readiness, and
graceful shutdown belong to the runtime manager.

## Proposed file layout

```text
llm/
  main.py                       # public gRPC facade and startup sequence
  config.py                     # typed environment/config loading
  registry.py                   # model-registry parsing and validation
  models.toml                   # version-controlled model definitions
  hardware.py                   # static + dynamic hardware snapshots
  compatibility.py              # engine capability matrix and hard gates
  estimator.py                  # weights/KV/runtime memory estimators
  planner.py                    # candidates, scores, deterministic selection
  runtime_manager.py            # worker process, readiness, restart handling
  status.py                     # current plan, status payload, metrics labels
  engines/
    base.py                     # EngineAdapter and worker protocol interfaces
    vllm_worker.py              # extracted/adapted current server.py logic
    llamacpp_worker.py          # extracted/adapted current server_unsloth.py logic
    sglang_worker.py            # later; registered as unavailable initially
  tests/
    test_registry.py
    test_hardware.py
    test_compatibility.py
    test_estimator.py
    test_planner.py
    test_runtime_manager.py
    fixtures/
      gtx1650_4gb.json
      rtx4090_24gb.json
```

The existing scripts should first become compatibility wrappers with clear
deprecation notices, then be removed only after the new facade is the documented
entry point. `query_profile.py` should be refactored to profile a selected
worker/plan rather than importing `server.py` directly.

## Model registry

`models.toml` is the source of truth. A registry entry describes facts about a
model, not an engine-specific launch command. It must support local paths and
optional Hugging Face download metadata.

```toml
[models.medgemma_1_5_4b_q4]
display_name = "MedGemma 1.5 4B IT Q4_K_S"
format = "gguf"
architecture = "gemma3"
parameter_count_b = 4.0
quantization = "Q4_K_S"
model_path = "models/unsloth/medgemma-1.5-4b-it-GGUF/medgemma-1.5-4b-it-Q4_K_S.gguf"
tokenizer = "google/medgemma-4b-it"
max_context_tokens = 2048
default_max_output_tokens = 256
chat_template = "tokenizer"

[models.qwen3_8b_bf16]
display_name = "Qwen3 8B Instruct BF16"
format = "safetensors"
architecture = "qwen3"
parameter_count_b = 8.0
weight_dtype = "bfloat16"
model_path = "/models/Qwen3-8B-Instruct"
max_context_tokens = 8192
default_max_output_tokens = 512
chat_template = "tokenizer"
```

At validation time, augment declared metadata from local artifacts when
possible: GGUF headers/file size for GGUF, and `config.json` plus Safetensors
index/shard sizes for Hugging Face models. A conflict is a startup error, not a
silent guess. `LLM_MODEL_ID` becomes the primary selector; `LLM_MODEL_PATH` is
supported only as a migration override that resolves to a generated local entry
and logs a warning.

## Hardware and resource profiling

### Static snapshot (collected once at startup)

- OS, kernel, Python/platform, CPU model, logical/physical cores, CPU ISA flags,
  total RAM, NUMA topology when discoverable.
- Per GPU: index/UUID, model, compute capability, total VRAM, driver/CUDA
  version, and available CUDA backend(s).

Prefer NVML (`pynvml`) for NVIDIA properties; fall back to `nvidia-smi` and
then a CPU-only result. CPU and RAM data come from `psutil` and `/sys`/`lscpu`
where available. Probing failures must be represented as unknown values, never
invented capabilities.

### Dynamic snapshot (sampled at startup and every configurable interval)

- Per GPU: free/used VRAM, utilization, temperature, and process allocations.
- Host: available RAM, CPU utilization/load, and NUMA-local available RAM where
  available.
- Runtime: selected model, active requests, queue depth, worker health, engine
  KV-cache utilization/capacity when the adapter can report it.

Use a short rolling average for utilization and retain instantaneous free memory
for allocation safety. Reserve configurable safety headroom from both RAM and
VRAM (defaults: 15% VRAM, 10% RAM, never less than 512 MiB) before planning.
Do not treat temperature as a performance score; it is a health gate/warning
with configurable critical thresholds.

## Compatibility matrix and hard gates

The compatibility layer yields an explainable reason for every rejected engine.
Hard failures eliminate a candidate before scoring.

| Engine | Initial formats | Key constraints | Planned modes |
|---|---|---|---|
| vLLM | HF Safetensors/BF16/FP16 and supported quantization | installed worker, supported architecture/dtype, CUDA capability, enough VRAM for weights + engine/KV reservation | `FULL_GPU`, later `MULTI_GPU`/`DISTRIBUTED` |
| SGLang | future supported HF formats | adapter not implemented in phase 1 | none initially |
| llama.cpp | GGUF | installed CUDA or CPU build, GGUF architecture support, enough combined RAM/VRAM | `FULL_GPU`, `PARTIAL_GPU`, `CPU_ONLY`, later `MULTI_GPU` |

Examples:

- A GGUF MedGemma model on a GTX 1650 is incompatible with the initial vLLM
  profile and SGLang adapter, but compatible with llama.cpp using GPU-layer
  offload and CPU-resident layers.
- A Qwen3-8B BF16/Safetensors model on an RTX 4090 is compatible with vLLM and
  should normally prefer it because it can reserve KV cache and batch requests
  on the GPU.

Engine availability is separate from compatibility: an otherwise valid vLLM
candidate is rejected with `worker environment unavailable` when its configured
venv or executable is missing. The response/log must list this reason.

## Resource estimation and execution modes

Use engine-specific estimators, with conservative estimates and explicit
uncertainty. The generic planning baseline is:

```text
weight_bytes ≈ parameter_count × bytes_per_parameter × overhead_factor
base_memory  ≈ weight_bytes × 1.10–1.30
total_memory ≈ base_memory + kv_cache_bytes + runtime/scratch_bytes
```

For local files, actual aggregate model size is preferred to a parameter-count
estimate. KV-cache estimation must use architecture/config values (layers,
hidden/KV heads, head dimension, cache dtype, context, concurrent sequences)
rather than a universal constant. The estimator records the inputs and its
confidence in the plan/status response.

Planning order:

1. Calculate context and output-token requirements from registry defaults plus
   allowed `LLM_*` limits.
2. Reserve OS/display/other-process headroom from free VRAM and free RAM.
3. Try full-GPU candidates only if model, runtime scratch, and initial KV cache
   fit within safe free VRAM.
4. For GGUF/llama.cpp, calculate the largest safe `n_gpu_layers` from measured
   per-layer GGUF sizes (or a conservative calibration), then reserve remaining
   model layers and KV memory in RAM. Reduce layers/context/batch in that order
   only within registry/configured floors.
5. Choose `CPU_ONLY` only if model plus KV/cache fits safely in available RAM.
6. Reject the model with a structured resource report if no mode fits. Do not
   start a worker that is expected to OOM.

The phase-1 llama.cpp execution-plan fields are `gpu_index`, `n_gpu_layers`,
`n_ctx`, `n_batch`, `n_ubatch`, `n_threads`, `main_gpu`, RAM/VRAM estimates,
and requested output tokens. vLLM fields include `gpu_indices`,
`gpu_memory_utilization`, `max_model_len`, `max_num_batched_tokens`, dtype,
and tensor parallel size (always one initially).

For the GTX 1650 example, the expected plan is approximately:

```text
engine=llama.cpp, mode=PARTIAL_GPU, gpu=0,
n_gpu_layers≈18–22 (calibrated from the actual GGUF), n_threads=6,
n_ctx=2048, n_batch=128, n_ubatch=64
```

The exact layer count must be measured/calibrated for the specific GGUF and
display-memory baseline; it must not be hard-coded as `20`. For the RTX 4090
example the expected plan is `vLLM + FULL_GPU`, subject to real model/KV
estimates and current free VRAM.

## Scoring and tie-breaking

After hard gates, normalize each component to 0–100 and retain the breakdown:

```text
score = 0.25 * compatibility
      + 0.20 * hardware_fit
      + 0.20 * memory_fit
      + 0.15 * performance
      + 0.10 * quantization_support
      + 0.05 * context_support
      + 0.05 * current_load
```

- `compatibility` is 100 after hard gates; capability caveats reduce it only
  when a safe fallback exists.
- `hardware_fit` accounts for CUDA capability, number of GPUs, CPU ISA, and
  whether the engine can use the detected hardware.
- `memory_fit` rewards headroom after all reservations; insufficient memory is
  a hard failure.
- `performance` is a versioned, locally benchmarked engine/model/profile score,
  falling back to conservative defaults until benchmark data exists.
- `quantization_support` favors native support over experimental conversion.
- `context_support` evaluates the requested context/concurrency, not merely a
  model's advertised maximum context.
- `current_load` uses dynamic memory and utilization, but does not select a
  different engine while a worker is serving requests.

Tie-break deterministically by: higher memory headroom, higher local benchmark
score, lower estimated time-to-first-token, then configured engine preference.
Add `LLM_ENGINE=auto|vllm|llamacpp|sglang` for an explicit operator override;
an override bypasses ranking but never hard safety gates unless
`LLM_ALLOW_UNSAFE_OVERRIDE=1` is set.

## Runtime behavior and failure policy

- Startup writes `llm/run/current-plan.json` atomically and logs human-readable
  candidates, rejected reasons, chosen plan, and all memory estimates.
- The manager starts the selected worker, waits for a private readiness RPC, and
  only then marks the public service ready. It forwards the public `Generate`
  request/response unmodified.
- A worker crash makes the facade return `UNAVAILABLE`; it must not silently
  switch engines during active sessions. Restart the same validated plan with
  exponential backoff and a crash-loop cap.
- A manual reload/control command or a planned future idle re-plan can take a
  fresh dynamic snapshot, drain requests, stop the worker, and select a new
  plan. Automatic engine switching while requests are active is out of scope.
- On llama.cpp OOM during first load, permit a bounded retry ladder generated
  by the planner: reduce GPU layers, then batch/ubatch, then context down to its
  configured floor. Persist every failed attempt. Do not use blind retries.
- On vLLM OOM, retry only planned safe profiles (for example lower
  `max_num_batched_tokens`, then context/KV allocation); otherwise report a
  precise startup failure.

## Observability

Keep the existing request metrics and add:

- `llm_engine_info{engine,mode,model_id,model_format}` = 1
- `llm_execution_plan_info{...}` = 1 (only stable, low-cardinality labels)
- `llm_planner_score{engine,component}` and
  `llm_planner_rejections_total{engine,reason}`
- `llm_worker_up`, `llm_worker_restarts_total`, `llm_active_requests`, and
  `llm_queue_depth`
- `llm_gpu_free_bytes`, `llm_gpu_utilization_ratio`, `llm_ram_available_bytes`
  and engine-reported KV cache metrics where available.

Expose a local-only `/status` endpoint (or an equivalent admin gRPC method in a
later protobuf revision) with the model metadata, static snapshot summary,
latest dynamic snapshot, candidate score breakdown, current plan, worker PID,
and health. Do not change `llm.proto` in the first implementation.

## Configuration and deployment

- `LLM_MODEL_ID` (required after migration), `LLM_ENGINE=auto`,
  `LLM_CONFIG=models.toml`, `LLM_STATE_DIR=./run`, `LLM_ADDR=[::]:50053`.
- `LLM_VLLM_PYTHON` and `LLM_LLAMACPP_PYTHON` point to dedicated worker venv
  interpreters. The manager validates both without importing their libraries.
- `LLM_GPU_SAFETY_MARGIN_MIB`, `LLM_RAM_SAFETY_MARGIN_MIB`,
  `LLM_MIN_CONTEXT`, `LLM_MAX_CONTEXT`, `LLM_MAX_CONCURRENCY`, and
  `LLM_DYNAMIC_SAMPLE_SECONDS` set safe policy bounds.
- Keep model-specific defaults in the registry; keep machine/site policy in an
  untracked `.env` or a deployment config. Never require a model path to be
  duplicated across worker-specific env files.

Provide example profiles for the GTX 1650 GGUF setup and RTX 4090 Qwen3 setup,
plus a CPU-only GGUF profile. Existing `.env.unsloth` values map directly to
the llama.cpp plan fields during migration.

## Implementation phases

1. **Foundation:** add typed configuration, TOML registry, model/artifact
   detection, static/dynamic profiler, compatibility result types, and unit
   fixtures. No serving behavior changes.
2. **Planning:** implement resource estimators, deterministic candidate scoring,
   explainable plans, `current-plan.json`, and CLI dry-run:
   `python -m llm.main plan --model medgemma_1_5_4b_q4`.
3. **Facade + llama.cpp:** extract reusable gRPC/metrics code from
   `server_unsloth.py`, add a private worker protocol and manager, and make the
   public facade launch the llama.cpp worker from its dedicated venv.
4. **vLLM adapter:** extract `server.py` into the vLLM worker adapter, pass only
   planner-approved arguments, preserve native continuous batching, and support
   vLLM safe retry profiles.
5. **Migration:** make `python -m llm.main serve` the documented entry point;
   retain old scripts as wrappers for one release and update both run guides,
   `.env.example`, and `query_profile.py`.
6. **Validation and operations:** collect calibrated profile data, add load/OOM
   tests, dashboard panels, worker crash recovery tests, and controlled idle
   reload. Add SGLang/multi-GPU/distributed adapters only with representative
   hardware tests and registry capability entries.

## Acceptance criteria

- The orchestrator sends exactly the same `GenerateRequest` to `:50053` before
  and after migration; existing LLM routing tests still pass.
- `plan` with the GTX 1650 fixture selects llama.cpp / `PARTIAL_GPU`, reports a
  safe layer count range/configuration and every rejected engine reason.
- `plan` with the RTX 4090/Qwen3 fixture selects vLLM / `FULL_GPU` with a
  higher score than supported alternatives.
- A GGUF that cannot fit wholly in VRAM can run through llama.cpp with planned
  partial offload when RAM is sufficient; an insufficient-RAM fixture fails
  before attempting worker startup.
- No worker engine package is imported by the facade process; vLLM and
  llama.cpp can remain in independent virtual environments.
- The selected plan, score breakdown, live resource data, worker health, and
  active-request count are observable without changing `llm.proto`.
- Concurrent vLLM requests remain un-serialized; llama.cpp retains explicit
  serialized execution until its concurrency model is independently validated.

## Decisions intentionally deferred

- Exact SGLang, multi-GPU, and distributed backend support: add only after
  adapter compatibility and hardware validation.
- A public control-plane API for reload/status: start with local status and
  process lifecycle; version a control RPC only when a remote caller needs it.
- The final benchmark-derived performance weights and llama.cpp per-layer
  calibration: collect these on each supported GPU/model pair rather than
  encoding speculative universal numbers.
