# Adaptive LLM: Architecture and Operations Guide

## 1. What this service is

`adaptive_llm` is a local LLM serving layer for the inference orchestrator. It
keeps the public LLM API stable while choosing the most suitable local inference
engine for the selected model and the machine it is running on.

The orchestrator calls one gRPC method:

```text
llm.v1.Llm/Generate
```

The orchestrator does not need to know whether the reply came from `llama.cpp`
or vLLM. It continues to call the public Adaptive LLM address (normally
`localhost:50053`), and Adaptive LLM handles the engine-specific details behind
that endpoint.

This is useful because models and hardware differ substantially:

- GGUF models can go to either engine. `llama.cpp` is the portable choice: it
  supports CPU-only inference, partial GPU offload, and files split across
  shards. vLLM serves a *single-file* GGUF with kernels ported from llama.cpp,
  trading their lower tuning for continuous batching and a paged KV cache,
  which is usually the better deal on a CUDA GPU. See
  [VLLM_GGUF.md](VLLM_GGUF.md).
- Hugging Face/Safetensors models are generally best served with vLLM when a
  CUDA GPU has enough available memory.
- A model that fits comfortably on a 24 GB GPU may not fit on a 4 GB GPU,
  especially with a large context window.

The goal is to make a safe, explainable selection at startup instead of relying
on a manually edited server command for each machine.

## 2. High-level architecture

```text
orchestrator
    |
    | gRPC: llm.v1.Llm/Generate
    v
Adaptive LLM facade                    Prometheus metrics
public address, e.g. :50053 ---------> <machine-ip>:9106/metrics
    |
    | private gRPC
    v
selected worker on 127.0.0.1:50054
    |
    +-- llama.cpp worker for GGUF
    |
    +-- vLLM worker for Safetensors, or a single-file GGUF
```

The facade is intentionally small. It plans the execution, starts or validates
the worker, then forwards each `Generate` request to the selected worker.
Workers implement the same gRPC contract, so the facade does not need to
translate protobuf messages.

## 3. Main pieces

| Component | File(s) | Responsibility |
| --- | --- | --- |
| Entrypoint and proxy | `main.py` | Builds the plan, starts the worker, serves the public gRPC endpoint, proxies requests, exports facade metrics. |
| Configuration | `config.py`, `.env`, `.env.example` | Reads environment settings and establishes defaults. |
| Model resolution and profiles | `model_store.py`, `model-profiles.json`, `registry.py` | Resolves Hugging Face references into the local cache and applies optional model-specific runtime profiles. |
| Hardware discovery | `hardware.py` | Detects host RAM, CPU information, and NVIDIA GPU memory through `nvidia-smi`. |
| Planner | `compatibility.py`, `estimator.py`, `planner.py` | Rejects unsafe engines, estimates memory use, scores the remaining candidates, and writes the selected plan. |
| Worker supervisor | `runtime_manager.py` | Starts/stops workers, applies the selected environment, and verifies readiness. |
| Engine workers | `engines/llamacpp_worker.py`, `engines/vllm_worker.py` | Load a model and generate replies using an engine-specific API. |
| Tests | `tests/test_planner.py` | Verifies planner decisions for representative GPU hardware. |

## 4. Startup flow

Run the service from the repository root:

```bash
./adaptive_llm/run.sh
```

`run.sh` loads `adaptive_llm/.env` without overwriting shell environment
variables already provided by the user. It creates the local virtual environment
when necessary, installs dependencies when `requirements.txt` changes, and runs
`python -m adaptive_llm.main`.

At startup, the Python service performs the following sequence:

1. Load configuration and optional JSON model profiles.
2. Resolve `LLM_MODEL_REF` into the service-owned local cache, downloading only when absent. The legacy `LLM_MODEL_ID` path remains available for local profiles.
3. Collect a static and dynamic hardware snapshot.
4. Build candidate engine plans for vLLM, SGLang, and `llama.cpp`.
5. Reject candidates that cannot safely serve the selected model.
6. Score compatible candidates and select the best one.
7. Write the explainable result to `adaptive_llm/run/current-plan.json`.
8. Start a managed worker, or check that an externally managed worker is ready.
9. Start the public gRPC facade and proxy requests to the worker.

SGLang is represented in the planner so its absence is explicit, but it is not
implemented yet and is always rejected.

## 5. Model resolution and profiles

`LLM_MODEL_REF` is the normal model selector. It accepts a Hugging Face
`owner/model` reference and stores fetched files under `adaptive_llm/models/huggingface/`.
Public models need no token; use `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`) for
private or gated models after accepting the model terms. For a GGUF repository
with multiple files, `LLM_MODEL_FILE` is required.

`model-profiles.json` is optional tuning data, not a model inventory users must
maintain. A profile can be automatically matched by `model_ref` or selected with
`LLM_MODEL_PROFILE`. It supplies known-safe sizing and runtime settings when
model metadata is incomplete, especially for GGUF files.

Important fields include:

- `format`: `safetensors` is vLLM-or-SGLang only; `gguf` can run on either
  `llama.cpp` or vLLM, and the planner scores the two.
- `model_path`: local GGUF file or local Hugging Face model directory.
- `tokenizer`: the repository a GGUF was converted from. A GGUF carries no chat
  template, so without one prompt rendering silently diverges from the model's
  instruction tuning. Optional only when the file records its own base model,
  which the worker falls back to.
- `parameter_count_b`, `file_size_bytes` when available, and quantization/dtype:
  used to estimate model weight memory.
- `layers`, `kv_heads`, `head_dim`, and `kv_cache_dtype_bytes`: used to estimate
  the KV cache for the planned context length.
- `max_context_tokens` and `default_context_tokens`: distinguish the model's
  theoretical upper limit from the practical default selected at startup.
- `runtime_profiles`: per-model configuration for known hardware ranges.

The supported bundled profiles currently cover:

- `medgemma_1_5_4b_q4`: a GGUF model intended for `llama.cpp`, including partial
  GPU offload on smaller cards.
- `medgemma_27b_it_gguf`: MedGemma 27B IT as a Q4_K_M GGUF for vLLM, paired with
  the tokenizer of `google/medgemma-27b-it`.
- `medgemma_27b_text_it_gguf`: the text-only MedGemma 27B variant, same shape,
  paired with `google/medgemma-27b-text-it`.
- `qwen3_8b_bf16`: a Safetensors model intended for vLLM on larger CUDA GPUs.
- `qwen3_0_6b_bf16`: a smaller vLLM model with a conservative GTX 4 GB profile.

Profiles are deliberately declarative: model-specific tuning belongs there,
not scattered across shell scripts or engine code.

## 6. Planning and resource safety

The planner first applies hard compatibility gates. A candidate is rejected when
its model format is wrong, CUDA is unavailable, the required Python engine is
not installed, the model path is missing, or estimated memory exceeds the safe
memory budget.

The safe budget reserves both a fraction of total memory and a minimum fixed
amount:

```text
safe VRAM = free VRAM - max(minimum headroom, total VRAM × VRAM headroom)
safe RAM  = available RAM - max(minimum headroom, total RAM × RAM headroom)
```

Defaults reserve 15% VRAM, 10% RAM, and at least 512 MiB. This avoids planning
against every last byte of memory, which is a common source of startup and
runtime out-of-memory failures.

The estimator accounts for:

- model weights;
- KV cache at the planned context length;
- engine runtime/scratch memory.

For GGUF models, the planner estimates how many transformer layers can be
offloaded to the GPU. It then chooses one of `FULL_GPU`, `PARTIAL_GPU`, or
`CPU_ONLY`. For vLLM, a compatible Safetensors model uses `FULL_GPU`.

Compatible candidates are scored by hardware fit, memory fit, expected engine
performance, model quantization/format, context suitability, and current load.
The selected plan is stored in `run/current-plan.json`; inspect it first when a
startup decision is surprising.

## 7. GPU selection

On a machine with several GPUs, Adaptive LLM chooses the CUDA GPU with the most
free VRAM and records its physical index as `gpu_index` in the execution plan.
The worker manager sets `CUDA_VISIBLE_DEVICES` to that physical GPU before
launching the worker. `llama.cpp` is additionally told to use its now-visible
GPU `0` through `LLM_MAIN_GPU=0`.

This matters because planning on one GPU and letting an engine silently use GPU
0 can otherwise cause an out-of-memory error or use a slower/busier card. Runtime
profiles are also matched against the selected GPU's total VRAM, not simply the
largest GPU in the machine.

## 8. Worker lifecycle and reliability behavior

By default, Adaptive LLM starts the selected engine as a child process. The
worker listens only on the private worker address, normally `127.0.0.1:50054`.
The public facade waits until that address is gRPC-ready before exposing the
public service.

The startup lifecycle has specific safeguards:

- If a managed worker exits while loading the model, startup fails immediately
  and reports its exit code instead of waiting for the full readiness timeout.
- If worker readiness fails for any reason, the facade stops the child process;
  failed startup does not leave an orphaned model process behind.
- If `ADAPTIVE_LLM_AUTO_START_WORKER=0`, the service still checks that the
  external worker is reachable before declaring itself ready.
- The facade and both workers treat a zero result from `add_insecure_port()` as
  a startup failure. A busy or invalid port therefore does not produce a false
  “listening” log message.
- The facade closes the private worker channel and stops the worker during
  shutdown.

These checks make startup failures actionable and prevent an unhealthy service
from accepting traffic only to fail on its first request.

## 9. Request handling

The public `Generate` RPC retains the existing request and response types:

```text
GenerateRequest  -> session_id, text
GenerateResponse -> session_id, text, ts_ms, proc_ms
```

For normal operation, the facade forwards the request directly to the selected
worker and returns its response. In dry-run mode, the facade returns a short
diagnostic reply describing the selected model, engine, and execution mode;
model weights are not loaded.

`llama.cpp` serializes generation through a one-thread executor because one
loaded `Llama` instance is used. vLLM uses its asynchronous engine and can take
advantage of its own scheduler/batching behavior.

## 10. Configuration reference

Place normal local settings in `adaptive_llm/.env`. Shell environment variables
take precedence over that file.

| Setting | Default | Meaning |
| --- | --- | --- |
| `LLM_MODEL_REF` | unset | Hugging Face `owner/model` reference; preferred model selector. |
| `LLM_MODEL_FILE` | unset | GGUF filename; required when its repository has multiple GGUF variants. |
| `LLM_MODEL_DIR` | `adaptive_llm/models` | Service-owned local model cache root. |
| `LLM_AUTO_DOWNLOAD` | `1` | Download a missing selected model from Hugging Face. |
| `HF_TOKEN` | unset | Optional Hugging Face token for gated/private models; never logged. |
| `LLM_TOKENIZER` | model path | Tokenizer/chat-template source. For a GGUF on vLLM, name the repository it was converted from; falls back to the profile, then to the base model the file records. |
| `LLM_MODEL_PROFILE` | unset | Optional JSON model-profile name. |
| `LLM_MODEL_ID` | `medgemma_1_5_4b_q4` | Legacy local profile entry to load. |
| `LLM_ADDR` | `[::]:50053` | Public facade gRPC listener. |
| `ADAPTIVE_LLM_WORKER_ADDR` | `127.0.0.1:50054` | Private worker gRPC listener. |
| `LLM_METRICS_HOST` | `0.0.0.0` | Facade Prometheus bind host. Set to `127.0.0.1` for local-only metrics. |
| `LLM_METRICS_PORT` | `9106` | Facade Prometheus port. |
| `LLM_ENGINE` | unset | Optional debugging override: `vllm` or `llama.cpp`. It does not bypass safety gates. |
| `ADAPTIVE_LLM_DRY_RUN` | `0` | Plan and serve a diagnostic response without loading weights. |
| `ADAPTIVE_LLM_AUTO_START_WORKER` | `1` | Start the engine worker as a child process. Set to `0` only when another process manages it. |
| `ADAPTIVE_LLM_WORKER_READY_TIMEOUT_S` | `900` | Worker startup/readiness timeout in seconds. |
| `ADAPTIVE_LLM_VRAM_HEADROOM` | `0.15` | Fraction of total VRAM left unplanned. |
| `ADAPTIVE_LLM_RAM_HEADROOM` | `0.10` | Fraction of total RAM left unplanned. |
| `ADAPTIVE_LLM_MIN_HEADROOM_MIB` | `512` | Minimum RAM/VRAM reserve. |
| `ADAPTIVE_LLM_MAX_INFLIGHT_REQUESTS` | `4` | Maximum requests executing through the facade at once. |
| `ADAPTIVE_LLM_MAX_QUEUED_REQUESTS` | `16` | Maximum additional admitted requests waiting for an execution slot. Excess requests receive gRPC `RESOURCE_EXHAUSTED`. |
| `ADAPTIVE_LLM_RESOURCE_METRICS_INTERVAL_S` | `5` | Interval for live CPU/RAM/VRAM metric collection. |
| `ADAPTIVE_LLM_PROFILE_BANDWIDTH` | `0` | Run short CPU-memory and PCIe host-to-GPU bandwidth benchmarks at startup. This is opt-in because it briefly allocates memory and uses the GPU. |
| `LLM_MAX_MODEL_LEN` | model/profile default | Planned maximum context/model length. |
| `LLM_MAX_TOKENS` | model/profile default | Maximum generated output tokens. |
| `LLM_TEMPERATURE` | engine default | Sampling temperature. |
| `LLM_LANGUAGE_MODEL_ONLY` | `1` | Disable image/video inputs in vLLM. The current gRPC contract is text-only, so this prevents unused multimodal encoder/cache allocation. |

Engine-specific settings such as `LLM_N_BATCH`, `LLM_N_GPU_LAYERS`,
`LLM_GPU_MEMORY_UTILIZATION`, and `LLM_ENFORCE_EAGER` are passed to their
respective worker. Prefer a `runtime_profiles` entry in `model-profiles.json` for tuning
that should be consistently applied to a model/hardware combination.

## 11. Runtime profiles

Runtime profiles let a model select safe engine environment variables based on
the selected GPU's VRAM. For example, the Qwen 0.6B profile for a 4 GB GPU lowers
vLLM's GPU memory utilization, limits batched tokens, and enables eager mode.

Profiles are evaluated in the order they appear in the model entry. A profile
applies only when its optional engine and VRAM constraints match. Profile values
use `setdefault`, so an explicitly supplied user environment variable can still
override a profile while debugging.

## 12. Metrics and observability

The facade exposes Prometheus metrics on:

```text
http://<machine-ip>:9106/metrics
```

The selected worker exposes its own metrics endpoint:

- vLLM worker: `9114` by default.
- llama.cpp worker: `9115` by default.

Useful facade metrics include proxy latency, successful proxied replies, errors,
the selected plan score, and selected model/engine/mode information. Worker
metrics include generation latency, output tokens, replies, and generation
errors.

The facade additionally exports admission-control metrics (`active`, `queued`,
and rejected requests), queue latency, input size, and sampled available RAM,
CPU utilization, GPU VRAM, and GPU utilization. Workers expose prompt size,
TTFT, approximate prefill time, decode duration, and decode tokens per second.
For the unary gRPC API, TTFT is measured internally at the first generated
chunk; the response is still returned only after generation finishes. Prefill is
reported as a TTFT-based approximation because engines do not expose a portable
separate prefill-completion callback. When bandwidth profiling is enabled, the
facade also exports the measured CPU memory-copy and PCIe host-to-device
bandwidth values, while the hardware snapshot records PCI bus ID and active
link generation/width when `nvidia-smi` provides them.

## 13. Typical operations

### Start with the default model

```bash
cp adaptive_llm/.env.example adaptive_llm/.env
./adaptive_llm/run.sh
```

The default configuration downloads Qwen3 0.6B if it is absent from the local cache.

### Select another registered model

```bash
LLM_MODEL_REF=Qwen/Qwen3-8B ./adaptive_llm/run.sh
```

### Use another environment file

```bash
ADAPTIVE_LLM_ENV_FILE=/path/to/custom.env ./adaptive_llm/run.sh
```

### Validate planning without loading weights

```bash
ADAPTIVE_LLM_DRY_RUN=1 ./adaptive_llm/run.sh
```

### Use different public and worker ports

```bash
LLM_ADDR=127.0.0.1:50063 \
ADAPTIVE_LLM_WORKER_ADDR=127.0.0.1:50064 \
./adaptive_llm/run.sh
```

## 14. Validation and troubleshooting

Run the local checks from the repository root:

```bash
bash -n adaptive_llm/run.sh
python -m compileall adaptive_llm
python -m adaptive_llm.tests.test_planner
```

The planner tests cover the intended decisions for a GTX 1650 4 GB, an RTX 4090
24 GB, GPU-selection behavior on a multi-GPU host, and missing model-path
rejection.

When troubleshooting a failed start:

1. Read the startup log; it prints the selected plan and any rejection reasons.
2. Open `adaptive_llm/run/current-plan.json` to see the selected engine,
   context length, memory estimate, GPU index, and rejected alternatives.
3. Confirm that the cached `model_path` is present and has the required type: a GGUF file
   for `llama.cpp`, or a model directory for vLLM.
4. Check `nvidia-smi` for free VRAM and competing processes.
5. Increase headroom or reduce `LLM_MAX_MODEL_LEN` if the model is close to the
   available-memory boundary.
6. Check that public, worker, and metrics ports are not already occupied.

## 15. Current scope and limitations

- SGLang selection is reserved but its worker adapter is not implemented.
- Hugging Face safetensors directories and explicitly selected GGUF files are
  supported. Other weight formats are rejected before a worker starts.
- The planner makes conservative estimates. Actual engine behavior can vary with
  CUDA versions, model architecture, quantization implementation, batching, and
  other processes using the GPU.
- `llama.cpp` generation is serialized for correctness with its shared model
  instance; high-throughput use cases should prefer a suitable vLLM deployment.

Despite those boundaries, the service provides one stable local API, explicit
engine selection, an inspectable plan, and safer failure behavior across the
supported GGUF and Safetensors workflows.
