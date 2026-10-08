# Adaptive LLM Setup

## Prerequisites

- Linux or WSL with Python 3.12 and `python3-venv`.
- Build tools such as `build-essential` and `cmake` for llama.cpp-based
  dependencies.
- NVIDIA driver and CUDA runtime for GPU-backed vLLM or SGLang.
- A Hugging Face model reference or local GGUF/model files under
  `adaptive_llm/models/`.
- Optional `HF_TOKEN` for private or gated Hugging Face models.

## Configure

From the repository root:

```bash
cp adaptive_llm/.env.example adaptive_llm/.env
```

Set the model selector:

```bash
LLM_MODEL_REF=Qwen/Qwen3-0.6B
LLM_AUTO_DOWNLOAD=1
```

For a GGUF repository with multiple files, set `LLM_MODEL_FILE` to the exact
`.gguf` file. For a private or gated model, set `HF_TOKEN` in the shell or in
`adaptive_llm/.env`.

A GGUF is not automatically a llama.cpp model: vLLM serves a single-file GGUF
too, and on a CUDA GPU the planner normally picks it. That path also needs the
repository the GGUF was converted from, because the file carries no usable chat
template — from `LLM_TOKENIZER`, a model profile's `tokenizer`, or the base
model the GGUF itself records. Those repositories are often gated, so accept
the model's terms and set `HF_TOKEN`. See
[docs/VLLM_GGUF.md](docs/VLLM_GGUF.md).

Important settings:

| Variable | Purpose |
| --- | --- |
| `LLM_MODEL_REF` | Hugging Face `owner/model` reference. |
| `LLM_MODEL_FILE` | GGUF filename when a repository has several model files. |
| `LLM_TOKENIZER` | Tokenizer/chat-template repository for a GGUF on vLLM; overrides the profile and the GGUF's own base model. |
| `LLM_MODEL_DIR` | Local model cache root. |
| `LLM_ENGINE` | Optional override: `vllm`, `sglang`, or `llama.cpp`. |
| `LLM_ADDR` | Public gRPC address, default `[::]:50053`. |
| `ADAPTIVE_LLM_WORKER_ADDR` | Internal worker gRPC address. |
| `ADAPTIVE_LLM_MAX_INFLIGHT_REQUESTS` | Concurrent generation cap. |
| `ADAPTIVE_LLM_MAX_QUEUED_REQUESTS` | Queue cap before rejection. |
| `LLM_PROMPT_ID` | Prompt catalog entry used for generation. |
| `LLM_MAX_TOKENS` | Output token budget. |

Keep model-specific runtime tuning in `model-profiles.json`; use `.env` for
developer and machine overrides.

## Run

```bash
./adaptive_llm/run.sh
```

The launcher creates `adaptive_llm/.venv`, installs dependencies when
`requirements.txt` or `requirements-overrides.txt` changes, loads
`adaptive_llm/.env`, starts the worker, and then exposes the public gRPC
service. The selected runtime plan is written to
`adaptive_llm/run/current-plan.json`.

`requirements-overrides.txt` holds packages deliberately installed at a version
one of the pinned dependencies forbids, and is applied with `--no-deps` after
`requirements.txt` — pip cannot resolve the two together. Installing only
`requirements.txt` by hand leaves GGUF-on-vLLM broken; see
[docs/VLLM_GGUF.md](docs/VLLM_GGUF.md).

Useful modes:

```bash
ADAPTIVE_LLM_DRY_RUN=1 ./adaptive_llm/run.sh
LLM_MODEL_REF=Qwen/Qwen3-0.6B ./adaptive_llm/run.sh
ADAPTIVE_LLM_SKIP_INSTALL=1 ./adaptive_llm/run.sh
```

Start this service before the orchestrator.

## SGLang Notes

SGLang should run in a separate virtual environment or managed process because
its CUDA/Torch dependency stack can conflict with the bundled vLLM runtime.

Use one of these:

```bash
LLM_ENGINE=sglang LLM_SGLANG_PYTHON=/path/to/sglang-venv/bin/python ./adaptive_llm/run.sh
LLM_ENGINE=sglang LLM_SGLANG_URL=http://127.0.0.1:30000 ./adaptive_llm/run.sh
```

Keep `LLM_ENABLE_PREFIX_CACHING=1` unless you are diagnosing cache behavior.
If FlashInfer fails while building kernels, compare `nvcc --version` with the
installed CUDA toolkits and set `LLM_SGLANG_CUDA_HOME=/usr/local/cuda-X.Y`.

## Verify

```bash
bash -n adaptive_llm/run.sh
(cd adaptive_llm && python -m pytest tests -q)
```

Benchmark conversational cache behavior with:

```bash
adaptive_llm/.venv/bin/python adaptive_llm/tools/conversation_cache_bench.py --include-cache-off
```

## Operational Notes

- The service disables unused multimodal encoders by default with
  `LLM_LANGUAGE_MODEL_ONLY=1`.
- On pre-Ampere GPUs, the worker may auto-disable vLLM features that require
  unsupported attention/head configurations.
- Do not clear SGLang cache between turns during conversation tests; radix
  prefix cache reuse is part of the runtime behavior being measured.
- Prometheus exposes facade metrics on `LLM_METRICS_PORT` and worker metrics on
  `LLM_WORKER_METRICS_PORT`.
