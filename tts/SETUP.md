# Text-to-Speech Setup

## Prerequisites

- Python 3.12 with `python3-venv`.
- Local DhVaani or GGUF/MioTTS model artifacts under `tts/models/`, or another
  ignored local path.
- A clean 3-10 second mono reference WAV and its exact transcript.
- CUDA-capable PyTorch and NVIDIA driver for GPU-backed PyTorch inference.
  CPU and llama.cpp-backed GGUF inference are supported.

## Configure

From the repository root:

```bash
cp tts/.env.example tts/.env
```

Set the model and voice prompt:

```bash
TTS_MODEL_DIR=./models/DhVaani-0.5-fp16
TTS_PROMPT_WAV=/path/to/reference.wav
TTS_PROMPT_TEXT=Reference voice transcript here
```

A path ending in `.gguf` selects the llama.cpp/GGUF backend unless
`TTS_FORMAT` overrides auto-detection. For the GGUF backend, `TTS_PROMPT_TEXT`
is unused; the reference WAV still provides the speaker embedding.

Important settings:

| Variable | Purpose |
| --- | --- |
| `TTS_ADDR` | gRPC listener, default `[::]:50054`. |
| `TTS_METRICS_PORT` | Prometheus metrics port, default `9104`. |
| `TTS_MODEL_DIR` | Model directory or `.gguf` file. |
| `TTS_FORMAT` | Optional override: `gguf` or `pytorch`. |
| `TTS_DEVICE` | `auto`, `cpu`, or `cuda`. |
| `TTS_N_GPU_LAYERS` | GGUF backend GPU offload layer count. |
| `TTS_NUM_THREADS` | CPU worker thread tuning. |
| `TTS_NUM_STEP` | DhVaani quality/latency setting. |
| `TTS_MAX_TEXT_CHARS` | Request text size cap. |
| `TTS_MAX_QUEUE` | Maximum queued synthesis requests. |
| `TTS_QUEUE_TIMEOUT_S` | Maximum wait for the synthesis worker. |
| `TTS_WEDGE_TIMEOUT_S` | Health timeout for a stuck synthesis worker. |

## Run

```bash
./tts/run.sh
```

The launcher creates `tts/.venv`, installs dependencies when
`requirements.txt` changes, loads `tts/.env`, caches the reference voice, warms
the model when `TTS_WARMUP=1`, and starts gRPC on `TTS_ADDR`.

Start TTS before the orchestrator. When using `scripts/run_pipeline.py`,
replica ports and CPU thread counts are assigned automatically from the
computed plan.

## Verify

```bash
bash -n tts/run.sh
(cd tts && python -m pytest tests -q)
```

With the service running, synthesize a WAV through gRPC:

```bash
cd tts
.venv/bin/python -m tests.test_client "Hello, how can I help you today?" --out reply.wav
```

Use `--target host:port` when the service is on another machine.

## Profiling

`tests.test_client` can run warm-ups and repeated sequential measurements:

```bash
cd tts
.venv/bin/python -m tests.test_client \
  "Hypertension is persistent high blood pressure. It should be monitored." \
  --expected-device cuda --warmup 1 --runs 5 \
  --out tts-profile-last.wav
```

It prints JSON per run plus mean, p50, and p95 summaries. New profiling output
should normally go to an ignored path.

## Operational Notes

- A GPU request that cannot be honored should fail at startup instead of
  silently falling back to CPU.
- The initial model and vocoder load can use the Hugging Face cache; populate it
  before offline deployment.
- Regenerate gRPC stubs after changes to `proto/tts.proto`:

  ```bash
  bash scripts/gen-proto.sh
  ```

- Keep local `.env`, `.venv/`, and model artifacts out of Git.
