# Speech-to-Text Setup

## Prerequisites

- Python 3.12 with `python3-venv`.
- A SraVaani model directory for the Transformers runtime, including
  `model-asr.fp16.ts` and the accompanying model files.
- CUDA-capable PyTorch and a compatible NVIDIA driver for GPU inference.
- CPU inference is available with `STT_DEVICE=cpu` or the ONNX CPU provider.

## Configure

From the repository root:

```bash
cp stt/.env.example stt/.env
```

For the default Transformers runtime, set:

```bash
STT_RUNTIME=transformers
SRAVAANI_MODEL=/path/to/SraVaani_portable/model
```

`STT_MODEL_DIR` is the preferred alias and overrides `SRAVAANI_MODEL` when set.

For the ONNX runtime:

```bash
STT_RUNTIME=onnx
STT_ONNX_VARIANT=int8
STT_ONNX_PROVIDER=auto
```

`STT_ONNX_VARIANT` supports `int8`, `fp8`, and `fp16`. `STT_ONNX_PROVIDER`
supports `auto`, `cpu`, and `cuda`. `STT_ONNX_MODEL_DIR` is optional; the
service defaults to bundled model directories under `stt/models/`.

Important settings:

| Variable | Purpose |
| --- | --- |
| `STT_ADDR` | gRPC listener, default `[::]:50052`. |
| `STT_METRICS_PORT` | Prometheus metrics port, default `9103`. |
| `STT_DEVICE` | `auto`, `cpu`, or `cuda` for the Transformers runtime. |
| `CUDA_VISIBLE_DEVICES` | GPU pinning on multi-GPU hosts. |
| `STT_CPU_THREADS` | CPU thread count for CPU runtimes and replicas. |
| `STT_MAX_SEGMENT_SEC` | Maximum speech window before cutting. |
| `STT_SEGMENT_GRACE_SEC` | Extra quiet-point search window before forced cut. |
| `STT_MAX_INFLIGHT` | Bound on queued and running model work. |
| `STT_WEDGE_TIMEOUT_S` | Health timeout for a stuck model forward. |
| `STT_TRANSCRIPT_LOG` | `none`, `partials`, `finals`, or `both`. |

Transcript logging can expose spoken content. Keep `STT_TRANSCRIPT_LOG=none`
unless debugging requires text output.

## Run

```bash
./stt/run.sh
```

The launcher creates `stt/.venv-sravaani`, installs the active dependency set,
loads `stt/.env`, checks the runtime, warms the model, and starts gRPC on
`STT_ADDR`.

Install CUDA PyTorch in the STT venv when using GPU Transformers inference:

```bash
stt/.venv-sravaani/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Start STT before the orchestrator. When using `scripts/run_pipeline.py`, replica
ports and CPU thread counts are assigned automatically from the computed plan.

## Verify

```bash
bash -n stt/run.sh
(cd stt && python -m tests.test_segment)
```

Run backend and concurrency tools when changing model runtime behavior:

```bash
stt/.venv-sravaani/bin/python stt/scripts/bench_backends.py
stt/.venv-sravaani/bin/python stt/scripts/bench_concurrency.py
```

## Operational Notes

- The first startup can be slow because model loading and warm-up are paid
  before the service reports ready.
- Regenerate gRPC stubs after changes to `proto/asr.proto`:

  ```bash
  bash scripts/gen-proto.sh
  ```

- Keep model artifacts under `stt/models/` or another ignored local path.
- On multi-GPU hosts, pin replicas with `CUDA_VISIBLE_DEVICES` when launching
  manually.
- ONNX still requires PyTorch for SraVaani preprocessing artifacts.
