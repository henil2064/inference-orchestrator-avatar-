# Setup

This file is the root setup guide for developers working on the full inference
pipeline. Service-specific setup details live beside each service:

- [media-edge/SETUP.md](media-edge/SETUP.md)
- [orchestrator/SETUP.md](orchestrator/SETUP.md)
- [stt/SETUP.md](stt/SETUP.md)
- [adaptive_llm/SETUP.md](adaptive_llm/SETUP.md)
- [tts/SETUP.md](tts/SETUP.md)
- [monitoring/SETUP.md](monitoring/SETUP.md)
- [pipeline_test/SETUP.md](pipeline_test/SETUP.md)

## Prerequisites

Install these before configuring the services:

- Docker Engine with Compose v2.
- Python 3.12 with `python3-venv` and `pip`.
- Go matching [media-edge/go.mod](media-edge/go.mod).
- Build tools: `build-essential`, `cmake`, `git`, `pkg-config`, `libopus-dev`,
  and `libopusfile-dev` on Ubuntu/Debian.
- NVIDIA driver and CUDA-compatible runtime for GPU inference. The target
  production topology is two NVIDIA L40S GPUs: LLM on GPU0, STT and TTS on
  GPU1.
- Local model artifacts for STT, TTS, and adaptive LLM, or network access and
  credentials to download them during service setup.

On Ubuntu/Debian, the media-edge helper installs the Go toolchain, Opus
dependencies, and ONNX Runtime used by edge VAD:

```bash
./media-edge/setup_ubuntu.sh
```

## Environment Files

Copy the tracked examples before first run:

```bash
cp media-edge/.env.example media-edge/.env
cp orchestrator/.env.example orchestrator/.env
cp stt/.env.example stt/.env
cp adaptive_llm/.env.example adaptive_llm/.env
cp tts/.env.example tts/.env
```

The `.env` files are ignored by Git. Shell environment variables override each
service `.env`. Docker Compose also overrides paths and service hostnames that
must differ inside containers.

For host-process development, set `REDIS_URL=redis://127.0.0.1:6380/0` in
`orchestrator/.env` after copying the example. Compose services use
`redis://redis:6379/0` internally.

## Infrastructure

For the host-process development flow, start infrastructure containers first:

```bash
docker compose up -d redis clickhouse prometheus grafana
```

Useful local endpoints:

| Service | URL |
| --- | --- |
| Redis | `redis://127.0.0.1:6380/0` from host processes, `redis://redis:6379/0` in Compose |
| ClickHouse | `http://127.0.0.1:8123` |
| Prometheus | `http://localhost:9090` |
| Grafana | `http://localhost:3000` |

Grafana defaults to `admin`/`admin` unless `GRAFANA_ADMIN_USER` and
`GRAFANA_ADMIN_PASSWORD` are set.

## One-Time Service Setup

Set up each service once:

```bash
./adaptive_llm/run.sh
./stt/run.sh
./tts/run.sh
./orchestrator/run.sh
./media-edge/run.sh
```

Each Python launcher creates its own virtual environment and installs
dependencies. Stop each service after its first successful startup if you are
only preparing the machine. The separate virtual environments are intentional:
the CUDA/Torch stacks for STT, TTS, adaptive LLM, and orchestration should not
share one environment.

For the default STT Transformers runtime, install CUDA PyTorch in
`stt/.venv-sravaani` and set `SRAVAANI_MODEL` or `STT_MODEL_DIR` to a local
SraVaani directory. For adaptive LLM, set `LLM_MODEL_REF` and optional
`HF_TOKEN`. For TTS, set `TTS_MODEL_DIR`, `TTS_PROMPT_WAV`, and
`TTS_PROMPT_TEXT`.

## Run The Full Pipeline

The current host-process launcher is:

```bash
python3 scripts/run_pipeline.py
```

It starts adaptive LLM, STT replicas, TTS replicas, the orchestrator, and
media-edge. It detects hardware, computes replica/thread counts unless
overridden, writes logs to `run/logs/`, and updates Prometheus file-discovery
targets under `monitoring/targets/`.

Preview the computed plan without launching services:

```bash
python3 scripts/run_pipeline.py --dry-run
```

Launch only selected service groups while iterating:

```bash
python3 scripts/run_pipeline.py --only stt,tts
python3 scripts/run_pipeline.py --no-edge
```

Common overrides:

```bash
STT_REPLICAS=2 TTS_REPLICAS=2 python3 scripts/run_pipeline.py
STT_THREADS_PER_REPLICA=4 TTS_THREADS_PER_REPLICA=4 python3 scripts/run_pipeline.py
RESERVED_CONTROL_CORES=4 python3 scripts/run_pipeline.py
```

The media edge prints the browser URL. HTTPS is the default because browsers
require a secure origin for microphone access from another device. Accept the
self-signed certificate once during local development.

## Manual Run Order

When debugging one service at a time, start services in this order:

```bash
docker compose up -d redis clickhouse prometheus grafana
./adaptive_llm/run.sh
./stt/run.sh
./tts/run.sh
./orchestrator/run.sh
./media-edge/run.sh
```

Default ports:

| Component | App port | Metrics |
| --- | --- | --- |
| media-edge | `8443` | `9101` |
| orchestrator | `50051` | `9102` |
| stt | `50052` | `9103` |
| adaptive_llm | `50053` | `9106`, worker `9114` |
| tts | `50054` | `9104` |

## Docker Compose Run

To run the containerized stack:

```bash
docker compose build
docker compose up -d
docker compose ps
```

Open `https://localhost:8443` for the media edge. See [docs/DOCKER.md](docs/DOCKER.md)
for model volume paths, GPU settings, and container-specific configuration.

Stop containers while keeping Redis, ClickHouse, Prometheus, and Grafana data:

```bash
docker compose down
```

Do not use `docker compose down -v` unless local state and telemetry volumes can
be discarded.

## Verification

Run focused checks after setup or a docs-affecting service change:

```bash
(cd media-edge && go test ./...)
(cd orchestrator && python -m tests.test_endpoint && python -m tests.test_llm_routing)
(cd stt && python -m tests.test_segment)
(cd adaptive_llm && python -m pytest tests -q)
(cd tts && python -m pytest tests -q)
```

Run a browser session through media-edge for manual end-to-end validation, or
use the WAV harness in [pipeline_test/SETUP.md](pipeline_test/SETUP.md).

## Common Gotchas

- Regenerate stubs after any `proto/*.proto` change, then restart both Python
  services and the Go edge:

  ```bash
  bash scripts/gen-proto.sh
  ```

- Keep service virtual environments separate. Sharing a venv across ML services
  often causes pip to re-resolve incompatible Torch/CUDA stacks.
- Keep model weights and local `.env` files out of Git.
- Pin GPU placement explicitly with `CUDA_VISIBLE_DEVICES` on multi-GPU hosts.
- Redis is required for orchestrator session state. ClickHouse can be disabled
  for development telemetry, but Redis cannot.
- If Grafana targets are down, confirm the pipeline services are running and
  that `scripts/run_pipeline.py` has refreshed `monitoring/targets/stt.json`
  and `monitoring/targets/tts.json`.
