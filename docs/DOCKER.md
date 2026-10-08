# Docker deployment

The active pipeline uses one container image per service:

```text
media-edge -> orchestrator -> stt
                         -> adaptive-llm
                         -> tts
```

Redis and ClickHouse run as separate infrastructure containers. Model weights
and local secrets are not copied into images; Compose mounts model directories
and optionally loads each service's local `.env` file.

Prometheus and Grafana are also wired into Compose for local monitoring.
Prometheus scrapes the service metrics ports over the Compose network, and
Grafana provisions the checked-in dashboards plus local Prometheus and
ClickHouse data sources.

## Prerequisites

- Docker Engine with Compose v2.
- NVIDIA Container Toolkit for STT, TTS, and adaptive-LLM GPU access.
- Local model files in the existing `stt/models`, `tts/models`, and
  `adaptive_llm/models` directories.
- Optional service settings in `media-edge/.env`, `orchestrator/.env`,
  `stt/.env`, `tts/.env`, and `adaptive_llm/.env`.

## Configuration convention

The service `.env` files are for normal settings such as logging, tuning, and
credentials. Compose supplies the values that must differ in a container:
service-to-service hostnames and paths under `/app`.

Do not put an absolute host path (for example `/home/name/...`) in a service
`.env` when starting the Docker stack. Compose overrides the standard container
paths below, but any extra local path variables in `.env` must also point inside
the container or be removed.

| Service | Host location | Location inside its container |
| --- | --- | --- |
| STT | `stt/models` | `/app/stt/models` |
| TTS | `tts/models` | `/app/tts/models` |
| Adaptive LLM | `adaptive_llm/models` | `/app/adaptive_llm/models` |

The checked-in Compose defaults select SraVaani ONNX INT8 for STT,
`tts/models/indic-mio-q8/indic-mio-q8.gguf`, the bundled `tts/sample/harvard.wav`
prompt, and `LLM_MODEL_REF=Qwen/Qwen3-0.6B` cached under
`adaptive_llm/models`. The standard deployment assigns the GPU to Adaptive
LLM/vLLM; STT uses the selected ONNX model on CPU, and TTS uses llama.cpp on
CPU. Change the corresponding Compose environment value if you intentionally
use another model.

## Build and run

From the repository root:

```bash
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f orchestrator
```

The media edge is available on port `8443`; the orchestrator is bound to
localhost on `50051`. STT, TTS, and LLM ports stay internal to the Compose
network. Redis and ClickHouse data use named volumes.

Prometheus is available at `http://localhost:9090`; open
`http://localhost:9090/targets` to confirm scrape status. Grafana is available
at `http://localhost:3000` with the default local credentials `admin`/`admin`
unless you override `GRAFANA_ADMIN_USER` and `GRAFANA_ADMIN_PASSWORD`. The
dashboards are provisioned under the **Inference Pipeline** folder. If you only
want to start the monitoring containers first, run:

```bash
docker compose up -d prometheus grafana
```

Prometheus targets will stay `DOWN` until the inference services are started.

Media Edge serves HTTPS by default using its persisted self-signed development
certificate. On another device, open `https://<host-LAN-IP>:8443` and accept or
trust that certificate before granting microphone access. Set `EDGE_NO_TLS=1`
only for local HTTP development.

To stop the stack while retaining data:

```bash
docker compose down
```

Do not use `docker compose down -v` unless the Redis and ClickHouse data can be
discarded.

## Runtime choices

Compose builds the STT image for the ONNX runtime by default. To build the
Transformers runtime instead:

```bash
STT_RUNTIME=transformers docker compose build stt
STT_RUNTIME=transformers docker compose up -d stt
```

The Compose GPU reservations assume an NVIDIA host. For CPU-only development,
remove the `deploy.resources.reservations.devices` block from the inference
services and use CPU-compatible dependency builds.

## Image boundaries

- `docker/media-edge.Dockerfile`: compiled Go edge gateway.
- `docker/orchestrator.Dockerfile`: lightweight Python routing service.
- `docker/stt.Dockerfile`: SraVaani inference runtime.
- `docker/tts.Dockerfile`: DhVaani inference runtime.
- `docker/adaptive-llm.Dockerfile`: adaptive LLM facade and worker runtime.

The Dockerfiles intentionally keep model files outside the image so rebuilding
application code does not duplicate large weights or credentials.
