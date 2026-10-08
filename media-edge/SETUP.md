# Media Edge Setup

## Prerequisites

- Go matching [go.mod](go.mod).
- `pkg-config`, `libopus-dev`, and `libopusfile-dev`.
- A running orchestrator, normally at `localhost:50051`.
- ONNX Runtime shared library for Silero VAD. Without it, the service starts
  with RMS VAD fallback.

## Ubuntu Setup

From the repository root:

```bash
./media-edge/setup_ubuntu.sh
```

The script installs the pinned Go toolchain, Opus build dependencies, ONNX
Runtime, and builds the service. It installs Go under `/usr/local/go`. If the
current shell cannot find `go` after setup:

```bash
export PATH="/usr/local/go/bin:$PATH"
```

If setup reports that `opusfile` is missing:

```bash
sudo apt-get update
sudo apt-get install -y libopusfile-dev
./media-edge/setup_ubuntu.sh
```

## Configure

From the repository root:

```bash
cp media-edge/.env.example media-edge/.env
```

Important settings:

| Variable | Purpose |
| --- | --- |
| `EDGE_ADDR` | HTTPS/HTTP listener, default `:8443`. |
| `EDGE_ORCHESTRATOR` | Orchestrator gRPC target, default `localhost:50051`. |
| `EDGE_METRICS_ADDR` | Prometheus metrics listener, default `:9101`. |
| `EDGE_NO_TLS` | Set `1` only for localhost HTTP development. |
| `EDGE_CERT_DIR` | Self-signed development certificate directory. |
| `EDGE_ONNX_LIB` | ONNX Runtime shared library for Silero VAD. |
| `EDGE_VAD_MODEL` | Silero ONNX model path. |
| `EDGE_PLAYOUT_PREBUFFER_MS` | Agent-audio jitter cushion. |
| `EDGE_PLAYOUT_MAX_MS` | Maximum buffered agent audio. |
| `EDGE_AVATAR_TARGET` | Avatar service gRPC target, e.g. `localhost:50056`. Unset disables video. |
| `EDGE_AVATAR_RENDER_MARGIN_MS` | Extra agent-audio delay on top of the avatar's lookahead, default `200`. |

CLI flags override shell variables and `.env` values.

## Run

```bash
./media-edge/run.sh
```

The service prints the browser URL. HTTPS is the default. Accept the
self-signed certificate warning once during development, then grant microphone
access in the browser.

For localhost-only HTTP development:

```bash
EDGE_NO_TLS=1 ./media-edge/run.sh
```

Or run the Go command directly while iterating:

```bash
cd media-edge
go run ./cmd/media-edge -orchestrator localhost:50051
```

## Verify

```bash
bash -n media-edge/run.sh
(cd media-edge && go test ./...)
```

For a full manual check, start the pipeline, open the printed URL, click start,
speak, and confirm the browser receives transcript/reply data plus agent audio.

## Operational Notes

- Rebuild or restart media-edge after any Go or generated proto change.
- `scripts/run_pipeline.py` builds a real `run/bin/media-edge` binary before
  supervising the service.
- Keep HTTPS enabled when testing from another device. Browsers require a
  secure origin for remote microphone access.
- If Silero fails to load, startup logs identify the fallback VAD mode.
