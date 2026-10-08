# Pipeline Test Setup

## Prerequisites

- Python 3.12 with `python3-venv`.
- The normal pipeline running: Redis, ClickHouse, STT, adaptive LLM, TTS,
  orchestrator, and media-edge.
- WAV recordings to replay.

## Setup

From the repository root:

```bash
python3 -m venv pipeline_test/.venv
pipeline_test/.venv/bin/python -m pip install -r pipeline_test/requirements.txt
```

Put input recordings in `pipeline_test/recordings/`. The harness replays every
`*.wav` file in alphabetical order unless another directory is supplied.

## Run

The default expects media-edge at its standard local self-signed HTTPS endpoint
and disables TLS verification for that development certificate:

```bash
pipeline_test/.venv/bin/python pipeline_test/stream_wav.py
```

Use a different recording directory and write results:

```bash
pipeline_test/.venv/bin/python pipeline_test/stream_wav.py \
  --wav-dir ./recordings \
  --output pipeline_test/results.json
```

If media-edge is running without TLS:

```bash
pipeline_test/.venv/bin/python pipeline_test/stream_wav.py --edge-url http://127.0.0.1:8443
```

`--playout-idle-s` controls how long the inbound agent track must be quiet
before the next turn begins. Increase it for bursty remote RTP delivery.

## Recent Performance Report

`report_recent_sessions.py` reads ClickHouse over HTTP using the Python standard
library and prints aggregate Markdown tables:

```bash
python3 pipeline_test/report_recent_sessions.py --sessions 2
```

Useful filters:

```bash
python3 pipeline_test/report_recent_sessions.py \
  --sessions 10 \
  --since-hours 24 \
  --engine vllm \
  --device cuda \
  --output pipeline_test/recent-performance.md
```

The report includes p50/p95 and average response latency, turn funnel,
interrupt/cancellation events, backend work, and LLM prefix-cache reuse.
Service timings may overlap, so do not add component rows to produce an
end-to-end total.

## Load Tools

Use the load tools only after the normal pipeline and monitoring stack are
running:

```bash
pipeline_test/.venv/bin/python pipeline_test/load_burst.py --help
pipeline_test/.venv/bin/python pipeline_test/open_loop_load.py --help
```

Prometheus and Grafana should show admitted, rejected, and saturated work while
these tools run.
