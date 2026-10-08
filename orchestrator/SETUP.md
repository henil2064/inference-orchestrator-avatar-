# Orchestrator Setup

## Prerequisites

- Python 3.12 with `python3-venv`.
- Redis. The full development stack starts it from the repository root with
  `docker compose up -d redis`.
- Running STT, adaptive LLM, and TTS services for the full pipeline.
- ClickHouse is optional but enabled by default in `.env.example` for
  development telemetry.

## Configure

From the repository root:

```bash
cp orchestrator/.env.example orchestrator/.env
```

For host-process development with the repository Compose Redis service, set:

```bash
REDIS_URL=redis://127.0.0.1:6380/0
```

Check these values first:

| Variable | Purpose |
| --- | --- |
| `REDIS_URL` | Required session store. Host-process default should usually be `redis://127.0.0.1:6380/0`. |
| `STT_TARGET` / `STT_TARGETS` | Single STT endpoint or comma-separated replica pool. |
| `LLM_TARGET` | Adaptive LLM endpoint. |
| `TTS_TARGET` / `TTS_TARGETS` | Single TTS endpoint or comma-separated replica pool. |
| `ORCH_MAX_SESSIONS` | Session admission cap. |
| `ORCH_MAX_ACTIVE_TURNS` | Concurrent LLM/TTS turn cap. |
| `ORCH_TURN_DEADLINE_S` | End-to-end turn deadline after endpointing. |
| `CLICKHOUSE_ENABLED` | Enables append-only telemetry writes. |

Set `LLM_ENABLED=0` or `TTS_ENABLED=0` only when intentionally bringing up a
partial pipeline.

## Run

```bash
./orchestrator/run.sh
```

The launcher creates `orchestrator/.venv`, installs dependencies when
`requirements.txt` changes, loads `orchestrator/.env`, and starts gRPC on
`ORCH_ADDR` or `[::]:50051`.

When using the Phase 3 launcher, do not set `STT_TARGETS` and `TTS_TARGETS`
manually unless you are overriding the computed plan:

```bash
python3 scripts/run_pipeline.py --only orchestrator
```

The full launcher sets replica target lists for the orchestrator automatically.

## Verify

```bash
bash -n orchestrator/run.sh
(cd orchestrator && python -m tests.test_endpoint && python -m tests.test_llm_routing)
```

Run the broader test set when changing shared routing, session state, or
telemetry code:

```bash
(cd orchestrator && python -m pytest tests -q)
```

## Operational Notes

- Regenerate Python gRPC stubs after `proto/*.proto` changes:

  ```bash
  bash scripts/gen-proto.sh
  ```

- Restart the orchestrator after changing generated proto files or environment
  values.
- Redis is required. If Redis is slow or unavailable, bounded timeouts prevent
  it from hanging a live call, but session persistence will fail.
- ClickHouse telemetry is best-effort. A ClickHouse outage should not stop
  audio, STT, LLM, or TTS work.
- Logs and runtime output from `scripts/run_pipeline.py` go under `run/logs/`.
