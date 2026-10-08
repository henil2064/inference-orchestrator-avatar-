# Monitoring Setup

## Prerequisites

- Docker Engine with Compose v2.
- Pipeline services exposing metrics on their configured metrics ports.
- ClickHouse enabled in the orchestrator if session dashboards are needed.

## Start

From the repository root:

```bash
docker compose up -d prometheus grafana clickhouse
```

For the full infrastructure stack:

```bash
docker compose up -d redis clickhouse prometheus grafana
```

Prometheus targets stay `DOWN` until the corresponding pipeline services are
running.

## Open

| Tool | URL |
| --- | --- |
| Prometheus targets | `http://localhost:9090/targets` |
| Prometheus graph | `http://localhost:9090/graph` |
| Grafana | `http://localhost:3000` |

Grafana defaults to `admin`/`admin` unless `GRAFANA_ADMIN_USER` and
`GRAFANA_ADMIN_PASSWORD` are set.

Dashboards are provisioned into the `Inference Pipeline` folder when using the
repository Compose stack.

## Data Sources

Grafana provisions two local data sources:

| Data source | Container URL | Used by |
| --- | --- | --- |
| `Prometheus` / uid `prometheus` | `http://prometheus:9090` | Saturation and service metrics dashboards. |
| `ClickHouse` / uid `clickhouse` | `http://clickhouse:8123`, database `pipeline` | Session, fleet, model, and version dashboards. |

The host browser opens Prometheus at `http://localhost:9090`; Grafana reaches it
inside Docker at `http://prometheus:9090`.

## Service Metrics Ports

Default metrics endpoints:

| Service | Metrics endpoint |
| --- | --- |
| media-edge | `:9101/metrics` |
| orchestrator | `:9102/metrics` |
| stt | `:9103/metrics` |
| tts | `:9104/metrics` |
| adaptive_llm facade | `:9106/metrics` |
| adaptive_llm worker | `:9114/metrics` |

Metrics ports bind to all interfaces by default. To keep metrics local-only,
set the corresponding service metrics host or bind address to `127.0.0.1`.

## Replica Targets

Phase 3 STT/TTS replicas are discovered through Prometheus file discovery.
`scripts/run_pipeline.py` rewrites these files on every launch:

```text
monitoring/targets/stt.json
monitoring/targets/tts.json
```

If replica panels are empty, run the launcher again or inspect those files.

## Useful PromQL

```promql
histogram_quantile(0.9, sum(rate(stt_wait_seconds_bucket[1m])) by (le))
histogram_quantile(0.9, sum(rate(stt_rtf_bucket[1m])) by (le))
orch_stt_queue_depth
histogram_quantile(0.95, sum(rate(stt_window_seconds_bucket[5m])) by (le))
rate(edge_vad_inference_errors_total[5m])
rate(orch_stt_stream_errors_total[5m])
rate(stt_model_errors_total[5m])
```

## ClickHouse Migration

For an existing ClickHouse data volume that predates TTS telemetry columns,
apply the migration once before restarting the orchestrator:

```bash
docker compose exec -T clickhouse clickhouse-client --multiquery < monitoring/clickhouse/migrate_tts.sql
```

## Manual Grafana Import

Use this only when not using the repository Compose provisioning:

1. Install the Grafana ClickHouse data source plugin.
2. Configure a read-only ClickHouse data source pointed at database `pipeline`.
3. Configure a Prometheus data source pointed at the active Prometheus server.
4. Import the JSON files from `monitoring/grafana/dashboards/`.

Session-detail dashboards can expose final transcripts and replies. Restrict
access accordingly.
