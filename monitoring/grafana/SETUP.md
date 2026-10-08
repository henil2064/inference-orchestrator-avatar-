# Grafana Setup

## Docker Compose

From the repository root:

```bash
docker compose up -d prometheus clickhouse grafana
```

Open `http://localhost:3000` and log in with `admin`/`admin` unless
`GRAFANA_ADMIN_USER` and `GRAFANA_ADMIN_PASSWORD` are set.

The Compose stack installs the ClickHouse data source plugin and provisions the
dashboards into the `Inference Pipeline` folder.

## Provisioning Mounts

For a custom Grafana container, mount:

```yaml
volumes:
  - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards:ro
  - ./monitoring/grafana/provisioning/dashboards:/etc/grafana/provisioning/dashboards:ro
  - ./monitoring/grafana/provisioning/datasources:/etc/grafana/provisioning/datasources:ro
```

Grafana normally detects dashboard JSON changes within ten seconds.

## Manual Import

Use manual import only outside the repository Compose stack:

1. Install the Grafana ClickHouse data source plugin.
2. Configure ClickHouse against database `pipeline`.
3. Configure Prometheus against the active Prometheus server.
4. In Grafana, select `Dashboards`, `New`, `Import`.
5. Upload each JSON file from `monitoring/grafana/dashboards/`.
6. Select the matching data source when Grafana prompts for it.

For Docker Compose Grafana, Prometheus should be `http://prometheus:9090`.
For a host-installed Grafana, use the host URL such as `http://localhost:9090`.

## Access Notes

The session-detail dashboards can include final transcripts, LLM responses, and
raw telemetry payloads. Restrict access to users who are allowed to inspect
conversation content.
