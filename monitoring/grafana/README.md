# Grafana Dashboards

`monitoring/grafana` contains the Grafana dashboard JSON and provisioning files
for the inference pipeline.

Setup and import instructions live in [SETUP.md](SETUP.md).

## Dashboard Set

| Dashboard | Data source | Purpose |
| --- | --- | --- |
| Voice Agent KPIs | ClickHouse | Fleet overview of caller-felt latency, answer rate, barge-ins, errors, and slowest turns. |
| Inference Session Detail | ClickHouse | One-session timeline, conversation, latency map, per-worker detail, and raw trace. |
| Model & Release Deep-Dive | ClickHouse | Per-model, per-engine, per-device, and per-version performance comparisons. |
| Saturation & Bounded Overload | Prometheus | Offered load, admitted work, rejections, queue depth, event-loop lag, and resource saturation. |

## Latency Semantics

The primary caller-felt latency is time-to-first-audio: endpointing through the
first PCM queued to media-edge. Full LLM decode and TTS synthesis can continue
while playback is already happening, so dashboards keep those worker timings
separate.

Historical ClickHouse rows from before the orchestrator timing fix can contain
LLM completion durations contaminated by overlapping TTS time. New rows record
LLM completion at the LLM stream boundary.

## Data Model

Fleet panels query typed ClickHouse columns such as model, engine, device,
queue time, TTFT, token counts, STT RTF, and stage timings. The restricted raw
audit trace is the only dashboard area that displays `payload_json`.

For cross-host GPU grouping, workers should set deployment and GPU identity
fields such as `TELEMETRY_DEPLOYMENT_ID`, `TELEMETRY_SERVICE_INSTANCE_ID`,
`TELEMETRY_GPU_UUID`, and `TELEMETRY_GPU_INDEX`. CUDA ordinal alone is not a
stable fleet identity because containers can remap it.

## Layout

```text
monitoring/grafana/
  dashboards/                 Dashboard JSON
  provisioning/dashboards/    Dashboard provider config
  provisioning/datasources/   Local Compose data source config
```
