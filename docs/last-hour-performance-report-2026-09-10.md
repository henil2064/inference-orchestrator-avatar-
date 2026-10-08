# Last 1 Hour System Performance Report

Queried ClickHouse via the `clickhouse` container. Window covered by available telemetry: `2026-09-10 11:44:30.642` to `2026-09-10 12:35:52.135` ClickHouse time.

| Metric | Value |
| --- | ---: |
| Events recorded | 990 |
| Sessions | 16 |
| Closed sessions | 16 |
| OK events | 977 |
| Error events | 8 |
| Interrupted events | 5 |
| Error rate | 0.8% |

## Pipeline Coverage

| Stage | Count |
| --- | ---: |
| Caller endpoints / VAD completed | 83 |
| STT completed | 86 |
| LLM completed | 47 |
| TTS first audio queued | 52 |
| TTS completed OK | 47 |
| Interrupt signals | 53 |
| TTS replies interrupted | 5 |
| Empty turns skipped | 3 |

## Latency

| Stage | Avg | p50 | p95 |
| --- | ---: | ---: | ---: |
| Endpoint to first agent PCM | 953.9 ms | 583.5 ms | 3350.8 ms |
| VAD completed duration | 162.1 ms | 103.0 ms | 408.6 ms |
| STT completed duration | 350.0 ms | 227.0 ms | 967.5 ms |
| LLM completed duration | 953.4 ms | 623.0 ms | 2494.9 ms |
| TTS first audio queued duration | 917.2 ms | 568.0 ms | 3335.2 ms |
| TTS full synthesis duration | 495.8 ms | 447.9 ms | 1670.9 ms |

## Throughput

| Metric | Value |
| --- | ---: |
| LLM avg input tokens | 190.5 |
| LLM avg output tokens | 13.1 |
| Total input / output tokens | 8952 / 614 |
| LLM decode throughput avg | 65.6 tokens/s |
| LLM decode throughput p50 / p95 | 68.0 / 81.5 tokens/s |
| LLM TTFT avg | 177.8 ms |
| LLM decode avg | 219.8 ms |
| TTS generated audio | 169.48 s total |
| TTS avg audio per reply | 3.61 s |
| TTS real-time factor avg | 0.25x |
| TTS sample rate | 24000 Hz |

## Deployment

| Component | Value |
| --- | --- |
| LLM model | `/home/aiplay/Desktop/atharva/inference-orchestrator/adaptive_llm/models/huggingface/unsloth--gemma-3-1b-it-GGUF/gemma-3-1b-it-Q4_K_M.gguf` |
| LLM engine / device / compute | `llama.cpp / cuda / gguf` |
| TTS model | `indic-mio-q8` |
| TTS engine / device / compute | `miotts-llamacpp / cuda / gguf` |
| Pipeline version | `ff09a476c7ee` |

## Assessment

The system was mostly healthy at the event level: only `8 / 990` events were marked error. The main user-facing latency was acceptable at median, with first agent audio at `583.5 ms` p50, but tail latency was high at `3350.8 ms` p95.

The largest latency pressure appears around LLM completion and TTS first-audio startup. STT was comparatively moderate, though it still had `176.3 ms` average queue time. LLM queue time was `0 ms`, so the LLM delay is mostly compute/decode rather than waiting.

Interrupt activity was high: `53` interrupt signals, but only `5` TTS replies ended as interrupted. Errors were concentrated in `turn.agent_ended` and `turn.rejected`, `4` each. LLM prefix-cache telemetry was not populated in this window, so cache reuse could not be evaluated.
