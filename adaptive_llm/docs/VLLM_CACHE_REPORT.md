# Qwen3-0.6B multi-turn voice cache report

Date: 2026-09-01. This is a controlled cache comparison on the measured vLLM
winner: GPU memory utilization 0.70, 512 batched tokens, four sequences,
prefix caching as the only changed setting, and four concurrent text-only voice
sessions. STT and TTS were not invoked.

## Workload

Every session used the same five-turn voice-style memory exchange. The returned
assistant response was appended to the next request as conversation history,
using one stable gRPC session ID per session:

1. Remember: my name is Maya.
2. What name did I ask you to remember?
3. Remember: I live in Paris.
4. What city do I live in?
5. What are both my name and city?

The factual checks were case-insensitive checks for `Maya`, `Paris`, or both;
they are transparent sanity checks, not a human-quality evaluation.

## Results

| Setting | Requests | Cached prompt tokens | Total prompt tokens | Token-weighted prefix hit rate | Requests with hit | p50 / p95 TTFT | p50 / p95 latency | Mean decode | Quality checks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Prefix cache enabled | 20 | 3,552 | 3,820 | 92.98% | 20 / 20 | 457 / 688 ms | 1.140 / 2.054 s | 5.75 tok/s | 11 / 12 |
| Prefix cache disabled | 20 | 0 | 3,820 | 0.00% | 0 / 20 | 1,661 / 2,572 ms | 3.026 / 4.922 s | 4.01 tok/s | 11 / 12 |

| Setting | Peak VRAM | Peak GPU utilization | Peak temperature | Peak power | Errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefix cache enabled | 3,025 MiB | 100% | 48 C | 56.96 W | 0 |
| Prefix cache disabled | 3,020 MiB | 100% | 50 C | 37.03 W | 0 |

## Interpretation

Keep `LLM_ENABLE_PREFIX_CACHING=1`. On this representative growing-history
voice workload it reused nearly all prompt tokens and reduced p95 TTFT by
1.884 seconds and p95 end-to-end latency by 2.868 seconds without increasing
observed peak VRAM materially (5 MiB difference).

One of twelve factual checks missed in each arm because of sampling variation;
the cache setting itself is not expected to change model semantics. This is why
the result supports prefix-cache performance, not a claim of perfect response
quality. Use a larger deterministic quality suite before evaluating dialogue
quality or changing sampling settings.

## Telemetry added

The vLLM worker now emits these per-response `InferenceProfile.attributes`:

- `cached_prompt_tokens`
- `history_turns_used`
- `history_turns_trimmed`

It also exposes Prometheus counters `llm_prefix_cached_tokens_total` and
`llm_prefix_cache_prompt_tokens_total` on the worker metrics endpoint. The
token-weighted hit rate is calculated as:

```text
sum(cached_prompt_tokens) / sum(input_tokens) * 100
```

Raw per-turn responses, token counts, resource samples, and logs are in
`adaptive_llm/run/vllm-cache-tuning-results.json` and
`adaptive_llm/run/vllm-cache-tuning-logs/`.

## Reproduction

```bash
adaptive_llm/.venv/bin/python adaptive_llm/tools/vllm_cache_tune.py --sessions 4
```
