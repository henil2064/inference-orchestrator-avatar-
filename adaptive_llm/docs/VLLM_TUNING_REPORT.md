# Qwen3-0.6B GTX 1650 vLLM tuning report

Date: 2026-09-01. All figures are from a clean vLLM service per candidate,
after a warm-up, with three short voice-style requests per session at
concurrency 1, 2, and 4. Sampling, prompts, model, and generation settings
were unchanged between candidates.

## Measured environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce GTX 1650, 4,096 MiB, compute capability 7.5 |
| Driver / CUDA | 580.173.02 / CUDA 13.0 |
| vLLM / PyTorch | 0.25.1 / 2.11.0+cu130 |
| CUDA check | `True NVIDIA GeForce GTX 1650` |
| CPU / RAM | 16 logical CPUs; 62 GiB RAM, 54 GiB available; 8 GiB swap unused |
| Desktop baseline | 222–258 MiB VRAM, including GNOME/Xwayland and the active desktop client |
| Model | Local `Qwen/Qwen3-0.6B` safetensors; vLLM correctly falls back BF16 weights to FP16 on Turing |

vLLM loaded with prefix caching and chunked prefill enabled, no quantization,
no swap, and no CPU offload. vLLM 0.25.1 removed the `swap_space` constructor
option; the worker now explicitly ignores its required neutral `0` value while
rejecting unsupported non-neutral arguments.

## Candidate results

Values are p50/p95 TTFT in ms, p50/p95 end-to-end latency in seconds, and mean
decode tokens/s. Each candidate had 0 errors and passed 4/4 transparent basic
answer checks (`12`, `Paris`, `cold`, `A B C`).

### Latency and throughput

| Candidate | Settings | C=1 TTFT p50/p95 (ms) | C=2 TTFT p50/p95 (ms) | C=4 TTFT p50/p95 (ms) | C=4 latency p50/p95 (s) | C=4 TPS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Safe | 0.65 / 256 / 2 | 181 / 182 | 249 / 463 | 452 / 464 | 3.851 / 4.115 | 5.31 |
| Balanced | 0.70 / 512 / 4 | 188 / 189 | 467 / 468 | 685 / 686 | 2.296 / 2.755 | 4.97 |
| Memory increment | 0.75 / 512 / 4 | 187 / 187 | 462 / 466 | 678 / 680 | 2.274 / 2.728 | 5.02 |
| Batch increment | 0.70 / 768 / 4 | 187 / 187 | 462 / 464 | 678 / 680 | 2.274 / 2.728 | 5.02 |

### Resource use and outcome

| Candidate | Peak VRAM | Peak temperature | Peak power | Outcome |
| --- | ---: | ---: | ---: | --- |
| Safe | 2,847 MiB | 51 C | 62.82 W | Stable; its two inflight slots queue four sessions. |
| Balanced | 3,031 MiB | 54 C | 63.67 W | Stable; selected winner. |
| Memory increment | 3,217 MiB | 52 C | 64.02 W | Stable; tail gain was too small to justify extra VRAM. |
| Batch increment | 3,017 MiB | 51 C | 63.53 W | Stable; tail gain was too small to justify extra batch pressure. |

Service CPU peaks were 6.90–8.27 logical-core equivalents; peak service RSS
was 9.43–9.45 GiB; total host-RAM peak was 10.31–10.37 GiB. GPU utilization
reached 100% for every candidate. Full per-request records and 250 ms resource
samples are in `adaptive_llm/run/vllm-tuning-results.json`.

The follow-up multi-turn voice-session prefix-cache comparison is documented in
`adaptive_llm/docs/VLLM_CACHE_REPORT.md`.

## Recommendation

Use the measured balanced setting already present in `.env`:

```dotenv
LLM_GPU_MEMORY_UTILIZATION=0.70
LLM_MAX_NUM_BATCHED_TOKENS=512
LLM_MAX_NUM_SEQS=4
ADAPTIVE_LLM_MAX_INFLIGHT_REQUESTS=4
ADAPTIVE_LLM_MAX_QUEUED_REQUESTS=16
LLM_MAX_TOKENS=128
LLM_CONTEXT_SAFETY_TOKENS=128
LLM_SWAP_SPACE_GIB=0
LLM_CPU_OFFLOAD_GB=0
LLM_QUANTIZATION=
LLM_ENABLE_PREFIX_CACHING=1
LLM_ENABLE_CHUNKED_PREFILL=1
```

This is the measured winner for a responsive four-session voice service:
unlike the safe two-sequence configuration it admits all four sessions, while
the 0.75-memory and 768-batch variants improved 4-session p95 TTFT by only
about 6 ms in this small fixed workload—too little evidence to spend extra
VRAM or batch pressure. Expect a safe capacity of four simultaneous sessions;
the observed 4-session p95 is about 686 ms to first token and 2.76 s end to
end for the 7–10-token test answers. Do not infer a guarantee for longer
prompts or 128-token outputs from this short-answer workload.

Sequences 5/6 were not tested: four sessions are stable, but the scheduler is
already GPU-saturated and the requested voice target prioritizes tail latency.

## Reproduction and verification

```bash
adaptive_llm/.venv/bin/python adaptive_llm/tools/vllm_tune.py \
  --candidate safe,0.65,256,2 \
  --candidate balanced,0.70,512,4 \
  --candidate memory-increment,0.75,512,4 \
  --candidate batch-increment,0.70,768,4
```

The harness starts each candidate in a separate process group, captures its
logs, waits for CUDA VRAM to return to the desktop baseline before the next
candidate, and records startup/resource/load failures instead of masking them.

Relevant tests: `adaptive_llm/.venv/bin/python -m pytest
adaptive_llm/tests/test_vllm_engine_args.py adaptive_llm/tests/test_planner.py
-q` -> 9 passed.
