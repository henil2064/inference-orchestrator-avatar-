# STT and TTS CPU profiling report

Date: 2026-09-02. All figures are from a clean service per candidate, started
via its own `run.sh`, after warm-up runs, on an otherwise idle box. Model,
prompt/audio input, and all other settings were unchanged between candidates.
The question being answered: **can STT and TTS be parallelised to serve more
concurrent sessions, and what does that cost in cores?**

## Measured environment

| Item | Value |
| --- | --- |
| CPU | Intel i9-11900 @ 2.50 GHz — **8 physical cores**, 16 logical (2 threads/core) |
| RAM | 62 GiB total, 53 GiB available |
| GPU | NVIDIA GeForce GTX 1650, 4,096 MiB (unused by STT/TTS in this report) |
| STT | `sravaani-onnx-fp8`, `onnxruntime-tdt`, CPU — onnxruntime 1.26.0, torch 2.6.0 |
| TTS | `indic-mio-q8`, `miotts-llamacpp`, CPU — llama-cpp-python 0.3.35, torch 2.13.0 |
| LLM | Qwen3-0.6B, vLLM, CUDA (out of scope here; see `adaptive_llm/docs/VLLM_TUNING_REPORT.md`) |

`nproc` reports 16, but that is hyperthreading. **The budget is 8 physical
cores.** That distinction is not cosmetic — see the STT 16-thread result.

Both audio models run on CPU on this box because the 4 GiB card is fully
occupied by the LLM. On the 2×L40S target this report's conclusions about
thread tuning do not transfer; its conclusions about *where the cost is* do.

## 1. TTS

### 1.1 Thread scaling saturates at 4

Isolated, ~100-character reply, 4–6 runs per candidate.

| `TTS_NUM_THREADS` | compute | RTF |
| --- | --- | --- |
| 1 | 3,460 ms | 0.732 |
| 2 | 2,555 ms | 0.527 |
| **4** | **1,770 ms** | **0.384** |
| 8 (current default) | 1,823 ms | 0.381 |

Eight threads buys nothing over four. `TTS_NUM_THREADS=0` in `tts/.env` resolves
to `cpu_count() // 2` = 8, so the shipped default is paying for four threads it
does not use.

### 1.2 The knob controls only part of the pipeline

`TTS_NUM_THREADS` sets llama.cpp's `n_threads`
(`tts/service/backends/llamacpp_backend.py:66`) — the backbone LM only. The
second stage, MioCodec decode (`llamacpp_backend.py:120-121`), is PyTorch with
its own thread pool and is unaffected by that variable.

An earlier revision of this report estimated from the thread curve that the codec
was roughly 90% of the cost. **That estimate was wrong**; see §1.5, which measures
it directly.

### 1.3 Multiple workers do not help, and are not safe

`POOL = ThreadPoolExecutor(max_workers=1)` (`tts/service/server.py:43`). The
`BoundedSemaphore(1 + TTS_MAX_QUEUE)` beside it is a **queue bound, not
parallelism**: five requests admitted, one running.

Two concurrent requests:

| Configuration | batch wall | aggregate RTF |
| --- | --- | --- |
| 1 worker × 8 threads (today) | 8.68 s | 0.858 |
| 2 workers × 4 threads | 9.37 s | 0.997 |

Worse, not better. The blocker is correctness rather than throughput:
`llamacpp_backend.py` holds no lock, so two pool workers would share one `Llama`
context — which carries mutable KV-cache state — and one codec instance. Making
`max_workers > 1` correct requires **per-worker model instances**, roughly 3.4 GiB
RSS each, not a configuration change.

A `TTS_POOL_WORKERS` environment knob was added to run this experiment and then
reverted. It should not be reintroduced without per-worker model instances.

### 1.4 A telemetry defect found and fixed

`SynthesizeStream` started its per-sentence timer before `run_in_executor`, so
time spent waiting for the single pool thread was billed as model time. Two
concurrent requests doing identical work reported `realtime_factor` 0.66 and
1.23. The unary `Synthesize` path was always correct; the streaming path now
matches it. Waiting moved to `queue_ms`, a `pool_wait_ms` profile attribute, and
the `tts_pool_wait_seconds` histogram. See DECISION.md D-26.

**Historical ClickHouse `realtime_factor` recorded under concurrency overstates
compute.** Most of the analysed session was single-request and is unaffected, but
any capacity conclusion drawn from a loaded period should be re-measured.

Regression coverage:
`tts/tests/test_backends.py::test_stream_profile_separates_pool_wait_from_compute`.
It asserts that identical work reports identical `compute_ms` regardless of
queueing, and was verified to fail against the original code
(`compute_ms spread 101..401ms for identical work`) before being kept.

### 1.5 Where the time actually goes: backbone decode, bandwidth-bound

The installed `llama-cpp-python` is a CPU-only build
(`llama_supports_gpu_offload()` is `False`), so setting `TTS_N_GPU_LAYERS>0`
moves *only* the MioCodec decoder to CUDA and leaves the backbone LM on CPU.
That accident isolates the codec cost exactly.

| configuration | compute | RTF |
| --- | --- | --- |
| all CPU, 4 threads | 1,814 ms | 0.3816 |
| codec on GPU, backbone on CPU | 1,860 ms | 0.3765 |

GPU memory rose from 296 MiB to 1,557 MiB, so the codec genuinely moved. It made
**no difference**. The codec is not the bottleneck; the backbone decode is
essentially the entire cost.

The arithmetic points at memory bandwidth rather than compute. MioCodec runs at
25 Hz, so 4.94 s of audio is about 123 audio tokens; at 1,860 ms that is roughly
15 ms per token. Batch-1 autoregressive decode reads the whole model per token —
653 MiB at q8 — which works out near 43 GB/s, about what dual-channel DDR4
delivers in practice. This explains both observations at once: extra threads do
not help because bandwidth is already saturated, and the codec offload did
nothing because the codec was never the constraint.

**Expected GPU gain.** An L40S has roughly 864 GB/s of HBM against ~50 GB/s of
DDR4, and bandwidth-bound decode scales close to linearly with it. Estimated
RTF 0.384 → **0.03–0.05, i.e. 8–12× faster**, which would move TTS from ~8
sessions per instance to ~80 and remove it as the system bottleneck. This is an
estimate from the mechanism, not a measurement — no L40S was available.

Two prerequisites, both currently unmet on the dev box:

1. `llama-cpp-python` must be built with CUDA:
   `CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir`.
2. Until DECISION D-27, `llamacpp_backend.py` set `self.device = "cuda"` from the
   environment variable alone, so a CPU-only wheel produced profiles reporting
   `device: cuda` while the backbone ran on CPU. The service now refuses to start
   in that configuration rather than emitting a false device into ClickHouse.

## 2. STT

### 2.1 Thread scaling saturates at 4; hyperthreading hurts

Isolated, 5 s of audio per forward, 5 runs per candidate.

| `STT_CPU_THREADS` | compute | RTF | vs. best |
| --- | --- | --- | --- |
| 1 | 456 ms | 0.0913 | 3.2× worse |
| 2 | 252 ms | 0.0504 | 1.8× worse |
| **4** | **156 ms** | **0.0313** | 1.11× |
| 8 | 149 ms | 0.0298 | 1.06× |
| 16 | 216 ms | 0.0432 | 1.5× worse |
| unset (default) | 140 ms | 0.0281 | best |

Setting 16 places two ONNX intra-op threads on each of the eight physical cores
and they contend. onnxruntime's own default selects the physical core count,
which is why leaving it unset is fastest. **Do not set this above 8.**

### 2.2 Two things STT already gets right

**The knob is real.** `STT_CPU_THREADS` sets both `torch.set_num_threads`
(mel preprocessing, `stt/service/server.py:135`) and onnxruntime's
`intra_op_num_threads` (encoder and decoder,
`stt/service/backends/onnx_backend.py:71`). Unlike the TTS equivalent it
governs the whole pipeline.

**The telemetry was already correct.** `stt/service/server.py:452` computes
`wait_ms = proc_ms - compute_ms`, with `compute_ms` measured inside the worker.
STT never had the TTS timing defect; its recorded RTF is trustworthy as-is.

### 2.3 Concurrency behaviour (4 threads, isolated)

| concurrent turns | aggregate RTF | compute | queue |
| --- | --- | --- | --- |
| 1 | 0.0317 | 159 ms | 0 ms |
| 2 | 0.0340 | 162 ms | 81 ms |
| 4 | 0.0339 | 160 ms | 242 ms |
| 8 | 0.0394 | 179 ms | 637 ms |

`compute_ms` stays flat while `queue_ms` absorbs the contention — serialisation
is clean and honestly reported. `GPU_SLOTS = Semaphore(STT_MAX_INFLIGHT)` blocks
rather than rejects, so excess load becomes backpressure on the gRPC reader
instead of errors.

This matches production telemetry for the live session analysed the same day:
123.7 s of audio against 3.49 s of compute, aggregate RTF 0.028.

## 3. Combined load: the 4+4 split shows no measured gain

Both services under simultaneous load (STT at concurrency 2, TTS at concurrency 2):

| Configuration | STT RTF | TTS aggregate RTF |
| --- | --- | --- |
| A: defaults (8 + 8) | 0.0497 | 0.599 |
| B: 4 + 4 | 0.0465 | 0.605 |

Within noise. The expectation going in was that 4+4 would win by removing 2×
oversubscription of the physical cores; it does not, because neither model scales
past four threads anyway and the scheduler absorbs the remainder. The split is
still worth setting for determinism and for leaving headroom to the orchestrator
and media edge — but it is hygiene, not a throughput improvement.

### Measurement variance

Repeat runs of the same TTS configuration varied between 0.60 and 0.86 aggregate
RTF. Treat any single TTS measurement as ±30%. STT was considerably more stable,
within about ±5%.

## 4. Resource plan

Per-session cost at four threads each. Duty cycles are taken from the live
session analysed the same day: user speech 13.98 s of 93 s (15%); agent speech
assumed at roughly 30%.

| | RTF | duty cycle | cost per session | one worker sustains |
| --- | --- | --- | --- | --- |
| STT | 0.031 | ~15% | 0.005 | ~200 sessions |
| TTS | 0.384 | ~30% | 0.115 | ~8 steady, ~5 practical |

**TTS costs roughly 25× more per session than STT** and accounts for about 92% of
audio-inference CPU. The practical ceiling is near five rather than eight because
`TTS_MAX_QUEUE=4` causes the sixth in-flight request to fail with
`RESOURCE_EXHAUSTED` — a dropped reply, not a delayed one.

### Recommendations

1. Set `STT_CPU_THREADS=4`. Costs about 10% STT latency, frees four cores.
2. Set `TTS_NUM_THREADS=4`. Free — identical RTF — and frees four cores.
3. Never set either above 8 on this box.
4. Stop spending effort on STT concurrency. With ~200 sessions of headroom it is
   not, and will not become, the constraint.
5. Treat every capacity decision as a TTS decision. CPU thread-slicing is a dead
   end there: the backbone decode is memory-bandwidth bound (§1.5), so more
   threads cannot help, and multi-worker requires duplicated models. The durable
   fix is TTS on the GPU — see §1.5 for the expected gain — with batching, per
   README's "GPU = batching, not locking" item, whose warned-against mutex is
   exactly the present 1-thread pool.
6. Open question worth measuring next: what `TTS_N_GPU_LAYERS > 0` buys on the
   1650, as a directional signal for the L40S.

## 5. Reproducing

Services must be started through `run.sh`, which loads the service `.env`;
invoking `server.py` directly does not. Set `STT_WARM_ALL_BUCKETS=0` to remove
roughly 60 s from STT startup when benchmarking a single input length.

The two probes used here were throwaway scripts, not committed:

- an STT client that streams a fixed-length turn followed by `END_TURN` — one
  forward per measurement, kept under `STT_MAX_SEGMENT_SEC` so nothing is
  mid-cut — reporting `compute_ms`, `queue_ms` and RTF from the returned
  `InferenceProfile`;
- a TTS client issuing N concurrent `SynthesizeStream` calls, reporting
  per-request RTF from the profile and aggregate RTF from wall time over total
  audio produced.

Both read their figures from the services' own `InferenceProfile`, which is the
same source the orchestrator forwards to ClickHouse.
