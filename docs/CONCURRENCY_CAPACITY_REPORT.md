# Concurrency & capacity report — single A100 box

Date: 2026-09-19. Question being answered: **how many instances of STT, TTS and
LLM should run on this box, and how much concurrent traffic can the pipeline
actually take?**

All figures below are measured on this box, on the date above. The LLM was
already running when this work started and was never restarted or
reconfigured — every number involving it comes from the instance as it was
deployed (MedGemma-27B GGUF via vLLM, `MAX_INFLIGHT=16`).

This report supersedes nothing. `CPU_PROFILING_REPORT.md` remains correct for
the box it was measured on (an 8-core i9 with a 4 GB card); its conclusions
about *where the cost is* mostly transfer, its absolute numbers do not, and its
central assumption — that TTS runs on CPU — is no longer true in this
deployment. See §5.

---

## 0. TL;DR

| | Recommendation | Why |
| --- | --- | --- |
| **STT** | 1 needed, **2** for redundancy | ~250 ms compute per 5.4 s of audio, flat to 32 concurrent. Never the bottleneck. |
| **TTS** | **1** (2 only as cold failover) | GPU-offloaded. A second *active* instance makes throughput worse, not better (§5). |
| **LLM** | **1** (as deployed) | One vLLM instance per GPU is a design guardrail; its 16-slot cap is not the binding constraint below 16. |

**Comfortable concurrency today: ~16 simultaneous voice turns.** Clean at 16
(16/16 completed). Degrades at 24 (19/24 completed, 4 client timeouts).

**But 16 is a throughput number, not a quality number.** Time-to-first-audio at
that load averages ~8.4 s (p95 ~20.9 s), which is not a usable conversational
experience. The two things making it slow are, in order: **TTS queueing**
(~3.1 s of pure wait at 16 concurrent, before the LLM cap is even touched) and
**LLM decode speed** (~9–17 tok/s, which is low for a 27B model on an A100 and
is a fixed cost on every turn at any concurrency). See §8.

---

## 1. Measured environment

| Item | Value |
| --- | --- |
| CPU | AMD EPYC 75F3 — **32 physical cores**, 64 logical (2 threads/core) |
| RAM | 503 GiB total, 447 GiB available |
| GPU | 1× NVIDIA A100 80GB PCIe (81,920 MiB), driver 550.127.08, CUDA 13.0 |
| Environment | Containerised (Kubernetes); **no host-level GPU privileges** — see §6 |
| STT | `sravaani-16-onnx` fp16, onnxruntime, **CPU**, `STT_CPU_THREADS=4`, `STT_MAX_INFLIGHT=8` |
| TTS | `indic-mio-q8` gguf, miotts-llamacpp, **CUDA** (`TTS_N_GPU_LAYERS=999`), `TTS_NUM_THREADS=4`, `TTS_MAX_QUEUE=16` |
| LLM | `medgemma-27b-text-it-Q4_K_M.gguf` via vLLM, CUDA, `GPU_MEMORY_UTILIZATION=0.45`, `MAX_INFLIGHT=16`, `MAX_NUM_SEQS=16`, `MAX_MODEL_LEN=9162` |

LLM VRAM footprint: 36,587 MiB held (plan estimate: 17,358 weights + 4,438 KV +
868 runtime = 22,664 MiB; the rest is vLLM's reserved pool at util 0.45).
That leaves ~44 GiB of the A100 free — which is *not* the constraint. GPU
*compute* time is. See §5.

### What the auto-sizing formula proposes

`scripts/orchestration/plan.py` on this hardware, with no overrides:

```
physical_cores=32 logical_cores=64
reserved_control_cores=5 llm_cpu_reserve=1 worker_cores=26
stt: 2 replica(s) x 4 threads
tts: 4 replica(s) x 4 threads
```

The STT number is fine. **The TTS number (4) is wrong for this deployment** and
should not be used as-is — the formula assumes TTS is CPU-bound and that
replicas therefore scale on cores. With `TTS_N_GPU_LAYERS=999` that assumption
does not hold; see §5. The formula is not buggy, its input model is stale.

---

## 2. Full-pipeline end-to-end results

Real WebRTC path, `pipeline_test/load_burst.py`, one instance of every service
(media-edge, orchestrator, STT, TTS, LLM). Two WAV turns per session.

| Concurrency | Outcome | Wall time |
| --- | --- | --- |
| 1 | 1 completed | 25.3 s |
| 2 | 2 completed | 27.6 s |
| 4 | 4 completed | 34.0 s |
| 8 | 8 completed | 45.6 s |
| 16 | **16 completed** | 53.4 s |
| 24 | 19 completed, 4 failed, 1 generator timeout | 105.0 s |

**Nothing was ever rejected server-side.** `edge_connection_rejected_total_delta`
was 0 at every level, and only 2 `turn.rejected` events were recorded all
session. All four failures at concurrency 24 were the *client* giving up:
`timed out waiting 12s for final orchestrator response` — i.e. `finalReplyWait`
in `media-edge/cmd/media-edge/bridge.go`. The system queues under overload
rather than shedding, so the observable failure mode is a late answer, not a
clean rejection. That is worth knowing before turning traffic up: the wall is
soft and it is made of latency.

---

## 3. Per-stage latency

Two views. The first is every session closed in the hour (58 sessions, all
concurrency levels blended — the tail is dominated by the 24-run):

| Phase | Average | p50 | p95 |
| --- | --- | --- | --- |
| **Time to first audio** | 8,372 ms | 7,781 ms | 20,868 ms |
| STT (endpoint→final) | 2,238 ms | 1,170 ms | 6,163 ms |
| LLM queue | 732 ms | 0.3 ms | 6,902 ms |
| LLM encode | 387 ms | 304 ms | 742 ms |
| LLM decode | 2,331 ms | 2,103 ms | 4,211 ms |
| Phrase generation | 977 ms | 938 ms | 2,032 ms |
| **TTS queue** | **3,542 ms** | **3,738 ms** | **9,105 ms** |
| TTS first PCM | 478 ms | 279 ms | 1,124 ms |
| Media-edge queue | 12 ms | 11 ms | 20 ms |

Supporting: LLM input/output 196.6 / 21.7 tokens, decode 14.3 tok/s,
prefix-cache reuse 86.9%. TTS full synthesis wall 4,447 ms for 5.9 s of audio.

The second view isolates the two runs that matter, by ClickHouse time window —
this is the more useful one, because it separates "inside the envelope" from
"past the cap":

| | @16 (zero failures) | @24 (failures start) |
| --- | --- | --- |
| LLM queue wait | **0.4 ms** | 1,488 ms |
| LLM decode | 3,339 ms | 2,351 ms |
| **TTS queue wait** | **3,076 ms** | 5,377 ms |

Read that carefully. **At 16 concurrent the LLM is not waiting at all** (0.4 ms)
— its 16-slot cap is exactly not binding. But TTS is already imposing ~3.1 s of
pure queue wait per turn. TTS is the tax you pay well before you reach any
configured limit. Past 16, the LLM cap engages too (1.5 s) and the two
compound.

(`llm_decode_ms` being higher at 16 than at 24 is reply-length mix, not a
trend — output length varies per turn and the samples are small.)

---

## 4. STT — measured, and not a problem

Direct gRPC to one instance, 5.43 s of audio per session, ONNX fp16 on CPU at
4 threads:

| concurrent sessions | ok | err | batch wall | avg compute | avg queue |
| --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0 | 0.26 s | 250.8 ms | 0.7 ms |
| 4 | 4 | 0 | 1.00 s | 247.9 ms | 372.6 ms |
| 8 | 8 | 0 | 2.00 s | 247.4 ms | 868.5 ms |
| 16 | 16 | 0 | 4.04 s | 250.1 ms | 1,881.4 ms |
| 32 | 32 | 0 | 8.02 s | 248.6 ms | 3,862.4 ms |

RTF ≈ 0.046. `compute_ms` is **flat to 32 concurrent** — the single worker
serialises cleanly and honestly reports the contention as `queue_ms`, exactly
as `CPU_PROFILING_REPORT.md` §2.3 described on the older box. No rejections at
any level: `GPU_SLOTS = Semaphore(STT_MAX_INFLIGHT)` blocks rather than
rejects, so excess load becomes backpressure.

**STT is not, and will not become, the constraint.** One instance is enough for
throughput; run 2 for crash redundancy. Cost: ~8 of 64 threads.

Caveat on the pipeline numbers in §3: the multi-second "STT" phase there is
VAD-endpoint-to-final-transcript, which is dominated by waiting for speech and
segment boundaries in fixed-length replayed recordings. It is not 2 s of STT
compute — compute is the 250 ms above. Do not read the §3 STT row as a
resourcing signal.

---

## 5. TTS — the actual bottleneck

### 5.1 One instance, increasing concurrent callers

GPU-offloaded, ~8.2 s of audio per call:

| concurrent calls | ok | err | batch wall | avg compute | avg queue | aggregate RTF |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 0 | 0.60 s | 576.8 ms | 0.3 ms | 0.073 |
| 2 | 2 | 0 | 1.14 s | 554.2 ms | 284.6 ms | 0.066 |
| 4 | 4 | 0 | 2.22 s | 543.7 ms | 804.1 ms | 0.066 |
| 8 | 8 | 0 | 4.43 s | 549.2 ms | 1,916.4 ms | 0.066 |
| 16 | 16 | 0 | 8.64 s | 537.9 ms | 4,070.4 ms | 0.065 |
| 24 | 17 | **7** | 9.21 s | 539.0 ms | 4,289.5 ms | 0.065 |

Compute is flat at ~540 ms; **every additional caller adds ~540 ms of queue to
everyone behind them.** The 7 errors at 24 are `RESOURCE_EXHAUSTED: TTS queue
is full` — admission is `1 + TTS_MAX_QUEUE` = 17, working exactly as designed.

Note how closely the isolated `queue_ms` at 16 (4,070 ms) matches the real
pipeline's `tts_queue_ms` at 16 concurrent (3,076 ms). The stage benchmark
predicted the production behaviour.

GPU cost: **~3.4 GiB VRAM per instance**. One instance under sustained backlog
drove `utilization.gpu` to **90%**.

### 5.2 Multiple GPU instances actively scale *negatively*

Separate processes, each with its own CUDA context, loaded simultaneously:

| active instances | avg compute per call | vs solo |
| --- | --- | --- |
| 1 | ~550 ms | 1.0× |
| 2 | 1,173–1,290 ms | ~2.2× slower |
| 4 | 2,904–2,993 ms | ~5.4× slower |

Aggregate throughput with 4 instances: 16 calls in 12.01 s = **1.33 calls/s** —
**lower than a single instance's own ceiling of ~1.8 calls/s.** Four instances
are worse than one.

The mechanism: without MPS, the GPU cannot run kernels from multiple processes
truly in parallel; it time-slices between CUDA contexts, and the switching
overhead grows slightly worse than linearly. VRAM is irrelevant here (4
instances = ~13.8 GiB, with ~31 GiB still free) — the contended resource is
GPU execution time.

**This is the single most important operational finding in this report.**
Scaling TTS by process count — the strategy `LOAD_SCALING_PLAN.md` prescribes,
and which is correct for CPU TTS — is actively counterproductive for
GPU-offloaded TTS on one card. Run one, and treat a second as a cold spare
only.

---

## 6. NVIDIA MPS — untested, blocked

MPS would be the obvious fix for §5.2: it lets multiple processes share one GPU
through a single server context instead of time-slicing. It could not be tested
here.

`nvidia-smi -c EXCLUSIVE_PROCESS` fails with **`Insufficient Permissions`**,
even as root, because this is a container with GPU passthrough and no
host-level capability to change compute mode. Nothing was changed; compute mode
remained `Default` throughout.

**Status: plausible but unverified.** If someone can run this on the host, or
grant the container `SYS_ADMIN`, it is worth measuring — the mechanism in §5.2
is exactly what MPS addresses. Do not assume it works until it is measured.
Note also that mixing MPS and non-MPS clients on one GPU is unsupported, so a
real test needs the LLM behind MPS too, which means restarting it.

---

## 7. TTS on CPU — a real but sub-linear second lane

Since GPU instances cannot scale by count, the alternative is spending idle CPU
cores. Same service, `TTS_N_GPU_LAYERS=0 TTS_DEVICE=cpu`:

| concurrent calls (1 instance) | batch wall | avg compute | avg queue | aggregate RTF |
| --- | --- | --- | --- | --- |
| 1 | 3.96 s | 3,933.5 ms | 0.2 ms | 0.480 |
| 2 | 7.71 s | 3,842.5 ms | 1,831.6 ms | 0.449 |
| 4 | 15.69 s | 3,914.9 ms | 5,983.4 ms | 0.456 |

One CPU instance is ~7× slower per call than one GPU instance (3,930 ms vs
550 ms), RTF 0.48. Notably this is *worse* than the old 8-core i9 box's 0.384 —
the extra memory channels did not help, which is consistent with the
bandwidth-bound decode described in `CPU_PROFILING_REPORT.md` §1.5.

Four CPU instances, run concurrently (one call each):

| | value |
| --- | --- |
| per-call compute | 9,575–9,886 ms (**~2.4× slower** than solo) |
| aggregate throughput | **0.404 calls/s** (vs 0.254 for one instance) |
| scaling efficiency | ~40% of linear (1.59× throughput from 4× instances) |

So CPU TTS **does** scale positively — unlike GPU (§5.2) — just inefficiently.
Each process showed ~430–460% CPU and **47 threads**, despite
`TTS_NUM_THREADS=4` / `OMP_NUM_THREADS=4`: those knobs govern the llama.cpp
backbone only, not the PyTorch MioCodec decode stage, which is the
long-standing gap noted in `CPU_PROFILING_REPORT.md` §1.2. RSS ~3.3 GiB each.

**Verdict:** a CPU pool is a legitimate overflow lane to relieve GPU TTS queue
pressure — additive, since it contends for nothing the GPU instance or the LLM
needs. It is a workaround for the single-worker design, not a fix for it.

---

## 8. What is actually slow, and what to fix

Ranked by measured impact.

**1. TTS queueing — confirmed, dominant, structural.** At 16 concurrent, before
any configured cap binds, TTS contributes ~3.1 s of pure queue wait per turn
(§3), because one instance synthesises one clause at a time
(`ThreadPoolExecutor(max_workers=1)`) and more instances make it worse (§5.2).
The real fix is **concurrent/batched synthesis inside one process** — what vLLM
does for the LLM — not more processes. That is engineering work on the TTS
backend, not configuration. Interim: the CPU pool in §7.

**2. LLM decode speed — strong lead, not yet isolated.** Measured
`decode_tokens_per_second`: 16.6 at concurrency 2, 9.1 at 16, 13.8 at 24;
overall average 13.5–14.3 tok/s. For a 27B model on an A100 this is low, and
critically **it does not collapse as concurrency rises**, which argues against
"it is just batching dilution" and for a fixed per-token overhead. The most
likely candidate is that the model runs as **GGUF through vLLM**, whose GGUF
path is not vLLM's optimised kernel (unlike AWQ/GPTQ/FP8 or native bf16). This
costs ~2–3 s on *every* turn at *any* concurrency — it is baseline latency, not
a scaling problem, and it is probably the largest single lever on TTFA.
**Unverified**: the LLM was never load-tested in isolation (it was off-limits
all session). Test by serving the same model in a native quant and comparing
`decode_tokens_per_second`.

**3. The LLM's 16-slot cap is fine and is not the problem below 16.** Queue wait
is 0.4 ms at 16 concurrent. It only engages past its cap (1,488 ms at 24).
Raising it would not help TTFA and would make TTS queueing worse.

**4. STT needs no work at all** (§4).

Do not spend effort on 3 or 4 before 1 and 2.

---

## 9. Bug found and fixed: media-edge send-after-close

While setting up the end-to-end runs, **every session beyond the first failed**
with `rpc error: code = Internal desc = SendMsg called after CloseSend`.
Concurrency 1 passed; concurrency 2 failed both attempts, deterministically.

Cause, in `media-edge/cmd/media-edge/bridge.go`: `runBridge` deliberately
half-closes its send side (`stream.CloseSend()`) once the inbound track ends,
leaving the response direction open so the orchestrator can still deliver the
reply. But the playback notification registered via `sink.SetPlaybackStarted`
fires asynchronously and calls `sendEvent` → `stream.Send()`. If playback starts
*after* the half-close, that is a `Send` after `CloseSend`, which grpc-go treats
as fatal. The existing `sendMu` serialised concurrent sends against each other
but did not encode "the send side is already closed". `CloseSend()` was also
called outside that mutex, so it could race a send in flight.

Why concurrency exposed it: at concurrency 1, STT→LLM→TTS completes fast enough
that playback typically starts before the track ends. Under load the pipeline
is slower, so playback lands after the half-close and the latent race becomes
the common case.

Fix (committed as `faec178`): a `sendClosed` flag guarded by the existing
`sendMu`, checked in `sendEvent` and `sendControlSafe` (late events are dropped
rather than sent), with `CloseSend()` now called under the same mutex. After the
fix, concurrency 2 → 16 all completed cleanly.

**This was a pre-existing defect, unrelated to instance counts.** Any prior
multi-session measurement taken against media-edge without this fix is invalid.

---

## 10. Gaps and caveats

- **The LLM was never benchmarked in isolation.** It was off-limits for the
  whole exercise. §8 item 2 is a lead built on pipeline telemetry, not a
  controlled measurement.
- **`batch_size` and `running_requests` are always NULL** in
  `pipeline.pipeline_events`. vLLM's batch depth is therefore not observable
  from ClickHouse, which is why §8 item 2 could not be settled from the data on
  hand. Worth wiring up — it is the field that would resolve it.
- **MPS untested** (§6).
- **Two-WAV synthetic sessions with near-simultaneous arrivals.** Real traffic
  arrives staggered; the synchronised endpoints in these runs inflate queueing
  tails relative to organic load. `open_loop_load.py` (Poisson arrivals) was not
  run — only closed-loop bursts.
- **Single runs per concurrency level**, no repeats. TTS measurement variance on
  the older box was ±30%; treat single points accordingly.
- **Soak and recovery untested.** Nothing here covers multi-hour stability,
  repeated over-limit bursts, or drain behaviour — see the capacity protocol in
  `LOAD_SCALING_PLAN.md`.

---

## 11. Reproducing

Stage benchmarks used throwaway gRPC clients (not committed, same practice as
`CPU_PROFILING_REPORT.md` §5): a TTS client issuing N concurrent
`SynthesizeStream` calls and reporting per-call `InferenceProfile`
compute/queue plus aggregate RTF from wall time over audio produced; and an STT
client driving N concurrent `Transcribe` bidi streams with `END_TURN`, reading
`compute_ms`/`queue_ms` from the returned profile. Services must be started via
their own `run.sh` (it loads the service `.env`).

Extra TTS instances for §5.2 / §7 were started with `TTS_ADDR` and
`TTS_METRICS_PORT` overridden per instance (the same port-stride scheme
`scripts/orchestration/plan.py` uses), and for §7 additionally
`TTS_N_GPU_LAYERS=0 TTS_DEVICE=cpu`.

End-to-end runs:

```bash
python3 -m venv pipeline_test/.venv
pipeline_test/.venv/bin/python -m pip install -r pipeline_test/requirements.txt
# WAVs in pipeline_test/recordings/
pipeline_test/.venv/bin/python pipeline_test/load_burst.py \
  --concurrency 16 --edge-url https://127.0.0.1:8443 \
  --wav-dir pipeline_test/recordings --out pipeline_test/recordings/burst_c16.jsonl
python3 pipeline_test/report_recent_sessions.py --since-hours 1
```

Per-run stage isolation (§3, second table) was a direct ClickHouse query
filtered on `event_time_ms` between the run's start and end, taken from the
harness's own `started_at`/`finished_at` in the JSONL:

```bash
curl -sG http://127.0.0.1:8123/ --data-urlencode \
  "query=SELECT avg(llm_queue_ms), avg(llm_decode_ms), avg(tts_queue_ms)
         FROM pipeline.pipeline_events
         WHERE event_time_ms BETWEEN <start_ms> AND <end_ms> FORMAT PrettyCompact"
```
