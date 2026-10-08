# Load & Horizontal-Scaling Plan

Status: draft for the L40S single-box load-hardening effort.
Context: single node, **1× L40S 48 GB**, ~256 vCPU (verify physical cores/SMT/NUMA),
256 GB RAM. LLM (vLLM) on GPU; STT + TTS on CPU (Compose defaults). Tester ramps
concurrent sessions 2 → 4 → 8 → … → 1024.

Goal: make outcomes and telemetry **truthful**, bound overload so bursts reject
cleanly instead of degrading everyone, then scale on **measured** numbers. Do not
promise a concurrency figure from hardware specs alone.

---

## How to use this doc

Work is grouped into four phases. **Finish a phase before the next one is meaningful.**
Each phase lists concrete work items — do them in the order written. Each item is
independent enough to ship and revert on its own (put it behind an env flag with a
safe default, add a focused test, run the affected service's suite).

| Phase | Theme | You're done when |
|-------|-------|------------------|
| **1 — Correctness & safety** | Stop OOM and stop the telemetry from lying | No unbounded queue; a failed/interrupted turn is never recorded as a heard answer; core counters count the right thing. |
| **2 — Visibility & bounded overload** | See the load, reject excess cleanly | Offered-vs-served + per-stage saturation are on a dashboard; bursts reject promptly; admitted calls keep their guarantees. |
| **3 — Measured scaling** | Add replicas on real numbers | Replicas discoverable & scraped; one measured baseline + agreed latency target exists. |
| **4 — Deferred / gated** | Optional polish | Only build after Phase-2 measurement shows the need. |

**If you only do one thing this week:** Phase 1 plus the Phase 2 dashboard. That turns
"we don't know what happened" into "we can see the wall" — the actual blocker today.

---

## Assessment (fact vs hypothesis)

**Fact (from code):** STT and TTS each serialize to one worker; the LLM concurrency is
capped at 4; and several telemetry defects make any load result untrustworthy.

| Stage | Runs on | Real concurrency | Hard limiter |
|-------|---------|------------------|--------------|
| STT | CPU (ONNX int8) | **1** | `ThreadPoolExecutor(max_workers=1)` (`stt/service/server.py:174`) |
| TTS | CPU (llama.cpp gguf) | **1** (admitted 1+4) | `ThreadPoolExecutor(max_workers=1)` (`tts/service/server.py:43,69`) |
| LLM | GPU (vLLM) | **4** | `ADAPTIVE_LLM_MAX_INFLIGHT_REQUESTS=4` |
| Orchestrator | single asyncio proc | unbounded accept | no `maximum_concurrent_rpcs`; no orchestrator session cap |

**Hypothesis (not yet measured):** the "4–16 concurrent conversation" ceiling and the
"idle GPU while CPU chokes" story. Plausible from the above — confirm on this box, do
not quote as a result. `256 vCPUs` may not be 256 physical cores; a duty-cycle average
is not a max-concurrency number.

**Correctness problems found in code (these drive Phase 1):**
- A turn whose TTS clauses all fail is still persisted/announced as a complete answer
  (completion is gated on LLM success; TTS failure is swallowed).
- `orch_tts_calls_total` increments per audio **chunk**, not per call (`server.py:425`).
- vLLM `GenerateStream` (the path actually used) records almost none of the metrics the
  unary `Generate` does (`vllm_worker.py`).
- STT releases its GPU slot on cancel while the native forward keeps running.
- TTS admission bounds *admitted* work but lets unlimited callers wait outside it.
- `EDGE_MAX_SESSIONS` **exists** (`media-edge/.../rtc.go:94`) but defaults to unlimited;
  there is no orchestrator-side cap.

---

## Env knobs — what scales, what doesn't

A *queue-depth* knob changes how many requests **wait**; an *execution-concurrency*
knob changes how many **run at once**. Only the latter raises capacity, and only up to
where latency/memory/bandwidth push back. Pick values by measured throughput of timely
answers, not by the number itself.

### Adaptive LLM facade — genuinely scales via `.env`
| Env var | Default | Effect | Action |
|---------|---------|--------|--------|
| `ADAPTIVE_LLM_MAX_INFLIGHT_REQUESTS` | 4 | Concurrent generations reaching vLLM; a slot is held for the whole stream. | Sweep up; 32–64 are **candidates**, not safe defaults. |
| `ADAPTIVE_LLM_MAX_QUEUED_REQUESTS` | 16 | Waiting room; `inflight+queued` reached → `RESOURCE_EXHAUSTED`. | Keep small (8–16); prefer fast reject. |
| `LLM_MAX_NUM_SEQS` | vLLM default | vLLM's own batch width. Must be ≥ facade inflight or vLLM is the limiter. | Tune jointly; don't inherit the 4 GB profile's `=1`. |
| `LLM_MAX_NUM_BATCHED_TOKENS` | vLLM default | Prefill budget/step. | Raise if prefill-bound (more VRAM). |
| `LLM_GPU_MEMORY_UTILIZATION` | 0.85 | KV-cache budget → concurrency. | Test 0.90 with measured headroom. |
| `LLM_MAX_MODEL_LEN` / `LLM_MAX_TOKENS` | model / 512 | Smaller context+output → more concurrency, faster turnover. | Cap to voice needs (e.g. 2048 / 256); verify quality. |

> ⚠️ `model-profiles.json` `gtx_4gb_vllm_fp16_unsafe` pins `LLM_MAX_NUM_SEQS=1`,
> `GPU_MEMORY_UTILIZATION=0.70`. Confirm the L40S deployment does not inherit it.

### STT / TTS — `.env` controls queue depth & per-forward cost, NOT throughput
| Env var | Default | Note |
|---------|---------|------|
| `STT_MAX_INFLIGHT` | 8 | Bounds submitted/running work, not the segment queue or waiters. Adds no workers. |
| `STT_CPU_THREADS` | runtime-picked | Threads per forward. Set explicitly per replica once scaled. |
| `STT_MAX_SEGMENT_SEC` / `_GRACE_SEC` | 8.0 / 2.0 | Bounds worst-case compute per forward. |
| `STT_DEVICE` / `STT_ONNX_PROVIDER` | cpu | Placement; GPU is a later measured experiment, not assumed spare. |
| `TTS_MAX_QUEUE` | 4 | Admitted = `1+MAX_QUEUE`; one inference runs. Doesn't bound callers waiting outside admission. |
| `TTS_QUEUE_TIMEOUT_S` | 15 | Admission-acquire timeout, **not** the turn deadline. |
| `TTS_NUM_THREADS` | ~half CPUs | llama.cpp threads per synth (could be ~128 here). Set explicitly per replica. |
| `TTS_N_GPU_LAYERS` / `TTS_DEVICE` | 0 / cpu | GPU offload; measured Phase-3 experiment. |

**Scaling STT/TTS = more processes, not bigger queue numbers.** The single-worker
serialization is not removable via `.env`.

### Orchestrator & edge
| Env var | Default | Note |
|---------|---------|------|
| `ORCH_INBOUND/STT/OUTBOUND_QUEUE_MAX` | 400 | Per-session buffers → grpc flow-control backpressure. Not a concurrency cap. |
| `TTS_TIMEOUT_S` | 45 | **Per-clause** RPC deadline, renewed each clause. Cap by remaining turn budget (Phase 2). |
| `LLM_HISTORY_MAX_TURNS` | 6 | Prompt history depth = LLM load. |
| `EDGE_MAX_SESSIONS` | 0 (unlimited) | **Already implemented** at the edge. Enable a conservative cap. |
| `ORCH_MAX_SESSIONS` | — (missing) | Add authoritative orchestrator admission (Phase 2). |

---

## Phase 1 — Correctness & safety (before any load push)

Small diffs, high confidence. Do them in this order: the outcome/metric fixes touch
the same orchestrator turn code, so land those together, then the queue bound that
relies on them.

**1. Make turn outcomes truthful.** In the orchestrator (`call_tts_clause`, `consume`,
`run_turn`): return an explicit per-clause result (success / partial / failed /
canceled); do **not** append a failed clause to `spoken_parts`; persist only what was
generated or sent, with a delivery status (sent/queued/unknown — never "heard" without
a playback ack); emit exactly one terminal turn outcome while keeping the existing
end-signal protocol. *Verify:* first-clause failure, mid-stream failure, later-clause
failure, and barge-in never record a complete spoken answer or failed text as heard.

**2. Fix the load-bearing metric miscounts** (do alongside #1 — same code region).
Count `orch_tts_calls_total` once per clause RPC, not per audio chunk. Instrument the
vLLM `GenerateStream` path with the same started/completed/tokens/TTFT semantics as
unary. Stop LLM latency timing at stream end, not after TTS consumption. Count a
barge-in only when speech overlaps an active agent turn (keep all speech-starts as a
separate counter). *Verify:* controlled calls with known chunk/token counts produce
correct counts; delayed TTS does not inflate LLM duration.

**3. Bound the STT audio backlog.** The per-stream `segments` queue in `Transcribe`
(`stt/service/server.py:387`) is unbounded; the reader fills it in real time while one
worker drains it. Add per-stream bounds by **count and buffered-audio age/bytes** (a
count bound alone can still hold minutes of stale speech) with enqueue timestamps and a
safe default. On overflow, stop the turn/stream with the explicit outcome from #1 —
never silently splice. *Verify:* fast producer + stalled consumer across many streams
stays bounded; accepted audio preserved; no hang on full-queue teardown.

**4. Hard-bound TTS admission.** In `tts/service/server.py`, replace the
unbounded-wait-outside-the-semaphore with an atomic admitted-count reservation plus a
bounded gate; reject overflow immediately for both unary and streaming; release exactly
once, with native-running work keeping its slot until it exits. Keep the existing static
`TTS_QUEUE_TIMEOUT_S` as the wait bound for now (deadline-derived timeout comes in Phase
2). *Verify:* with `1+MAX_QUEUE` capacity, exactly that many admit during a stalled
worker; extras reject promptly.

**5. Fix cancellation cleanup.** STT must hold its GPU slot until the executor forward
actually exits (TTS already does this), releasing exactly once. A native forward can't
be killed by a timeout, so a wedged worker must **report itself unready** (the
supervisor that restarts it comes in Phase 3). *Verify:* cancel during a forward; the
slot is not freed early; no leaked slots.

**6. Stop STT copying the whole buffer on the event loop.** `_find_quiet_cut(bytes(buf),
…)` (`:398`) copies the entire buffer each frame though it only scans the tail. Copy
only the trailing scan window — but the returned offset is **absolute** into the buffer
passed, so a tail-only copy must rebase before `del buf[:cut]`, and no numpy/memoryview
view may outlive the bytearray resize. *Verify:* byte-identical cut offsets vs the
current implementation for quiet/forced/long cases; lower loop lag under concurrent
readers.

---

## Phase 2 — Visibility & bounded overload (before the real ramp)

Now make the system observable and overload bounded, so you can push traffic and
believe the numbers.

**1. Add loop-lag & queue metrics.** An event-loop drift gauge in each Python service;
per-queue count / capacity / oldest-age / executing gauges plus wait/reject/expiry
counters. Note `stt_gpu_inflight` includes waiters — add a true "executing" gauge and an
explicit TTS occupancy gauge.

**2. Add offered-load counting & reconciliation.** Add an attempts counter at the edge
**before** rejection/upgrade (`edge_sessions_total` starts too late). Extend the load
harness (`pipeline_test/stream_wav.py` or a sibling) to write durable per-attempt
records with distinct attempt IDs. Reconcile generator-started vs edge-observed vs
admitted vs rejected. A generator schedule is offered demand, not proof it hit the
server.

**3. Build the saturation dashboard and repair the KPI queries.** New
Prometheus-datasource dashboard in `monitoring/grafana/dashboards/` (wired via the
provisioning yaml), with rows for: offered vs served (attempts vs `orch_active_sessions`);
rejections per stage; per-stage saturation (executing/queued/oldest-age, LLM active vs
queued); loop lag; resources (GPU util/VRAM vs CPU/RAM — the idle-GPU story in one
glance); and latency p50/p95 (kept, but demoted). Repair the ClickHouse KPI queries to
use the truthful outcomes from Phase 1. Aggregate histogram buckets across replicas
before computing percentiles; missing data must read **unknown**, not a healthy zero.
*Verify:* a run with known failures/interrupts matches the panels; no "No data".

**4. Add session & turn admission.** Enable `EDGE_MAX_SESSIONS`; add an authoritative
`ORCH_MAX_SESSIONS` using an **atomic reservation** (not a read of the Prometheus gauge)
acquired **before** opening downstream STT work and released on every close/fail path;
set `maximum_concurrent_rpcs` on the orchestrator server. Also bound active/queued
**turns** so many idle sessions can't all launch inference at once. *Verify:* N+1
simultaneous arrivals never exceed N; rejected sessions open no STT stream; live calls
survive a connection storm.

**5. Add turn deadlines.** Start a configurable turn deadline at turn commitment,
account for already-aged audio, and propagate the remaining budget through admission and
each downstream RPC (each stage's timeout ≤ remaining budget; this is where TTS's
per-clause timeout gets capped). Reject aged STT results before starting the LLM. Use
monotonic clocks. *Verify:* a timeout during admission / inference / clause production
leaves no leaked tasks/slots or stale audio.

**6. Un-throttle the LLM (env only).** Now that metrics are honest, set the candidate
values in the deployment `.env`: `ADAPTIVE_LLM_MAX_INFLIGHT_REQUESTS=48`,
`LLM_MAX_NUM_SEQS≥48`, `LLM_GPU_MEMORY_UTILIZATION=0.90`, `LLM_MAX_MODEL_LEN=2048`,
`LLM_MAX_TOKENS=256`. Starting points to sweep in Phase 3, not final. Don't touch the
facade's slot logic.

**Done when:** a *guarded* 2/4/8/16 ramp runs with offered-vs-served, per-stage
saturation, and clean rejection all visible. This is where a real capacity number starts
to exist.

---

## Phase 3 — Measured scaling & ops readiness

Do these once Phase 2 gives a baseline and the dashboard shows where the wall is.
Replica counts and caps come from **measurement**, not this doc.

Status: items 1-5 shipped and verified against a real 5-service bare-process
run over the actual WebRTC path (multiple real pre-existing bugs found and
fixed along the way — model/engine mismatch, port collisions, orphaned
subprocesses on shutdown, two benign-race telemetry false-failures). Items 6-7
partially done (a manual 2/4/8 ramp, not the full automated sweep tool). Full
writeup: **[docs/PHASE3_RESULTS.md](PHASE3_RESULTS.md)**.

**1. Storage & telemetry health.** [x] `session_store.py`: bounded Redis connection/op
waits, sized pool, atomic append+trim, retention separate from `LLM_HISTORY_MAX_TURNS`.
`session_logger.py`: keep inserts off the request path, export queue-age/drops/last-flush,
clean up sequence state on every teardown. Test *slow* Redis/ClickHouse, not just
connection-refused. Telemetry loss must be visible, never block audio.

**2. Exporters, readiness, alerts.** [x] Wire host/container + GPU exporters (DCGM tried
first per this doc's original wording; it reports zero metrics on consumer/GeForce GPUs,
`nvidia_gpu_exporter` used instead — see results doc); real backend readiness that
detects worker death behind a live facade; graceful drain + supervisor restart for a
wedged worker (this completes the "report unready" fix from Phase 1); alerts on
oldest-work-near-deadline, no-audio/failure rate, telemetry loss, scrape failure,
memory/disk. Distinguish an expected overload rejection from an admitted-call failure.

**3. Replica ports & scrape discovery.** [x] Remove fixed host-published STT/TTS metrics
ports from scaled Compose services (they collide); scrape each replica's internal
endpoint via DNS/Docker discovery; bare-process mode needs distinct app + metrics ports.
(`scripts/run_pipeline.py` + `monitoring/prometheus.yml` `file_sd_configs`.)

**4. Replica fan-out (round-robin).** [x] Point the orchestrator's STT/TTS channels at
`dns:///<host>:<port>` with `grpc.lb_policy_name=round_robin` + keepalive (default gRPC
uses only the first address); verify the resolver in the deployed runtime. STT sticks
per session, TTS routes per **clause** (preserve clause order); keep **one** LLM worker
on the GPU. No auto-retry after partial speech (it duplicates audio). *Verify:* two
replicas both get work; a dead replica is avoided on new streams. (Used gRPC's
multi-address `ipv4:` scheme rather than a `dns:///` resolver — no DNS server in this
bare-process topology; verified live against a two-backend channel.)

**5. Load & recovery harness.** [x] A reproducible driver over the **real WebRTC path** with
independently scheduled (open-loop) arrivals, a workload seed, and durable outcomes. Run
generators on separate machines and measure their own CPU/scheduling lag so generator
saturation can't masquerade as server capacity. Stage-only benchmarks are diagnostics,
not the SLO. (`pipeline_test/open_loop_load.py`; no separate generator machine on this
dev box, so its lag reporting is the substitute — see results doc's concurrency=8 finding.)

**6. CPU replica/thread sweep (bounded — not a NUMA research project).** [ ] Record physical
cores/SMT/NUMA/cgroup quota (done — `scripts/orchestration/hardware.py`). Sweep replica count × per-worker threads (e.g. 2/4/8),
warming each; control the ONNX/OpenMP/BLAS/llama.cpp thread pools; measure combined
STT+TTS+LLM-host contention, peak RSS, tail latency. Pick the **smallest** allocation
meeting the target; stop when useful throughput plateaus. A dedicated `sweep_replicas.py`
automating this matrix is not built yet — deferred, see results doc.

**7. Tune the one vLLM instance.** [ ] Sweep facade inflight + `LLM_MAX_NUM_SEQS` together,
then batched-tokens/context/output/mem; test chunked prefill and cold/warm prefix cache;
avoid KV preemption and TTS backlog. Keep shared source defaults conservative; write
measured values to the deployment `.env` only. `adaptive_llm/tools/vllm_tune.py` already
exists for this; not yet rerun with this box's 4 GB card / `gemma-3-1b-it` in mind.

---

## Phase 4 — Deferred / gated (build only if measurement shows the need)

Do **not** build these before Phase-3 data justifies them. They are correct ideas for
problems we have not yet observed on this box.

**Playback flow-control credits.** `media-edge/.../playout.go` drops oldest audio when
TTS outruns playback (`:131`). Add playback-occupancy-driven credits + client
first-audible measurement — **only if** the dashboard shows real overflow/underrun under
target load. Until then, marking an overflowed turn "incomplete" (Phase 1) is enough.

**Playback-aware scheduling & exact-audio cache.** Prioritize first clauses; cache
greeting/overload audio. Pure optimization; gated on evidence of starvation or synth
saturation at the overload boundary.

**Full soak/burst certification.** The multi-hour soak + repeated over-limit bursts (see
the capacity protocol below) is the release gate, run once a candidate operating point
exists — not upfront work.

---

## Horizontal-scaling design (reference for Phases 3–4)

### What pins where (statefulness)
- **STT:** per-session PCM lives inside the `Transcribe` bidi RPC → session pinned to one
  STT replica for the call.
- **TTS:** a new stream **per clause** → round-robins per clause; preserve order.
- **LLM (vLLM):** stateless; prefix cache is a per-replica optimization.
- **Orchestrator:** in-memory only for the RPC's life; durable state in **Redis** → any
  instance can serve any session (connection stickiness).
- **Redis/ClickHouse:** the only true cross-instance state.

Redis does **not** restore live PCM, cancellation tasks, or playback position after a
crash — reconnect/failover needs explicit behavior + tests, not just the state audit.

### One worker per instance
Each STT/TTS instance = **one in-flight forward + a genuinely bounded waiting room**.
Scale by process count, not threads inside a process.
- **TTS (llama.cpp):** context not thread-safe; multiple concurrent forwards need
  multiple model copies = multi-process without the isolation. Multi-process wins.
- **STT (ONNX):** concurrent `Run()` is safe, but on CPU a forward already spans many
  cores — cores are the budget, not processes. Keep one worker; revisit only after
  reentrancy + throughput/memory tests.
Benefits: no thread-safety risk, clean core-pinning, crash isolation, cheap RAM copies.

### Fan-out and load balancing
Default gRPC name resolution uses only the first address. Use client-side round-robin on
a single box; an Envoy/nginx proxy is the heavier alternative (zero orchestrator change)
— defer to multi-node. Topology: **N× STT + N× TTS + 1× LLM.**

### Prometheus is not optional here
Each service exposes a fixed metrics port; `--scale` makes replicas share it on distinct
IPs, so a static scrape target sees only one. Use service discovery, or you scale blind —
the exact failure this whole effort exists to fix.

### Sizing
`replicas ≥ ceil(offered_work_per_s / measured_work_per_replica_per_s)` with headroom —
an **estimate only**; bandwidth, queueing, and burst synchronization make it nonlinear.
RTF < 1 for one synthesis does not prove concurrent callers meet first-audio deadlines.
Publish a measured envelope (connected sessions, speaking/answering turns, arrivals/s,
timely-complete-answers/s, latency distribution, rejections, recovery time).

### Multi-node (future)
L4 LB in front of media-edge; shared/clustered Redis+ClickHouse; stages as K8s
Deployments behind headless Services; autoscale on the saturation metrics. The state
audit enables it but doesn't establish reconnect/ICE reachability — test those.

---

## Guardrails (violating these breaks the design)
- **One vLLM instance.** The LLM scales inside it (env), never by duplicate weights on
  one GPU.
- **STT/TTS worker stays single.** Scale by process count.
- **Never free native-running capacity because a coroutine timed out.** Hold the slot
  until the forward exits; mark a wedged worker unready for restart.
- **Never silently drop speech, persist failed text as heard, or mark a partial answer
  complete.** Interruption is accounted separately from failure.
- **Telemetry/Redis stay best-effort & bounded** — a dependency failure gets bounded
  waits + explicit status, never tears down a live call.
- **Don't raise buffers/threads/LLM concurrency just to make rejections fall.** Reject
  cleanly instead.
- Correctness fixes don't need a flag that reverts to false-success behavior; new
  limits/tuning are configurable with validated safe defaults.

---

## Capacity test protocol (the release gate)

1. **Freeze a manifest:** git SHA, model/revision/quant, runtime/driver versions,
   effective config, CPU/NUMA/quota/RAM, warmup/cache state, corpus, prompt/output
   distributions, seed, run ID. Agree first-audio / completion / interruption latency
   targets, allowed admitted-turn failure rate, queue-age ceiling, recovery deadline —
   **no capacity is certified without these thresholds.**
2. **Workload matrix:** isolated stages → combined; conversation-paced; open-loop
   arrivals incl. random bursts; synchronized speech-endings; short/long + warm/cold
   cache; silence/noise + rapid interrupts; slow receivers/loss/reconnect storms;
   dependency faults (worker crash/hang, slow Redis/ClickHouse, scrape loss). Powers of
   two to locate the knee, then finer steps; hold each step until backlog behavior is
   clear.
3. **Certify steady + burst + recovery:** multi-hour soak at the candidate limit;
   repeated over-limit bursts must reject promptly while protecting admitted work;
   measure drain time and that sessions/slots/RSS/Redis/telemetry/playback return to
   baseline unaided. A nice p95 on a small successful minority is **not** a pass.
4. **Publish:** manifest + offered/admitted/terminal counts + timely-complete-answers/s +
   latency/sample counts + bottleneck + rejection & recovery behavior. Unreconciled
   attempts or missing telemetry = invalid measurement, not a result.

---

## Summary
Make it **truthful** (Phase 1), make it **visible and bounded** (Phase 2), then scale on
**measurement** (Phase 3) and only gold-plate if the data demands it (Phase 4). `.env`
scales the LLM inside one instance; STT/TTS scale by replicas + round-robin; the whole
thing lifts to multi-node because only Redis/ClickHouse hold cross-instance state.
