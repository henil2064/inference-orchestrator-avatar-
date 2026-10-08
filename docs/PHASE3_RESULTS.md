# Phase 3 Results — Measured Scaling & Ops Readiness

Date: 2026-09-09. Box: Intel i9-11900 (8 physical / 16 logical cores), 62 GiB
RAM, 1× GTX 1650 4 GB. Same machine `docs/CPU_PROFILING_REPORT.md` used.

## What shipped

- **Storage & telemetry health** (`orchestrator/service/session_store.py`,
  `session_logger.py`): sized Redis pool + bounded per-op timeouts (a slow,
  not just refused, Redis can no longer hang a call), atomic turn-list
  trimming (`REDIS_TURN_RETENTION`, decoupled from `LLM_HISTORY_MAX_TURNS`),
  and three previously-invisible telemetry gauges
  (`orch_telemetry_queue_depth/dropped_total/last_flush_age_seconds`) that
  also fixed a real per-session memory leak (`SessionLogger._sequences` never
  shrank).
- **gRPC health checking** on STT, TTS, adaptive-llm, and the orchestrator,
  with STT/TTS wedge watchdogs (self-report `NOT_SERVING` when the sole
  worker's current forward/synthesis has run past a bound) and an
  adaptive-llm watchdog that detects the local worker subprocess dying behind
  a still-live facade.
- **Replica launcher & supervisor** (`scripts/orchestration/`,
  `scripts/run_pipeline.py`): hardware-detected (stdlib-only) auto-sizing of
  STT/TTS replica counts and per-replica thread counts, backward-compatible
  port allocation, process supervision (crash-restart with backoff, health-
  triggered kill-and-restart), graceful drain on SIGINT/SIGTERM, and
  Prometheus `file_sd` target generation.
- **Orchestrator replica fan-out**: `STT_TARGETS`/`TTS_TARGETS` build a
  multi-address gRPC channel (`ipv4:` scheme) with `round_robin` +
  client-side health checking — verified against a live two-backend channel
  that traffic splits across both and a `NOT_SERVING` backend gets excluded.
- **Monitoring**: `node_exporter`, `nvidia_gpu_exporter` (see below),
  `cadvisor`, `file_sd`-based STT/TTS scrape discovery, `monitoring/alerts.yml`
  (Prometheus-native rules), and 4 new dashboard panels (per-replica
  executing gauges, host CPU/mem, GPU util/VRAM).
- **Load harness**: `pipeline_test/open_loop_load.py` — seeded Poisson/
  jittered open-loop arrivals over the real WebRTC path, reporting its own
  scheduling lag so generator saturation can't be mistaken for server
  capacity (no separate generator machine on this box).

## Real bugs found and fixed along the way

Running the full 5-service pipeline together, bare-process, for the first
time surfaced several genuine pre-existing defects — not hypothetical, each
reproduced and fixed:

1. **STT's `.env` never loaded.** `_load_dotenv()` computed `stt/service/.env`
   (one directory too shallow) instead of `stt/.env`; every prior run relied
   on the shell already having the right variables exported.
2. **adaptive-llm's `.env` never loaded at all**, in any bare-process run —
   no loader existed. Docker deployments never noticed (Compose's `env_file:`
   loads it before Python starts). This silently ran the wrong model/engine
   (`medgemma_1_5_4b_q4` via llama.cpp instead of the configured
   `google/gemma-3-1b-it` via vLLM) with no error.
3. **The llama.cpp engine doesn't implement `GenerateStream`** (only
   `Generate`) — the orchestrator's actual call path — so bug #2 compounded
   into every real LLM call failing outright until the model/engine mismatch
   was fixed.
4. **`ADAPTIVE_LLM_WORKER_ADDR` defaulted to the same port as `TTS_ADDR`**
   (`127.0.0.1:50054`) — silently broken the moment both services ran on the
   same box together, which bare-process mode always does.
5. **No SIGTERM handler in `adaptive_llm/main.py`** — a supervisor's plain
   SIGTERM killed the facade via the OS default action, skipping its
   `finally` block, orphaning the local worker subprocess (observed directly:
   `vllm_worker.py`/`llamacpp_worker.py` processes surviving, reparented,
   still holding the worker port).
5b. **`go run` orphaned media-edge on shutdown** the same way — it spawns
   the compiled binary as a child rather than exec-replacing itself, and that
   child doesn't reliably receive a signal sent to the `go run` process.
   Fixed by having the supervisor launch a real compiled binary
   (`scripts/run_pipeline.py` now runs `go build` once and execs the result)
   instead of `go run`.
6. **media-edge misreported two benign end-of-call races as errors**: the
   client's own courtesy `SESSION_CLOSE` send failing with `io.EOF` (the
   orchestrator had already closed its side first) and the client's `"stop"`
   WebSocket message cancelling the *entire* gRPC stream context out from
   under a still-draining final-reply wait, not just the audio read loop it
   was meant to stop. Both turned already-successful turns into reported
   failures — this is what `docs/LOAD_SCALING_PLAN.md`'s Phase 1 truthful-
   telemetry principle is about, just found one layer further out than
   Phase 1 looked.
7. **The health watchdog used one timeout for two different questions.**
   `wedge_restart_after_s` (15s default) governed both "this was healthy and
   stopped responding" (a real wedge — 15s is appropriate) *and* "this has
   never responded healthy yet" (cold model load — vLLM alone measured
   ~20-30s on this box). A slow-but-fine cold start could have been killed
   and restarted, plausibly forever if cold-start time stayed near the
   threshold. Fixed with a separate, much longer `startup_grace_s` (180s
   default) that applies only before the *first* successful health check;
   the tight bound now only fires for a genuine post-startup wedge.
8b. **No circuit breaker on crash-looping.** A fundamentally broken replica
   (bad config, missing model file) would have retried forever on capped
   backoff, never giving up or telling the operator clearly. Fixed:
   `max_consecutive_failures` (default 8, resets on any run that stays up
   past `stable_uptime_s`) stops retrying that one replica and marks
   `given_up`, logging loudly — every other replica keeps running normally,
   and `run_pipeline.py` prints a summary of any given-up replicas and exits
   non-zero.
9. **`ProcessSupervisor`'s own event log was silently swallowed** —
   `logging.info()` with no handler configured — so `scripts/run_pipeline.py`
   had zero visibility into start/restart/health-kill events until wired to
   a plain `print()`.

None of these are Phase-3-specific corner cases: #2–#6 would have broken the
*first* real attempt to run this project's own documented "no Docker for the
5 services" bare-process workflow with all 5 services live together.

## Measured envelope (this box)

Auto-sized replica plan on this box: 1 STT replica × 4 threads, 1 TTS
replica × 4 threads (`worker_cores=5` after reserving 2 control-plane + 1 GPU
facade core leaves too little room for a second replica of either at the
measured-safe 4-thread width — see `scripts/orchestration/plan.py`). vLLM
auto-selected `LLM_MAX_NUM_SEQS=1` (the `model-profiles.json` GTX-4GB
profile) for `google/gemma-3-1b-it`.

Closed-loop ramp (`pipeline_test/load_burst.py`, real WebRTC path via
media-edge, 6 s test utterance):

| Concurrency | Elapsed | Outcome |
|---|---|---|
| 2 | 18.3 s | 2/2 completed |
| 4 | 22.1 s | 4/4 completed |
| 8 | 71.4 s | 7/8 completed, 1 failed (client-side timeout, no turns) |

At concurrency 8, `orch_active_sessions` reached 8 but `orch_turns_active`
capped at 4 (Phase 2's `ORCH_MAX_ACTIVE_TURNS` admission gate doing exactly
its job — queuing the rest, not failing them), while `adaptive_llm_active_requests`
(max 2), `tts_admission_occupancy` (max 4/5), and `orch_event_loop_lag_seconds`
(~0) all stayed comfortably inside their bounds. **No server-side queue
overflowed and nothing was rejected** — the one failure was a client-side
generator timeout with no turn ever recorded. Per the capacity protocol's own
rule, this is read as **inconclusive, not a certified wall**: 8 concurrent
real `aiortc` WebRTC clients (DTLS/SRTP + Opus codec work each) on the *same*
8 physical cores as the service under test is exactly the generator-
saturation-masquerading-as-server-capacity scenario `open_loop_load.py`'s own
lag reporting exists to catch — there is no separate generator machine here.

**This box's number is not the point.** The replica/thread auto-sizing
formula (`scripts/orchestration/plan.py`) is: given the same code, more
physical cores yield proportionally more STT/TTS replicas automatically
(verified: 128 simulated physical cores → 10 STT + 16 TTS replicas from the
same formula, no code change). What should change on a bigger box is the
number of replicas and the vLLM concurrency profile — not this document.

## Explicitly deferred (not built this pass)

- **Full automated CPU replica/thread sweep** (`docs/LOAD_SCALING_PLAN.md`
  Phase 3 item 6): the matrix-driving harness described in the plan (M7)
  wasn't built as a standalone script this pass — the manual 2/4/8 ramp above
  used the tooling that exists (`load_burst.py`, `open_loop_load.py`,
  `run_pipeline.py --only`) but a dedicated `sweep_replicas.py` automating
  the matrix is a reasonable next increment, not attempted here given this
  box's single-replica-each auto-sizing leaves little to sweep.
- **vLLM engine-parameter tuning** (`adaptive_llm/tools/vllm_tune.py` already
  exists and sweeps `gpu_memory_utilization` × batched-tokens × `max_num_seqs`
  at concurrency 1/2/4 — rerun it with candidates sized for this 4 GB card
  and `gemma-3-1b-it` before trusting `LLM_MAX_NUM_SEQS` above 1).
- **DCGM-based GPU metrics**: tried first per the plan's original wording,
  returns zero metrics on this GTX 1650 (DCGM's field set doesn't cover
  consumer/GeForce GPUs) — `nvidia_gpu_exporter` (nvidia-smi-based) is wired
  instead and verified working.
- **Alertmanager routing**: `monitoring/alerts.yml` rules are live and
  verified firing correctly (caught this box's own 94%-full root filesystem
  during testing), but no notification channel is wired since none was
  specified.
