# Hardware Topology Tuner Plan

Status: design proposal.

This document describes an offline tuner that qualifies new hardware and
produces a measured deployment topology for the inference pipeline. The tuner
chooses service placement, replica counts, resource assignments, concurrency
limits, and routing weights for a defined workload and service-level objective
(SLO).

The tuner is not a production autoscaler. It runs controlled experiments,
keeps the currently promoted configuration recoverable, and promotes a new
configuration only after an independent confirmation run.

## Problem statement

The pipeline has three inference stages with different resource behavior:

- Adaptive LLM is a singleton and must run on a compatible GPU.
- STT can run one or more instances on CPU or GPU.
- TTS can run one or more instances on CPU or GPU.

The required STT and TTS capacity cannot be derived from model size or a single
inference-time measurement. It depends on the current machine and workload,
including:

- Turns and synchronized speech endings per second.
- Speech duration and STT segments per turn.
- Generated text length and TTS clauses per turn.
- Warm and cold cache behavior.
- CPU, memory-bandwidth, GPU, and VRAM contention between colocated services.
- Batching behavior and latency under concurrency.
- The latency deadline and permitted rejection rate.

For example, if a hypothetical STT request takes 200 ms and a hypothetical TTS
request takes 800 ms, TTS initially appears to require four times the service
capacity. That ratio is only a seed for experimentation. TTS may run multiple
times per turn, GPU batching may be nonlinear, and colocated processes may
interfere with each other. No example number in this document is a measured
result.

The tuner must find the best complete topology for a named workload profile,
not a universally "perfect" configuration.

## Goals

- Detect the effective CPU, memory, GPU, VRAM, driver, and runtime environment.
- Measure each service on every supported device class on the current machine.
- Choose per-instance CPU/GPU placement, replica count, threads, and admission
  limits.
- Reserve one compatible GPU placement for the singleton Adaptive LLM.
- Measure interference instead of adding isolated benchmark results together.
- Route work according to measured replica capacity when CPU and GPU replicas
  coexist.
- Select only configurations that satisfy correctness, resource, latency, and
  recovery gates.
- Save enough evidence to reproduce and compare every decision.
- Emit an explicit deployment manifest and generated environment configuration.

## Non-goals

- Changing resource allocation continuously during live user traffic.
- Claiming capacity from hardware specifications without measurement.
- Maximizing queue sizes to hide overload.
- Retrying partially emitted speech on another TTS instance.
- Running multiple LLM weight copies on one GPU unless a future measurement and
  design explicitly justify it.
- Replacing the end-to-end WebRTC load test with stage-only microbenchmarks.

## Existing foundation

The first implementation should reuse the repository's existing components:

- `scripts/orchestration/hardware.py` detects CPUs, memory, and NVIDIA GPUs.
- `scripts/orchestration/plan.py` provides the current homogeneous STT/TTS
  replica and thread plan.
- `scripts/run_pipeline.py` builds process specifications, supervises replicas,
  performs health checks, and writes Prometheus discovery targets.
- `pipeline_test/load_burst.py` drives simultaneous real WebRTC sessions.
- `pipeline_test/open_loop_load.py` drives reproducible independent arrivals.
- `pipeline_test/report_recent_sessions.py` reads end-to-end ClickHouse results.
- `adaptive_llm/tools/vllm_tune.py` already evaluates predefined LLM candidates.
- Prometheus exposes per-stage queues, execution, rejection, saturation, and
  resource metrics.

The current planner and launcher create homogeneous replicas: all STT replicas
share one thread configuration and all TTS replicas share another. The tuner
requires a per-instance topology manifest so that CPU and GPU instances can be
mixed safely.

## Architecture

```text
hardware and software inventory
              |
              v
single-instance characterization
              |
              v
candidate topology generator <---- previous experiment results
              |
              v
apply -> start -> ready -> warm -> load -> settle -> collect
              |                                      |
              +--------------------------------------+
                              |
                              v
                 gates and candidate ranking
                              |
                 +------------+------------+
                 |                         |
             next candidate        confirmation run
                                           |
                                           v
                                  promoted manifest
```

The controller should be a Python CLI under `scripts/tuner/`. It controls the
existing process supervisor and test tools, but does not run inside a latency-
sensitive service.

Suggested package layout:

```text
scripts/tuner/
  cli.py                 command-line entry point
  inventory.py           hardware/software/model fingerprint
  manifest.py            topology schema and validation
  candidates.py          candidate generation and pruning
  controller.py          experiment state machine
  metrics.py             Prometheus and ClickHouse collection
  scoring.py             gates, confidence, and Pareto ranking
  storage.py             SQLite experiment history and artifacts
  routing.py             measured capacity data for runtime routing
  adapters/
    stt.py                STT benchmark and quality checks
    tts.py                TTS benchmark and quality checks
    llm.py                LLM benchmark and response checks
    pipeline.py           end-to-end WebRTC tests
```

## Topology manifest

A candidate represents the whole process topology. Replica count is derived
from its instance list rather than stored as one global number.

```yaml
schema_version: 1
profile: interactive_voice

workload:
  target_arrivals_per_second: 2.0
  burst_sessions: 16
  duration_seconds: 300
  seed: 4172

slo:
  timely_completion_rate_min: 0.99
  admitted_failure_rate_max: 0.005
  first_audio_p95_ms_max: 1500
  queue_oldest_age_ms_max: 500
  memory_utilization_max: 0.90
  gpu_memory_utilization_max: 0.90

instances:
  - id: llm-0
    service: adaptive_llm
    device: cuda
    gpu: 0
    env:
      ADAPTIVE_LLM_MAX_INFLIGHT_REQUESTS: "16"
      LLM_MAX_NUM_SEQS: "16"

  - id: stt-cpu-0
    service: stt
    device: cpu
    cpu_cores: [4, 5, 6, 7]
    threads: 4
    env:
      STT_MAX_INFLIGHT: "2"

  - id: tts-gpu-0
    service: tts
    device: cuda
    gpu: 1
    env:
      TTS_MAX_QUEUE: "2"

routing:
  stt:
    policy: predicted_earliest_finish
  tts:
    policy: predicted_earliest_finish
```

This is a schema example only. Values must come from measurement. The manifest
validator must reject duplicate ports, overlapping exclusive CPU allocations,
unsupported devices, unavailable GPUs, and total model memory above the safe
VRAM or RAM budget.

The launcher must translate each instance into a `ReplicaSpec`, including:

- Unique application and metrics ports.
- `CUDA_VISIBLE_DEVICES` and telemetry GPU identity.
- `STT_DEVICE` or `TTS_DEVICE`.
- CPU affinity and controlled Torch, ONNX, OpenMP, BLAS, and llama.cpp threads.
- Instance-specific concurrency, queue, and timeout settings.
- Generated `STT_TARGETS` and `TTS_TARGETS` for the orchestrator.

## Workload profiles

Optimization is meaningless without a workload contract. A profile must pin:

- Arrival process, rate, burst distribution, and random seed.
- Concurrent connected sessions and simultaneously active turns.
- Audio corpus and speech-duration distribution.
- Prompt history and requested output-length distribution.
- TTS clauses per turn and voice configuration.
- Warm/cold model and cache state.
- Target latency, completion, rejection, and recovery thresholds.

Useful initial profiles are:

- `interactive_voice`: prioritizes time to first audio and protects admitted
  turns during bursts.
- `steady_capacity`: measures maximum timely completed turns per second.
- `burst_recovery`: exceeds capacity deliberately and measures prompt rejection,
  queue drainage, and recovery to idle.

The tuner may produce a different recommended manifest for each profile.

## Measurement model

### Hardware and software fingerprint

Every tuning run records:

- Physical and logical cores, CPU affinity/cgroup limits, RAM, and NUMA layout.
- GPU model, UUID, compute capability, total VRAM, clocks, and power limit.
- OS, kernel, NVIDIA driver, CUDA, Python, Torch, ONNX Runtime, vLLM, SGLang,
  and llama.cpp versions.
- Git SHA and dirty-worktree marker.
- Model identity, revision, quantization, engine, and artifact checksum.
- Effective configuration, corpus checksum, workload seed, and run ID.

Results may be reused only when the relevant fingerprint is unchanged.

### Single-instance characterization

Benchmark every supported `(service, device, runtime, knob set)` combination.
At minimum this includes STT and TTS on CPU and each compatible GPU, and the LLM
on every GPU that can load it safely.

Record:

- Startup and warmup time.
- Warm and cold p50, p95, and p99 execution latency.
- Throughput and latency as concurrency increases.
- Peak and steady CPU, RAM, GPU utilization, and VRAM.
- Queue time, queue growth, rejections, failures, and timeouts.
- STT real-time factor, LLM TTFT/tokens per second, and TTS real-time factor.
- Service-specific quality or response-sanity results.

Compute time provides an initial service-rate estimate, but measured sustained
throughput is authoritative.

### Demand estimate

For initial candidate sizing, estimate stage demand from the workload:

```text
STT request rate = turns/s * mean STT segments/turn
LLM request rate = turns/s
TTS request rate = turns/s * mean TTS clauses/turn

offered service demand = request rate * mean service time
```

For approximately independent, non-batched workers, a first replica estimate is:

```text
replicas >= ceil(offered service demand / target utilization)
```

Target utilization should leave headroom, initially 0.65 to 0.75 rather than
1.0. This calculation only seeds the search. GPU batching and shared-resource
contention make the real system nonlinear.

### Colocation and interference

The tuner must measure promising colocations, including:

- LLM plus host-side STT/TTS CPU load.
- STT and TTS sharing one GPU.
- Multiple CPU replicas competing for cores and memory bandwidth.
- GPU work plus CPU tokenization, audio processing, and orchestration.

Do not calculate a topology's capacity by summing isolated benchmark rates.
Reject colocations that cause unsafe VRAM pressure, repeated allocation failure,
large tail-latency regression, or starvation of the LLM.

Keep explicit memory headroom for runtime allocations and fragmentation. A
model merely fitting in VRAM is not evidence that it is deployable.

## Multi-GPU placement

Adaptive LLM placement is a hard constraint:

- Exactly one Adaptive LLM service instance.
- It must run on a compatible GPU.
- Its model, KV cache, and runtime headroom must fit before other placements on
  that GPU are considered.

On a two-GPU machine, the candidate generator should compare topologies such as:

```text
A: GPU 0 = LLM, GPU 1 = TTS, STT = CPU
B: GPU 0 = LLM, GPU 1 = STT, TTS = CPU
C: GPU 0 = LLM, GPU 1 = STT + TTS, plus CPU replicas
D: GPU 0 = LLM, GPU 1 = TTS, plus CPU STT and TTS replicas
```

The labels do not imply preference. TTS may deserve the second GPU when its
measured demand is higher, but only end-to-end testing can establish that.

The candidate generator should normally test the LLM on each GPU first, select
the compatible placement with the best measured headroom/performance, and then
search remaining STT/TTS placements. It may consider sharing the LLM GPU only
when measured VRAM and compute headroom make it a valid candidate.

## Capacity-aware replica routing

The current round-robin fan-out assumes roughly homogeneous replicas. Equal
round robin is not suitable for a mixed pool containing a fast GPU replica and
slower CPU replicas.

The orchestrator needs a replica registry containing at least:

- Instance identity and health.
- Device and measured baseline capacity.
- Current queued and executing work.
- An exponentially weighted moving average of execution time.
- Recent failure, rejection, and deadline-expiry state.

The preferred routing policy is predicted earliest finish:

```text
estimated finish time = queued work estimate + predicted request duration
```

Static measured capacity weights can be the first implementation. Dynamic
least-work or predicted-finish routing is the target because request lengths and
runtime performance vary.

Routing requirements:

- STT selects a replica at stream/session creation and remains pinned because
  live PCM and segmentation state reside in that process.
- TTS may select per clause, but clause output order must remain correct.
- A replica reporting `NOT_SERVING` is immediately excluded from new work.
- Work is never retried after partial text or audio has been emitted.
- A scheduler must not send work to a replica whose predicted completion is
  beyond the remaining turn deadline.

## Candidate search

The initial implementation should use a transparent, bounded search rather
than an opaque optimizer.

### 1. Generate feasible candidates

Apply hard constraints before starting processes:

- LLM singleton and GPU requirement.
- Runtime and model compatibility.
- RAM and VRAM budgets with headroom.
- CPU/core and port availability.
- Maximum experiment replica count.
- Valid service and queue limits.

Use the demand estimates and single-instance profiles to generate likely
placements near the required capacity. Include a few candidates on both sides
of every estimate.

### 2. Successive evaluation

Evaluate candidates with increasing cost:

1. Startup, readiness, and warmup validation.
2. Short stage-level correctness and load test.
3. Short combined-pipeline test.
4. Repeated representative workload test.
5. Burst, overload, recovery, and soak certification for finalists.

Discard clearly invalid or dominated candidates early. Do not eliminate a
candidate because of an invalid benchmark run or missing telemetry.

### 3. Neighborhood refinement

Starting from the best survivors, test bounded single changes:

- Add or remove one STT/TTS replica.
- Move one replica between CPU and GPU.
- Change CPU cores or threads per instance.
- Change an execution-concurrency or batching setting.
- Adjust a routing weight.
- Reduce a queue or admission bound to protect the SLO.

Continue until no neighboring candidate produces a statistically meaningful
improvement, the experiment budget is exhausted, or all candidates fail gates.

### 4. Future optimizer

Once enough reliable history exists, a constrained solver or Bayesian optimizer
may propose candidates. It must remain subordinate to manifest validation,
hard gates, and empirical end-to-end confirmation.

## Gates and ranking

A candidate is eligible only when all hard gates pass:

- Test quality/correctness passes for every service.
- Offered load is reconciled against edge-observed attempts.
- Minimum admitted and timely completed work is met.
- Failure, timeout, and unexpected rejection rates remain below limits.
- Queue age remains within the turn deadline and drains after load stops.
- CPU, RAM, GPU, and VRAM stay below safety ceilings.
- No process crashes, OOMs, wedges, or unhealthy periods occur.
- Telemetry is complete enough to support the decision.
- The system returns to its idle resource and slot baseline within the recovery
  deadline.

Do not use one arbitrary weighted score for pass/fail decisions. First filter by
hard gates, then rank feasible candidates lexicographically:

1. Maximize timely completed turns per second at the target offered load.
2. Minimize burst p95 time to first audio.
3. Minimize p99 caller-visible latency.
4. Minimize rejection rate within the permitted overload policy.
5. Minimize CPU, RAM, GPU, and VRAM cost.
6. Prefer fewer replicas and a simpler topology when results are equivalent.

Also preserve a Pareto frontier so operators can compare, for example, a
lower-latency topology with a lower-resource topology.

A latency-only score is invalid: a configuration that rejects most requests
can report excellent latency for the small successful subset.

## Failure classification and pruning

Failures require different actions:

| Failure | Tuner action |
| --- | --- |
| Invalid manifest or incompatible runtime | Reject candidate before launch. |
| Startup/readiness failure | Reject candidate and save diagnostics. |
| OOM or hard memory boundary | Reject and prune descendants that consume more of the same resource. |
| SLO or quality failure | Reject this candidate, but continue other placements. |
| Intermittent request failure | Repeat according to the confidence policy. |
| Load-generator or telemetry failure | Mark experiment invalid; retry and never score it as healthy. |
| Worker wedge or failed recovery | Reject candidate and restore baseline. |

Do not always stop at the first failed configuration. Replica count, placement,
batching, and thread count are not globally monotonic. A failed high-thread
candidate does not prove that more replicas with fewer threads will fail.

## Experiment lifecycle

Every trial follows one state machine:

```text
planned -> validated -> applied -> ready -> warmed -> running
        -> settling -> collected -> gated -> ranked
```

Any failure transitions through cleanup and baseline restoration. Before a test:

- Verify that expected ports are free and no stale supervised processes exist.
- Snapshot the effective baseline configuration.
- Start only the candidate's declared processes.
- Wait for real backend readiness, not just an open port.
- Warm all relevant models and configured input-size buckets.
- Verify Prometheus targets and telemetry sinks before generating load.

After a test:

- Stop arrivals and allow bounded queue drainage.
- Collect the final metrics window and ClickHouse run/session records.
- Check recovery to idle and for leaked processes, slots, memory, or GPU usage.
- Stop the candidate and restore the previous promoted topology.

The test controller must survive interruption and restore the baseline on the
next invocation if the prior experiment did not finish cleanup.

## Result storage and promotion

Use SQLite for indexed experiment state and immutable JSON artifacts for raw
inputs/results. Store artifacts under a hardware fingerprint:

```text
tuning/results/<hardware-id>/
  inventory.json
  experiments.sqlite3
  runs/<run-id>/
    candidate.yaml
    effective-env.json
    workload.json
    raw-results.jsonl
    metrics.json
    decision.json
  recommended.yaml
  recommended.env
```

Each decision records:

- Candidate and parent candidate IDs.
- Complete hardware/software/model fingerprint.
- Workload profile and seed.
- Raw sample counts and metric windows.
- Gate outcomes with explicit reasons.
- Ranking values and confidence intervals.
- Whether the run was cold, warm, isolated, colocated, or end to end.

Promotion requires a new confirmation run of the selected topology. The
confirmation run must not reuse the exact candidate-selection samples. Write
the promoted manifest atomically and retain the previous recommendation for
rollback.

## Queue and admission policy

Queue limits, timeouts, and retry counts are protection policies, not resources
to maximize. Larger queues can reduce immediate rejection while making admitted
turns miss their caller-visible deadline.

The tuner should derive queue candidates from:

- Remaining turn deadline.
- Measured service-time distribution.
- Maximum admitted work that can finish within that deadline.
- Required burst tolerance and rejection policy.

Prefer prompt, explicit overload rejection over accepting work that is predicted
to expire in a queue. Increasing a queue is not evidence of increased capacity.

## Implementation phases

### Phase 1: Deterministic experiment runner

- Add versioned workload and topology manifest schemas.
- Add inventory capture and fingerprinting.
- Add the experiment state machine, SQLite storage, and safe cleanup/restore.
- Wrap existing stage and pipeline load tools as adapters.
- Save raw results and evaluate explicit gates.
- Support predefined candidate lists only.

Done when the same candidate can be run repeatedly, interrupted safely, and
compared from stored evidence without manual metric transcription.

### Phase 2: Per-instance topology launch

- Extend planning from homogeneous counts to per-instance `ReplicaSpec` entries.
- Add CPU affinity, device assignment, GPU pinning, and per-instance settings.
- Generate unique ports and Prometheus discovery targets.
- Validate RAM/VRAM/core constraints before startup.
- Emit complete `recommended.yaml` and `recommended.env` artifacts.

Done when one manifest can launch a mixed CPU/GPU STT/TTS pool and restore the
prior topology reliably.

### Phase 3: Capacity-aware routing

- Add an explicit replica registry to the orchestrator.
- Start with tuner-generated static capacity weights.
- Add live queue/executing state and EWMA execution estimates.
- Implement STT pinning and deadline-aware TTS selection.
- Verify health removal, no partial retry, and correct TTS clause ordering.

Done when mixed-speed replicas receive work proportional to useful capacity and
one slow replica does not dominate tail latency.

### Phase 4: Automated characterization and search

- Characterize service/device pairs automatically.
- Generate feasible candidates from measured demand and resource profiles.
- Add successive evaluation, dominance pruning, and neighborhood refinement.
- Measure selected CPU/GPU colocation interference.
- Preserve the feasible Pareto frontier.

Done when a new machine can progress from baseline to a defensible candidate
without a manually authored topology matrix.

### Phase 5: Certification and operationalization

- Run independent confirmation, burst/recovery, and soak tests.
- Publish a human-readable tuning report and measured capacity envelope.
- Add comparison against the previously promoted topology.
- Add regression thresholds for software/model changes on known hardware.

Done when the promoted result states the workload and SLO it supports, includes
rollback artifacts, and can be reproduced from its recorded manifest.

## Verification strategy

Unit tests:

- Manifest parsing, versioning, and constraint validation.
- Candidate generation and safe pruning.
- Gate evaluation, lexicographic ranking, and Pareto selection.
- Failure classification and interrupted-run recovery.
- Port, CPU, and GPU assignment.
- Capacity-weight and predicted-finish routing.

Integration tests:

- Fake services with configurable speed, memory, failure, and health behavior.
- A mixed fast/slow replica pool to prove routing is not equal round robin.
- Startup failure, OOM classification, wedge, and telemetry-loss handling.
- Controller interruption during every lifecycle state.
- Atomic promotion and rollback.

Hardware tests:

- Single-service CPU/GPU characterization.
- LLM plus CPU-worker contention.
- Shared-GPU STT/TTS contention where memory permits.
- Combined real WebRTC steady, synchronized burst, overload, and recovery runs.
- Repeated finalist runs in randomized order to reduce cache and thermal bias.

## Acceptance criteria

The tuner is complete when it can take a new supported machine and a versioned
workload profile, then:

1. Produce a complete and reproducible machine fingerprint.
2. Measure valid service/device combinations without assuming example timings.
3. Generate and run safe mixed CPU/GPU topology candidates.
4. Route work appropriately across heterogeneous replicas.
5. Reject candidates with incorrect, incomplete, unsafe, or unrecoverable runs.
6. Select a topology using timely completed work and caller-visible latency.
7. Confirm the selection independently and atomically publish rollback-capable
   deployment artifacts.
8. Report a measured capacity envelope rather than a hardware-derived promise.

## Recommended first increment

Build Phase 1 around a small explicit candidate file and the existing launcher
and load harness. Run each candidate repeatedly, apply hard gates, and generate a
comparison report. This creates trustworthy experiment data before introducing
per-instance placement and automated search. The next increment should add the
topology manifest and mixed-resource launcher, because capacity-aware routing and
the smarter search both depend on that representation.
