# Decision log

Important decisions and *why*, newest section last. The canonical design lives in
[docs/PRD.md](docs/PRD.md); this file records the choices — especially the
implementation ones not spelled out there — so we don't relitigate them later.

Format per entry: **decision**, why, and the alternative we rejected.

---

## Architecture (locked — see PRD for full rationale)

**D-1 · Polyglot by plane.** Go + Pion for the WebRTC media edge (+ VAD
fast-path) → gRPC bidi (localhost/UDS) → Python asyncio orchestrator → Python GPU
backends.
*Why:* real-time media termination and per-packet work want a compiled,
goroutine-per-session runtime with no GIL; the orchestration/ML ecosystem is
Python. Split at the gRPC seam so each plane uses its best-fit runtime.
*Rejected:* all-Python edge (GIL + GC jitter on the media path), all-Go stack
(no ML ecosystem).

**D-2 · LLM = MedGemma-27B FP8 on one GPU via vLLM.** Continuous batching +
prefix cache.
*Why:* fits one L40S at FP8; continuous batching is how many sessions share one
GPU without per-session locking.
*Rejected:* tensor-parallel across both GPUs (TP=2) — the box has **no NVLink**,
so cross-GPU P2P over PCIe would bottleneck. Keep each model on one GPU.

**D-3 · GPU Topology A baseline** (LLM on GPU0; STT+TTS on GPU1), with Topology B
(2nd LLM replica on GPU1) as a config flip for the high-concurrency case.
*Why:* isolates the latency-critical LLM; the flip is a knob, not a rewrite.

**D-4 · Full-duplex + barge-in, multi-turn, English-only phase 1, medical
domain. Tools/RAG built but feature-flagged OFF for beta. No auth yet. Single
node, Docker Compose.**
*Why:* scope control — ship the hard real-time path first; defer breadth
(languages, tools, authz) until the core loop is proven.

---

## VAD & Silero integration (2026-08-20)

**D-5 · Energy RMS gate is the default VAD; Silero neural VAD is opt-in.**
Silero engages only when `--vad-model` / `vad_model_path` is set (plus
`--onnx-lib`); otherwise the pure-Go RMS gate runs.
*Why:* the binary must build and run with zero native dependencies. Silero drags
in cgo + an onnxruntime shared library; making it opt-in keeps dev/CI/local
frictionless and lets prod turn it on when the `.so` is in place.
*Rejected:* Silero always-on (crashes every run without the native lib);
energy-only (can't tell speech from speech-level noise — music, other voices).

**D-6 · One shared Schmitt-trigger state machine (`step()`); only the metric
differs.** Both paths reuse the same attack/hangover/hysteresis debounce; the
energy path feeds it RMS-vs-threshold, the Silero path feeds it
probability-vs-threshold.
*Why:* the debounce logic is the genuinely tricky, tested part — write it once.
The ponytail comment already anticipated "swap in Silero behind this same seam."
*Rejected:* a second parallel state machine for Silero (duplicate logic, two
things to keep correct).

**D-7 · Vendored the two files we use from `plandem/silero-go` instead of
importing it.** `internal/silero/silero.go` = its `model.go` + `onnx.go` (MIT),
depending directly on `yalue/onnxruntime_go`.
*Why:* upstream ships **no `go.mod`**, so `go get` synthesizes a `go1.16` module
and the library's own generics / `max` / `unsafe.Slice` fail to compile — it
cannot be a dependency as-is. We only need `Model.Predict` (~190 lines); its
streaming `Detector` + miniaudio deps are unused.
*Rejected:* forking upstream to add a `go.mod` (more maintenance surface); a local
`replace` pointing at a checkout (drags in the whole repo).

**D-8 · Model.Predict (per-chunk probability), not the high-level Detector.** We
accumulate 10 ms RTP frames into Silero's required 512-sample windows and predict
per window, feeding our own state machine.
*Why:* the Detector's callback fires on *completed* speech segments (start+end
together) — too late for low-latency barge-in, which needs the **start** edge
ASAP. Per-chunk probability gives us that; our debounce owns endpointing.
*Rejected:* Detector callback (wrong latency shape for barge-in).

**D-9 · Lock-free, per-session concurrency; no shared mutable state.** Each
session's VAD struct, Silero `Model`, and accumulator live inside its one
goroutine. The Silero model is stateful and not thread-safe, so it is **not**
shared.
*Why:* the whole point of edge VAD is per-session locality that scales across
cores with no contention. A shared model would need a lock that serializes
inference — the exact "resource locked while others wait" anti-pattern we're
avoiding.
*Rejected:* a shared model pool (reintroduces a lock; deferred to only-if-RAM-bites,
see README scaling TODOs).

**D-10 · onnxruntime loaded at runtime via `--onnx-lib`, installed per machine,
git-ignored.** The C API header is vendored in `onnxruntime_go`; nothing links at
build time — the library is `dlopen`ed at runtime.
*Why:* keeps the repo and binary free of a ~100 MB platform-specific blob; each
OS installs its own from Microsoft's releases. cgo means build **on** the target
(don't cross-compile). The 2.3 MB Silero model *is* vendored (`models/`) — it's
tiny and platform-independent.

**D-11 · `VAD_DEBUG` env var + a startup line naming the active VAD.** Throttled
per-session `speaking=… level=…` log; startup states energy vs Silero.
*Why:* the silent energy fallback once hid a misconfigured Silero (level read as
raw RMS, never crossed threshold). A loud "which VAD is active" line + a live
metric make "is it working / how do I tune thresholds" answerable in seconds.

---

## Release (2026-08-20)

**D-12 · Semver minor bump per feature; `v0.5.0 → v0.6.0` for Silero VAD.** Build
on the target OS because of cgo; release artifacts are per-platform, not
`GOOS`-flipped from one host.
*Why:* cgo can't be trivially cross-compiled; a Linux prod binary must be built on
Linux with a C toolchain present.

---

## STT model swap (2026-08-21)

**D-13 · STT model = SraVaani (portable TorchScript TDT/RNNT, fp16), not
Qwen3-ASR/vLLM.** Runs via `transformers` in its own `.venv-sravaani`, GPU-pinned,
behind a model-agnostic gRPC seam (`proto/asr.proto`).
*Why:* SraVaani is multilingual (65 Indian langs + English), fits the 4 GB box in
~0.9 GB fp16, and profiled at RTF ~0.04 (25× real-time) with the encoder as the only
cost (see `stt/sravaani_profile.py`). Qwen/vLLM's KV-cache tuning was fighting the
small card for no benefit at this model size.
*Rejected:* Qwen3-ASR via vLLM (the original PLAN.md decision) — heavier deps,
vLLM/torch pinning pain, and the streaming path's concurrency was unproven.

**D-14 · "Streaming" = growing-window re-transcription, since SraVaani is
batch-only.** Buffer each session's PCM; every ~0.7s re-transcribe the whole turn and
emit a partial; one final at `SESSION_CLOSE`. Snapshots are zero-padded up to 1 s
buckets to bound the distinct input shapes.
*Why:* SraVaani has no incremental `streaming_transcribe`/state API. Turns are short
and VAD-gated, so re-running the fast encoder each tick is cheap. Bucket-padding
avoids the per-new-shape kernel-autotune stall the profiler surfaced.
*Rejected:* fixed sliding window (needs word-stitching, drops early context);
per-chunk committed decoding (no partial revision). **Phase 0's vLLM concurrency gate
is therefore moot** — SraVaani has no scheduler; GPU forwards serialize behind a
single-thread pool (a 4 GB card can't run parallel forwards anyway).

**D-15 · Silero VAD is now the DEFAULT (was opt-in in D-5), best-effort with
energy fallback; VAD_DEBUG on by default.** `--vad-model` defaults to the vendored
`models/silero_vad.onnx` and `--onnx-lib` to `third_party/onnxruntime/libonnxruntime.so`
(where `setup_ubuntu.sh` puts it). If onnxruntime can't load, the edge logs a
warning and falls back to the energy RMS gate instead of `Fatalf`-ing.
*Why:* energy-only can't tell speech from speech-level noise, which drives the
endpointer (D-14) — the better detector should be the default now that STT is live.
The graceful fallback preserves D-5's "runs with zero native deps" property, and
`VAD_DEBUG` defaulting on (set `VAD_DEBUG=0` to silence) makes the active VAD and
live metric obvious. *Amends D-5, which made Silero opt-in.*

---

## Pipeline, state, and observability (2026-08-27)

**D-16 · SraVaani emits bounded, immutable STT segments, not repeated
growing-window partials.** A session buffer is cut near a quiet point once it
reaches `STT_MAX_SEGMENT_SEC` (or forcibly after a short grace period); the
orchestrator joins ordered segments until the VAD end-of-turn segment
(`turn_complete=true`). Inputs are padded to one-second buckets and representative
buckets are warmed at startup.
*Why:* retranscribing an ever-growing turn made both per-forward work and latency
grow with a user's uninterrupted speech. Bounded segments cap the worst case,
while quiet-point cuts preserve words better than arbitrary cuts. Bucketing avoids
new-shape autotune stalls. *Supersedes D-14's growing-window re-transcription and
revisable partials.*
*Rejected:* unbounded re-transcription (increasing cost), arbitrary fixed cuts
(more mid-word errors), and pretending batch-only SraVaani has incremental state.

**D-17 · Gate STT with edge VAD, retaining a configurable PCM pre-roll.** The
orchestrator buffers roughly `STT_VAD_PREROLL_MS` while the gate is closed,
flushes it on `USER_SPEECH_STARTED`, and sends `END_TURN` to STT when the edge
endpointer fires.
*Why:* it avoids transcribing silence and silence hallucinations without losing the
start of a word that began before the causal VAD decision.
*Rejected:* forwarding every frame (wastes the single STT worker) and dropping all
pre-speech audio (clips initial phonemes).

**D-18 · Redis is the sole durable source of logical session state and history.**
The orchestrator is its sole writer; it keeps metadata, bounded UI events,
coordination state, and completed user/assistant turns with active/closed TTLs.
Live PCM, WebRTC peers, gRPC streams, VAD/STT buffers, and model handles remain
process-local. The old JSON session log is retired.
*Why:* a reconnect or replacement orchestrator can recover the useful session
state, while putting live media objects in Redis would add latency, memory cost,
and fragile serialization.
*Rejected:* append-only local JSON as the source of truth, and persisting live
audio or transport objects.

**D-19 · Keep the `llm.v1/Llm.Generate` gRPC contract stable and put runtime
selection behind an adaptive LLM facade.** The facade reads a versioned model
registry, checks hardware and memory headroom, selects `llama.cpp` for GGUF or
vLLM for compatible Safetensors, pins the selected GPU, applies a model/GPU
runtime profile, and enforces bounded in-flight plus queued requests. The
orchestrator always calls the same endpoint.
*Why:* development hardware and production GPUs need different engines and
placement, but changing orchestration code or the wire contract for each model
experiment would make the pipeline brittle. Admission control turns saturation
into an explicit `RESOURCE_EXHAUSTED` result.
*Rejected:* embedding engine/model selection in the orchestrator, and accepting
an unbounded LLM queue.

**D-20 · Send structured completed-turn history to the LLM; enforce the final
token budget in the worker that owns the tokenizer.** The orchestrator loads only
a bounded number of Redis turns and sends USER/ASSISTANT messages; the worker
reserves output and safety tokens, then removes oldest complete pairs before
truncating current input. System prompts are selected by stable IDs from the
versioned prompt catalog (with an explicit environment override for experiments).
*Why:* the service rendering the exact chat template is the only component that
can count the real prompt safely, while stable prompt IDs make performance and
quality comparisons reproducible.
*Rejected:* raw transcript concatenation in the orchestrator and character-based
context limits.

**D-21 · Deliver completed text replies on the existing edge stream and ordered
WebRTC data channel.** The orchestrator writes `AgentReply` to the edge gRPC
bidi stream; the media edge relays it to the browser's data channel. TTS/audio
egress and interruption cancellation are still future work.
*Why:* it gives the client an immediate, session-scoped reply path without
misusing the signaling WebSocket or waiting for TTS.
*Rejected:* sending agent replies over signaling, and adding a parallel browser
transport solely for text.

**D-22 · Treat ClickHouse telemetry as an asynchronous, lossy observability
sink, separate from Redis state.** The orchestrator queues structured per-session
events and summaries, batches inserts in the background, and drops telemetry on
overflow or sink failure rather than blocking the voice path. Events carry typed
model/engine/device/version fields and timing profiles; Grafana queries
ClickHouse for session drill-down and fleet/version comparisons.
*Why:* measurement must not increase audio, STT, or LLM latency, and typed fields
make cross-model comparisons reliable without parsing JSON.
*Rejected:* synchronous telemetry writes in the media path, and using ClickHouse
as session-state storage.

## Turn lifecycle signalling (2026-09-02)

**D-23 · The orchestrator, not each service, tells the client when a turn is
over.** `edge.proto` gains a `turn_id` on `AudioFrame`, `Control` and
`AgentReply`, plus `AGENT_TURN_STARTED` / `AGENT_TURN_ENDED` (orch → edge, with
`metadata["reason"] = complete | interrupted | error`). `AGENT_TURN_ENDED` is
emitted on every exit path of `run_turn`, including a failed LLM that produced no
audio at all.
*Why:* a client needs to know whether the agent still owes it work. Per-service
"I have no pending work" cannot answer that: TTS is genuinely idle in the gap
between two clauses of the *same* turn, so such a broadcast is a lie about the
turn, and joining N services' idle claims lands back on the orchestrator — which
already knows the answer directly. Existing per-call completion (`is_final` on
`GenerateChunk`, `is_last` on `SynthesizeChunk`) is correctly scoped but
per-clause; only the orchestrator sees "no more clauses". `turn_id` is what keeps
a barged-in turn's trailing signal distinguishable from the next turn's.
*Rejected:* per-service idle/readiness broadcasts on the client-facing path.
Service-level idleness is still the right signal for **capacity** (admission
control, autoscaling) — a different consumer, a different cadence, and it must
not ride on the turn path.

**D-24 · The client gates on `PLAYBACK_COMPLETE` from the edge, not on
`AGENT_TURN_ENDED` from the orchestrator.** The edge holds the orchestrator's
`AGENT_TURN_ENDED` as a drain marker on the playout buffer and emits
`PLAYBACK_COMPLETE` over the data channel once the audio has actually gone out —
or immediately, as `interrupted`, when barge-in flushes the buffer.
*Why:* `AGENT_TURN_ENDED` means "last frame queued", which is up to
`EDGE_PLAYOUT_PREBUFFER_MS` earlier than "the user has heard it". Gating a mic or
a speaking indicator on it unmutes over the agent's own tail. The marker also
fixes a real bug: a final clause shorter than the prebuffer cushion used to sit
in the buffer unplayed, since the cushion could never fill again.
*Rejected:* forwarding `AGENT_TURN_ENDED` straight through to the browser, and
having the browser guess at drain with a timer.

**D-25 · Bound how long the playout cushion may hold buffered audio
(`EDGE_PLAYOUT_MAX_STALL_MS`, default 2x the prebuffer).** The cushion was
re-armed on every underrun, so each clause independently had to reach
`EDGE_PLAYOUT_PREBUFFER_MS` before it played. A clause shorter than the cushion
could therefore never satisfy it on its own.
*Why:* the LLM streams clause by clause, so a short sentence ("Okay.") is often
all that arrives before the next synthesis gap, and the cushion it waits on can
only be filled by audio that is still seconds away. The observable symptom was
that the first (long) clause of a turn played and the short ones after it did
not. The ceiling keeps the smoothness bias for the normal case — audio still
arriving gets the full cushion — while guaranteeing buffered speech is never held
indefinitely. `edge_playout_stall_releases_total` counts the releases.
*Rejected:* dropping the cushion (choppy playback), and relying on D-24's
end-of-turn drain alone (it un-strands the tail, but only at turn end, which
bunches short mid-turn clauses into the end of the reply).

**D-26 · `compute_ms` is measured inside the model worker, never from
submission.** `SynthesizeStream` started its per-sentence timer before
`run_in_executor`, so whenever the single-thread pool was busy the wait for a
free worker was billed as model time. Two concurrent requests reported
`realtime_factor` 0.66 and 1.23 for identical work. The timer now starts inside
the worker (matching the unary `Synthesize` path, which was always correct), the
wait is reported as `queue_ms` plus a `pool_wait_ms` attribute, and
`tts_pool_wait_seconds` exposes it to Prometheus. The orchestrator sums
`queue_ms` across a turn's sentences the same way it already summed `compute_ms`.
*Why:* `realtime_factor` is the number capacity planning reads. It has to be a
property of the model, not of how busy the service happened to be, or every
measurement taken under load argues for more hardware than the workload needs —
and the error grows with concurrency, exactly when the decision matters most.
*Rejected:* deriving compute by subtracting a separately measured queue time
(same clock, more places to drift), and leaving it and correcting in the
dashboard query (the raw column stays wrong for everyone else).

**D-27 · A GPU request that cannot be honoured fails at startup rather than
reporting a false device.** `llamacpp_backend.py` derived `self.device` from
`TTS_N_GPU_LAYERS` alone. A CPU-only `llama-cpp-python` wheel ignores that flag
in silence, so the backbone LM kept running on CPU while every emitted profile
claimed `cuda`. The backend now checks `llama_supports_gpu_offload()` and
`torch.cuda.is_available()` and raises with the remedy in the message; profiles
additionally carry `llm_device` and `codec_device` attributes.
*Why:* the backbone decode is nearly all of TTS compute and is memory-bandwidth
bound, so the silent fallback costs roughly an order of magnitude — and the wrong
device lands in ClickHouse, where it is read back as capacity data. A service
that will not start is a five-minute fix; a fleet dashboard confidently
attributing CPU numbers to GPU is not. Mirrors the existing STT ONNX provider
check, which already fails loudly on the same class of fallback.
*Rejected:* warning and continuing on CPU (the warning is invisible in a
container log six weeks later), and reporting the true device while honouring the
request (it hides a 10x regression behind a correct-looking metric).
