# Design — Real-Time Voice Pipeline: Barge-in, Pipelining, WebRTC Return Path, Backpressure

- **Date:** 2026-09-01
- **Status:** Approved for planning
- **Scope target:** Works on the 4 GB dev box; architecture carries to the L40S topology unchanged.

## Problem

Code review found that three headline product behaviors are scaffolded but not
wired, and they are code-structure gaps, not hardware gaps:

1. **Barge-in does not exist end to end.** `USER_SPEECH_STARTED` only bumps a
   turn counter (`orchestrator/server.py:583`); nothing aborts the in-flight
   LLM/TTS, flushes egress, or stops playback. The TTS server's `CancelledError`
   handling (`tts/server.py:158`) is dead — nothing triggers it. The browser
   schedules Web Audio buffers it never stops (`web/index.html:84`).
2. **No LLM→TTS pipelining.** The orchestrator drains the entire LLM stream, then
   opens TTS (`server.py:325-373`). Time-to-first-audio = full decode + first TTS
   sentence. The streaming LLM interface is currently pointless.
3. **Return audio bypasses WebRTC.** TTS PCM goes back as base64 over the
   signaling WebSocket and is played by the browser's Web Audio clock
   (`rtc.go:367`, `index.html:72`). No Opus/RTP/jitter; the edge cannot own the
   barge-in cut.

Plus two load/quality gaps: unbounded orchestrator queues (`server.py:192`), and
a VAD-only endpointer.

## Decisions (locked)

- **Return path:** real WebRTC outbound Opus track at the edge.
- **Opus encoder:** CGo libopus via `hraban/opus` (adds `libopus-dev` build dep).
- **Barge-in history:** persist user text + the assistant prefix actually spoken,
  flagged `interrupted`.
- **Endpointing:** SraVaani is batch (no revisable partials), so true STT-stability
  endpointing is out of reach. Ship the achievable subset (empty-turn gate,
  min-speech gate) behind the same VAD seam; streaming-ASR swap is future work.

## Architecture

Per-session turn processing changes from *inline-blocking* to a *cancellable
streaming pipeline* with an interrupt controller. The return audio plane moves
from browser Web Audio to a real edge-owned Opus track with a flushable playout
buffer. Each component below is independently testable behind its current seams.

### A. Orchestrator — interrupt controller + cancellable turns

**File:** `orchestrator/server.py` (the `Stream` coroutine).

- Introduce a per-session **turn epoch** (monotonic int) and a single
  **current-turn `asyncio.Task`** holding the LLM+TTS work.
- `from_stt` no longer awaits LLM/TTS inline. On `turn_complete` it *spawns* the
  turn task (epoch = active_turn) and keeps ingesting STT.
- **Interrupt trigger** — on `USER_SPEECH_STARTED`, or a newer end-of-turn while a
  turn task is live: call `interrupt(epoch)`:
  1. `task.cancel()` → cancels the LLM `GenerateStream` grpc call and the TTS
     `SynthesizeStream` grpc call (grpc-aio cancels on task cancellation; TTS
     server already holds its slot correctly on `CancelledError`).
  2. **Flush `outbound`** of frames tagged with the cancelled epoch.
  3. Emit a `STOP_PLAYBACK` control on `outbound` (edge already has the enum).
  4. Persist the spoken prefix (see D/persistence).
- **Race guard (R4):** every `outbound` audio item carries its turn epoch; the
  main relay loop drops any item whose epoch != current before yielding. Late
  frames from a cancelled turn never reach the edge. This is the invariant the
  correctness test asserts.

### B. Orchestrator — LLM→TTS pipelining

**File:** `orchestrator/server.py`.

- The LLM token loop splits `text_delta`s into **clauses** at sentence
  punctuation, with a max-char safety flush (so a punctuation-free reply still
  streams). Clauses go onto a **bounded clause queue** (maxsize small, e.g. 4).
- A **clause consumer** pulls clauses and calls `SynthesizeStream(clause)`
  sequentially, streaming 20 ms frames to `outbound` as they arrive. Producer and
  consumer run concurrently inside the turn task; both die on turn cancel.
- No proto change — reuses the existing per-sentence TTS stream. TTFA becomes
  first-clause latency. Trade-off accepted: N TTS calls/turn (serialize on the
  single TTS worker) and slightly reduced cross-sentence prosody.
- **Text reply** (`AgentReply`) is still sent once, when the full reply text is
  known (assembled from clauses), so the UI transcript is unaffected.

### C. Orchestrator — bounded queues / backpressure

**File:** `orchestrator/server.py`.

- `q` (STT ingest) and `outbound` (to edge) become **bounded** `asyncio.Queue`s.
- When `q` is full, stop draining `request_iterator`; grpc-aio flow control then
  backpressures the edge — bounded memory, no silent degradation (N3).
- `outbound` bound paces TTS production to edge drain speed naturally.
- GPU admission stays with the backends (STT `MAX_INFLIGHT`, TTS `MAX_QUEUE`
  semaphores already exist). The orchestrator owns per-session buffer bounds +
  stale-turn drop only; documented as such (no fake "GPU admission" claim).

### D. Edge — WebRTC Opus return path + flushable playout

**Files:** `media-edge/cmd/media-edge/rtc.go`, `bridge.go`, new `playout.go`;
`media-edge/web/index.html`; `go.mod`.

- **Track:** add a local Opus `TrackLocalStaticSample` to the PeerConnection
  before `CreateAnswer` (today the PC is receive-only, `rtc.go:29`).
- **Playout buffer (`playout.go`):** receives TTS PCM (24 kHz mono) from the
  orchestrator over the existing `ServerEvent.audio`; resamples 24k→48k;
  Opus-encodes 20 ms frames (libopus); writes to the track on a steady 20 ms
  ticker. This IS the jitter buffer. Bounded ring; overflow drops oldest.
- **Barge-in cut (local, instant):** the edge already detects speech onset
  (`bridge.go:371`). On that edge it **flushes the playout ring and stops
  emitting** — sub-frame, before any network round-trip. Also honors an incoming
  `STOP_PLAYBACK` control from the orchestrator (orchestrator-initiated stop).
- **Browser:** delete the base64/Web Audio path (`index.html:72,84`); attach the
  inbound remote track to an `<audio>` element via `srcObject`. Net deletion.

### E. Endpointing — achievable subset

**Files:** `orchestrator/server.py`, `media-edge/cmd/media-edge/bridge.go`.

- **Empty-turn gate (orchestrator):** if a turn's joined STT text is empty/
  whitespace, do not spawn a turn task (silence false-endpoint no longer burns an
  LLM+TTS cycle). Still recorded as an event for auditing.
- **Min-speech-duration gate (edge):** require a minimum cumulative speech length
  within a candidate before `END_TURN` fires, so a cough/click doesn't open a
  turn. Extends the existing VAD attack/hangover; a new tunable env.
- VAD-hangover endpoint stays, tunable, behind the same seam — the documented
  swap point for a streaming ASR later.

### F. Enum rename (`START_TURN` → `END_TURN`, edge→orch)

**Files:** `proto/edge.proto`, regenerated `edge_pb2*.py` / `edge.pb.go`, callers.

- The edge's `START_TURN` means "turn ended" (`server.py:597`). Rename to
  `END_TURN` to match the STT contract and kill the landmine. Mechanical; new
  field number reuses the slot (wire value unchanged) to stay compatible.

## Data flow (post-change, one turn with barge-in)

1. Edge: speech → PCM up (gRPC) + `USER_SPEECH_STARTED`; on end-of-speech,
   `END_TURN`.
2. Orchestrator: STT segments join; on `turn_complete` with non-empty text →
   spawn turn task (epoch N).
3. Turn task: LLM stream → clauses → TTS per clause → epoch-tagged PCM frames on
   `outbound` → edge playout → Opus track → browser `<audio>`.
4. User barges in: edge flushes playout locally (instant) + sends
   `USER_SPEECH_STARTED`. Orchestrator `interrupt(N)`: cancel task, flush
   `outbound` epoch N, send `STOP_PLAYBACK`, persist spoken prefix as
   `interrupted`. New turn epoch N+1 begins.

## Persistence changes

**File:** `orchestrator/session_store.py`. `record_turn` gains an optional
`interrupted: bool = False` flag (stored in the turn record + `turn_complete`
event). The reserved `control.cancel_generation` field (already present,
`session_store.py:48`) is unused by this design — the interrupt is in-process via
task cancellation, not a Redis-polled flag; leave it or remove in cleanup.

## Error handling

- LLM/TTS per-turn isolation preserved: a failed turn task logs + emits an error
  event and does not kill `from_stt` or the session.
- `CancelledError` from a deliberate interrupt is distinguished from a real error
  (no error metric/event on intentional cancel).
- Playout: a libopus encode error drops that frame (mirrors the existing
  decode-error skip), never kills the session.
- Bounded-queue full is backpressure, not an error.

## Testing (production-load bar)

- **Interrupt correctness (R4):** unit/integration test — inject a
  `USER_SPEECH_STARTED` mid-reply; assert (a) no epoch-N frames yielded after the
  cut, (b) `STOP_PLAYBACK` sent, (c) spoken-prefix persisted `interrupted=True`,
  (d) turn N+1 proceeds. Add a rapid double-barge-in case.
- **Pipelining:** assert first `outbound` audio frame is emitted before the LLM
  stream completes (first-clause latency).
- **Backpressure:** fill `q`; assert the edge iterator stops being drained and
  memory stays bounded (no unbounded growth).
- **Playout (Go):** table test for resample + 20 ms framing + flush-on-barge-in
  (ring emptied, ticker stops emitting). Seam: encode func injectable.
- **Endpointing:** empty-turn text → no turn task; sub-min-speech candidate → no
  `END_TURN`.

## Risks

- **R-Opus:** CGo libopus adds a build dep; CI/setup scripts must install
  `libopus-dev`. Mitigation: update `setup_ubuntu.sh`; document in edge README.
- **R-Prosody:** per-clause TTS is choppier than whole-utterance. Mitigation:
  clause min-length so we don't synthesize 2-word fragments.
- **R-Interrupt-races:** the epoch guard is the whole correctness story; covered
  by the dedicated test above before this is called done.
- **R-Playout-drift:** 24k→48k resample + fixed 20 ms ticker can drift vs. TTS
  arrival rate. Mitigation: bounded ring with drop-oldest + underrun = silence
  frames; never block the ticker.

## Out of scope

- Streaming ASR / true stability endpointing (model swap).
- Tools/RAG, auth, multi-node — unchanged from existing phase plan.
