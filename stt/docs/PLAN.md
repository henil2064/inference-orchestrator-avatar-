# STT service — implementation plan

Streaming speech-to-text for the voice agent. This directory holds a **separate,
GPU-pinned Python process** that transcribes each session's audio live and streams
partial + final transcripts back to the orchestrator. Read the repo `docs/PRD.md`
(§4.4, §8.3) and `DECISION.md` before starting; this plan assumes that context.

> **STATUS / PIVOT (see DECISION.md D-13).** The model was switched from
> **Qwen3-ASR/vLLM to SraVaani** (portable TorchScript TDT/RNNT, fp16). Everything
> below that is *vLLM/`streaming_transcribe`-specific is now stale*:
> - **Phase 0 is MOOT** — its gate tested vLLM scheduler interleaving; SraVaani has no
>   scheduler. `bench_concurrency.py` + the vLLM `requirements.txt` are dead weight.
> - **Streaming = bounded-segment transcription** (SraVaani is batch-only): buffer PCM
>   per session, cut a segment once it exceeds STT_MAX_SEGMENT_SEC (quiet-point-seeking,
>   bounded fallback) or the turn ends, transcribe each segment exactly once. No more
>   growing-window re-transcription or provisional partials — every emitted Transcript
>   is complete; see `stt/service/server.py`, `stt/tests/test_segment.py`, `stt/docs/RUN_SRAVAANI.md`.
> - **Cold start solved** by a warm-up forward at service startup.
> - The **gRPC seam is model-agnostic** and matches Phase 1/2 as written — only the
>   *inside* of the service changed. Built: `proto/asr.proto`, `stt/service/server.py`,
>   orchestrator wiring in `orchestrator/server.py`. Run `scripts/gen-proto.sh` to
>   emit stubs before starting either process.

## Historical Qwen/vLLM plan (superseded by D-13/D-16)

- **Model:** `Qwen/Qwen3-ASR-0.6B`, served via **vLLM**.
- **Path: true-streaming, in-process.** Use `Qwen3ASRModel.LLM(...).streaming_transcribe(seg, state)`
  with a **per-session `state`** object, NOT the OpenAI-compatible `vllm serve` HTTP
  endpoint. The HTTP endpoint is segment-only (no incremental partials, forces us to
  re-window). The streaming API keeps per-session state and emits growing transcripts
  as audio arrives — that is what a full-duplex barge-in agent needs.
- **Topology:** STT runs on **GPU1** (`CUDA_VISIBLE_DEVICES=1`), a separate process from
  the orchestrator. Orchestrator ↔ STT over **gRPC bidi streaming** (same seam style as
  `proto/edge.proto`). Keeps the engine + state isolated, restartable, and GPU-pinned.
- **Input audio:** mono 16 kHz PCM16 — exactly what the media edge already decodes and
  the orchestrator already forwards. `streaming_transcribe` wants a float32 numpy array
  at 16 kHz, so convert PCM16 → float32 (`/32768.0`) at the service boundary.

### The catch the next agent MUST understand

The clean "vLLM serve owns concurrency, orchestrator is a thin HTTP client" story only
holds for the **segment** path. We chose the **streaming** path, so **we host the engine
and the per-session state ourselves**, and concurrency across sessions is **unproven** —
the upstream example only shows one session in a loop. **Phase 0 exists to settle this
before any product code is written.** Do not skip it.

## Environment — use a venv (hard requirement)

This process pulls heavy, version-sensitive deps (vLLM, torch, CUDA) that MUST NOT mix
with the orchestrator's env. Keep it isolated in **this directory's own venv**.

```bash
cd stt
python -m venv .venv
# Linux/prod:   source .venv/bin/activate
# Windows dev:  .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

`.venv/` is git-ignored (add it if the root `.gitignore` doesn't already cover it).
`requirements.txt` starts from the Qwen3-ASR toolkit's pinned vLLM + torch versions —
do NOT float these; vLLM/torch/CUDA compatibility is exact. Pin from the upstream repo.

> Note on OS: vLLM is Linux/CUDA. On the Windows dev box, Phase 0 may only run against
> CPU/segment mode or not at all — the real Phase 0 numbers must come from the L40S Linux
> node. Don't take a Windows result as the concurrency verdict.

## Phase 0 — De-risk (do this first, throwaway code, no product wiring)

Goal: prove the streaming path works on real audio AND handles concurrent sessions.
Use the existing captures in `../orchestrator/received/*.wav` as input — they're real
decoded session audio.

1. Load `Qwen3ASRModel.LLM(model="Qwen/Qwen3-ASR-0.6B", gpu_memory_utilization=<low>)`.
   **Tune `gpu_memory_utilization` down** (target ~0.3–0.4). The upstream example uses
   0.8, which would starve TTS on the shared GPU1. A 0.6B model does not need 0.8.
2. Run the streaming example against 2–3 `received/*.wav` files. Confirm partials look
   sane on real audio. Feed in `step`-sized chunks (start ~500 ms).
3. **The gate — concurrency probe:** drive **N concurrent "sessions"** (start N=4, push
   to the phase-1 target) through `streaming_transcribe`, each with its OWN `state`
   object, dispatched from a thread pool. Measure whether it **interleaves** (throughput
   scales, per-session latency stable) or **serializes** (latency stacks with N).
   - Interleaves → proceed with the streaming path as planned below.
   - Serializes → STOP and pick a fallback before building: (a) K engine replicas, or
     (b) the segment HTTP path (`vllm serve` + orchestrator-side windowing). Record the
     decision in `DECISION.md`.
4. Leave the probe script in `stt/scripts/` as `bench_concurrency.py` with an assert-based check
   (it's the one runnable check that proves the load-bearing assumption). Write the
   measured numbers into this PLAN or DECISION.md.

**Do not start Phase 1 until Phase 0's concurrency number is in hand.**

## Phase 1 — STT gRPC service

1. **Proto** — add `proto/asr.proto` (bidi stream, mirror `edge.proto` style):
   - client → server: `oneof { AudioChunk audio; Control control }` where `AudioChunk`
     is `{ string session_id; bytes pcm16; int64 ts_ms }` and `Control` carries
     `SESSION_OPEN` / `SESSION_CLOSE` per `session_id`.
   - server → client: `Transcript { string session_id; string text; bool is_final;
     int64 ts_ms }`.
   - Regenerate Python stubs with the existing `scripts/gen-proto.sh` (extend it to emit
   `asr_pb2*.py` into `stt/proto/`).
2. **Service** (`stt/service/server.py`):
   - Load the vLLM engine **once** at startup (GPU1).
   - On `SESSION_OPEN`: create `state = asr.init_streaming_state(...)`, store in
     `{session_id → state}`.
   - Per `AudioChunk`: accumulate PCM into a small per-session buffer; once ≥ `step`
     samples, convert to float32 numpy and call `streaming_transcribe(seg, state)` in a
     **thread-pool executor** (the call is blocking — never run it on the event loop).
     Emit a `Transcript` partial from `state.text`.
   - On `SESSION_CLOSE`: emit the final transcript, then **delete the state** (guard the
     leak before it exists).
   - Concurrency model (one shared stream+dispatcher vs. one stream per session) is
     decided by Phase 0's interleave result — do not hardcode before that's known.

## Phase 2 — Wire the orchestrator

Edit `../orchestrator/server.py`:
- Open one gRPC channel to the STT service at startup.
- In `MediaBridge.Stream`, **replace the debug-wav buffering** with: forward each
  session's PCM chunks to STT, receive `Transcript`s, and **log them per session**
  (`session_id: "<text>" (partial|final)`). That logged live transcript IS the phase
  deliverable — no WebSocket needed yet.
- **Gate STT on speech:** only forward audio when the edge VAD says the user is speaking
  (the edge already emits `USER_SPEECH_STARTED`). Prevents transcribing/hallucinating on
  silence. Keep the debug-wav path behind an env flag if still useful for debugging.

## Phase 3 — Verify at load

- Run several concurrent live WebRTC sessions end-to-end. Confirm partials stay live and
  independent per session.
- Measure STT final latency + partial cadence; report against `PRD.md` §7 targets.

## Deliberately deferred (do NOT build now — YAGNI)

- **WS `/v1/sessions/{id}/events`** (PRD §8.1) — logging partials proves the block; add
  the WebSocket when a real client consumes transcripts.
- **Endpointer / end-of-turn → LLM trigger** — belongs to the LLM block. STT only emits
  `(text, is_final)`; something downstream decides "turn is over."
- **Speech-*stopped* control event edge→orch** — the streaming path feeds audio
  continuously, so no segment-boundary signal is needed for STT. (It'll matter for the
  endpointer later, not here.)
- **Formal swappable-adapter interface** — one concrete service is enough; formalize a
  protocol only when a second STT backend actually appears.

## Open risks to watch

- **R-STT-1 · Concurrency of `streaming_transcribe`** (the Phase 0 gate). Highest risk.
- **R-STT-2 · Medical-vocab WER.** Qwen3-ASR is a strong *general* model, not medical.
  Accept the gap for phase 1; domain fine-tuning / biasing is a later track. (MedASR was
  evaluated and rejected — dictation-only, not conversational, non-OSS license.)
- **R-STT-3 · Silence hallucination.** Mitigated by VAD-gating in Phase 2.
- **R-STT-4 · VRAM on shared GPU1** with TTS — mitigated by low `gpu_memory_utilization`.

## References

- Model: https://huggingface.co/Qwen/Qwen3-ASR-0.6B
- Toolkit + streaming example: https://github.com/QwenLM/Qwen3-ASR
  (`examples/example_qwen3_asr_vllm_streaming.py`)
- Repo context: `docs/PRD.md` §4.4/§6/§8, `DECISION.md`, `proto/edge.proto`,
  `orchestrator/server.py`.
