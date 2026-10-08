# MedGemma-4B (GGUF) on vLLM

`server.py` hosts [bartowski/google_medgemma-4b-it-GGUF](https://huggingface.co/bartowski/google_medgemma-4b-it-GGUF)
via vLLM's async engine and generates one reply per completed turn, routed to it by
the orchestrator (see `orchestrator/server.py`'s `MediaBridge.Stream` — `call_llm`
fires once a Transcript with `turn_complete=True` arrives; PRD §4.2's
`context_builder`/multi-turn-history sub-module is separate, not-yet-built work;
this is single-turn only).

**Honesty check before you start:** this was written against vLLM's documented
`AsyncLLMEngine` API and GGUF-loading conventions, but has **not** been run
end-to-end anywhere in this repo's development — no GPU and no vLLM Windows
support in the dev sandbox that wrote it. Treat the first run as a real test, not
a formality. vLLM's Python API also shifts across versions; if `server.py` errors
on import or engine construction, check `AsyncEngineArgs`'/`AsyncLLMEngine`'s
current signature against whatever vLLM version you land on and adjust — the
surface used here is a handful of lines in `load_engine()`.

## Why a dedicated venv

vLLM installs its own CUDA-matched torch build. Keep it out of `stt/`'s SraVaani
env and `orchestrator/`'s env, same reasoning as `stt/requirements-sravaani.txt`.

**vLLM has no official Windows support.** Set this up on the Linux/WSL GPU box,
same as `stt/.venv-sravaani` (see `stt/RUN_SRAVAANI.md`).

**GGUF on vLLM is one of its less-traveled paths** — historically limited
quantization/parallelism support compared to native safetensors models, and
occasionally slower. If it turns out to be too rough in practice, the fallback is
converting/using a native HF-format MedGemma-4B checkpoint instead of the GGUF —
that only changes `LLM_MODEL_PATH`/`LLM_QUANTIZATION`, not `server.py`'s structure.

## Setup (Ubuntu GPU box)

```bash
cd llm
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

vLLM's own install may need a CUDA-version-matched extra index, e.g.:
```bash
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu124
```
(exact flag depends on your CUDA version and vLLM release — check vLLM's install
docs if the plain `pip install vllm` doesn't pull a working torch build).

## Point it at the model

```bash
export LLM_MODEL_PATH=/path/to/medgemma-4b-it.Q4_K_M.gguf
```

Or copy `.env.example` to `.env` and fill it in — `server.py` auto-loads it.

## One-time: generate the gRPC stubs

```bash
cd ..                       # repo root
source orchestrator/.venv/bin/activate 2>/dev/null || true
bash scripts/gen-proto.sh   # emits llm_pb2*.py into llm/ and orchestrator/
```

## Run

```bash
cd llm && source .venv/bin/activate
export LLM_MODEL_PATH=/path/to/medgemma-4b-it.Q4_K_M.gguf
# CUDA_VISIBLE_DEVICES=0  LLM_ADDR=[::]:50053  LLM_MAX_TOKENS=512   (optional)
python server.py            # logs "engine ready", then listens
```

Then point the orchestrator at it (`LLM_TARGET=localhost:50053`, on by default —
see `orchestrator/.env.example`) and drive a real session. The reply lands in the
orchestrator log and in the session JSON log's turn-closing entry (`llm_text`,
`llm_proc_ms`).

## Verify it's actually working

- Startup log should show `engine ready` then `LLM gRPC listening on ...`.
- If it OOMs on load, lower `LLM_GPU_MEMORY_UTILIZATION` or `LLM_MAX_MODEL_LEN`.
- If generation is very slow, try `LLM_ENFORCE_EAGER=1` first to isolate whether
  CUDA graph capture itself is the problem, and check `nvidia-smi` for actual GPU
  utilization during a call (near-zero util during generation usually means the
  request isn't reaching the GPU at all — a config/quantization mismatch, not a
  performance issue).

## Known rough edges (hit on a real GTX 1650 4GB box, vLLM 0.25.1)

- **`Unknown quantization method: gguf`** — don't set `LLM_QUANTIZATION=gguf`
  explicitly; leave it unset so vLLM auto-detects (GGUF is self-describing).
- **vLLM tries to JSON-parse the `.gguf` file itself and crashes** — a vLLM
  engine-construction bug (`maybe_override_with_speculators` reads `model=`
  expecting a real HF config). Fix: point `LLM_MODEL_PATH` at a *directory*
  containing the `.gguf` + a `config.json` copied from the original (non-GGUF)
  HF repo, not the bare `.gguf` file.
- **`max_model_len` defaults to the model's full native context** (131072 for
  Gemma3/MedGemma) if `LLM_MAX_MODEL_LEN` is unset — set it explicitly (e.g.
  `4096`) on small-VRAM boxes or the KV cache allocation will OOM.
- **`google/medgemma-4b-it` is a gated HF repo** — `LLM_TOKENIZER` pointing at
  it needs `huggingface-cli login` with an approved account, or point it at a
  local directory with the tokenizer files instead (`tokenizer.json`,
  `tokenizer.model`, `tokenizer_config.json`, `special_tokens_map.json`,
  `added_tokens.json`, `chat_template.jinja` — same directory as the model is
  fine).
- **CUDA OOM on a large single allocation at startup**, even with
  `LLM_GPU_MEMORY_UTILIZATION` and `LLM_MAX_MODEL_LEN` both turned down —
  that setting only bounds the KV cache reservation, not vLLM's
  chunked-prefill scratch buffer (`max_num_batched_tokens`, defaults to an
  internal heuristic like 2048). On a small/pre-Ampere card (forced to
  `float32` — no bf16 support, ~2x the memory of a card that has it) that
  buffer alone can exceed what's left. Try, roughly in order: `LLM_ENFORCE_EAGER=1`
  (skip CUDA graph capture's own upfront reservation), then lower
  `LLM_MAX_NUM_BATCHED_TOKENS` (e.g. 512, then 256). Also check `nvidia-smi`
  for other processes already holding GPU memory — every MiB counts on a
  ~3.6 GiB-usable 4GB card.

## Notes

- No conversation history, no RAG, no streaming tokens — single-turn request/reply
  by design (see `proto/llm.proto`'s docstring). PRD's eventual topology is
  MedGemma-27B FP8 behind an OpenAI-compatible HTTP surface with continuous
  batching + prefix caching; this local GGUF-on-vLLM setup is the practical
  smaller-scale stand-in for the same role during development.
- Unlike `stt/server.py`, there's no single-worker GPU lock here — vLLM's
  `AsyncLLMEngine` does its own continuous batching, so concurrent sessions'
  `Generate()` calls are NOT serialized by this service.
