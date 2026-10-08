# SraVaani STT on the small-VRAM GPU box

`scripts/sravaani_gpu.py` runs the SraVaani transducer in **fp16 on the GPU**. This is a
TorchScript RNNT model through `transformers` — **not vLLM**, so none of the
`bench_concurrency.py` / KV-cache / `gpu_memory_utilization` knobs apply. It fits a
4 GB card easily (~0.9 GB in fp16).

Why a dedicated venv: `AutoModel.from_pretrained(...).to("cuda")` can silently
**upcast the fp16 model to fp32** (doubles VRAM, slower); this runner explicitly
requests fp16. Keep SraVaani isolated from the orchestrator's environment.

## Setup (Ubuntu GPU box)

```bash
cd stt
python3 -m venv .venv-sravaani
source .venv-sravaani/bin/activate
pip install -U pip
pip install -r requirements-sravaani.txt

# CUDA torch (NOT the default CPU wheel). cu124 supports pre-Ampere cards (sm_75) fine:
pip install torch --index-url https://download.pytorch.org/whl/cu124

# confirm the GPU is visible — must print: True <gpu name>
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Point it at the model

Set the folder that contains `model-asr.fp16.ts` (from the SraVaani bundle you copied
onto this box):

```bash
export SRAVAANI_MODEL=/home/aiplay/SraVaani_portable/model
```

## Run

```bash
# transcribe a file
python scripts/sravaani_gpu.py clip.wav

# batch
python scripts/sravaani_gpu.py a.wav b.flac c.ogg

# reuse an existing capture from the bench dir
python scripts/sravaani_gpu.py ../orchestrator/received/*.wav

# live mic (record 7s)
python scripts/sravaani_gpu.py --record 7
```

## Verify it's really fp16 on GPU

The run prints `device=cuda, dtype=torch.float16` and `VRAM: ~900 MiB`.
- `device=cpu` → CUDA torch install failed (redo the torch step).
- `VRAM: ~1700 MiB` → it loaded fp32 (you're not running this script / dtype was overridden).

## Live STT service (gRPC) — the streaming pipeline

`server.py` hosts SraVaani once and emits immutable bounded transcript segments to
the orchestrator. Since SraVaani is batch-only, it buffers a turn and transcribes a
segment at a quiet point after `STT_MAX_SEGMENT_SEC` (or force-cuts after its grace
period), then emits the final segment at the VAD end-of-turn. Startup warms the
runtime input shapes so calls do not pay first-shape autotune stalls.

**One-time: generate the gRPC stubs** (needs `grpcio-tools`, which lives in the
orchestrator venv):

```bash
cd ..                       # repo root
source orchestrator/.venv/bin/activate 2>/dev/null || true
bash scripts/gen-proto.sh   # emits bindings into stt/proto/ and orchestrator/
```

**Run the STT service** (own venv; pin GPU on a multi-GPU box):

```bash
cd stt && source .venv-sravaani/bin/activate
export SRAVAANI_MODEL=/path/to/SraVaani_portable/model
# CUDA_VISIBLE_DEVICES=1  STT_ADDR=[::]:50052  STT_MAX_SEGMENT_SEC=8.0   (optional)
python server.py            # logs "warmup done — ready", then listens
```

**Run the orchestrator** (its own venv, points at the STT service):

```bash
cd orchestrator && source .venv/bin/activate
# STT_TARGET=localhost:50052   STT_VAD_GATE=1 (only forward after speech)   (optional)
python server.py
```

Then drive real sessions from the media edge. Transcript shows up in the
orchestrator log as `session_id seg=N: "<text>" (turn-end|mid-cut)`.

## Notes
- 65 Indian languages + English. The gRPC service expects mono 16 kHz PCM16;
  the orchestrator/media edge supplies that format.
- Pre-Ampere GPUs (compute < 8.0, GTX 16xx / RTX 20xx) run fp16 fine here; bf16 is not used.
- Long turns are auto-split into bounded segments (STT_MAX_SEGMENT_SEC, default
  8s) so a single continuous utterance can't grow one giant window / OOM the
  encoder — see stt/service/server.py's module docstring.
