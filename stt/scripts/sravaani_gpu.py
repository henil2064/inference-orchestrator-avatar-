"""SraVaani STT runner tuned for the small-VRAM GPU box (e.g. 4 GB).

This is a TorchScript TDT/RNNT transducer run through `transformers` — NOT vLLM.
None of bench_concurrency.py's KV-cache / gpu_memory_utilization tuning applies here;
the model just loads and runs. It fits 4 GB comfortably (~0.9 GB in fp16).

The one trap: AutoModel.from_pretrained(...).to("cuda") UPCASTS the fp16 graphs to
fp32 (see modeling_sravaani._ensure_loaded — fp16 is kept only if the caller asks).
fp32 doubles VRAM and runs slower, so this runner forces fp16 on CUDA (fp32 on CPU,
which has no fp16 conv kernel).

Model dir: set $SRAVAANI_MODEL or pass --model PATH (the folder holding model-asr.fp16.ts).

Usage:
    export SRAVAANI_MODEL=/path/to/SraVaani_portable/model
    python scripts/sravaani_gpu.py clip.wav                # transcribe a file
    python scripts/sravaani_gpu.py a.wav b.flac            # batch of files
    python scripts/sravaani_gpu.py --record 7              # record 7s from mic, transcribe
    python scripts/sravaani_gpu.py --model /m clip.wav     # explicit model dir
"""
import argparse
import os
import sys
import time

import torch
from transformers import AutoModel

CUDA = torch.cuda.is_available()
DEV = "cuda" if CUDA else "cpu"
SR = 16000


def resolve_model_dir(cli_path):
    d = cli_path or os.environ.get("SRAVAANI_MODEL")
    if not d:
        sys.exit("no model dir: set $SRAVAANI_MODEL or pass --model PATH "
                 "(the folder containing model-asr.fp16.ts)")
    if not os.path.isfile(os.path.join(d, "model-asr.fp16.ts")):
        sys.exit(f"{d} has no model-asr.fp16.ts — point --model at the SraVaani model/ folder")
    return d


def load(model_dir):
    dtype = torch.float16 if CUDA else torch.float32  # fp16 on GPU, fp32 on CPU (no fp16 conv)
    print(f"Loading {model_dir}  (device={DEV}, dtype={dtype}) ...")
    t0 = time.time()
    m = AutoModel.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=dtype).to(DEV).eval()
    if CUDA:
        torch.cuda.synchronize()
        print(f"Model loaded in {time.time() - t0:.1f}s. VRAM: {torch.cuda.memory_allocated() / 2**20:.0f} MiB")
    else:
        print(f"Model loaded in {time.time() - t0:.1f}s (CPU).")
    return m


def record(seconds, out="mic.wav"):
    import numpy as np
    import sounddevice as sd
    import soundfile as sf
    for c in ("3", "2", "1"):
        print(f"  starting in {c}...", end="\r", flush=True)
        time.sleep(1)
    print(f"\n>>> Recording {seconds}s -- SPEAK NOW <<<          ")
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    sf.write(out, audio, SR)
    print(f">>> Done (peak={float(np.abs(audio).max()):.3f}, >0.01 = heard you). Transcribing...")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="*", help="audio files to transcribe")
    ap.add_argument("--model", default=None, help="model dir (else $SRAVAANI_MODEL)")
    ap.add_argument("--record", type=int, metavar="SEC", help="record SEC seconds from mic instead")
    args = ap.parse_args()

    model_dir = resolve_model_dir(args.model)
    files = list(args.audio)
    if args.record:
        files.append(record(args.record))
    if not files:
        sys.exit("nothing to transcribe: pass audio file(s) or --record SEC")

    model = load(model_dir)
    if CUDA:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    hyps = model.transcribe(files, return_hypotheses=True)
    if CUDA:
        torch.cuda.synchronize()
    dt = time.time() - t0

    for path, h in zip(files, hyps):
        print(f"\n{path}\n  -> {getattr(h, 'text', h)!r}")
    tail = f", peak VRAM {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB" if CUDA else ""
    print(f"\ntranscribed {len(files)} file(s) in {dt:.1f}s{tail}")
    assert all(getattr(h, "text", h) is not None for h in hyps), "transcribe returned no text"


if __name__ == "__main__":
    main()
