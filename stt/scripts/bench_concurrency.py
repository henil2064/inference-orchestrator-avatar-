"""Phase 0 probe (throwaway, see PLAN.md). Answers the one question that gates
Phase 1: does Qwen3ASRModel.streaming_transcribe INTERLEAVE across concurrent
sessions (throughput scales, per-session latency stable) or SERIALIZE (latency
stacks with N)? The streaming API is per-session state driven by us, not
`vllm serve` — concurrency is unproven until this runs.

vLLM is Linux/CUDA only — no Windows wheels. On a Windows dev box this needs
WSL2 (Ubuntu, with `nvidia-smi` working inside WSL), not native Windows.

On a small-VRAM dev GPU (e.g. a 4 GB laptop card), use --enforce-eager and a
low --max-model-len/--gpu-mem-frac to fit the model at all, and run with
--sessions 1 or 2 as a functional smoke test only. A "SERIALIZES" verdict from
a VRAM-starved dev GPU is a memory-pressure artifact, not evidence about
vLLM's scheduler — the real N=4+ concurrency numbers must still come from the
L40S Linux node (see PLAN.md). Requires stt/.venv.

Usage:
    python scripts/bench_concurrency.py [--sessions 4] [--step-ms 500] [--gpu-mem-frac 0.35]
    python scripts/bench_concurrency.py --sessions 1 --enforce-eager --max-model-len 2048 --gpu-mem-frac 0.5  # 4GB laptop GPU
"""

import argparse
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from qwen_asr import Qwen3ASRModel

MODEL = "Qwen/Qwen3-ASR-0.6B"
WAV_DIR = Path(__file__).resolve().parents[2] / "orchestrator" / "received"
SAMPLE_RATE = 16000


def load_wav_f32(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getframerate() == SAMPLE_RATE, f"{path} is not 16 kHz"
        pcm = w.readframes(w.getnframes())
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def transcribe_one(asr: Qwen3ASRModel, wav: np.ndarray, step: int) -> tuple[float, str]:
    """Drive one session start-to-finish with its own state. Returns (wall_seconds, text)."""
    state = asr.init_streaming_state(unfixed_chunk_num=2, unfixed_token_num=5, chunk_size_sec=2.0)
    t0 = time.monotonic()
    pos = 0
    while pos < wav.shape[0]:
        seg = wav[pos : pos + step]
        pos += seg.shape[0]
        asr.streaming_transcribe(seg, state)
    asr.finish_streaming_transcribe(state)
    return time.monotonic() - t0, state.text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=4, help="concurrent sessions to probe with")
    ap.add_argument("--step-ms", type=int, default=500, help="audio pushed per streaming_transcribe call")
    ap.add_argument("--gpu-mem-frac", type=float, default=0.35)
    ap.add_argument("--enforce-eager", action="store_true",
                     help="skip CUDA graph capture; needed to fit small-VRAM dev GPUs (e.g. 4GB laptop cards)")
    ap.add_argument("--max-model-len", type=int, default=None,
                     help="cap KV cache allocation; set low (e.g. 2048) on small-VRAM dev GPUs")
    ap.add_argument("--dtype", default=None,
                     help="force e.g. float16 on pre-Ampere GPUs (compute capability <8.0, no native bf16)")
    ap.add_argument("--max-num-seqs", type=int, default=None,
                     help="cap concurrent-sequence memory profiling; vLLM's default (256) OOMs small-VRAM "
                          "dev GPUs regardless of --gpu-mem-frac — set to >= --sessions")
    ap.add_argument("--max-audio-sec", type=float, default=None,
                     help="truncate each capture to this many seconds. The streaming state's prompt grows "
                          "with the whole session (running transcript + recent-audio window), so a long "
                          "capture eventually exceeds any fixed --max-model-len — shorten the input instead "
                          "of chasing that number upward on small-VRAM dev GPUs")
    args = ap.parse_args()

    wavs = sorted(WAV_DIR.glob("*.wav"))[:3]
    assert wavs, f"no captures in {WAV_DIR} — need real session audio to probe against"
    step = int(round(args.step_ms / 1000.0 * SAMPLE_RATE))
    audio = [load_wav_f32(w) for w in wavs]
    if args.max_audio_sec is not None:
        cap = int(args.max_audio_sec * SAMPLE_RATE)
        audio = [a[:cap] for a in audio]

    extra = {"enforce_eager": True} if args.enforce_eager else {}
    if args.max_model_len is not None:
        extra["max_model_len"] = args.max_model_len
    if args.dtype is not None:
        extra["dtype"] = args.dtype
    if args.max_num_seqs is not None:
        extra["max_num_seqs"] = args.max_num_seqs

    print(f"loading {MODEL} (gpu_memory_utilization={args.gpu_mem_frac}, {extra})...")
    asr = Qwen3ASRModel.LLM(model=MODEL, gpu_memory_utilization=args.gpu_mem_frac, max_new_tokens=32, **extra)

    print("\n== sanity: partials on real captures ==")
    for w, a in zip(wavs, audio):
        elapsed, text = transcribe_one(asr, a, step)
        print(f"  {w.name}: {elapsed:.2f}s -> {text!r}")

    print("\n== the gate: solo vs concurrent ==")
    solo, _ = transcribe_one(asr, audio[0], step)
    print(f"solo session: {solo:.2f}s")

    n = args.sessions
    if n < 2:
        print(f"--sessions {n}: nothing to compare against, skipping the interleave verdict "
              "(this is a functional smoke test only — the real gate needs --sessions >= 2)")
        return

    pool_audio = [audio[i % len(audio)] for i in range(n)]
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(lambda a: transcribe_one(asr, a, step), pool_audio))
    concurrent_wall = time.monotonic() - t0
    latencies = [r[0] for r in results]

    print(f"{n} concurrent sessions: wall={concurrent_wall:.2f}s, "
          f"per-session latencies={[f'{l:.2f}' for l in latencies]}")

    # Interleaves: wall time stays near solo, not near n * solo. 0.7x*n*solo is a
    # generous line between "clearly scaled" and "clearly stacked" — tune once real
    # numbers are in hand.
    interleave_threshold = solo * n * 0.7
    interleaves = concurrent_wall < interleave_threshold
    print(f"verdict: {'INTERLEAVES' if interleaves else 'SERIALIZES'} "
          f"(wall {concurrent_wall:.2f}s vs {n}x-solo {solo * n:.2f}s, threshold {interleave_threshold:.2f}s)")
    print("-> record this verdict + numbers in DECISION.md (see PLAN.md Phase 0 step 4)")

    assert interleaves, (
        "streaming_transcribe SERIALIZED across sessions — the streaming-path plan's "
        "load-bearing assumption doesn't hold. Do not proceed to Phase 1 as planned; "
        "pick a fallback in PLAN.md (K engine replicas, or the segment HTTP path)."
    )


if __name__ == "__main__":
    main()
