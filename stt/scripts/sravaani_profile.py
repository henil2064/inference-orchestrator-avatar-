"""Deep-dive profiler for SraVaani STT: where does the time actually go, and what
does each parameter mean. Runs the SAME pipeline as sravaani_gpu.py but breaks
transcribe() into its real stages and times/measures each one.

The pipeline for one clip:
    audio -> mel features -> ENCODER (1 big GPU forward) -> GREEDY TDT DECODE
    (a Python loop calling decoder_joint per frame) -> detokenize

Why this matters: the encoder is one batched GPU op (compute-bound, fast). The
greedy decoder is a *sequential* loop — one tiny decoder_joint call per emission,
each launch-bound (Python + kernel-launch overhead dominates, not math). On a
transducer that loop, not the encoder, is usually the wall-clock cost. This script
proves which one it is on your box.

CUDA timing note: GPU ops are async — we torch.cuda.synchronize() around every
stage, otherwise the numbers are meaningless.

Usage:
    export SRAVAANI_MODEL=/path/to/SraVaani_portable/model
    python scripts/sravaani_profile.py clip.wav [more.wav ...]
"""
import argparse
import os
import shutil
import warnings
import subprocess
import sys
import threading
import time

import torch
from transformers import AutoModel

CUDA = torch.cuda.is_available()
DEV = "cuda" if CUDA else "cpu"

# The decoder's LSTM fires a cuDNN "weights not contiguous, compacting every call"
# warning on every step — real overhead (see report note), but here it's just noise.
warnings.filterwarnings("ignore", message="RNN module weights are not part")


def sync():
    if CUDA:
        torch.cuda.synchronize()


class _CountCalls:
    """Wrap a callable to count how many times it's invoked (delegates everything else)."""
    def __init__(self, fn):
        self.fn = fn
        self.n = 0

    def __call__(self, *a, **k):
        self.n += 1
        return self.fn(*a, **k)

    def __getattr__(self, name):
        return getattr(self.fn, name)


class _GpuSampler(threading.Thread):
    """Poll nvidia-smi in the background for GPU utilization % and memory (MiB)."""
    def __init__(self, idx, period=0.05):
        super().__init__(daemon=True)
        # NB: do NOT name this _stop — threading.Thread._stop is a real method join() calls.
        self.idx, self.period, self.util, self.mem, self._stopflag = idx, period, [], [], False

    def run(self):
        q = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", "-i", str(self.idx)]
        while not self._stopflag:
            try:
                u, m = subprocess.check_output(q, text=True).strip().split(",")
                self.util.append(float(u)); self.mem.append(float(m))
            except Exception:
                pass
            time.sleep(self.period)

    def stop(self):
        self._stopflag = True


def _peak_mib():
    return torch.cuda.max_memory_allocated() / 2**20 if CUDA else 0.0


def _reset_peak():
    if CUDA:
        torch.cuda.reset_peak_memory_stats()


def resolve_model_dir(cli):
    d = cli or os.environ.get("SRAVAANI_MODEL")
    if not d or not os.path.isfile(os.path.join(d, "model-asr.fp16.ts")):
        sys.exit("set $SRAVAANI_MODEL (or --model) to the folder with model-asr.fp16.ts")
    return d


def print_params(model, mod):
    """Dissect the config: what each number is and what it drives at runtime."""
    c = model.config
    p = model._p
    fd = model._frame_seconds()
    print("\n=== MODEL PARAMETERS (what each one means) ===")
    rows = [
        ("dtype (compute)", model._io_dtype, "fp16 on GPU keeps weights ~0.9GB & uses tensor cores"),
        ("feat_in", c.feat_in, "mel bins fed to the encoder (feature dim F)"),
        ("enc_d_model", c.enc_d_model, "encoder hidden width — cost per encoder frame"),
        ("subsampling_factor", c.subsampling_factor, f"audio frames -> encoder frames (÷{c.subsampling_factor}); 1 enc frame ≈ {fd*1000:.0f} ms audio"),
        ("pred_hidden", c.pred_hidden, "LSTM predictor width (decoder state)"),
        ("pred_rnn_layers", c.pred_rnn_layers, "LSTM layers in the predictor"),
        ("durations", c.durations, "TDT frame-skips per emission — >0 lets decode SKIP frames (fewer steps)"),
        ("num_durations", c.num_durations, "size of the duration head"),
        ("max_symbols", c.max_symbols, "cap on tokens emitted at one frame (runaway guard)"),
        ("vocab_size / blank_id", f"{c.vocab_size} / {c.blank_id}", "token count; blank = 'emit nothing, advance'"),
        ("sample_rate", c.sample_rate, "audio is resampled to this before features"),
        ("n_fft / hop / win", f"{p['n_fft']} / {p['hop_length']} / {p['win_length']}", "STFT: hop sets feature-frame rate"),
    ]
    for name, val, meaning in rows:
        print(f"  {name:24} = {str(val):18} {meaning}")


def profile_file(model, mod, path, target_sr, sp):
    """Run one clip stage-by-stage, timing + measuring each. Returns a dict of stats."""
    dev = model._anchor.device

    # --- stage 1: read + resample audio (CPU) ---
    t = time.perf_counter()
    wav, sr = mod._read_audio(path)
    if sr != target_sr:
        wav = mod._resample(wav, sr, target_sr)
    t_read = time.perf_counter() - t
    audio_sec = wav.shape[0] / target_sr

    # --- stage 2: mel feature extraction ---
    _reset_peak(); sync(); t = time.perf_counter()
    feats, slen = model.extract_features(wav.unsqueeze(0), torch.tensor([wav.shape[0]]))
    sync(); t_feat = time.perf_counter() - t
    feat_mib = _peak_mib()
    F, T_feat = feats.shape[1], feats.shape[2]

    # --- stage 3: encoder (one GPU forward) ---
    _reset_peak(); sync(); t = time.perf_counter()
    out = model.forward(feats, slen)
    sync(); t_enc = time.perf_counter() - t
    enc_mib = _peak_mib()
    enc, enc_len = out.last_hidden_state, out.encoder_lengths
    T_enc = int(enc_len[0].item())

    # --- stage 4: greedy TDT decode (sequential loop) ---
    # decoder_joint is a registered submodule; nn.Module.__setattr__ rejects a plain
    # wrapper, so slip the counter into _modules directly (the loop only reads the
    # attribute) and restore right after. ponytail: works because we never iterate
    # modules while wrapped; if that changes, use a forward hook instead.
    orig_dj = model._modules["decoder_joint"]
    counter = _CountCalls(orig_dj)
    model._modules["decoder_joint"] = counter
    _reset_peak(); sync(); t = time.perf_counter()
    toks, frames = model._greedy_one(enc[0:1], T_enc)
    sync(); t_dec = time.perf_counter() - t
    dec_mib = _peak_mib()
    n_calls = counter.n
    model._modules["decoder_joint"] = orig_dj  # unwrap

    # --- stage 5: detokenize ---
    t = time.perf_counter()
    text = sp.decode([int(x) for x in toks])
    t_tok = time.perf_counter() - t

    total = t_read + t_feat + t_enc + t_dec + t_tok
    return {
        "path": os.path.basename(path), "audio_sec": audio_sec, "text": text,
        "T_feat": T_feat, "T_enc": T_enc, "n_tokens": len(toks), "n_calls": n_calls,
        "t_read": t_read, "t_feat": t_feat, "t_enc": t_enc, "t_dec": t_dec, "t_tok": t_tok,
        "total": total, "feat_mib": feat_mib, "enc_mib": enc_mib, "dec_mib": dec_mib,
    }


def report(s):
    print(f"\n=== {s['path']}  ({s['audio_sec']:.2f}s audio) ===")
    print(f"  transcript: {s['text']!r}")
    print(f"\n  shapes:  mel frames T={s['T_feat']}  ->  encoder frames T_enc={s['T_enc']}"
          f"  (÷{round(s['T_feat']/max(s['T_enc'],1))} subsample)")
    print(f"  emitted: {s['n_tokens']} tokens over {s['n_calls']} decoder_joint calls"
          f"  ({s['n_calls']/max(s['T_enc'],1):.2f} calls/enc-frame; "
          f"TDT skips let this be < {s['T_enc']} calls)")

    def line(name, t, extra=""):
        pct = 100 * t / s["total"] if s["total"] else 0
        bar = "#" * int(pct / 2)
        rtf = t / s["audio_sec"] if s["audio_sec"] else 0
        print(f"  {name:14} {t*1000:8.1f} ms  {pct:5.1f}%  RTF {rtf:5.2f}x  {bar} {extra}")

    print(f"\n  stage            time      share    (RTF = stage_sec / audio_sec)")
    line("read+resample", s["t_read"], "CPU audio load")
    line("mel features", s["t_feat"], f"peak {s['feat_mib']:.0f} MiB")
    runs = s.get("_enc_runs", [s["t_enc"]])
    spread = (f"peak {s['enc_mib']:.0f} MiB — 1 GPU forward"
              + (f"; {len(runs)} runs min/med/max {min(runs)*1000:.0f}/{s['t_enc']*1000:.0f}/{max(runs)*1000:.0f} ms"
                 if len(runs) > 1 else ""))
    line("ENCODER", s["t_enc"], spread)
    line("greedy DECODE", s["t_dec"], f"peak {s['dec_mib']:.0f} MiB — {s['n_calls']} serial calls")
    line("detokenize", s["t_tok"], "CPU")
    print(f"  {'TOTAL':14} {s['total']*1000:8.1f} ms         RTF {s['total']/s['audio_sec']:5.2f}x"
          f"  ({'faster' if s['total'] < s['audio_sec'] else 'SLOWER'} than real-time)")
    if s["t_dec"] > 0.5 * s["total"]:
        per = s["t_dec"] / max(s["n_calls"], 1) * 1000
        print(f"\n  >> DECODE dominates ({100*s['t_dec']/s['total']:.0f}%): {s['n_calls']} serial "
              f"decoder_joint calls @ ~{per:.2f} ms each. This is launch-bound (Python loop +\n"
              f"     tiny kernels), NOT GPU-compute-bound — which is why GPU util stays low here.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="+", help="audio files to profile")
    ap.add_argument("--model", default=None)
    ap.add_argument("--repeat", type=int, default=1,
                    help="run each clip N times, report the median (exposes first-shape autotune stalls)")
    args = ap.parse_args()
    # Default is already False, but pin it: with autotuning off, each new audio length
    # uses a heuristic kernel instead of a one-time benchmarking stall inside our timing.
    if CUDA:
        torch.backends.cudnn.benchmark = False
    model_dir = resolve_model_dir(args.model)

    dtype = torch.float16 if CUDA else torch.float32
    print(f"loading {model_dir} (device={DEV}, dtype={dtype}) ...")
    _reset_peak(); sync(); t = time.perf_counter()
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=dtype).to(DEV).eval()
    model._ensure_loaded()  # force graph load now so load time is honest
    sync(); load_s = time.perf_counter() - t
    mod = sys.modules[type(model).__module__]      # reuse the model's own audio helpers
    sp = model._get_tokenizer()
    target_sr = int(model.config.sample_rate)

    print(f"model loaded + graphs ready in {load_s:.1f}s"
          + (f" | weights resident {torch.cuda.memory_allocated()/2**20:.0f} MiB" if CUDA else ""))
    print_params(model, mod)

    # Warm-up: the FIRST forward pays CUDA context init + cuDNN/TorchScript specialization.
    # Measuring it as 'stage time' would be a lie — run it once, discarded, then measure warm.
    print("\nwarming up (first pass pays one-time CUDA/cuDNN init — discarded) ...")
    sync(); t = time.perf_counter()
    profile_file(model, mod, args.audio[0], target_sr, sp)
    sync(); warm_s = time.perf_counter() - t
    print(f"  cold first-pass took {warm_s:.2f}s (one-time cost, not per-clip)")

    sampler = None
    if CUDA and shutil.which("nvidia-smi"):
        sampler = _GpuSampler(torch.cuda.current_device())
        sampler.start()

    stats = []
    for p in args.audio:
        runs = [profile_file(model, mod, p, target_sr, sp) for _ in range(max(args.repeat, 1))]
        runs.sort(key=lambda r: r["t_enc"])
        s = runs[len(runs) // 2]           # median run (by encoder time)
        s["_enc_runs"] = [r["t_enc"] for r in runs]
        stats.append(s)

    if sampler:
        sampler.stop(); sampler.join(timeout=1.0)

    for s in stats:
        report(s)

    # aggregate
    tot_audio = sum(s["audio_sec"] for s in stats)
    tot_time = sum(s["total"] for s in stats)
    print(f"\n=== TOTAL: {len(stats)} clip(s), {tot_audio:.1f}s audio in {tot_time:.2f}s "
          f"(RTF {tot_time/tot_audio:.2f}x) ===")
    if sampler and sampler.util:
        print(f"  GPU during run: util avg {sum(sampler.util)/len(sampler.util):.0f}% / "
              f"peak {max(sampler.util):.0f}%   |   mem peak {max(sampler.mem):.0f} MiB")
        print("  (low avg util is EXPECTED for transducer decode — the serial loop leaves the GPU idle between tiny calls.)")


if __name__ == "__main__":
    main()
