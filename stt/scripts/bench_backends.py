#!/usr/bin/env python3
"""Benchmark every supported SraVaani backend against one WAV file.

Run from the repository root with the STT virtual environment:

    stt/.venv-sravaani/bin/python stt/scripts/bench_backends.py

The parent process starts one fresh child process per configuration.  This is
important: PyTorch and ONNX Runtime both retain CUDA allocations and provider
state, so in-process switching would make the later measurements incomparable.
Configuration is passed as environment overrides, exactly matching the keys in
``stt/.env``; the existing .env is never modified.
"""
from __future__ import annotations

import argparse
import audioop
import json
import os
import re
import subprocess
import sys
import time
import wave
from pathlib import Path
from statistics import median

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STT_DIR = ROOT / "stt"
sys.path.insert(0, str(STT_DIR))
DEFAULT_AUDIO = ROOT / "tts/sample/harvard.wav"
DEFAULT_REPORT = STT_DIR.parent / "docs/STT_BACKEND_BENCHMARK.md"
EXPECTED = (
    "The stale smell of old beer lingers. It takes heat to bring out the odor. "
    "A cold dip restores health and zest. A salt pickle tastes fine with ham. "
    "Tacos al pastor are my favorite. A zestful food is the hot cross bun."
)

CONFIGURATIONS = (
    ("Transformers FP16 (CUDA)", {"STT_RUNTIME": "transformers", "STT_DEVICE": "cuda"}),
    ("ONNX INT8 (CPU)", {"STT_RUNTIME": "onnx", "STT_ONNX_VARIANT": "int8", "STT_ONNX_PROVIDER": "cpu"}),
    ("ONNX FP8 storage (CPU)", {"STT_RUNTIME": "onnx", "STT_ONNX_VARIANT": "fp8", "STT_ONNX_PROVIDER": "cpu"}),
    ("ONNX FP16 (CPU)", {"STT_RUNTIME": "onnx", "STT_ONNX_VARIANT": "fp16", "STT_ONNX_PROVIDER": "cpu"}),
    ("ONNX INT8 (CUDA)", {"STT_RUNTIME": "onnx", "STT_ONNX_VARIANT": "int8", "STT_ONNX_PROVIDER": "cuda"}),
    ("ONNX FP8 storage (CUDA)", {"STT_RUNTIME": "onnx", "STT_ONNX_VARIANT": "fp8", "STT_ONNX_PROVIDER": "cuda"}),
    ("ONNX FP16 (CUDA)", {"STT_RUNTIME": "onnx", "STT_ONNX_VARIANT": "fp16", "STT_ONNX_PROVIDER": "cuda"}),
)


def load_audio(path: Path) -> tuple[np.ndarray, float]:
    """Read PCM WAV and convert any channel/rate to mono 16 kHz float32."""
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise ValueError("only uncompressed 16-bit PCM WAV input is supported")
        rate, channels, frames = wav.getframerate(), wav.getnchannels(), wav.getnframes()
        pcm = np.frombuffer(wav.readframes(frames), dtype="<i2").reshape(-1, channels)
    # Convert to mono before rate conversion.  ``audioop.ratecv`` is part of
    # Python 3.12 (the interpreter used by this project) and avoids adding a
    # benchmark-only SciPy dependency to the service environment.
    mono_pcm = np.rint(pcm.astype(np.float32).mean(axis=1)).clip(-32768, 32767).astype("<i2")
    if rate != 16000:
        converted, _ = audioop.ratecv(mono_pcm.tobytes(), 2, 1, rate, 16000, None)
        mono_pcm = np.frombuffer(converted, dtype="<i2")
    samples = mono_pcm.astype(np.float32) / 32768.0
    return np.ascontiguousarray(samples), len(samples) / 16000.0


def normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalized_words(reference), normalized_words(hypothesis)
    previous = list(range(len(hyp) + 1))
    for i, word in enumerate(ref, start=1):
        current = [i]
        for j, candidate in enumerate(hyp, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (word != candidate)))
        previous = current
    return previous[-1] / len(ref) if ref else 0.0


def child(args: argparse.Namespace) -> int:
    # Import only here, after the parent has supplied the configuration env.
    sys.path.insert(0, str(STT_DIR))
    from service.backends import load_backend

    audio, duration = load_audio(Path(args.audio))
    started = time.perf_counter()
    backend = load_backend()
    load_seconds = time.perf_counter() - started

    # The warm-up is deliberately excluded: it pays CUDA/provider setup and
    # JIT/autotuning costs that occur once per long-lived service process.
    warm_started = time.perf_counter()
    transcript = backend.transcribe_waveform(audio)
    warmup_seconds = time.perf_counter() - warm_started

    elapsed: list[float] = []
    for _ in range(args.runs):
        started = time.perf_counter()
        transcript = backend.transcribe_waveform(audio)
        elapsed.append(time.perf_counter() - started)
    elapsed.sort()
    result = {
        "name": args.name,
        "status": "ok",
        "audio_seconds": duration,
        "load_seconds": load_seconds,
        "warmup_seconds": warmup_seconds,
        "median_seconds": median(elapsed),
        "p95_seconds": elapsed[min(len(elapsed) - 1, int(np.ceil(len(elapsed) * .95)) - 1)],
        "rtf": median(elapsed) / duration,
        "wer": wer(EXPECTED, transcript),
        "transcript": transcript,
        "backend": {key: getattr(backend, key, None) for key in
                    ("engine", "model_name", "device", "compute_type", "profile_attributes")},
    }
    print(json.dumps(result), flush=True)
    return 0


def md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def report(results: list[dict], audio: Path, runs: int) -> str:
    lines = [
        "# STT backend benchmark",
        "",
        f"Audio: `{audio.relative_to(ROOT)}`. Each result is the median of {runs} post-warm-up direct backend inferences.",
        "Audio decoding/downmixing/resampling, gRPC transport, and service segmentation are excluded from latency; model load and first inference are shown separately.",
        "",
        "| Configuration | Status | Device/provider | Load (s) | Warm-up (s) | Median (s) | P95 (s) | RTF | WER |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results:
        if item["status"] != "ok":
            lines.append(f"| {md_cell(item['name'])} | unavailable | {md_cell(item['error'])} | — | — | — | — | — | — |")
            continue
        info = item["backend"]
        provider = info["profile_attributes"].get("provider", info["device"])
        lines.append(
            f"| {md_cell(item['name'])} | ok | {md_cell(provider)} | {item['load_seconds']:.3f} | "
            f"{item['warmup_seconds']:.3f} | {item['median_seconds']:.3f} | {item['p95_seconds']:.3f} | "
            f"{item['rtf']:.3f} | {item['wer']:.1%} |"
        )
    lines += ["", "## Transcripts", ""]
    for item in results:
        if item["status"] == "ok":
            lines += [f"### {item['name']}", "", item["transcript"] or "*(empty transcript)*", ""]
    lines += [
        "## Notes", "",
        "- RTF is median inference time divided by audio duration; lower is faster than real time.",
        "- WER is case/punctuation-insensitive word error rate against the supplied reference transcript; lower is better.",
        "- The repository’s ONNX choices are `int8`, `fp8` (the files are named `*_real_fp8_storage.onnx`), and `fp16` (the files are in `srvaani-16-onnx`).",
    ]
    return "\n".join(lines) + "\n"


def parent(args: argparse.Namespace) -> int:
    results: list[dict] = []
    for name, overrides in CONFIGURATIONS:
        env = os.environ.copy()
        env.update(overrides)
        # The legacy backend needs an explicit bundled-model default.
        env.setdefault("STT_MODEL_DIR", str(STT_DIR.parent / "models/sravaani"))
        # Let the backend select the bundled folder that matches each variant.
        # An explicit shell value still takes precedence for custom artifacts.
        cmd = [sys.executable, str(Path(__file__).resolve()), "--child", "--name", name,
               "--audio", str(args.audio), "--runs", str(args.runs)]
        print(f"\n== {name} ==", flush=True)
        completed = subprocess.run(cmd, env=env, text=True, capture_output=True)
        payload = next((line for line in reversed(completed.stdout.splitlines()) if line.startswith("{")), None)
        if completed.returncode or not payload:
            error = (completed.stderr or completed.stdout or f"child exited {completed.returncode}").strip()
            results.append({"name": name, "status": "unavailable", "error": error[-800:]})
            print(f"unavailable: {error[-300:]}", file=sys.stderr)
        else:
            item = json.loads(payload)
            results.append(item)
            print(f"median={item['median_seconds']:.3f}s, RTF={item['rtf']:.3f}, WER={item['wer']:.1%}")
    output = report(results, Path(args.audio), args.runs)
    Path(args.report).write_text(output)
    print(f"\nWrote {args.report}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--runs", type=int, default=3, help="post-warm-up iterations per configuration")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--name", default="", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be at least one")
    return child(args) if args.child else parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
