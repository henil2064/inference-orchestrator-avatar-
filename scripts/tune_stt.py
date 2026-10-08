#!/usr/bin/env python3
"""Tune STT settings on the current host.

This offline calibration tool starts one STT service per candidate, waits for
gRPC health SERVING, streams known PCM audio through the real Asr service, then
scores latency plus transcript WER. It writes a JSON report and a suggested env
file under run/tuning without modifying stt/.env.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
import warnings
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import audioop  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
STT_DIR = REPO_ROOT / "stt"
STT_ENV = STT_DIR / ".env"
STT_PYTHON = STT_DIR / ".venv-sravaani" / "bin" / "python"
OUT_DIR = REPO_ROOT / "run" / "tuning"
DEFAULT_AUDIO = REPO_ROOT / "tts" / "sample" / "harvard.wav"
DEFAULT_REFERENCE = (
    "The stale smell of old beer lingers. It takes heat to bring out the odor. "
    "A cold dip restores health and zest. A salt pickle tastes fine with ham. "
    "Tacos al pastor are my favorite. A zestful food is the hot cross bun."
)

USED_KEYS = {
    "LOG_LEVEL",
    "STARTUP_BANNER",
    "STT_TRANSCRIPT_LOG",
    "STT_RUNTIME",
    "SRAVAANI_MODEL",
    "STT_MODEL_DIR",
    "STT_ONNX_MODEL_DIR",
    "STT_ONNX_VARIANT",
    "STT_ONNX_PROVIDER",
    "STT_ADDR",
    "STT_METRICS_HOST",
    "STT_METRICS_PORT",
    "STT_DEVICE",
    "STT_CPU_THREADS",
    "CUDA_VISIBLE_DEVICES",
    "STT_MAX_SEGMENT_SEC",
    "STT_SEGMENT_GRACE_SEC",
    "STT_QUIET_SCAN_SEC",
    "STT_QUIET_RMS_THRESHOLD",
    "STT_QUIET_MIN_SEC",
    "STT_MAX_INFLIGHT",
    "STT_WARM_ALL_BUCKETS",
    "STT_SEGMENT_QUEUE_MAX",
    "STT_SEGMENT_QUEUE_MAX_AGE_SEC",
    "STT_WEDGE_TIMEOUT_S",
    "STT_LOG_INFERENCE",
    "STT_PROFILE_RES",
    "STT_PROFILE_INTERVAL",
}

TUNABLES = {
    "STT_RUNTIME",
    "STT_ONNX_VARIANT",
    "STT_ONNX_PROVIDER",
    "STT_DEVICE",
    "STT_CPU_THREADS",
    "STT_MAX_SEGMENT_SEC",
    "STT_SEGMENT_GRACE_SEC",
    "STT_QUIET_SCAN_SEC",
    "STT_QUIET_RMS_THRESHOLD",
    "STT_QUIET_MIN_SEC",
    "STT_MAX_INFLIGHT",
    "STT_WARM_ALL_BUCKETS",
}


@dataclass(frozen=True)
class Candidate:
    name: str
    env: dict[str, str]


def parse_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.is_file():
        return env
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key not in USED_KEYS:
            continue
        try:
            parts = shlex.split(value, posix=True)
            env[key] = parts[0] if parts else ""
        except ValueError:
            env[key] = value.strip().strip("'\"")
    return env


def parse_fixed(values: Iterable[str]) -> tuple[dict[str, str], list[str]]:
    fixed: dict[str, str] = {}
    warnings: list[str] = []
    for item in values:
        if "=" not in item:
            raise SystemExit(f"--fixed expects KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise SystemExit(f"--fixed has an empty key: {item!r}")
        if key not in USED_KEYS:
            warnings.append(f"{key} is not read by today's STT code; ignoring it")
            continue
        if key not in TUNABLES:
            warnings.append(f"{key} is used by STT but is not searched; passing fixed value through")
        fixed[key] = value
    return fixed, warnings


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def load_pcm16_mono_16k(path: Path) -> tuple[bytes, float]:
    with wave.open(str(path), "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise ValueError("only uncompressed 16-bit PCM WAV input is supported")
        rate = wav.getframerate()
        channels = wav.getnchannels()
        frames = wav.getnframes()
        pcm = wav.readframes(frames)
    if channels != 1:
        pcm = audioop.tomono(pcm, 2, 0.5, 0.5)
    if rate != 16000:
        pcm, _ = audioop.ratecv(pcm, 2, 1, rate, 16000, None)
    return pcm, len(pcm) / 2 / 16000


def normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def wer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalized_words(reference), normalized_words(hypothesis)
    if not ref:
        return 0.0
    previous = list(range(len(hyp) + 1))
    for i, word in enumerate(ref, start=1):
        current = [i]
        for j, candidate in enumerate(hyp, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (word != candidate),
            ))
        previous = current
    return previous[-1] / len(ref)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * p)]


def summarize(rows: list[dict]) -> dict:
    summary: dict[str, dict[str, float]] = {}
    for key in ("proc_ms", "compute_ms", "queue_ms", "audio_seconds", "padded_seconds", "realtime_factor"):
        values = [float(row[key]) for row in rows if key in row]
        summary[key] = {
            "mean": round(statistics.fmean(values), 3) if values else 0.0,
            "p50": round(percentile(values, 0.5), 3),
            "p95": round(percentile(values, 0.95), 3),
        }
    return summary


def score_result(result: dict, max_wer: float) -> float:
    if result["status"] != "ok":
        return 1_000_000.0
    if result["wer"] > max_wer:
        return 500_000.0 + result["wer"] * 100_000.0
    s = result["summary"]
    return (
        s["proc_ms"]["p50"]
        + 0.5 * s["proc_ms"]["p95"]
        + 0.25 * s["compute_ms"]["p50"]
        + result["wer"] * 2000.0
    )


def health_check(stt_python: Path, target: str, timeout_s: float = 1.0) -> bool:
    code = """
import sys, grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc
target = sys.argv[1]
with grpc.insecure_channel(target) as channel:
    stub = health_pb2_grpc.HealthStub(channel)
    resp = stub.Check(health_pb2.HealthCheckRequest(service=""), timeout=float(sys.argv[2]))
    raise SystemExit(0 if resp.status == health_pb2.HealthCheckResponse.SERVING else 1)
"""
    proc = subprocess.run(
        [str(stt_python), "-c", code, target, str(timeout_s)],
        cwd=str(STT_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return proc.returncode == 0


def wait_ready(stt_python: Path, target: str, process: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"STT exited before ready with code {process.returncode}")
        if health_check(stt_python, target):
            return
        time.sleep(1.0)
    raise TimeoutError(f"STT did not become ready within {timeout_s:.0f}s")


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait(timeout=10)


def run_stt_client(stt_python: Path, target: str, audio_path: Path, runs: int, timeout_s: float,
                   chunk_ms: int) -> list[dict]:
    code = r"""
import asyncio, json, sys, time, uuid, wave, audioop, grpc
from proto import asr_pb2, asr_pb2_grpc

def load_pcm(path):
    with wave.open(path, "rb") as wav:
        if wav.getsampwidth() != 2 or wav.getcomptype() != "NONE":
            raise RuntimeError("only uncompressed 16-bit PCM WAV input is supported")
        rate, channels, frames = wav.getframerate(), wav.getnchannels(), wav.getnframes()
        pcm = wav.readframes(frames)
    if channels != 1:
        pcm = audioop.tomono(pcm, 2, 0.5, 0.5)
    if rate != 16000:
        pcm, _ = audioop.ratecv(pcm, 2, 1, rate, 16000, None)
    return pcm

async def once(stub, target, audio_path, chunk_ms, timeout_s):
    sid = "stt-tune-" + str(uuid.uuid4())
    pcm = load_pcm(audio_path)
    chunk_bytes = max(2, int(16000 * chunk_ms / 1000) * 2)
    async def requests():
        now = int(time.time() * 1000)
        yield asr_pb2.AsrClientEvent(control=asr_pb2.Control(session_id=sid, type=asr_pb2.Control.SESSION_OPEN, ts_ms=now))
        for offset in range(0, len(pcm), chunk_bytes):
            yield asr_pb2.AsrClientEvent(audio=asr_pb2.AudioChunk(session_id=sid, pcm16=pcm[offset:offset + chunk_bytes], ts_ms=int(time.time() * 1000)))
        yield asr_pb2.AsrClientEvent(control=asr_pb2.Control(session_id=sid, type=asr_pb2.Control.END_TURN, ts_ms=int(time.time() * 1000)))
        yield asr_pb2.AsrClientEvent(control=asr_pb2.Control(session_id=sid, type=asr_pb2.Control.SESSION_CLOSE, ts_ms=int(time.time() * 1000)))
    started = time.perf_counter()
    segments = []
    async for transcript in stub.Transcribe(requests(), timeout=timeout_s):
        profile = transcript.profile
        segments.append({
            "text": transcript.text,
            "turn_complete": transcript.turn_complete,
            "segment_index": transcript.segment_index,
            "proc_ms": transcript.proc_ms,
            "compute_ms": profile.compute_ms,
            "queue_ms": profile.queue_ms,
            "audio_seconds": profile.audio_seconds,
            "padded_seconds": profile.padded_seconds,
            "realtime_factor": profile.realtime_factor,
            "model": profile.model,
            "engine": profile.engine,
            "device": profile.device,
            "compute_type": profile.compute_type,
            "attributes": dict(profile.attributes),
        })
    text = " ".join(seg["text"] for seg in segments if seg["text"]).strip()
    return {"wall_ms": round((time.perf_counter() - started) * 1000, 3), "text": text, "segments": segments}

async def main():
    target, audio_path, runs, timeout_s, chunk_ms = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
    async with grpc.aio.insecure_channel(target) as channel:
        stub = asr_pb2_grpc.AsrStub(channel)
        for i in range(runs):
            row = await once(stub, target, audio_path, chunk_ms, timeout_s)
            row["run"] = i + 1
            print(json.dumps(row), flush=True)

asyncio.run(main())
"""
    proc = subprocess.run(
        [str(stt_python), "-c", code, target, str(audio_path), str(runs), str(timeout_s), str(chunk_ms)],
        cwd=str(STT_DIR),
        capture_output=True,
        text=True,
        timeout=timeout_s * (runs + 1),
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip() or f"STT client exited {proc.returncode}")
    rows = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not rows:
        raise RuntimeError("STT client produced no measurement rows")
    return rows


def benchmark_candidate(candidate: Candidate, args: argparse.Namespace, base_env: dict[str, str],
                        index: int, out_dir: Path) -> dict:
    app_port = free_port()
    metrics_port = free_port()
    target = f"127.0.0.1:{app_port}"
    env = os.environ.copy()
    env.update(base_env)
    env.update(candidate.env)
    env["STT_ADDR"] = f"[::]:{app_port}"
    env["STT_METRICS_PORT"] = str(metrics_port)
    env["STT_TRANSCRIPT_LOG"] = "none"
    env["LOG_LEVEL"] = env.get("LOG_LEVEL", "ERROR")
    log_path = out_dir / f"stt-tune-{index:03d}-{candidate.name}.log"
    rows: list[dict] = []
    flat_segments: list[dict] = []
    started = time.monotonic()
    log_file = log_path.open("wb")
    process = subprocess.Popen(
        [str(args.stt_python), "server.py"],
        cwd=str(STT_DIR),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        wait_ready(args.stt_python, target, process, args.ready_timeout)
        rows = run_stt_client(args.stt_python, target, args.audio, args.runs, args.client_timeout, args.chunk_ms)
        for row in rows:
            flat_segments.extend(row["segments"])
        if not flat_segments:
            raise RuntimeError("STT returned no transcript/profile segments")
        transcript = rows[-1]["text"]
        item = {
            "name": candidate.name,
            "status": "ok",
            "env": candidate.env,
            "summary": summarize(flat_segments),
            "wer": round(wer(args.reference, transcript), 6),
            "transcript": transcript,
            "rows": rows,
            "startup_s": round(time.monotonic() - started, 3),
            "log": str(log_path.relative_to(REPO_ROOT)),
            "backend": {
                key: flat_segments[-1].get(key)
                for key in ("model", "engine", "device", "compute_type", "attributes")
            },
        }
        item["score"] = round(score_result(item, args.max_wer), 3)
        return item
    except Exception as exc:
        return {
            "name": candidate.name,
            "status": "failed",
            "env": candidate.env,
            "error": str(exc),
            "startup_s": round(time.monotonic() - started, 3),
            "log": str(log_path.relative_to(REPO_ROOT)),
            "score": 1_000_000.0,
        }
    finally:
        stop_process(process)
        log_file.close()


def candidate_name(env: dict[str, str]) -> str:
    parts = []
    for key in sorted(env):
        short = key.removeprefix("STT_").lower()
        value = env[key].replace(".", "p").replace("/", "-")
        parts.append(f"{short}-{value}")
    return "_".join(parts)[:140]


def normalize_candidate_env(env: dict[str, str]) -> dict[str, str]:
    normalized = dict(env)
    runtime = normalized.get("STT_RUNTIME", "onnx")
    if runtime == "transformers":
        normalized.pop("STT_ONNX_VARIANT", None)
        normalized.pop("STT_ONNX_PROVIDER", None)
    elif runtime == "onnx":
        normalized.pop("STT_DEVICE", None)
    return normalized


def build_candidates(args: argparse.Namespace, base_env: dict[str, str], fixed: dict[str, str]) -> list[Candidate]:
    baseline = {
        "STT_RUNTIME": base_env.get("STT_RUNTIME", "onnx").split()[0],
        "STT_ONNX_VARIANT": base_env.get("STT_ONNX_VARIANT", "fp16").split()[0],
        "STT_ONNX_PROVIDER": base_env.get("STT_ONNX_PROVIDER", "cpu").split()[0],
        "STT_CPU_THREADS": base_env.get("STT_CPU_THREADS", "4").split()[0],
        "STT_MAX_SEGMENT_SEC": base_env.get("STT_MAX_SEGMENT_SEC", "8.0").split()[0],
        "STT_SEGMENT_GRACE_SEC": base_env.get("STT_SEGMENT_GRACE_SEC", "2.0").split()[0],
        "STT_QUIET_SCAN_SEC": base_env.get("STT_QUIET_SCAN_SEC", "1.0").split()[0],
        "STT_QUIET_RMS_THRESHOLD": base_env.get("STT_QUIET_RMS_THRESHOLD", "400").split()[0],
        "STT_QUIET_MIN_SEC": base_env.get("STT_QUIET_MIN_SEC", "0.06").split()[0],
        "STT_MAX_INFLIGHT": base_env.get("STT_MAX_INFLIGHT", "8").split()[0],
        "STT_WARM_ALL_BUCKETS": "1",
    }
    if args.cpu_only:
        baseline["STT_ONNX_PROVIDER"] = "cpu"
        baseline["STT_DEVICE"] = "cpu"
    if args.cuda_only:
        baseline["STT_ONNX_PROVIDER"] = "cuda"
        baseline["STT_DEVICE"] = "cuda"
    baseline.update({k: v for k, v in fixed.items() if k in TUNABLES})

    spaces = {
        "STT_RUNTIME": ["onnx"],
        "STT_ONNX_VARIANT": ["int8", "fp8", "fp16"],
        "STT_ONNX_PROVIDER": ["cpu", "cuda"] if not args.cpu_only else ["cpu"],
        "STT_CPU_THREADS": ["1", "2", "4", "6", "8"],
        "STT_MAX_SEGMENT_SEC": ["4.0", "6.0", "8.0"],
        "STT_SEGMENT_GRACE_SEC": ["0.5", "1.0", "2.0"],
        "STT_QUIET_MIN_SEC": ["0.03", "0.06", "0.12"],
        "STT_WARM_ALL_BUCKETS": ["1"],
    }
    if args.cuda_only:
        spaces["STT_ONNX_PROVIDER"] = ["cuda"]
    if baseline.get("STT_RUNTIME") == "transformers":
        spaces.pop("STT_ONNX_VARIANT", None)
        spaces.pop("STT_ONNX_PROVIDER", None)
        if not args.cpu_only and not args.cuda_only:
            spaces["STT_DEVICE"] = ["cpu", "cuda"]
    for key in list(spaces):
        if key in fixed:
            spaces.pop(key)

    raw: list[dict[str, str]] = [dict(baseline)]
    for key, values in spaces.items():
        for value in values:
            env = dict(baseline)
            env[key] = value
            raw.append(env)

    if args.full_grid:
        keys = list(spaces)
        for values in itertools.product(*(spaces[key] for key in keys)):
            env = dict(baseline)
            env.update(dict(zip(keys, values)))
            raw.append(env)

    seen = set()
    candidates: list[Candidate] = []
    for env in raw:
        env = normalize_candidate_env(env)
        key = tuple(sorted(env.items()))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(Candidate(candidate_name(env), env))
    return candidates[:args.max_candidates]


def write_env_file(path: Path, env: dict[str, str]) -> None:
    lines = [
        "# Suggested STT tuning generated by scripts/tune_stt.py",
        "# Review before using in production.",
    ]
    for key in sorted(env):
        lines.append(f"{key}={env[key]}")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=STT_ENV)
    parser.add_argument("--stt-python", type=Path, default=STT_PYTHON)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--ready-timeout", type=float, default=240)
    parser.add_argument("--client-timeout", type=float, default=120)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--max-wer", type=float, default=0.35)
    parser.add_argument("--fixed", action="append", default=[], help="fix a parameter, e.g. --fixed STT_CPU_THREADS=4")
    parser.add_argument("--cpu-only", action="store_true", help="search only CPU providers/devices")
    parser.add_argument("--cuda-only", action="store_true", help="search only CUDA providers/devices")
    parser.add_argument("--full-grid", action="store_true", help="run the full cartesian grid instead of one-knob sweeps")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="print candidates without starting STT")
    args = parser.parse_args()

    if args.cpu_only and args.cuda_only:
        raise SystemExit("cannot use --cpu-only and --cuda-only together")
    if args.runs < 1 or args.max_candidates < 1 or args.chunk_ms < 10:
        raise SystemExit("--runs and --max-candidates must be >= 1; --chunk-ms must be >= 10")
    if not args.stt_python.exists():
        raise SystemExit(f"missing STT venv Python: {args.stt_python}")
    if not args.audio.exists():
        raise SystemExit(f"missing benchmark audio: {args.audio}")
    load_pcm16_mono_16k(args.audio)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    base_env = parse_env_file(args.env_file)
    fixed, warnings = parse_fixed(args.fixed)
    base_env.update(fixed)
    candidates = build_candidates(args, base_env, fixed)

    print(f"tuning {len(candidates)} candidate(s); output={out_dir.relative_to(REPO_ROOT)}", flush=True)
    for warning in warnings:
        print(f"warning: {warning}", flush=True)
    if args.dry_run:
        for index, candidate in enumerate(candidates, start=1):
            print(f"[{index}/{len(candidates)}] {candidate.name} {json.dumps(candidate.env, sort_keys=True)}")
        return 0

    results = []
    for index, candidate in enumerate(candidates, start=1):
        print(f"[{index}/{len(candidates)}] {candidate.name}", flush=True)
        result = benchmark_candidate(candidate, args, base_env, index, out_dir)
        results.append(result)
        if result["status"] == "ok":
            s = result["summary"]
            print(
                "  ok score={score:.1f} proc_p50={proc:.1f}ms compute_p50={comp:.1f}ms rtf_p50={rtf:.4f} wer={wer:.1%}".format(
                    score=result["score"],
                    proc=s["proc_ms"]["p50"],
                    comp=s["compute_ms"]["p50"],
                    rtf=s["realtime_factor"]["p50"],
                    wer=result["wer"],
                ),
                flush=True,
            )
        else:
            print(f"  failed: {result['error']}", flush=True)

    successful = [r for r in results if r["status"] == "ok"]
    best = min(successful, key=lambda r: r["score"]) if successful else None
    report = {
        "model": "stt",
        "env_file": str(args.env_file),
        "audio": str(args.audio),
        "reference": args.reference,
        "fixed": fixed,
        "results": sorted(results, key=lambda r: r["score"]),
        "best": best,
    }
    report_path = out_dir / "stt-tuning-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {report_path.relative_to(REPO_ROOT)}", flush=True)
    if best is None:
        print("no successful candidates", flush=True)
        return 1
    suggested_env = dict(base_env)
    suggested_env.update(best["env"])
    env_path = out_dir / "stt.env.tuned"
    write_env_file(env_path, {k: suggested_env[k] for k in sorted(TUNABLES | set(fixed)) if k in suggested_env})
    print(f"best={best['name']}", flush=True)
    print(f"wrote {env_path.relative_to(REPO_ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
