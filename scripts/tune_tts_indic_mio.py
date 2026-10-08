#!/usr/bin/env python3
"""Tune Indic-Mio/GGUF TTS settings on the current host.

This is an offline calibration tool. It starts the TTS service in isolation for
each candidate env, waits for gRPC health SERVING, runs the existing TTS client,
scores latency with simple audio sanity checks, then writes a JSON report and a
suggested env file. It does not modify tts/.env unless a caller chooses to copy
the suggested settings.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shlex
import signal
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
TTS_DIR = REPO_ROOT / "tts"
TTS_ENV = TTS_DIR / ".env"
TTS_PYTHON = TTS_DIR / ".venv" / "bin" / "python"
OUT_DIR = REPO_ROOT / "run" / "tuning"

TUNABLES = {
    "TTS_N_GPU_LAYERS",
    "TTS_NUM_THREADS",
    "TTS_N_CTX",
    "TTS_MAX_AUDIO_TOKENS",
    "TTS_LLAMACPP_TEMPERATURE",
    "TTS_TOP_P",
    "TTS_MAX_QUEUE",
}
IGNORED_FOR_INDIC_MIO = {
    "TTS_NUM_STEP",
    "TTS_GUIDANCE_SCALE",
    "TTS_SPEED",
    "TTS_DEVICE",
}

DEFAULT_PROMPTS = [
    "Fever and throat pain can happen with a viral infection. Drink fluids and rest.",
    "Your appointment is confirmed for tomorrow morning. Please bring your reports.",
    "Blood pressure should be checked regularly, especially when medication changes.",
]


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
        if not key:
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
        if key in IGNORED_FOR_INDIC_MIO:
            warnings.append(f"{key} is ignored by the Indic-Mio GGUF backend; keeping it fixed has no effect")
        elif key not in TUNABLES:
            warnings.append(f"{key} is not a known Indic-Mio tuning knob; it will be passed through but not searched")
        fixed[key] = value
    return fixed, warnings


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * p)]


def summarize(rows: list[dict]) -> dict:
    summary: dict[str, dict[str, float]] = {}
    for key in ("tts_first_audio_ms", "tts_compute_ms", "tts_wall_ms", "audio_seconds", "rtf"):
        values = [float(row[key]) for row in rows if key in row]
        summary[key] = {
            "mean": round(statistics.fmean(values), 3) if values else 0.0,
            "p50": round(percentile(values, 0.5), 3),
            "p95": round(percentile(values, 0.95), 3),
        }
    return summary


def audio_penalty(rows: list[dict], max_audio_tokens: int) -> float:
    penalty = 0.0
    max_audio_s = max_audio_tokens / 25.0
    for row in rows:
        audio_s = float(row.get("audio_seconds", 0.0))
        chars = max(1, int(row.get("chars", 1)))
        chars_per_s = chars / max(audio_s, 0.001)
        if audio_s < 0.3:
            penalty += 5000.0
        if audio_s > max_audio_s * 1.05:
            penalty += 3000.0
        if chars_per_s < 4.0 or chars_per_s > 35.0:
            penalty += 1000.0
    return penalty


def score_result(result: dict) -> float:
    if result["status"] != "ok":
        return 1_000_000.0
    s = result["summary"]
    return (
        s["tts_first_audio_ms"]["p50"]
        + 0.5 * s["tts_first_audio_ms"]["p95"]
        + 0.25 * s["tts_compute_ms"]["p50"]
        + result["audio_penalty"]
    )


def health_check(tts_python: Path, target: str, timeout_s: float = 1.0) -> bool:
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
        [str(tts_python), "-c", code, target, str(timeout_s)],
        cwd=str(TTS_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return proc.returncode == 0


def wait_ready(tts_python: Path, target: str, process: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"TTS exited before ready with code {process.returncode}")
        if health_check(tts_python, target):
            return
        time.sleep(1.0)
    raise TimeoutError(f"TTS did not become ready within {timeout_s:.0f}s")


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


def run_client(tts_python: Path, target: str, prompt: str, runs: int, warmup: int, timeout_s: float,
               out_wav: Path | None) -> list[dict]:
    cmd = [
        str(tts_python), "-m", "tests.test_client", prompt,
        "--target", target,
        "--runs", str(runs),
        "--warmup", str(warmup),
        "--timeout", str(timeout_s),
    ]
    if out_wav is not None:
        cmd.extend(["--out", str(out_wav)])
    proc = subprocess.run(cmd, cwd=str(TTS_DIR), capture_output=True, text=True, timeout=timeout_s * (runs + warmup + 1))
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout).strip() or f"test_client exited {proc.returncode}")
    rows = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "run" in obj:
            rows.append(obj)
    if not rows:
        raise RuntimeError("test_client produced no measurement rows")
    return rows


def benchmark_candidate(candidate: Candidate, args: argparse.Namespace, base_env: dict[str, str],
                        index: int, out_dir: Path) -> dict:
    app_port = free_port()
    metrics_port = free_port()
    target = f"127.0.0.1:{app_port}"
    env = os.environ.copy()
    env.update(base_env)
    env.update(candidate.env)
    env["TTS_ADDR"] = f"[::]:{app_port}"
    env["TTS_METRICS_PORT"] = str(metrics_port)
    env["TTS_FORMAT"] = "gguf"
    env["TTS_WARMUP"] = "1"
    log_path = out_dir / f"tts-tune-{index:03d}-{candidate.name}.log"
    wav_dir = out_dir / "audio"
    wav_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    started = time.monotonic()
    log_file = log_path.open("wb")
    process = subprocess.Popen(
        [str(args.tts_python), "server.py"],
        cwd=str(TTS_DIR),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        wait_ready(args.tts_python, target, process, args.ready_timeout)
        for prompt_index, prompt in enumerate(args.prompt):
            wav_path = wav_dir / f"{index:03d}-{candidate.name}-p{prompt_index}.wav" if args.save_audio else None
            all_rows.extend(run_client(args.tts_python, target, prompt, args.runs, args.warmup,
                                       args.client_timeout, wav_path))
        max_audio_tokens = int(env.get("TTS_MAX_AUDIO_TOKENS", "1500"))
        summary = summarize(all_rows)
        result = {
            "name": candidate.name,
            "status": "ok",
            "env": candidate.env,
            "summary": summary,
            "rows": all_rows,
            "audio_penalty": audio_penalty(all_rows, max_audio_tokens),
            "startup_s": round(time.monotonic() - started, 3),
            "log": str(log_path.relative_to(REPO_ROOT)),
        }
        result["score"] = round(score_result(result), 3)
        return result
    except Exception as exc:
        result = {
            "name": candidate.name,
            "status": "failed",
            "env": candidate.env,
            "error": str(exc),
            "startup_s": round(time.monotonic() - started, 3),
            "log": str(log_path.relative_to(REPO_ROOT)),
            "score": 1_000_000.0,
        }
        return result
    finally:
        stop_process(process)
        log_file.close()


def candidate_name(env: dict[str, str]) -> str:
    parts = []
    for key in sorted(env):
        short = key.removeprefix("TTS_").lower()
        value = env[key].replace(".", "p")
        parts.append(f"{short}-{value}")
    return "_".join(parts)[:140]


def build_candidates(args: argparse.Namespace, base_env: dict[str, str], fixed: dict[str, str]) -> list[Candidate]:
    baseline = {
        "TTS_N_GPU_LAYERS": "0",
        "TTS_NUM_THREADS": "4",
        "TTS_N_CTX": "2048",
        "TTS_MAX_AUDIO_TOKENS": "300",
        "TTS_LLAMACPP_TEMPERATURE": "0.7",
        "TTS_TOP_P": "0.85",
        "TTS_MAX_QUEUE": "1",
    }
    for key in TUNABLES:
        if key in base_env:
            baseline[key] = base_env[key].split()[0]
    baseline.update({k: v for k, v in fixed.items() if k in TUNABLES})
    if args.cpu_only:
        baseline["TTS_N_GPU_LAYERS"] = "0"
    if args.gpu_layers is not None:
        baseline["TTS_N_GPU_LAYERS"] = str(args.gpu_layers)

    spaces = {
        "TTS_NUM_THREADS": ["1", "2", "4", "6", "8"],
        "TTS_N_CTX": ["1024", "2048", "4096"],
        "TTS_MAX_AUDIO_TOKENS": ["200", "300", "500"],
        "TTS_LLAMACPP_TEMPERATURE": ["0.6", "0.7", "0.8", "0.9"],
        "TTS_TOP_P": ["0.8", "0.85", "0.9"],
        "TTS_MAX_QUEUE": ["1"],
    }
    if not args.cpu_only and args.gpu_layers is None:
        spaces["TTS_N_GPU_LAYERS"] = ["0", "4", "8", "16", "24", "32", "999"]

    for key in list(spaces):
        if key in fixed:
            spaces.pop(key)

    raw: list[dict[str, str]] = [dict(baseline)]
    for key, values in spaces.items():
        for value in values:
            env = dict(baseline)
            env[key] = value
            raw.append(env)

    if "TTS_LLAMACPP_TEMPERATURE" in spaces or "TTS_TOP_P" in spaces:
        temp_values = spaces.get("TTS_LLAMACPP_TEMPERATURE", [baseline["TTS_LLAMACPP_TEMPERATURE"]])
        top_p_values = spaces.get("TTS_TOP_P", [baseline["TTS_TOP_P"]])
        for temp, top_p in itertools.product(temp_values, top_p_values):
            env = dict(baseline)
            env["TTS_LLAMACPP_TEMPERATURE"] = temp
            env["TTS_TOP_P"] = top_p
            raw.append(env)

    seen = set()
    candidates: list[Candidate] = []
    for env in raw:
        key = tuple(sorted(env.items()))
        if key in seen:
            continue
        seen.add(key)
        candidates.append(Candidate(candidate_name(env), env))
    return candidates[:args.max_candidates]


def write_env_file(path: Path, env: dict[str, str]) -> None:
    lines = [
        "# Suggested Indic-Mio TTS tuning generated by scripts/tune_tts_indic_mio.py",
        "# Review before using in production.",
    ]
    for key in sorted(env):
        lines.append(f"{key}={env[key]}")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=TTS_ENV)
    parser.add_argument("--tts-python", type=Path, default=TTS_PYTHON)
    parser.add_argument("--prompt", action="append", default=[], help="benchmark prompt; may be repeated")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--ready-timeout", type=float, default=240)
    parser.add_argument("--client-timeout", type=float, default=180)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--fixed", action="append", default=[], help="fix a parameter, e.g. --fixed TTS_NUM_THREADS=4")
    parser.add_argument("--cpu-only", action="store_true", help="fix TTS_N_GPU_LAYERS=0")
    parser.add_argument("--gpu-layers", type=int, help="fix TTS_N_GPU_LAYERS to this value")
    parser.add_argument("--save-audio", action="store_true", help="save candidate WAV files under run/tuning/audio")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dry-run", action="store_true", help="print candidates without starting TTS")
    args = parser.parse_args()

    if args.cpu_only and args.gpu_layers is not None:
        raise SystemExit("cannot use --cpu-only and --gpu-layers together")
    if not args.tts_python.exists():
        raise SystemExit(f"missing TTS venv Python: {args.tts_python}")
    if args.runs < 1 or args.warmup < 0 or args.max_candidates < 1:
        raise SystemExit("--runs and --max-candidates must be >= 1; --warmup must be >= 0")

    args.prompt = args.prompt or DEFAULT_PROMPTS
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base_env = parse_env_file(args.env_file)
    fixed, warnings = parse_fixed(args.fixed)
    for key, value in fixed.items():
        base_env[key] = value

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
                "  ok score={score:.1f} first_audio_p50={fa:.1f}ms compute_p50={comp:.1f}ms rtf_p50={rtf:.3f}".format(
                    score=result["score"],
                    fa=s["tts_first_audio_ms"]["p50"],
                    comp=s["tts_compute_ms"]["p50"],
                    rtf=s["rtf"]["p50"],
                ),
                flush=True,
            )
        else:
            print(f"  failed: {result['error']}", flush=True)

    successful = [r for r in results if r["status"] == "ok"]
    best = min(successful, key=lambda r: r["score"]) if successful else None
    report = {
        "model": "indic-mio",
        "env_file": str(args.env_file),
        "prompts": args.prompt,
        "fixed": fixed,
        "results": sorted(results, key=lambda r: r["score"]),
        "best": best,
    }
    report_path = out_dir / "tts-indic-mio-tuning-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {report_path.relative_to(REPO_ROOT)}", flush=True)
    if best is None:
        print("no successful candidates", flush=True)
        return 1
    suggested_env = dict(base_env)
    suggested_env.update(best["env"])
    env_path = out_dir / "tts.env.tuned"
    write_env_file(env_path, {k: suggested_env[k] for k in sorted(TUNABLES | set(fixed)) if k in suggested_env})
    print(f"best={best['name']}", flush=True)
    print(f"wrote {env_path.relative_to(REPO_ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
