#!/usr/bin/env python3
"""Repeatable load, resource, and response-sanity benchmark for Adaptive LLM.

Run on the GPU host from the repository root. Default target is Qwen3-0.6B
(--model-path); pass --model-ref to benchmark any registry model instead
(goes through the same LLM_MODEL_REF + model-profiles.json path production
uses, so runtime_profiles env like LLM_ALLOW_UNSAFE_FLOAT16 is applied):

  adaptive_llm/.venv/bin/python adaptive_llm/tools/vllm_tune.py \
    --model-ref google/gemma-3-1b-it \
    --candidate safe,0.70,1024,2 --candidate balanced,0.75,1536,4 \
    --candidate high-concurrency,0.80,2048,8

Each candidate is name,gpu_memory_utilization,max_batched_tokens,max_num_seqs.
The script starts a fresh service for each candidate, runs warm-up, then sends
short voice-style requests at 1, 2, and 4 concurrent sessions.  It writes JSON
with raw request timings, system-resource maxima, and transparent quality checks.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time
import uuid

import grpc

ROOT = Path(__file__).resolve().parents[2]
QWEN_MODEL_PATH = ROOT / "adaptive_llm/models/Qwen/Qwen3-0.6B"
sys.path.insert(0, str(ROOT))
from adaptive_llm.proto import llm_pb2, llm_pb2_grpc  # noqa: E402

PROMPTS = [
    "Reply in one short sentence: what is 7 plus 5?",
    "Reply in one short sentence: name the capital of France.",
    "Reply in one short sentence: what is the opposite of hot?",
    "Reply in one short sentence: list the first three letters of the English alphabet.",
]
QUALITY_EXPECTATIONS = [("12",), ("paris",), ("cold",), ("a", "b", "c")]


def percentile(values, p):
    if not values:
        return None
    data = sorted(values)
    return data[min(len(data) - 1, round((len(data) - 1) * p))]


def proc_stat(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split()
        # /proc/<pid>/stat fields 14/15 are CPU ticks and field 24 is RSS.
        # The list is zero-indexed after split(), hence 13, 14, and 23.
        return int(fields[13]) + int(fields[14]), int(fields[23])
    except (FileNotFoundError, ValueError):
        return 0, 0


def descendants(root_pid):
    """Return the launch process plus its live descendants using /proc PPIDs."""
    parents = {}
    for entry in Path("/proc").glob("[0-9]*"):
        try:
            parents[int(entry.name)] = int((entry / "stat").read_text().split()[3])
        except (FileNotFoundError, ValueError, IndexError):
            pass
    result = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, ppid in parents.items():
            if ppid in result and pid not in result:
                result.add(pid)
                changed = True
    return result


class ResourceMonitor:
    def __init__(self, service_pid, interval):
        self.service_pid, self.interval, self.samples = service_pid, interval, []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.started = False
        self._last_cpu, self._last_t = 0, 0.0

    def start(self):
        self.thread.start()
        self.started = True

    def close(self):
        self.stop.set()
        if self.started:
            self.thread.join(timeout=2)

    def _gpu(self):
        cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
               "--format=csv,noheader,nounits"]
        try:
            row = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=3).splitlines()[0]
            gpu, used, total, temp, power = [x.strip() for x in row.split(",")]
            return {"gpu_util_pct": float(gpu), "vram_used_mib": float(used), "vram_total_mib": float(total),
                    "gpu_temp_c": float(temp), "gpu_power_w": float(power)}
        except (FileNotFoundError, subprocess.SubprocessError, IndexError, ValueError):
            return {"gpu_available": False}

    def _run(self):
        clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
        while not self.stop.is_set():
            now = time.monotonic()
            pids = descendants(self.service_pid)
            ticks, rss_pages = 0, 0
            for pid in pids:
                t, r = proc_stat(pid); ticks += t; rss_pages += r
            cpu_pct = 0.0 if not self._last_t else 100 * ((ticks - self._last_cpu) / clock_ticks) / (now - self._last_t)
            self._last_cpu, self._last_t = ticks, now
            mem = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, value = line.split(":", 1); mem[key] = int(value.split()[0]) / 1024
            sample = {"t_s": now, "service_cpu_pct": cpu_pct, "service_rss_mib": rss_pages * os.sysconf("SC_PAGE_SIZE") / 1048576,
                      "ram_used_mib": mem["MemTotal"] - mem["MemAvailable"], "ram_total_mib": mem["MemTotal"], "processes": len(pids)}
            sample.update(self._gpu()); self.samples.append(sample)
            self.stop.wait(self.interval)

    def summary(self):
        keys = ("service_cpu_pct", "service_rss_mib", "ram_used_mib", "gpu_util_pct", "vram_used_mib", "gpu_temp_c", "gpu_power_w")
        return {f"max_{key}": max((x[key] for x in self.samples if key in x), default=None) for key in keys}


def request(target, prompt):
    started = time.perf_counter()
    with grpc.insecure_channel(target) as channel:
        reply = llm_pb2_grpc.LlmStub(channel).Generate(llm_pb2.GenerateRequest(session_id=str(uuid.uuid4()), text=prompt), timeout=120)
    elapsed = time.perf_counter() - started
    profile = reply.profile
    return {"elapsed_s": elapsed, "text": reply.text, "output_tokens": profile.output_tokens,
            "ttft_ms": profile.ttft_ms, "decode_tps": profile.decode_tokens_per_second}


def recorded_request(target, prompt):
    """Keep individual RPC failures in the raw result instead of aborting a load."""
    try:
        return {"ok": True, **request(target, prompt)}
    except grpc.RpcError as exc:
        return {"ok": False, "error": str(exc), "grpc_code": exc.code().name}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def wait_ready(target, process):
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("service exited during startup")
        try:
            with grpc.insecure_channel(target) as channel:
                grpc.channel_ready_future(channel).result(timeout=1)
            return
        except grpc.FutureTimeoutError:
            time.sleep(1)
    raise RuntimeError("service did not become ready within 180 seconds")


def quality(results):
    checks = []
    for result, expected in zip(results, QUALITY_EXPECTATIONS):
        text = result.get("text", "").lower()
        checks.append({"pass": result.get("ok", False) and all(word in text for word in expected),
                       "expected_terms": expected, "response": result.get("text"),
                       "error": result.get("error")})
    return {"passed": sum(x["pass"] for x in checks), "total": len(checks), "checks": checks,
            "note": "Sanity checks catch empty/incorrect basic answers; they are not a substitute for human domain evaluation."}


def parse_candidate(raw):
    name, gpu, batch, seqs = raw.split(",")
    if int(seqs) < 1 or int(batch) < 1 or not 0 < float(gpu) <= 1:
        raise ValueError(f"invalid candidate {raw!r}")
    return {"name": name, "LLM_GPU_MEMORY_UTILIZATION": gpu, "LLM_MAX_NUM_BATCHED_TOKENS": batch, "LLM_MAX_NUM_SEQS": seqs,
            "ADAPTIVE_LLM_MAX_INFLIGHT_REQUESTS": seqs}


def stop(process):
    if process.poll() is None:
        # The facade starts a worker which starts EngineCore.  A signal only to
        # the facade can leave descendants holding CUDA allocations after the
        # parent exits, contaminating the next clean candidate.
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5)


def log_tail(path, lines=80):
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except FileNotFoundError:
        return ""


def target_is_free(target):
    # Do not open a test socket: that is forbidden in some restricted runners
    # and it can create noisy connection errors in the service logs.
    try:
        port = target.rsplit(":", 1)[1]
        listeners = subprocess.run(
            ["ss", "-ltnH"], text=True, capture_output=True, check=True, timeout=3
        ).stdout.splitlines()
        # ``ss`` ends each row with the peer address (usually ``*:*``), so
        # inspect every column rather than just the final token.
        return not any(
            any(token.endswith(f":{port}") for token in line.split())
            for line in listeners
        )
    except (FileNotFoundError, subprocess.SubprocessError, IndexError):
        # Binding will still make an occupied port fail loudly.  Do not reject a
        # benchmark merely because this optional preflight is unavailable.
        return True


def gpu_memory_used_mib():
    try:
        row = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL, timeout=3,
        ).splitlines()[0]
        return float(row.strip())
    except (FileNotFoundError, subprocess.SubprocessError, IndexError, ValueError):
        return None


def wait_for_gpu_release(baseline_mib, timeout_s=90):
    """Wait for the CUDA context to release before starting the next candidate."""
    if baseline_mib is None:
        return
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        used_mib = gpu_memory_used_mib()
        if used_mib is not None and used_mib <= baseline_mib + 128:
            return
        time.sleep(1)
    raise RuntimeError(
        f"GPU memory did not return near its {baseline_mib:.0f} MiB baseline within {timeout_s}s"
    )


def summarize_load(rows):
    successful = [row for row in rows if row["ok"]]
    return {
        "requests": len(rows), "errors": len(rows) - len(successful), "raw_requests": rows,
        "p50_elapsed_s": percentile([r["elapsed_s"] for r in successful], .5),
        "p95_elapsed_s": percentile([r["elapsed_s"] for r in successful], .95),
        "p50_ttft_ms": percentile([r["ttft_ms"] for r in successful], .5),
        "p95_ttft_ms": percentile([r["ttft_ms"] for r in successful], .95),
        "mean_decode_tps": statistics.fmean(r["decode_tps"] for r in successful if r["decode_tps"] > 0)
        if any(r["decode_tps"] > 0 for r in successful) else 0,
    }


def run_candidate(candidate, args):
    if not target_is_free(args.target):
        return {"candidate": candidate, "status": "failed", "error": f"target {args.target} is already occupied; refusing to benchmark a non-clean service"}
    baseline_vram_mib = gpu_memory_used_mib()
    env = os.environ.copy(); env.update(candidate)
    # Pin the model/engine: an accidental planner fallback invalidates comparison.
    env.update({"ADAPTIVE_LLM_SKIP_INSTALL": "1", "LLM_ENGINE": "vllm"})
    if args.model_ref:
        # Go through the same LLM_MODEL_REF + registry path production uses,
        # NOT LLM_MODEL_PATH: a raw path bypasses model-profiles.json's
        # runtime_profiles entirely, silently dropping model-specific
        # compatibility env (e.g. LLM_ALLOW_UNSAFE_FLOAT16 for gemma3 on a
        # pre-Ampere GPU) that this exact model needs to load at all. The
        # candidate's own gpu_util/batch/seqs still win over the profile's
        # baked-in defaults (start_worker() layers plan.engine_env in via
        # setdefault(), after this env is already set).
        env["LLM_MODEL_REF"] = args.model_ref
        env.pop("LLM_MODEL_PATH", None)
        if args.model_profile:
            env["LLM_MODEL_PROFILE"] = args.model_profile
    else:
        if not (args.model_path / "config.json").is_file() or not list(args.model_path.glob("*.safetensors")):
            return {"candidate": candidate, "status": "failed",
                    "error": f"checkpoint is incomplete: {args.model_path}"}
        env["LLM_MODEL_PATH"] = str(args.model_path)
        env.pop("LLM_MODEL_REF", None)
    log_path = args.log_dir / f"{candidate['name']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen([str(ROOT / "adaptive_llm/run.sh")], cwd=ROOT, env=env, text=True,
                               stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
    monitor = ResourceMonitor(process.pid, args.sample_interval)
    monitor.start()  # Include model load and CUDA graph capture, not only steady state.
    try:
        wait_ready(args.target, process)
        request(args.target, PROMPTS[0])  # warm-up, excluded from aggregate figures
        quality_results = [recorded_request(args.target, p) for p in PROMPTS]
        loads = {}
        for concurrency in args.concurrency:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                rows = list(pool.map(lambda i: recorded_request(args.target, PROMPTS[i % len(PROMPTS)]), range(concurrency * args.requests_per_session)))
            loads[str(concurrency)] = summarize_load(rows)
        status = "completed" if all(load["errors"] == 0 for load in loads.values()) else "failed"
        return {"candidate": candidate, "status": status, "quality": quality(quality_results), "loads": loads,
                "resources": {"maxima": monitor.summary(), "samples": monitor.samples},
                "log_path": str(log_path)}
    except Exception as exc:
        return {"candidate": candidate, "status": "failed", "error": str(exc),
                "resources": {"maxima": monitor.summary(), "samples": monitor.samples},
                "log_path": str(log_path), "log_tail": log_tail(log_path)}
    finally:
        stop(process); monitor.close(); log_file.close()
        # vLLM's child EngineCore can release its CUDA context after the facade
        # exits. Starting early makes the next candidate fail the planner's
        # free-VRAM guard and is not a clean-service comparison.
        wait_for_gpu_release(baseline_vram_mib)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", default=[], help="name,gpu_util,batch_tokens,max_seqs")
    parser.add_argument("--target", default="127.0.0.1:50053")
    parser.add_argument("--model-ref", default=None,
                        help="HF model ref (e.g. google/gemma-3-1b-it) to benchmark via the same "
                             "LLM_MODEL_REF + model-profiles.json registry path production uses -- "
                             "preferred over --model-path so runtime_profiles env (e.g. "
                             "LLM_ALLOW_UNSAFE_FLOAT16) is applied. Mutually exclusive with --model-path.")
    parser.add_argument("--model-profile", default=None,
                        help="optional LLM_MODEL_PROFILE to pin a specific model-profiles.json entry "
                             "when --model-ref alone is ambiguous")
    parser.add_argument("--model-path", type=Path, default=QWEN_MODEL_PATH,
                        help="local HF safetensors directory to benchmark directly via LLM_MODEL_PATH, "
                             "bypassing the registry (default: Qwen3-0.6B). Ignored if --model-ref is set.")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--requests-per-session", type=int, default=3)
    parser.add_argument("--sample-interval", type=float, default=.25)
    parser.add_argument("--output", type=Path, default=ROOT / "adaptive_llm/run/vllm-tuning-results.json")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "adaptive_llm/run/vllm-tuning-logs")
    args = parser.parse_args()
    candidates = [parse_candidate(c) for c in args.candidate] or [parse_candidate("balanced,0.70,512,4")]
    results = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "target": args.target,
               "model_ref": args.model_ref, "model_path": None if args.model_ref else str(args.model_path),
               "concurrency": args.concurrency, "results": [run_candidate(c, args) for c in candidates]}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__": main()
