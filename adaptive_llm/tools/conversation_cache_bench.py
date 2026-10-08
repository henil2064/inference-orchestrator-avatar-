#!/usr/bin/env python3
"""Compare vLLM prefix caching and SGLang radix caching on voice turns.

Each arm starts a clean service and sends an ever-growing complete dialogue,
which is the production request shape for cache-aware conversational memory.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from adaptive_llm.tools.vllm_cache_tune import (  # noqa: E402
    TURN_SETS, ResourceMonitor, gpu_memory_used_mib, run_session, summarize,
    target_is_free, wait_for_gpu_release, wait_ready,
)
from adaptive_llm.tools.vllm_tune import stop  # noqa: E402


def run_arm(engine: str, cache_enabled: bool, args) -> dict:
    name = f"{engine}-cache-{'on' if cache_enabled else 'off'}"
    if not target_is_free(args.target):
        return {"name": name, "status": "failed", "error": f"target {args.target} is occupied"}
    if not (args.model_path / "config.json").is_file() or not list(args.model_path.glob("*.safetensors")):
        return {"name": name, "status": "failed", "error": f"incomplete safetensors checkpoint: {args.model_path}"}
    baseline = gpu_memory_used_mib()
    env = os.environ.copy()
    env.update({"ADAPTIVE_LLM_SKIP_INSTALL": "1", "LLM_ENGINE": engine, "LLM_MODEL_REF": "", "LLM_MODEL_PATH": str(args.model_path),
                "LLM_GPU_MEMORY_UTILIZATION": str(args.gpu_memory_utilization), "LLM_MAX_MODEL_LEN": str(args.context_tokens),
                "LLM_MAX_TOKENS": str(args.max_tokens), "LLM_MAX_NUM_BATCHED_TOKENS": str(args.max_prefill_tokens),
                "LLM_MAX_NUM_SEQS": str(args.sessions), "ADAPTIVE_LLM_MAX_INFLIGHT_REQUESTS": str(args.sessions),
                "LLM_ENABLE_PREFIX_CACHING": "1" if cache_enabled else "0"})
    # The GTX 1650 test host lacks Triton's bundled ptxas. Eager mode makes a
    # clean comparison possible there; leave the default unchanged elsewhere.
    if args.enforce_eager: env["LLM_ENFORCE_EAGER"] = "1"
    if engine == "sglang": env["LLM_SGLANG_PORT"] = str(args.sglang_port)
    if engine == "sglang" and args.sglang_python: env["LLM_SGLANG_PYTHON"] = str(args.sglang_python)
    log_path = args.log_dir / f"{name}.log"; log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen([str(ROOT / "adaptive_llm/run.sh")], cwd=ROOT, env=env, text=True,
                                   stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
        monitor = ResourceMonitor(process.pid, args.sample_interval); monitor.start()
        try:
            turns = TURN_SETS[args.turns]
            wait_ready(args.target, process)
            run_session(args.target, "warmup", args.stream, turns)
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.sessions) as pool:
                sessions = list(pool.map(lambda index: run_session(args.target, index, args.stream, turns), range(args.sessions)))
            summary = summarize(sessions)
            return {"name": name, "engine": engine, "cache_enabled": cache_enabled,
                    "status": "completed" if not summary["errors"] else "failed", "summary": summary, "sessions": sessions,
                    "resources": {"maxima": monitor.summary(), "samples": monitor.samples}, "log_path": str(log_path)}
        except Exception as exc:
            return {"name": name, "engine": engine, "cache_enabled": cache_enabled, "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}", "resources": {"maxima": monitor.summary(), "samples": monitor.samples}, "log_path": str(log_path)}
        finally:
            stop(process); monitor.close(); wait_for_gpu_release(baseline)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=ROOT / "adaptive_llm/models/huggingface/Qwen--Qwen3-0.6B")
    parser.add_argument("--target", default="127.0.0.1:50053"); parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--context-tokens", type=int, default=2048); parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--max-prefill-tokens", type=int, default=512); parser.add_argument("--gpu-memory-utilization", type=float, default=.70)
    parser.add_argument("--sglang-port", type=int, default=30000); parser.add_argument("--sample-interval", type=float, default=.25)
    parser.add_argument("--sglang-python", type=Path, help="Python executable in the isolated SGLang environment")
    parser.add_argument("--enforce-eager", action="store_true", help="disable vLLM CUDA graph capture for older CUDA toolchains")
    parser.add_argument("--engines", nargs="+", choices=("vllm", "sglang"), default=["vllm", "sglang"])
    parser.add_argument("--include-cache-off", action="store_true")
    parser.add_argument("--stream", action="store_true", help="use GenerateStream for a true prefill/decode split (needed for SGLang TTFT)")
    parser.add_argument("--turns", choices=list(TURN_SETS), default="basic")
    parser.add_argument("--output", type=Path, default=ROOT / "adaptive_llm/run/conversation-cache-benchmark.json")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "adaptive_llm/run/conversation-cache-benchmark-logs")
    args = parser.parse_args()
    if args.sessions < 1 or not 0 < args.gpu_memory_utilization <= 1: parser.error("invalid sessions or GPU memory utilization")
    arms = [(engine, True) for engine in args.engines] + ([(engine, False) for engine in args.engines] if args.include_cache_off else [])
    results = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model_path": str(args.model_path),
               "turns": [turn for turn, _ in TURN_SETS[args.turns]],
               "sessions": args.sessions, "results": [run_arm(engine, cached, args) for engine, cached in arms]}
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    for arm in results["results"]:
        summary = arm.get("summary", {})
        print(f"{arm['name']}: {arm['status']} | cache={summary.get('token_weighted_prefix_cache_hit_rate_pct')}% | "
              f"p50 prefill={summary.get('p50_prefill_ms')} ms | p50 decode={summary.get('p50_decode_ms')} ms | "
              f"p95 TTFT={summary.get('p95_ttft_ms')} ms | p95 latency={summary.get('p95_elapsed_s')} s")
    print(args.output)


if __name__ == "__main__": main()
