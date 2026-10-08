#!/usr/bin/env python3
"""A/B one Adaptive LLM model/quantization against another on the real gRPC facade.

Each arm starts a clean service with an environment overlay, waits for the
worker to load, then runs the same fixed workload through ``GenerateStream``
-- the path production actually uses. Two things are measured on purpose:

* latency/throughput, including model load time, single-stream TTFT and decode
  rate, and a concurrency sweep that shows how the arm behaves under batching;
* output text, captured verbatim so a quantization change can be judged on
  answer quality and not only on tokens per second.

Arms are compared by running this script once per arm and pointing
``--compare`` at the resulting JSON files.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from adaptive_llm.proto import llm_pb2  # noqa: E402
from adaptive_llm.tools.vllm_cache_tune import request_stream  # noqa: E402
from adaptive_llm.tools.vllm_tune import (  # noqa: E402
    ResourceMonitor, gpu_memory_used_mib, percentile, stop, target_is_free,
    wait_for_gpu_release,
)

import grpc  # noqa: E402


def wait_ready(target, process, timeout_s):
    """vllm_tune.wait_ready hardcodes 180s, which a 27B checkpoint loading off
    network storage exceeds routinely. Same contract, caller-set deadline."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"service exited during startup (exit code {process.returncode})")
        try:
            with grpc.insecure_channel(target) as channel:
                grpc.channel_ready_future(channel).result(timeout=1)
            return
        except grpc.FutureTimeoutError:
            time.sleep(1)
    raise RuntimeError(f"service did not become ready within {timeout_s} seconds")


# Single-turn prompts. ``expect`` terms are a coarse smoke check for an answer
# that collapsed into nonsense; they are not a clinical quality metric.
SINGLE_TURN = [
    {
        "id": "greeting",
        "text": "Hi there, can you introduce yourself in one sentence?",
        "expect": (),
    },
    {
        "id": "arithmetic",
        "text": "What is 17 minus 9? Reply with just the number.",
        "expect": ("8",),
    },
    {
        "id": "general-knowledge",
        "text": "Which planet is closest to the sun? Reply in one word.",
        "expect": ("mercury",),
    },
    {
        "id": "med-terminology",
        "text": "In plain language, what does the term hypertension mean?",
        "expect": ("blood pressure",),
    },
    {
        "id": "med-symptom-triage",
        "text": (
            "A patient says they have crushing chest pain radiating to the left arm "
            "with sweating. How urgent is this and what should they do?"
        ),
        "expect": ("emergency",),
    },
    {
        "id": "med-drug-class",
        "text": "What class of medication is metformin, and what is it normally used for?",
        "expect": ("diabet",),
    },
    {
        "id": "med-lab-value",
        "text": "A fasting blood glucose of 190 mg/dL: is that normal, and what does it suggest?",
        "expect": (),
    },
    {
        "id": "instruction-following",
        "text": "List exactly three common symptoms of iron deficiency anemia, as a plain comma-separated list.",
        "expect": (),
    },
    {
        "id": "refusal-boundary",
        "text": "My friend asked me to tell them a safe dose of insulin to take. What should I say?",
        "expect": (),
    },
    {
        "id": "long-form",
        "text": (
            "Explain in about four sentences how a pulse oximeter works and why a reading "
            "can be misleading in a patient with poor peripheral circulation."
        ),
        "expect": (),
    },
]

# A multi-turn session in the production request shape: the full completed
# dialogue plus one new user message. This is what exercises the context
# window, prefix caching, and the chat template.
MULTI_TURN = [
    ("Hi, please remember two things: my name is Jordan, and I am a nurse at Sunrise Hospital.", ()),
    ("Also remember my patient in bed 4 is on warfarin.", ()),
    ("What is my name and where do I work? Reply in one short sentence.", ("jordan", "sunrise")),
    ("Which medication did I say the patient in bed 4 is on? Reply briefly.", ("warfarin",)),
    ("Name one common food or drug interaction to watch for with that medication.", ()),
    ("Summarize in two short sentences everything you know about me and my patient.", ("jordan", "warfarin")),
]


def _messages(history, user_text):
    """Build the structured request: completed history, then the new turn."""
    messages = []
    for role, content in history:
        messages.append(llm_pb2.ChatMessage(role=role, content=content))
    messages.append(llm_pb2.ChatMessage(role=llm_pb2.ChatMessage.USER, content=user_text))
    return messages


def run_single_turn(target, repeats):
    """One independent request per prompt, no history, repeated for stability."""
    rows = []
    for repeat in range(repeats):
        for prompt in SINGLE_TURN:
            session = f"single-{prompt['id']}-{repeat}"
            try:
                result = request_stream(target, session, _messages([], prompt["text"]))
                text = result["text"]
                rows.append({
                    "prompt_id": prompt["id"], "repeat": repeat, "ok": True,
                    "prompt": prompt["text"], "text": text,
                    "expect_pass": all(term in text.lower() for term in prompt["expect"]),
                    "expect_terms": list(prompt["expect"]),
                    **{k: v for k, v in result.items() if k != "text"},
                })
            except Exception as exc:  # noqa: BLE001 - recorded, not raised, so one bad prompt doesn't void the arm
                rows.append({"prompt_id": prompt["id"], "repeat": repeat, "ok": False,
                             "prompt": prompt["text"], "error": f"{type(exc).__name__}: {exc}"})
    return rows


def run_multi_turn(target, session_suffix=""):
    """Walk the dialogue, feeding each reply back as completed history."""
    history = []
    rows = []
    session = f"multi{session_suffix}"
    for index, (user_text, expect) in enumerate(MULTI_TURN):
        try:
            result = request_stream(target, session, _messages(history, user_text))
            text = result["text"]
            history.append((llm_pb2.ChatMessage.USER, user_text))
            history.append((llm_pb2.ChatMessage.ASSISTANT, text))
            rows.append({"turn": index, "ok": True, "prompt": user_text, "text": text,
                         "expect_pass": all(term in text.lower() for term in expect),
                         "expect_terms": list(expect),
                         **{k: v for k, v in result.items() if k != "text"}})
        except Exception as exc:  # noqa: BLE001
            rows.append({"turn": index, "ok": False, "prompt": user_text,
                         "error": f"{type(exc).__name__}: {exc}"})
            break
    return rows


def run_concurrency(target, levels, prompt_index=9):
    """Fire N identical long-form requests at once and time the whole batch.

    Wall time across the batch -- not the mean of per-request latencies -- is
    what shows whether an engine is actually batching or serializing.
    """
    prompt = SINGLE_TURN[prompt_index]
    sweep = []
    for level in levels:
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=level) as pool:
            futures = [
                pool.submit(request_stream, target, f"conc-{level}-{i}", _messages([], prompt["text"]))
                for i in range(level)
            ]
            results, errors = [], []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{type(exc).__name__}: {exc}")
        wall_s = time.perf_counter() - started
        output_tokens = sum(r["output_tokens"] for r in results)
        sweep.append({
            "concurrency": level, "wall_s": wall_s, "completed": len(results), "errors": errors,
            "total_output_tokens": output_tokens,
            "aggregate_output_tps": output_tokens / wall_s if wall_s > 0 else 0.0,
            "ttft_ms_mean": statistics.fmean([r["ttft_ms"] for r in results]) if results else None,
            "ttft_ms_p95": percentile([r["ttft_ms"] for r in results], .95) if results else None,
            "elapsed_s_mean": statistics.fmean([r["elapsed_s"] for r in results]) if results else None,
            "elapsed_s_p95": percentile([r["elapsed_s"] for r in results], .95) if results else None,
        })
    return sweep


def summarize(single_rows, multi_rows, sweep):
    ok_single = [r for r in single_rows if r.get("ok")]
    ok_multi = [r for r in multi_rows if r.get("ok")]
    latency_rows = ok_single + ok_multi

    def stat(key, rows, p=None):
        values = [r[key] for r in rows if r.get(key) is not None]
        if not values:
            return None
        return percentile(values, p) if p else statistics.fmean(values)

    checked = [r for r in latency_rows if r.get("expect_terms")]
    return {
        "requests_ok": len(latency_rows),
        "requests_failed": len(single_rows) + len(multi_rows) - len(latency_rows),
        "ttft_ms_mean": stat("ttft_ms", latency_rows),
        "ttft_ms_p50": stat("ttft_ms", latency_rows, .50),
        "ttft_ms_p95": stat("ttft_ms", latency_rows, .95),
        "decode_tps_mean": stat("decode_tps", latency_rows),
        "decode_tps_p50": stat("decode_tps", latency_rows, .50),
        "elapsed_s_mean": stat("elapsed_s", latency_rows),
        "elapsed_s_p95": stat("elapsed_s", latency_rows, .95),
        "output_tokens_mean": stat("output_tokens", latency_rows),
        "input_tokens_mean": stat("input_tokens", latency_rows),
        "cached_prompt_tokens_total": sum(r.get("cached_prompt_tokens", 0) for r in latency_rows),
        "smoke_checks_passed": sum(1 for r in checked if r.get("expect_pass")),
        "smoke_checks_total": len(checked),
        "concurrency": sweep,
        "note": "Smoke checks catch collapsed/incorrect basic answers only; they are not a clinical quality metric.",
    }


def run_arm(args) -> dict:
    if not target_is_free(args.target):
        return {"arm": args.arm, "status": "failed", "error": f"target {args.target} is occupied"}

    env = os.environ.copy()
    env["ADAPTIVE_LLM_SKIP_INSTALL"] = "1"
    for item in args.set:
        key, value = item.split("=", 1)
        env[key] = value

    baseline_mib = gpu_memory_used_mib()
    log_path = args.log_dir / f"{args.arm}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen([str(ROOT / "adaptive_llm/run.sh")], cwd=ROOT, env=env, text=True,
                                   stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
        monitor = ResourceMonitor(process.pid, args.sample_interval)
        monitor.start()
        try:
            wait_ready(args.target, process, args.ready_timeout)
            load_s = time.perf_counter() - started_at
            vram_after_load_mib = gpu_memory_used_mib()
            # Warmup is excluded from every statistic: the first request pays
            # for CUDA graph/kernel autotuning that no later request repeats.
            run_single_turn(args.target, repeats=1)
            single_rows = run_single_turn(args.target, repeats=args.repeats)
            multi_rows = run_multi_turn(args.target)
            sweep = run_concurrency(args.target, args.concurrency)
            summary = summarize(single_rows, multi_rows, sweep)
            failed = summary["requests_failed"] or any(s["errors"] for s in sweep)
            return {
                "arm": args.arm, "status": "failed" if failed else "completed",
                "env_overlay": dict(item.split("=", 1) for item in args.set),
                "load_seconds": load_s,
                "vram_after_load_mib": vram_after_load_mib,
                "vram_baseline_mib": baseline_mib,
                "vram_peak_mib": max((s.get("vram_used_mib", 0) for s in monitor.samples), default=None),
                "summary": summary,
                "single_turn": single_rows, "multi_turn": multi_rows,
                "resources": {"samples": monitor.samples},
                "log_path": str(log_path),
            }
        except Exception as exc:  # noqa: BLE001
            return {"arm": args.arm, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                    "log_path": str(log_path), "resources": {"samples": monitor.samples}}
        finally:
            stop(process)
            monitor.close()
            wait_for_gpu_release(baseline_mib)


def compare(paths) -> dict:
    arms = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    keys = ("load_seconds", "vram_after_load_mib", "vram_peak_mib")
    summary_keys = ("ttft_ms_mean", "ttft_ms_p95", "decode_tps_mean", "elapsed_s_mean",
                    "elapsed_s_p95", "smoke_checks_passed", "smoke_checks_total")
    table = []
    for arm in arms:
        row = {"arm": arm.get("arm"), "status": arm.get("status")}
        row.update({key: arm.get(key) for key in keys})
        row.update({key: (arm.get("summary") or {}).get(key) for key in summary_keys})
        row["concurrency"] = {
            str(s["concurrency"]): round(s["aggregate_output_tps"], 1)
            for s in (arm.get("summary") or {}).get("concurrency", [])
        }
        table.append(row)
    return {"arms": table}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", default="arm", help="name for this arm; also names its log file")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="environment overlay for this arm; repeatable")
    parser.add_argument("--target", default="127.0.0.1:50053")
    parser.add_argument("--repeats", type=int, default=3, help="measured passes over the single-turn suite")
    parser.add_argument("--concurrency", type=int, nargs="*", default=[1, 4, 8])
    parser.add_argument("--sample-interval", type=float, default=2.0)
    parser.add_argument("--ready-timeout", type=float, default=1200.0,
                        help="seconds to wait for the worker to finish loading the model")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "adaptive_llm/run/bench")
    parser.add_argument("--out", type=Path, help="write this arm's JSON result here")
    parser.add_argument("--compare", nargs="+", metavar="RESULT.json",
                        help="print a comparison of previously written arm results and exit")
    args = parser.parse_args()

    if args.compare:
        print(json.dumps(compare(args.compare), indent=2))
        return

    result = run_arm(args)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in {"single_turn", "multi_turn", "resources"}}, indent=2))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
