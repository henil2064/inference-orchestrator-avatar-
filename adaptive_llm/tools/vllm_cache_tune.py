#!/usr/bin/env python3
"""Multi-turn, cache-aware voice-session benchmark for Adaptive LLM."""
from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid

import grpc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from adaptive_llm.proto import llm_pb2, llm_pb2_grpc  # noqa: E402
from adaptive_llm.tools.vllm_tune import (  # noqa: E402
    QWEN_MODEL_PATH, ResourceMonitor, gpu_memory_used_mib, percentile, stop,
    target_is_free, wait_for_gpu_release, wait_ready,
)

# The same turns are used for every session and cache setting.  Responses are
# appended to each session's next request, exactly like a text-only voice agent.
TURNS_BASIC = [
    ("Remember this preference: my name is Maya.", ()),
    ("What name did I ask you to remember? Reply in one short sentence.", ("maya",)),
    ("Please remember that I live in Paris.", ()),
    ("What city do I live in? Reply in one short sentence.", ("paris",)),
    ("Reply briefly: what are both my name and city?", ("maya", "paris")),
]

# A longer, harder session: interleaves multiple facts to remember with
# recall questions, plain reasoning/trivia unrelated to memory, and a
# combined memory+reasoning question, then a full recall at the end. This
# mirrors a real voice-agent conversation far better than TURNS_BASIC.
TURNS_VOICE_AGENT = [
    ("Hi, please remember two things: my name is Jordan, and I work as a nurse at Sunrise Hospital.", ()),
    ("Also remember that my dog is named Biscuit, a three-year-old golden retriever.", ()),
    ("Quick check: what's my name and where do I work? Reply in one short sentence.", ("jordan", "sunrise")),
    ("Math check: what is 17 minus 9? Reply with just the number.", ("8",)),
    ("Please also remember I have a dentist appointment this coming Tuesday at 3 in the afternoon.", ()),
    ("Trivia: which planet is closest to the sun? Reply in one word.", ("mercury",)),
    ("Remind me, what's my dog's name and breed? Reply briefly.", ("biscuit", "golden retriever")),
    ("If today is Monday, is my dentist appointment tomorrow or the day after? Reply in one word.", ("tomorrow",)),
    ("Summarize everything you know about me in two short sentences: my name, job, pet, and appointment day and time.",
     ("jordan", "biscuit")),
]

TURN_SETS = {"basic": TURNS_BASIC, "voice-agent": TURNS_VOICE_AGENT}
TURNS = TURNS_BASIC  # backward-compatible default


def request(target, session_id, messages):
    """Unary Generate call. vLLM measures true per-token TTFT internally even
    here (it iterates its async token stream before returning), but SGLang's
    ``/generate`` HTTP endpoint only returns the finished text, so for SGLang
    ``ttft_ms``/``decode_ms`` collapse to the full request latency here. Use
    ``request_stream`` for a genuine prefill/decode split on both engines.
    """
    started = time.perf_counter()
    with grpc.insecure_channel(target) as channel:
        reply = llm_pb2_grpc.LlmStub(channel).Generate(
            llm_pb2.GenerateRequest(session_id=session_id, messages=messages), timeout=120
        )
    profile = reply.profile
    attrs = dict(profile.attributes)
    return {
        "elapsed_s": time.perf_counter() - started, "text": reply.text,
        "input_tokens": profile.input_tokens, "output_tokens": profile.output_tokens,
        "cached_prompt_tokens": int(attrs.get("cached_prompt_tokens", "0")),
        "history_turns_used": int(attrs.get("history_turns_used", "0")),
        "history_turns_trimmed": int(attrs.get("history_turns_trimmed", "0")),
        "ttft_ms": profile.ttft_ms, "prefill_ms": profile.prefill_ms, "decode_ms": profile.decode_ms,
        "decode_tps": profile.decode_tokens_per_second,
    }


def request_stream(target, session_id, messages):
    """GenerateStream call. Both workers time their first emitted token from
    their own event loop, so this is the only RPC that gives SGLang a real
    prefill/decode split (its unary endpoint can't see per-token timing).
    """
    started = time.perf_counter()
    text_parts = []
    profile = None
    client_first_token_s = None
    with grpc.insecure_channel(target) as channel:
        stub = llm_pb2_grpc.LlmStub(channel)
        for chunk in stub.GenerateStream(
            llm_pb2.GenerateRequest(session_id=session_id, messages=messages), timeout=120
        ):
            if chunk.text_delta:
                if client_first_token_s is None:
                    client_first_token_s = time.perf_counter() - started
                text_parts.append(chunk.text_delta)
            if chunk.is_final:
                profile = chunk.profile
    elapsed_s = time.perf_counter() - started
    attrs = dict(profile.attributes) if profile else {}
    return {
        "elapsed_s": elapsed_s, "text": "".join(text_parts),
        "input_tokens": profile.input_tokens if profile else 0,
        "output_tokens": profile.output_tokens if profile else 0,
        "cached_prompt_tokens": int(attrs.get("cached_prompt_tokens", "0")),
        "history_turns_used": int(attrs.get("history_turns_used", "0")),
        "history_turns_trimmed": int(attrs.get("history_turns_trimmed", "0")),
        "ttft_ms": profile.ttft_ms if profile else 0.0,
        "prefill_ms": profile.prefill_ms if profile else 0.0,
        "decode_ms": profile.decode_ms if profile else 0.0,
        "decode_tps": profile.decode_tokens_per_second if profile else 0.0,
        # Client-observed wall-clock time to the first delta, as a sanity
        # cross-check against the server-reported ttft_ms (network/gRPC
        # framing overhead sits between the two).
        "client_ttft_ms": client_first_token_s * 1000 if client_first_token_s is not None else None,
    }


def run_session(target, index, stream=False, turns=None):
    fetch = request_stream if stream else request
    session_id = f"voice-cache-{index}-{uuid.uuid4()}"
    messages, rows = [], []
    for turn, expected in (turns if turns is not None else TURNS):
        messages.append(llm_pb2.ChatMessage(role=llm_pb2.ChatMessage.USER, content=turn))
        try:
            row = {"ok": True, "turn": len(rows) + 1, "expected_terms": expected,
                   **fetch(target, session_id, messages)}
            row["quality_pass"] = all(term in row["text"].lower() for term in expected)
            messages.append(llm_pb2.ChatMessage(role=llm_pb2.ChatMessage.ASSISTANT, content=row["text"]))
        except grpc.RpcError as exc:
            row = {"ok": False, "turn": len(rows) + 1, "expected_terms": expected,
                   "error": str(exc), "grpc_code": exc.code().name, "quality_pass": False}
            rows.append(row)
            break
        rows.append(row)
    return {"session_id": session_id, "requests": rows}


def summarize(sessions):
    rows = [row for session in sessions for row in session["requests"]]
    good = [row for row in rows if row["ok"]]
    total_prompt = sum(row["input_tokens"] for row in good)
    cached = sum(row["cached_prompt_tokens"] for row in good)
    expected = [row for row in good if row["expected_terms"]]
    return {
        "requests": len(rows), "errors": len(rows) - len(good),
        "total_prompt_tokens": total_prompt, "cached_prompt_tokens": cached,
        "token_weighted_prefix_cache_hit_rate_pct": 100 * cached / total_prompt if total_prompt else 0,
        "requests_with_prefix_hit": sum(row["cached_prompt_tokens"] > 0 for row in good),
        "request_prefix_hit_rate_pct": 100 * sum(row["cached_prompt_tokens"] > 0 for row in good) / len(good) if good else 0,
        "quality": {"passed": sum(row["quality_pass"] for row in expected), "total": len(expected)},
        "p50_ttft_ms": percentile([row["ttft_ms"] for row in good], .5),
        "p95_ttft_ms": percentile([row["ttft_ms"] for row in good], .95),
        "p50_prefill_ms": percentile([row.get("prefill_ms", row["ttft_ms"]) for row in good], .5),
        "p50_decode_ms": percentile([row.get("decode_ms", 0.0) for row in good], .5),
        "p95_decode_ms": percentile([row.get("decode_ms", 0.0) for row in good], .95),
        "p50_elapsed_s": percentile([row["elapsed_s"] for row in good], .5),
        "p95_elapsed_s": percentile([row["elapsed_s"] for row in good], .95),
        "mean_decode_tps": statistics.fmean(row["decode_tps"] for row in good if row["decode_tps"] > 0) if any(row["decode_tps"] > 0 for row in good) else 0,
    }


def run_candidate(name, prefix_caching, args):
    if not target_is_free(args.target):
        return {"name": name, "status": "failed", "error": "target is already occupied"}
    baseline = gpu_memory_used_mib()
    env = os.environ.copy()
    env.update({"ADAPTIVE_LLM_SKIP_INSTALL": "1", "LLM_ENGINE": "vllm", "LLM_MODEL_PATH": str(QWEN_MODEL_PATH),
                "LLM_GPU_MEMORY_UTILIZATION": "0.70", "LLM_MAX_NUM_BATCHED_TOKENS": "512",
                "LLM_MAX_NUM_SEQS": "4", "ADAPTIVE_LLM_MAX_INFLIGHT_REQUESTS": "4",
                "LLM_ENABLE_PREFIX_CACHING": "1" if prefix_caching else "0"})
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
                sessions = list(pool.map(lambda i: run_session(args.target, i, args.stream, turns), range(args.sessions)))
            return {"name": name, "prefix_caching": prefix_caching, "status": "completed",
                    "sessions": sessions, "summary": summarize(sessions),
                    "resources": {"maxima": monitor.summary(), "samples": monitor.samples}, "log_path": str(log_path)}
        except Exception as exc:
            return {"name": name, "prefix_caching": prefix_caching, "status": "failed", "error": str(exc),
                    "resources": {"maxima": monitor.summary(), "samples": monitor.samples}, "log_path": str(log_path)}
        finally:
            stop(process); monitor.close(); wait_for_gpu_release(baseline)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="127.0.0.1:50053")
    parser.add_argument("--sessions", type=int, default=4)
    parser.add_argument("--sample-interval", type=float, default=.25)
    parser.add_argument("--output", type=Path, default=ROOT / "adaptive_llm/run/vllm-cache-tuning-results.json")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "adaptive_llm/run/vllm-cache-tuning-logs")
    parser.add_argument("--stream", action="store_true", help="use GenerateStream for a true prefill/decode split")
    parser.add_argument("--turns", choices=list(TURN_SETS), default="basic")
    args = parser.parse_args()
    results = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sessions": args.sessions,
               "turns": [turn for turn, _ in TURN_SETS[args.turns]],
               "results": [run_candidate("prefix-on", True, args), run_candidate("prefix-off", False, args)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
