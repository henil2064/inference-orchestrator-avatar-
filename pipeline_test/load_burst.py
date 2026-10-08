#!/usr/bin/env python3
"""Concurrent-attempt load generator with durable per-attempt records and
offered-load reconciliation against the edge's own Prometheus counters.

Where stream_wav.py drives ONE call end to end, this launches many
concurrent attempts at once -- each a stream_wav.py subprocess, so admission
rejection and connection failures are observed exactly as a real client
would see them -- and reconciles what the generator scheduled against what
the edge actually observed. Per the Load & Horizontal-Scaling Plan: "a
generator schedule is offered demand, not proof it hit the server" -- this
is the proof.

Every attempt gets a distinct attempt_id and a durable JSONL record
(--out, default pipeline_test/recordings/load_attempts.jsonl) written as
each attempt finishes, so a run's outcomes survive even if the burst itself
is interrupted. The trailing reconciliation line compares:
  generator_scheduled   - how many attempts this run launched
  generator_completed / generator_rejected / generator_failed_other
                         - what each attempt's own subprocess reported
  edge_connection_attempts_total_delta
                         - what the edge itself observed reaching it
  edge_connection_rejected_total_delta / edge_sessions_total_delta
                         - the edge's own admission accounting

A None delta means the edge metrics endpoint was unreachable -- that is
logged and left as unknown, never coerced to 0.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path


def _fetch_metrics(url: str) -> dict[str, float]:
    """Parse a Prometheus text-exposition response into {metric_name: value}
    for bare (unlabeled) series -- enough for the plain Counters this script
    reconciles against. Returns {} (not a fabricated healthy state) if the
    endpoint can't be reached."""
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            text = resp.read().decode()
    except OSError:
        return {}
    values: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or "{" in line:
            continue
        try:
            name, value = line.rsplit(" ", 1)
            values[name] = float(value)
        except ValueError:
            continue
    return values


def _classify(returncode: int, stdout_results: list[dict], stderr_tail: str) -> str:
    """The generator only sees stream_wav.py's exit code and output -- a
    503 from admission control surfaces as a WebSocket handshake failure
    (an exception during connect, before any turn result is ever printed),
    not a distinct exit code, so it's identified from the error text."""
    if returncode == 0:
        return "completed"
    if not stdout_results and ("503" in stderr_tail or "server busy" in stderr_tail
                                or "at session capacity" in stderr_tail):
        return "rejected"
    return "failed"


async def _run_attempt(attempt_id: str, wav_dir: str, edge_url: str, timeout_s: float,
                       out_path: Path, script_dir: Path) -> dict:
    started_at = time.time()
    proc = await asyncio.create_subprocess_exec(
        sys.executable, str(script_dir / "stream_wav.py"),
        "--wav-dir", wav_dir, "--edge-url", edge_url, "--timeout-s", str(timeout_s),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s + 15)
        returncode = proc.returncode
    except asyncio.TimeoutError:
        proc.kill()
        stdout, stderr = await proc.communicate()
        returncode = None  # the generator itself gave up -- distinct from the subprocess failing cleanly

    results = []
    for line in stdout.decode(errors="replace").splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    stderr_tail = stderr.decode(errors="replace")[-500:]
    outcome = "generator_timeout" if returncode is None else _classify(returncode, results, stderr_tail)

    record = {
        "attempt_id": attempt_id,
        "started_at": started_at,
        "finished_at": time.time(),
        "outcome": outcome,
        "returncode": returncode,
        "turns": results,
        "session_id": results[0].get("session_id") if results else None,
        "edge_error": results[-1].get("edge_error") if results else None,
        "stderr_tail": stderr_tail,
    }
    with out_path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    return record


async def run(args: argparse.Namespace) -> tuple[list[dict], dict]:
    script_dir = Path(__file__).parent
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.append:
        out_path.unlink()

    metrics_before = _fetch_metrics(args.edge_metrics_url)

    attempt_ids = [str(uuid.uuid4()) for _ in range(args.concurrency)]
    print(f"launching {args.concurrency} concurrent attempts against {args.edge_url} "
          f"(run_id={args.run_id})", flush=True)
    t0 = time.time()
    records = await asyncio.gather(*(
        _run_attempt(aid, args.wav_dir, args.edge_url, args.timeout_s, out_path, script_dir)
        for aid in attempt_ids
    ))
    elapsed = time.time() - t0

    # Let Prometheus scrape the tail of this burst's activity before reading
    # the edge's own counters, or the delta would undercount the last events.
    await asyncio.sleep(args.scrape_settle_s)
    metrics_after = _fetch_metrics(args.edge_metrics_url)

    def delta(name: str) -> float | None:
        if name not in metrics_before or name not in metrics_after:
            return None  # unreachable/missing -- unknown, never a fabricated 0
        return metrics_after[name] - metrics_before[name]

    outcomes: dict[str, int] = {}
    for r in records:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1

    reconciliation = {
        "run_id": args.run_id,
        "elapsed_s": round(elapsed, 2),
        "generator_scheduled": args.concurrency,
        "generator_outcomes": outcomes,
        "edge_connection_attempts_total_delta": delta("edge_connection_attempts_total"),
        "edge_connection_rejected_total_delta": delta("edge_connection_rejected_total"),
        "edge_ws_upgrade_errors_total_delta": delta("edge_ws_upgrade_errors_total"),
        "edge_sessions_total_delta": delta("edge_sessions_total"),
    }
    with out_path.open("a") as f:
        f.write(json.dumps({"reconciliation": reconciliation}) + "\n")

    print(json.dumps(reconciliation, indent=2), flush=True)
    attempts_delta = reconciliation["edge_connection_attempts_total_delta"]
    if attempts_delta is None:
        print("WARNING: could not reach the edge metrics endpoint -- offered load is UNKNOWN, "
              "not reconciled (do not read this as zero attempts).", file=sys.stderr)
    elif attempts_delta < args.concurrency:
        print(f"WARNING: the edge observed only {attempts_delta:.0f} attempts but the generator "
              f"scheduled {args.concurrency} -- some attempts never reached the server "
              "(connection refused, DNS, or TLS failure before the HTTP request landed).",
              file=sys.stderr)
    return records, reconciliation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--concurrency", type=int, required=True,
                        help="number of concurrent connection attempts to launch at once")
    parser.add_argument("--wav-dir", default=str(Path(__file__).with_name("recordings")),
                        help="passed through to each stream_wav.py attempt")
    parser.add_argument("--edge-url", default="https://127.0.0.1:8443")
    parser.add_argument("--edge-metrics-url", default="http://127.0.0.1:9101/metrics",
                        help="the edge's Prometheus /metrics endpoint (EDGE_METRICS_ADDR)")
    parser.add_argument("--timeout-s", type=float, default=90,
                        help="passed through as each attempt's own --timeout-s")
    parser.add_argument("--scrape-settle-s", type=float, default=6.0,
                        help="wait this long after the burst before reading edge metrics, "
                             "so the scrape interval doesn't undercount the tail")
    parser.add_argument("--run-id", default=None, help="defaults to a fresh UUID")
    parser.add_argument("--out", default=str(Path(__file__).with_name("recordings") / "load_attempts.jsonl"),
                        help="durable per-attempt JSONL record")
    parser.add_argument("--append", action="store_true",
                        help="append to --out instead of truncating it (keep history across runs)")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    if not args.run_id:
        args.run_id = str(uuid.uuid4())
    return args


def main() -> int:
    args = parse_args()
    records, reconciliation = asyncio.run(run(args))
    outcomes = reconciliation["generator_outcomes"]
    unexpected_failures = outcomes.get("failed", 0) + outcomes.get("generator_timeout", 0)
    return 1 if unexpected_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
