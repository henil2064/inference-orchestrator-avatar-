#!/usr/bin/env python3
"""Open-loop load generator: independently scheduled arrivals over the real
WebRTC path, as opposed to load_burst.py's "launch N at once" closed loop.

Per docs/LOAD_SCALING_PLAN.md Phase 3 item 5 ("Load & recovery harness"): a
reproducible driver with independently scheduled (open-loop) arrivals, a
workload seed, and durable outcomes. Reuses load_burst.py's per-attempt
subprocess launcher (each attempt is its own stream_wav.py process, so
admission rejection and connection failures are observed exactly as a real
client would see them) and its edge-metrics reconciliation -- this only
replaces *when* attempts are dispatched.

There is no separate generator machine here (single dev box) -- this script
measures and reports its own scheduling lag (intended vs. actual dispatch
time per attempt) explicitly so generator-side saturation can never be
mistaken for server capacity. A growing lag over the run means the generator
itself is falling behind, not that the server is.

Usage:
  python3 open_loop_load.py --rate 1.0 --duration-s 60 --seed 42
  python3 open_loop_load.py --arrivals-file schedule.json   # reuse an exact
                                                              # prior schedule
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from load_burst import _classify, _fetch_metrics, _run_attempt  # noqa: E402


def poisson_schedule(rate_per_s: float, duration_s: float, seed: int) -> list[float]:
    """Intended dispatch offsets (seconds from t0) for a Poisson arrival
    process at the given mean rate, reproducible from --seed."""
    rng = random.Random(seed)
    offsets: list[float] = []
    t = 0.0
    while t < duration_s:
        t += rng.expovariate(rate_per_s)
        if t < duration_s:
            offsets.append(t)
    return offsets


def jittered_fixed_schedule(rate_per_s: float, duration_s: float, jitter_frac: float, seed: int) -> list[float]:
    """Fixed inter-arrival spacing (1/rate) with uniform +/-jitter_frac jitter
    -- a gentler, less bursty alternative to Poisson for the same mean rate."""
    rng = random.Random(seed)
    interval = 1.0 / rate_per_s
    offsets: list[float] = []
    t = 0.0
    while t < duration_s:
        jitter = interval * jitter_frac * (2 * rng.random() - 1)
        t += max(0.001, interval + jitter)
        if t < duration_s:
            offsets.append(t)
    return offsets


async def _dispatch_at(offset_s: float, t0: float, attempt_id: str, args: argparse.Namespace,
                       out_path: Path, script_dir: Path, lag_records: list[dict]) -> dict:
    target = t0 + offset_s
    now = time.monotonic()
    if target > now:
        await asyncio.sleep(target - now)
    actual = time.monotonic()
    lag_records.append({"attempt_id": attempt_id, "intended_offset_s": offset_s,
                        "lag_ms": (actual - target) * 1000})
    return await _run_attempt(attempt_id, args.wav_dir, args.edge_url, args.timeout_s, out_path, script_dir)


async def run(args: argparse.Namespace) -> tuple[list[dict], dict]:
    script_dir = Path(__file__).parent
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not args.append:
        out_path.unlink()

    if args.arrivals_file:
        offsets = json.loads(Path(args.arrivals_file).read_text())
    elif args.schedule == "poisson":
        offsets = poisson_schedule(args.rate, args.duration_s, args.seed)
    else:
        offsets = jittered_fixed_schedule(args.rate, args.duration_s, args.jitter_frac, args.seed)

    schedule_path = out_path.with_name(out_path.stem + "_schedule.json")
    schedule_path.write_text(json.dumps(offsets, indent=2))
    print(f"scheduled {len(offsets)} arrivals over {args.duration_s:.0f}s "
          f"(seed={args.seed}, schedule={args.schedule}, saved to {schedule_path.name})", flush=True)

    metrics_before = _fetch_metrics(args.edge_metrics_url)
    t0 = time.monotonic()
    lag_records: list[dict] = []
    attempt_ids = [str(uuid.uuid4()) for _ in offsets]
    tasks = [
        asyncio.create_task(_dispatch_at(offset, t0, aid, args, out_path, script_dir, lag_records))
        for offset, aid in zip(offsets, attempt_ids)
    ]
    records = await asyncio.gather(*tasks)
    elapsed = time.monotonic() - t0

    await asyncio.sleep(args.scrape_settle_s)
    metrics_after = _fetch_metrics(args.edge_metrics_url)

    def delta(name: str) -> float | None:
        if name not in metrics_before or name not in metrics_after:
            return None
        return metrics_after[name] - metrics_before[name]

    outcomes: dict[str, int] = {}
    for r in records:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1

    lags = [r["lag_ms"] for r in lag_records]
    lags_sorted = sorted(lags)
    generator_lag = {
        "mean_ms": round(sum(lags) / len(lags), 1) if lags else 0.0,
        "p95_ms": round(lags_sorted[int(0.95 * (len(lags_sorted) - 1))], 1) if lags_sorted else 0.0,
        "max_ms": round(max(lags), 1) if lags else 0.0,
    }

    reconciliation = {
        "run_id": args.run_id,
        "schedule": args.schedule,
        "seed": args.seed,
        "rate_per_s": args.rate,
        "duration_s": args.duration_s,
        "elapsed_s": round(elapsed, 2),
        "generator_scheduled": len(offsets),
        "generator_outcomes": outcomes,
        "generator_scheduling_lag": generator_lag,
        "edge_connection_attempts_total_delta": delta("edge_connection_attempts_total"),
        "edge_connection_rejected_total_delta": delta("edge_connection_rejected_total"),
        "edge_ws_upgrade_errors_total_delta": delta("edge_ws_upgrade_errors_total"),
        "edge_sessions_total_delta": delta("edge_sessions_total"),
    }
    with out_path.open("a") as f:
        f.write(json.dumps({"reconciliation": reconciliation}) + "\n")

    print(json.dumps(reconciliation, indent=2), flush=True)
    if generator_lag["p95_ms"] > 500:
        print(f"WARNING: generator scheduling lag p95={generator_lag['p95_ms']}ms -- this single dev box's "
              "own CPU/asyncio scheduling is falling behind its intended arrival schedule. Treat throughput "
              "results as a diagnostic, not a certified SLO, per the capacity protocol (no separate generator "
              "machine here).", file=sys.stderr)
    attempts_delta = reconciliation["edge_connection_attempts_total_delta"]
    if attempts_delta is None:
        print("WARNING: could not reach the edge metrics endpoint -- offered load is UNKNOWN.", file=sys.stderr)
    return records, reconciliation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rate", type=float, default=1.0, help="mean arrivals per second")
    parser.add_argument("--duration-s", type=float, default=60.0, help="schedule arrivals over this window")
    parser.add_argument("--schedule", choices=["poisson", "jittered"], default="poisson")
    parser.add_argument("--jitter-frac", type=float, default=0.3, help="jittered schedule only: +/- fraction of the mean interval")
    parser.add_argument("--seed", type=int, default=0, help="reproducible arrival schedule")
    parser.add_argument("--arrivals-file", default=None, help="JSON list of offset-seconds, overrides --rate/--duration-s/--schedule/--seed")
    parser.add_argument("--wav-dir", default=str(Path(__file__).with_name("recordings")))
    parser.add_argument("--edge-url", default="https://127.0.0.1:8443")
    parser.add_argument("--edge-metrics-url", default="http://127.0.0.1:9101/metrics")
    parser.add_argument("--timeout-s", type=float, default=90)
    parser.add_argument("--scrape-settle-s", type=float, default=6.0)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--out", default=str(Path(__file__).with_name("recordings") / "open_loop_attempts.jsonl"))
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()
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
