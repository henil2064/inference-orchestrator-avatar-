#!/usr/bin/env python3
"""Phase 3 pipeline launcher: bare-process, hardware-auto-sized, multi-replica.

Runs all 5 pipeline services (media-edge, orchestrator, STT, TTS,
adaptive-llm) as bare host processes -- never Docker; Docker stays reserved
for Redis/ClickHouse/Grafana/Prometheus (docker compose up -d redis
clickhouse prometheus grafana, run separately, before this).

STT/TTS replica counts and per-replica thread counts are computed from
detected hardware (scripts/orchestration/plan.py) unless overridden by env
(STT_REPLICAS, TTS_REPLICAS, STT_THREADS_PER_REPLICA, TTS_THREADS_PER_REPLICA,
RESERVED_CONTROL_CORES, LLM_CPU_RESERVE, STT_TTS_CORE_SPLIT). Replica 0 always
gets today's pre-Phase-3 default ports, so STT_REPLICAS=1 TTS_REPLICAS=1 (or
no overrides on a small box) is unaffected.

Usage:
  python3 scripts/run_pipeline.py                # auto-sized, run everything
  python3 scripts/run_pipeline.py --dry-run       # print the plan, launch nothing
  STT_REPLICAS=2 TTS_REPLICAS=2 python3 scripts/run_pipeline.py

AVATAR_ENABLED=1 additionally launches the avatar service (after a one-time
avatar/setup.sh on this box) and points media-edge at it.

Each replica's stdout/stderr goes to run/logs/<name>.log; Prometheus file_sd
targets for STT/TTS are (re)written to monitoring/targets/{stt,tts}.json on
every launch (see monitoring/prometheus.yml's file_sd_configs jobs).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from orchestration.hardware import detect_cpu, detect_gpus, detect_mem  # noqa: E402
from orchestration.plan import compute_replica_plan, stt_ports, tts_ports  # noqa: E402
from orchestration.supervisor import ProcessSupervisor, ReplicaSpec  # noqa: E402
from orchestration.targets import write_file_sd  # noqa: E402


def _has_grpc_health() -> bool:
    try:
        import grpc  # noqa: F401
        from grpc_health.v1 import health_pb2, health_pb2_grpc  # noqa: F401
    except ImportError:
        return False
    return True


def _ensure_grpc_health_runtime() -> None:
    """Run the supervisor under a Python env that can poll gRPC health."""
    if _has_grpc_health():
        return
    fallback = REPO_ROOT / "orchestrator" / ".venv" / "bin" / "python"
    if Path(sys.executable).resolve() == fallback.resolve():
        raise SystemExit(
            "current Python cannot import grpc/grpc_health, so readiness checks cannot run. "
            "Install orchestrator requirements first."
        )
    if not fallback.exists():
        raise SystemExit(
            "current Python cannot import grpc/grpc_health and orchestrator/.venv/bin/python is missing. "
            "Create the orchestrator venv first, or run the launcher with a Python that has grpcio-health-checking."
        )
    print(f"  re-execing launcher with {fallback} for gRPC health readiness checks", flush=True)
    os.execv(str(fallback), [str(fallback), *sys.argv])


def _port_from_addr(addr: str, default: int) -> int:
    """Pull the port out of a "[::]:PORT" / "host:PORT" style address."""
    m = re.search(r":(\d+)$", addr)
    return int(m.group(1)) if m else default


def _localhost_url_from_addr(addr: str, default_port: int, path: str) -> str:
    return f"http://127.0.0.1:{_port_from_addr(addr, default_port)}{path}"


def _venv_python(service_dir: Path, venv_name: str) -> Path:
    path = service_dir / venv_name / "bin" / "python"
    if not path.exists():
        raise SystemExit(f"missing venv interpreter: {path} (set it up per {service_dir.name}/README first)")
    return path


def build_specs(plan, args: argparse.Namespace) -> list[ReplicaSpec]:
    base_env = dict(os.environ)
    logs_dir = REPO_ROOT / "run" / "logs"
    specs: list[ReplicaSpec] = []

    # --- adaptive-llm (singleton: one vLLM instance on the GPU) ---
    llm_addr = base_env.get("LLM_ADDR", "[::]:50053")
    llm_port = _port_from_addr(llm_addr, 50053)
    llm_env = dict(base_env)
    specs.append(ReplicaSpec(
        name="adaptive-llm", cwd=REPO_ROOT, python=_venv_python(REPO_ROOT / "adaptive_llm", ".venv"),
        args=["-m", "adaptive_llm.main"], env=llm_env,
        log_path=logs_dir / "adaptive-llm.log", health_addr=f"127.0.0.1:{llm_port}",
    ))

    # --- STT replicas ---
    stt_dir = REPO_ROOT / "stt"
    stt_python = _venv_python(stt_dir, ".venv-sravaani")
    stt_addresses = []
    for i in range(plan.stt_replicas):
        app_port, metrics_port = stt_ports(i)
        stt_addresses.append(f"127.0.0.1:{app_port}")
        env = dict(base_env)
        env["STT_ADDR"] = f"[::]:{app_port}"
        env["STT_METRICS_PORT"] = str(metrics_port)
        env["STT_CPU_THREADS"] = str(plan.stt_threads)
        specs.append(ReplicaSpec(
            name=f"stt-{i}", cwd=stt_dir, python=stt_python, args=["server.py"], env=env,
            log_path=logs_dir / f"stt-{i}.log", health_addr=f"127.0.0.1:{app_port}",
        ))

    # --- TTS replicas ---
    tts_dir = REPO_ROOT / "tts"
    tts_python = _venv_python(tts_dir, ".venv")
    tts_addresses = []
    for i in range(plan.tts_replicas):
        app_port, metrics_port = tts_ports(i)
        tts_addresses.append(f"127.0.0.1:{app_port}")
        env = dict(base_env)
        env["TTS_ADDR"] = f"[::]:{app_port}"
        env["TTS_METRICS_PORT"] = str(metrics_port)
        env["TTS_NUM_THREADS"] = str(plan.tts_threads)
        specs.append(ReplicaSpec(
            name=f"tts-{i}", cwd=tts_dir, python=tts_python, args=["server.py"], env=env,
            log_path=logs_dir / f"tts-{i}.log", health_addr=f"127.0.0.1:{app_port}",
        ))

    # --- orchestrator (singleton; fans out to all STT/TTS replicas) ---
    orch_dir = REPO_ROOT / "orchestrator"
    orch_env = dict(base_env)
    orch_env["STT_TARGETS"] = ",".join(stt_addresses)
    orch_env["TTS_TARGETS"] = ",".join(tts_addresses)
    orch_env["LLM_TARGET"] = f"127.0.0.1:{llm_port}"
    orch_addr = base_env.get("ORCH_ADDR", "[::]:50051")
    orch_port = _port_from_addr(orch_addr, 50051)
    specs.append(ReplicaSpec(
        name="orchestrator", cwd=orch_dir, python=_venv_python(orch_dir, ".venv"),
        args=["server.py"], env=orch_env,
        log_path=logs_dir / "orchestrator.log", health_addr=f"127.0.0.1:{orch_port}",
    ))

    # --- avatar (optional singleton; shares the GPU with the LLM) ---
    # Opt-in: it needs its own one-time setup (avatar/setup.sh) and an
    # Ampere-or-newer GPU, neither of which a default launch may assume.
    avatar_target = None
    if base_env.get("AVATAR_ENABLED", "0") == "1":
        avatar_dir = REPO_ROOT / "avatar"
        avatar_port = _port_from_addr(base_env.get("AVATAR_ADDR", "[::]:50056"), 50056)
        avatar_target = f"127.0.0.1:{avatar_port}"
        # run.sh activates upstream AVTR-1's pixi environment; invoked through
        # bash so it works whether or not the checkout kept its exec bit.
        specs.append(ReplicaSpec(
            name="avatar", cwd=avatar_dir, python=Path(shutil.which("bash") or "/bin/bash"),
            args=[str(avatar_dir / "run.sh")], env=dict(base_env),
            log_path=logs_dir / "avatar.log", health_addr=avatar_target,
        ))

    # --- media-edge (singleton; Go) ---
    if not args.no_edge:
        binary = _build_media_edge_binary()
        edge_env = dict(base_env)
        if avatar_target:
            edge_env["EDGE_AVATAR_TARGET"] = avatar_target
        edge_metrics_addr = base_env.get("EDGE_METRICS_ADDR", ":9101")
        specs.append(ReplicaSpec(
            name="media-edge", cwd=REPO_ROOT / "media-edge", python=binary,
            args=["-orchestrator", f"localhost:{orch_port}"],
            env=edge_env, log_path=logs_dir / "media-edge.log", health_addr=None,
            http_health_url=_localhost_url_from_addr(edge_metrics_addr, 9101, "/health"),
        ))

    return specs


def _build_media_edge_binary() -> Path:
    """A real compiled binary, not `go run`, for the supervised process: `go
    run` spawns the actual media-edge binary as a *child* process rather than
    exec-replacing itself, and that child does not reliably receive a signal
    sent to the `go run` process's group -- observed directly, killing the
    supervisor left a real media-edge process orphaned and holding :8443
    across a restart, immune to both SIGTERM and a follow-up SIGKILL sent to
    the go wrapper's group. One real process, launched directly, has no such
    parent/child signal-forwarding gap."""
    go = shutil.which("go")
    if go is None:
        raise SystemExit("`go` not found on PATH; pass --no-edge to skip media-edge")
    out = REPO_ROOT / "run" / "bin" / "media-edge"
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"  building media-edge -> {out} ...", flush=True)
    result = subprocess.run(
        [go, "build", "-o", str(out), "./cmd/media-edge"],
        cwd=str(REPO_ROOT / "media-edge"), capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"media-edge build failed:\n{result.stdout}\n{result.stderr}")
    return out


def main() -> int:
    # Line-buffer stdout even when redirected to a file/pipe (the common case
    # for a long-running supervisor): otherwise every print() below sits in a
    # block buffer and an operator tailing the log sees nothing for minutes.
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print the computed plan and exit")
    parser.add_argument("--no-edge", action="store_true", help="skip media-edge (STT/TTS/LLM/orchestrator only)")
    parser.add_argument("--only", default=None,
                        help="comma-separated name prefixes to actually launch (e.g. 'stt,tts'); "
                             "everything else is still planned/printed but not started -- for "
                             "iterating on a subset without a full pipeline bring-up")
    parser.add_argument("--ready-timeout", type=float, default=float(os.environ.get("PIPELINE_READY_TIMEOUT_S", "180")),
                        help="seconds to wait for all launched replicas to report ready before warning "
                             "(default: PIPELINE_READY_TIMEOUT_S or 180)")
    args = parser.parse_args()
    if not args.dry_run:
        _ensure_grpc_health_runtime()

    cpu = detect_cpu()
    mem = detect_mem()
    gpus = detect_gpus()
    plan = compute_replica_plan(cpu, gpus, os.environ)

    print("=== Welcome To Worlds's Best Inference Platform ===")
    print("""
    █████ █   █ ████  ███ ████  
      █   █   █ █   █  █  █   █ 
      █   █ █ █ ████   █  ████  
      █   ██ ██ █   █  █  █     
      █   █   █ ████  ███ █     
    """)
    for line in plan.to_lines():
        print(f"  {line}")
    print(f"  total RAM: {mem.total_mib} MiB ({mem.available_mib} MiB available)")
    print(f"  GPUs: {[g.name for g in gpus] or 'none detected'}")
    print("=============================")

    if args.dry_run:
        return 0

    specs = build_specs(plan, args)
    if args.only:
        prefixes = tuple(p.strip() for p in args.only.split(",") if p.strip())
        specs = [s for s in specs if s.name.startswith(prefixes)]
        if not specs:
            raise SystemExit(f"--only={args.only!r} matched no replicas")
    for spec in specs:
        print(f"  {spec.name}: {spec.python} {' '.join(spec.args)} (cwd={spec.cwd})"
              f"{f' health={spec.health_addr}' if spec.health_addr else ''}")

    stt_metrics_ports = [stt_ports(i)[1] for i in range(plan.stt_replicas)]
    tts_metrics_ports = [tts_ports(i)[1] for i in range(plan.tts_replicas)]
    write_file_sd(REPO_ROOT / "monitoring" / "targets" / "stt.json", stt_metrics_ports, job="stt")
    write_file_sd(REPO_ROOT / "monitoring" / "targets" / "tts.json", tts_metrics_ports, job="tts")
    print(f"  wrote Prometheus file_sd targets: monitoring/targets/{{stt,tts}}.json")

    def _log_event(msg: str) -> None:
        # ProcessSupervisor defaults to logging.info(), which prints nothing
        # unless a handler is configured -- every start/restart/health-kill
        # event was silently swallowed until this was wired up. Plain print()
        # keeps this script dependency-free and matches its own style above.
        print(f"[supervisor] {msg}", flush=True)

    supervisor = ProcessSupervisor(specs, on_event=_log_event)

    async def _run_with_ready_banner() -> None:
        run_task = asyncio.create_task(supervisor.run())
        ready_task = asyncio.create_task(supervisor.wait_until_ready(timeout_s=args.ready_timeout))
        try:
            done, _ = await asyncio.wait({run_task, ready_task}, return_when=asyncio.FIRST_COMPLETED)
            if ready_task in done:
                try:
                    ready = ready_task.result()
                    print(f"\n=== PIPELINE READY: {', '.join(ready)} ===", flush=True)
                except TimeoutError as exc:
                    print(f"\n  WARNING: {exc}. Supervisor is still running; check run/logs/ for slow or failed startup.",
                          flush=True)
                except asyncio.CancelledError:
                    pass
        finally:
            ready_task.cancel()
            await run_task

    asyncio.run(_run_with_ready_banner())

    given_up = [r.spec.name for r in supervisor.replicas if r.given_up]
    if given_up:
        print(f"\n  NOTE: {len(given_up)} replica(s) gave up during this run and were not "
              f"restarted after the run began: {', '.join(given_up)}. Check their logs in "
              f"run/logs/ before the next launch.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
