"""Self-check for ProcessSupervisor: crash-restart and health-watchdog kill.
Needs grpc + grpcio-health-checking -- run with a venv that has them, e.g.:
  orchestrator/.venv/bin/python scripts/orchestration/test_supervisor.py
"""
import asyncio
import os
import sys
import tempfile
import threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from orchestration.supervisor import ProcessSupervisor, ReplicaSpec

PYTHON = Path(sys.executable)


def test_crashing_replica_gets_restarted():
    async def scenario():
        with tempfile.TemporaryDirectory() as tmp:
            events = []
            spec = ReplicaSpec(
                name="flaky", cwd=Path(tmp), python=PYTHON,
                args=["-c", "import sys; sys.exit(1)"],
                env=dict(os.environ), log_path=Path(tmp) / "flaky.log",
            )
            sup = ProcessSupervisor([spec], restart_backoff_base_s=0.05,
                                    restart_backoff_max_s=0.1, on_event=events.append)
            run_task = asyncio.create_task(sup.run())
            for _ in range(200):
                if sup.replicas[0].restart_count >= 3:
                    break
                await asyncio.sleep(0.05)
            assert sup.replicas[0].restart_count >= 3, \
                f"expected several restarts, got {sup.replicas[0].restart_count}; events={events}"
            sup.request_stop()
            await asyncio.wait_for(run_task, timeout=5)

    asyncio.run(scenario())


def test_crash_looping_replica_gives_up_after_max_consecutive_failures():
    """A fundamentally broken replica (bad config, missing dependency --
    simulated here by a script that always exits immediately) must not be
    retried forever. After max_consecutive_failures back-to-back unsuccessful
    runs, the supervisor stops restarting it and marks it given_up, but
    keeps running (never raises out of .run())."""
    async def scenario():
        with tempfile.TemporaryDirectory() as tmp:
            events = []
            spec = ReplicaSpec(
                name="broken", cwd=Path(tmp), python=PYTHON,
                args=["-c", "import sys; sys.exit(1)"],
                env=dict(os.environ), log_path=Path(tmp) / "broken.log",
            )
            sup = ProcessSupervisor([spec], restart_backoff_base_s=0.02,
                                    restart_backoff_max_s=0.02, max_consecutive_failures=3,
                                    on_event=events.append)
            run_task = asyncio.create_task(sup.run())
            for _ in range(500):
                if sup.replicas[0].given_up:
                    break
                await asyncio.sleep(0.02)
            assert sup.replicas[0].given_up, f"expected given_up after 3 failures; events={events}"
            assert sup.replicas[0].restart_count == 3, \
                f"expected exactly 3 attempts before giving up, got {sup.replicas[0].restart_count}"
            assert any("GIVING UP" in e for e in events), f"expected a loud give-up event; events={events}"
            restart_count_at_giveup = sup.replicas[0].restart_count
            await asyncio.sleep(0.2)  # long enough for several more attempts if it were still retrying
            assert sup.replicas[0].restart_count == restart_count_at_giveup, \
                "a given-up replica must not be restarted again"
            sup.request_stop()
            await asyncio.wait_for(run_task, timeout=5)  # .run() must still return normally

    asyncio.run(scenario())


def test_one_replica_giving_up_does_not_affect_another():
    """The whole point: one permanently broken service must not take the
    rest of the pipeline down with it."""
    async def scenario():
        with tempfile.TemporaryDirectory() as tmp:
            broken = ReplicaSpec(
                name="broken", cwd=Path(tmp), python=PYTHON,
                args=["-c", "import sys; sys.exit(1)"],
                env=dict(os.environ), log_path=Path(tmp) / "broken.log",
            )
            healthy = ReplicaSpec(
                name="healthy", cwd=Path(tmp), python=PYTHON,
                args=["-c", "import time; time.sleep(30)"],
                env=dict(os.environ), log_path=Path(tmp) / "healthy.log",
            )
            sup = ProcessSupervisor([broken, healthy], restart_backoff_base_s=0.02,
                                    restart_backoff_max_s=0.02, max_consecutive_failures=2,
                                    on_event=lambda m: None)
            run_task = asyncio.create_task(sup.run())
            broken_r = next(r for r in sup.replicas if r.spec.name == "broken")
            healthy_r = next(r for r in sup.replicas if r.spec.name == "healthy")
            for _ in range(500):
                if broken_r.given_up:
                    break
                await asyncio.sleep(0.02)
            assert broken_r.given_up
            assert healthy_r.process is not None and healthy_r.process.returncode is None, \
                "the healthy replica must keep running after the broken one gives up"
            assert healthy_r.restart_count == 0
            sup.request_stop()
            await asyncio.wait_for(run_task, timeout=5)

    asyncio.run(scenario())


def test_healthy_replica_is_left_alone():
    """A long-running (never-exiting) process with no health_addr configured
    must never be restarted -- the crash-restart path only fires on exit."""
    async def scenario():
        with tempfile.TemporaryDirectory() as tmp:
            spec = ReplicaSpec(
                name="steady", cwd=Path(tmp), python=PYTHON,
                args=["-c", "import time; time.sleep(30)"],
                env=dict(os.environ), log_path=Path(tmp) / "steady.log",
            )
            sup = ProcessSupervisor([spec])
            run_task = asyncio.create_task(sup.run())
            await asyncio.sleep(0.3)
            assert sup.replicas[0].restart_count == 0
            assert sup.replicas[0].process is not None and sup.replicas[0].process.returncode is None
            sup.request_stop()
            await asyncio.wait_for(run_task, timeout=5)

    asyncio.run(scenario())


def test_http_health_readiness_is_supported():
    class Handler(BaseHTTPRequestHandler):
        ready = False

        def do_GET(self):
            self.send_response(200 if self.ready else 503)
            self.end_headers()

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    async def scenario():
        with tempfile.TemporaryDirectory() as tmp:
            events = []
            spec = ReplicaSpec(
                name="http-ready", cwd=Path(tmp), python=PYTHON,
                args=["-c", "import time; time.sleep(30)"],
                env=dict(os.environ), log_path=Path(tmp) / "http-ready.log",
                http_health_url=f"http://127.0.0.1:{server.server_port}/health",
            )
            sup = ProcessSupervisor([spec], on_event=events.append)
            run_task = asyncio.create_task(sup.run())
            try:
                await asyncio.sleep(0.2)
                Handler.ready = True
                ready = await asyncio.wait_for(sup.wait_until_ready(timeout_s=2), timeout=3)
                assert ready == ["http-ready"]
                assert any("HTTP health OK" in e for e in events)
            finally:
                sup.request_stop()
                await asyncio.wait_for(run_task, timeout=5)

    try:
        asyncio.run(scenario())
    finally:
        server.shutdown()
        server.server_close()


def test_wedged_replica_is_killed_and_restarted_via_health_check():
    """A process that never exits on its own but whose gRPC health reports
    sustained NOT_SERVING must get killed (letting the crash-restart path
    bring it back) -- this is the Phase 1/3 "supervisor restart" guardrail."""
    import grpc
    from grpc_health.v1 import health, health_pb2, health_pb2_grpc

    async def scenario():
        health_servicer = health.aio.HealthServicer()
        # Starts SERVING: this test is specifically "was healthy, then
        # wedged" -- a service that has never yet become healthy is a
        # different case (startup grace), covered by the next test.
        await health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
        grpc_server = grpc.aio.server()
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, grpc_server)
        port = grpc_server.add_insecure_port("127.0.0.1:0")
        await grpc_server.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                spec = ReplicaSpec(
                    name="wedged", cwd=Path(tmp), python=PYTHON,
                    args=["-c", "import time; time.sleep(30)"],
                    env=dict(os.environ), log_path=Path(tmp) / "wedged.log",
                    health_addr=f"127.0.0.1:{port}",
                )
                events = []
                sup = ProcessSupervisor([spec], restart_backoff_base_s=0.05,
                                        health_poll_interval_s=0.02, wedge_restart_after_s=0.1,
                                        startup_grace_s=5.0, on_event=events.append)
                run_task = asyncio.create_task(sup.run())
                # Let it register as healthy at least once before wedging it.
                await asyncio.sleep(0.2)
                await health_servicer.set("", health_pb2.HealthCheckResponse.NOT_SERVING)
                for _ in range(300):
                    if sup.replicas[0].restart_count >= 1:
                        break
                    await asyncio.sleep(0.02)
                assert sup.replicas[0].restart_count >= 1, f"expected a restart from the health watchdog; events={events}"
                assert any("wedged" in e for e in events), f"expected the fast wedge path, not startup grace; events={events}"
                sup.request_stop()
                await asyncio.wait_for(run_task, timeout=5)
        finally:
            await grpc_server.stop(grace=0)

    asyncio.run(scenario())


def test_never_healthy_gets_startup_grace_not_a_fast_kill():
    """A service that hasn't responded healthy YET (cold model load) must
    not be killed on the tight wedge_restart_after_s budget -- that budget
    is for a service that WAS healthy and stopped responding. This is the
    real fix: vLLM/TTS cold starts measured 15-30s+ on the dev box, longer
    than the previous single wedge_restart_after_s default (15s) -- the old
    code would have killed a still-loading service and likely repeated
    forever if cold start stayed near/above that threshold."""
    async def scenario():
        with tempfile.TemporaryDirectory() as tmp:
            # Nothing listens on this port at all -- simulates "still
            # starting, port not open yet" for the whole test.
            spec = ReplicaSpec(
                name="slow-starter", cwd=Path(tmp), python=PYTHON,
                args=["-c", "import time; time.sleep(30)"],
                env=dict(os.environ), log_path=Path(tmp) / "slow.log",
                health_addr="127.0.0.1:1",  # nothing listens here
            )
            events = []
            sup = ProcessSupervisor([spec], health_poll_interval_s=0.02,
                                    wedge_restart_after_s=0.1, startup_grace_s=5.0,
                                    on_event=events.append)
            run_task = asyncio.create_task(sup.run())
            try:
                # Comfortably longer than wedge_restart_after_s (0.1s) but
                # well inside startup_grace_s (5.0s): must NOT restart yet.
                await asyncio.sleep(1.0)
                assert sup.replicas[0].restart_count == 0, \
                    f"a never-yet-healthy service must get startup_grace_s, not wedge_restart_after_s; events={events}"
            finally:
                sup.request_stop()
                await asyncio.wait_for(run_task, timeout=5)

    asyncio.run(scenario())
