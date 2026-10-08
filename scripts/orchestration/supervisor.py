"""Bare-process supervisor: start/restart/drain replicas, and kill a replica
whose gRPC health reports sustained NOT_SERVING (a wedged worker -- see
docs/LOAD_SCALING_PLAN.md Phase 1 item 5 and Phase 3 item 2/3).

Deliberately not using Docker: the 5 pipeline services run as bare host
processes per the user's requirement; this is the process-management layer
Docker's own restart policy would otherwise provide.
"""
from __future__ import annotations

import asyncio
import dataclasses
import http.client
import logging
import os
import signal
import time
from pathlib import Path
from typing import Callable

log = logging.getLogger("orchestration.supervisor")


@dataclasses.dataclass
class ReplicaSpec:
    name: str
    cwd: Path
    python: Path
    args: list[str]
    env: dict[str, str]
    log_path: Path
    # "host:port" of a gRPC health endpoint to poll, or None to skip health
    # watchdog restarts for this replica (still gets crash-restart either way).
    health_addr: str | None = None
    # HTTP health URL to poll for readiness-only checks. This does not drive
    # watchdog restarts; those remain tied to gRPC health for model services.
    http_health_url: str | None = None


class ManagedReplica:
    def __init__(self, spec: ReplicaSpec):
        self.spec = spec
        self.process: asyncio.subprocess.Process | None = None
        self.restart_count = 0
        # Consecutive unsuccessful runs (crashed or wedge-killed without
        # first staying up past stable_uptime_s) -- resets to 0 the moment
        # a run IS stable. Distinct from restart_count, which never resets
        # and is purely informational. See given_up below.
        self.consecutive_failures = 0
        self.last_start_monotonic: float | None = None
        self.stopping = False
        # Set once consecutive_failures reaches max_consecutive_failures:
        # the supervisor stops retrying this one replica (crash-looping
        # forever on a fundamentally broken config helps no one), but every
        # other replica keeps running normally -- one broken service doesn't
        # take the whole pipeline down. Check this after the fact (or watch
        # for the "giving up" event) to know a replica needs manual attention.
        self.given_up = False


def _signal_group(process: asyncio.subprocess.Process, sig: signal.Signals) -> None:
    """Signal the whole process group, not just the immediate child. `go run`
    (media-edge's dev invocation) spawns a compiled-binary child that does not
    reliably receive a signal sent only to the `go` process -- observed
    directly: killing just the parent left the child binary running and
    holding the listen port, orphaned, across a supervisor restart."""
    try:
        os.killpg(os.getpgid(process.pid), sig)
    except ProcessLookupError:
        pass


def _http_health_ok(url: str, timeout_s: float) -> bool:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    conn_cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path += "?" + parsed.query
    conn = conn_cls(parsed.hostname, port, timeout=timeout_s)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        resp.read()
        return 200 <= resp.status < 300
    except Exception:
        return False
    finally:
        conn.close()


async def _terminate(process: asyncio.subprocess.Process, grace_s: float = 5.0) -> None:
    if process.returncode is not None:
        return
    _signal_group(process, signal.SIGTERM)
    try:
        await asyncio.wait_for(process.wait(), timeout=grace_s)
    except asyncio.TimeoutError:
        _signal_group(process, signal.SIGKILL)
        await process.wait()


async def _start(spec: ReplicaSpec) -> asyncio.subprocess.Process:
    spec.log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(spec.log_path, "ab", buffering=0)
    return await asyncio.create_subprocess_exec(
        str(spec.python), *spec.args, cwd=str(spec.cwd), env=spec.env,
        stdout=log_file, stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,  # own process group, so _signal_group can reach any child it spawns
    )


class ProcessSupervisor:
    def __init__(self, specs: list[ReplicaSpec], *,
                 restart_backoff_base_s: float = 1.0, restart_backoff_max_s: float = 30.0,
                 health_poll_interval_s: float = 2.0, wedge_restart_after_s: float = 15.0,
                 startup_grace_s: float = 180.0, stable_uptime_s: float = 60.0,
                 max_consecutive_failures: int = 8,
                 on_event: Callable[[str], None] | None = None):
        self.specs = specs
        self.restart_backoff_base_s = restart_backoff_base_s
        self.restart_backoff_max_s = restart_backoff_max_s
        self.health_poll_interval_s = health_poll_interval_s
        # A run that stays up at least this long resets both the backoff
        # delay and the consecutive-failure count -- "it was fine for a
        # while, then something new went wrong" doesn't count toward
        # crash-loop detection the way a rapid-fire loop does.
        self.stable_uptime_s = stable_uptime_s
        # Crash-looping forever on a fundamentally broken config (bad
        # model path, missing dependency, port permanently taken by
        # something else) helps no one -- after this many *consecutive*
        # unsuccessful runs, stop retrying that one replica and say so
        # loudly, but keep supervising every other replica normally. With
        # the default backoff schedule this is roughly 2 minutes of
        # retrying (1+2+4+8+16+30+30+30s) before giving up -- generous
        # enough not to abandon a transient issue, bounded enough not to
        # spin forever.
        self.max_consecutive_failures = max_consecutive_failures
        # Two different questions, two different budgets. "Never yet become
        # healthy" is cold start (model load: vLLM alone measured ~20-30s on
        # this box, worse on a cold HF/page cache) -- wedge_restart_after_s
        # (tuned for catching a truly wedged worker fast) is too short for
        # that and would kill a service still legitimately loading, restart
        # it, and very possibly repeat forever if cold-start time is
        # consistently close to or over the threshold. "Was healthy, then
        # stopped responding" is the real wedge case and keeps the tight
        # bound.
        self.wedge_restart_after_s = wedge_restart_after_s
        self.startup_grace_s = startup_grace_s
        self._on_event = on_event or (lambda msg: log.info(msg))
        self.replicas = [ManagedReplica(spec) for spec in specs]
        self._stop_event = asyncio.Event()

    def request_stop(self) -> None:
        self._stop_event.set()

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self.request_stop)
            except NotImplementedError:
                pass  # non-POSIX platforms; not a target here
        tasks = [asyncio.create_task(self._supervise_one(r), name=f"supervise-{r.spec.name}")
                 for r in self.replicas]
        tasks.append(asyncio.create_task(self._wait_and_drain(), name="drain-on-stop"))
        try:
            await asyncio.gather(*tasks)
        finally:
            for sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.remove_signal_handler(sig)
                except (NotImplementedError, ValueError):
                    pass

    async def wait_until_ready(self, timeout_s: float = 180.0) -> list[str]:
        """Wait until every managed replica is usable.

        For replicas with a gRPC health address, usable means the health
        service reports SERVING. For replicas with an HTTP health URL, usable
        means that URL returns 2xx. For replicas without a health check, usable
        only means the process has started and is still running.

        Returns the names that became ready. Raises TimeoutError if the whole
        set does not become ready before timeout_s.
        """
        import grpc
        from grpc_health.v1 import health_pb2, health_pb2_grpc

        deadline = time.monotonic() + timeout_s
        ready: set[str] = set()
        while time.monotonic() < deadline and not self._stop_event.is_set():
            for r in self.replicas:
                if r.spec.name in ready:
                    continue
                process = r.process
                if process is None:
                    continue
                if process.returncode is not None or r.given_up:
                    continue
                if r.spec.health_addr is not None:
                    try:
                        async with grpc.aio.insecure_channel(r.spec.health_addr) as channel:
                            stub = health_pb2_grpc.HealthStub(channel)
                            resp = await asyncio.wait_for(
                                stub.Check(health_pb2.HealthCheckRequest(service="")),
                                timeout=1.0,
                            )
                        if resp.status == health_pb2.HealthCheckResponse.SERVING:
                            ready.add(r.spec.name)
                            self._on_event(f"{r.spec.name}: ready (gRPC health SERVING)")
                    except Exception:
                        pass
                    continue
                if r.spec.http_health_url is not None:
                    if await asyncio.to_thread(_http_health_ok, r.spec.http_health_url, 1.0):
                        ready.add(r.spec.name)
                        self._on_event(f"{r.spec.name}: ready (HTTP health OK)")
                    continue
                ready.add(r.spec.name)
                self._on_event(f"{r.spec.name}: ready (process started; no health endpoint)")
            if len(ready) == len(self.replicas):
                return [r.spec.name for r in self.replicas if r.spec.name in ready]
            await asyncio.sleep(min(1.0, self.health_poll_interval_s))
        pending = [r.spec.name for r in self.replicas if r.spec.name not in ready]
        if self._stop_event.is_set():
            raise asyncio.CancelledError
        raise TimeoutError(
            f"timed out after {timeout_s:.0f}s waiting for replica readiness: {', '.join(pending)}"
        )

    async def _wait_and_drain(self) -> None:
        await self._stop_event.wait()
        self._on_event("stop requested; draining all replicas")
        for r in self.replicas:
            r.stopping = True
        await asyncio.gather(*(self._terminate_replica(r) for r in self.replicas),
                             return_exceptions=True)

    async def _terminate_replica(self, r: ManagedReplica) -> None:
        if r.process is not None:
            await _terminate(r.process)

    async def _supervise_one(self, r: ManagedReplica) -> None:
        backoff = self.restart_backoff_base_s
        watchdog_task: asyncio.Task | None = None
        while not r.stopping and not self._stop_event.is_set():
            r.process = await _start(r.spec)
            r.last_start_monotonic = time.monotonic()
            self._on_event(f"{r.spec.name}: started pid={r.process.pid} "
                           f"(attempt {r.restart_count + 1})")
            if r.spec.health_addr:
                watchdog_task = asyncio.create_task(self._health_watchdog(r))
            returncode = await r.process.wait()
            if watchdog_task is not None:
                watchdog_task.cancel()
                watchdog_task = None
            if r.stopping or self._stop_event.is_set():
                self._on_event(f"{r.spec.name}: exited (code={returncode}) during shutdown")
                return
            uptime = time.monotonic() - (r.last_start_monotonic or time.monotonic())
            stable = uptime > self.stable_uptime_s
            r.restart_count += 1
            r.consecutive_failures = 0 if stable else r.consecutive_failures + 1
            if r.consecutive_failures >= self.max_consecutive_failures:
                r.given_up = True
                self._on_event(
                    f"{r.spec.name}: GIVING UP after {r.consecutive_failures} consecutive "
                    f"unsuccessful runs (last exit code={returncode}, uptime={uptime:.1f}s each "
                    f"time) -- not restarting again. Other replicas are unaffected. Check "
                    f"{r.spec.log_path} for why it keeps failing, fix it, then restart the "
                    f"whole supervisor.")
                return
            self._on_event(f"{r.spec.name}: exited unexpectedly (code={returncode}, "
                           f"uptime={uptime:.1f}s); restarting in {backoff:.1f}s "
                           f"({r.consecutive_failures}/{self.max_consecutive_failures} consecutive failures)")
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=backoff)
                return  # stop requested during the backoff wait
            except asyncio.TimeoutError:
                pass
            # A replica that stayed up a while resets the backoff; one that
            # keeps crash-looping immediately backs off further, capped.
            backoff = self.restart_backoff_base_s if stable else min(backoff * 2, self.restart_backoff_max_s)

    async def _health_watchdog(self, r: ManagedReplica) -> None:
        """Poll gRPC health on r.spec.health_addr; kill the process (letting
        _supervise_one's own loop restart it) once NOT_SERVING or unreachable
        has been sustained past the applicable budget. Import grpc lazily so
        hardware/plan can be unit-tested without a grpc install present.

        Before the first-ever SERVING response, "unhealthy" just means
        "still starting" (model load -- not a wedge), so that gets the long
        startup_grace_s budget instead of the tight wedge_restart_after_s
        one; only a wedge *after* having been healthy once uses the tight
        bound, restarting a truly stuck worker quickly."""
        import grpc
        from grpc_health.v1 import health_pb2, health_pb2_grpc

        unhealthy_since: float | None = None
        ever_healthy = False
        while True:
            await asyncio.sleep(self.health_poll_interval_s)
            if r.process is None or r.process.returncode is not None:
                return
            healthy = False
            try:
                async with grpc.aio.insecure_channel(r.spec.health_addr) as channel:
                    stub = health_pb2_grpc.HealthStub(channel)
                    resp = await asyncio.wait_for(
                        stub.Check(health_pb2.HealthCheckRequest(service="")),
                        timeout=min(3.0, self.health_poll_interval_s),
                    )
                    healthy = resp.status == health_pb2.HealthCheckResponse.SERVING
            except Exception:
                healthy = False  # unreachable counts the same as NOT_SERVING
            now = time.monotonic()
            if healthy:
                unhealthy_since = None
                ever_healthy = True
                continue
            if unhealthy_since is None:
                unhealthy_since = now
                continue
            budget = self.wedge_restart_after_s if ever_healthy else self.startup_grace_s
            if now - unhealthy_since > budget:
                phase = "wedged" if ever_healthy else "never became healthy after starting"
                self._on_event(f"{r.spec.name}: {phase} for >{budget:.0f}s; killing for restart")
                if r.process is not None and r.process.returncode is None:
                    _signal_group(r.process, signal.SIGKILL)
                return
