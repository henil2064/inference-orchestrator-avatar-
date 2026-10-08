"""Focused checks for facade admission control without loading an engine."""

from __future__ import annotations

import asyncio

from adaptive_llm.main import AdaptiveLlmServicer, _worker_watchdog
from adaptive_llm.planning.planner import ExecutionPlan
from adaptive_llm.proto import llm_pb2
from grpc_health.v1 import health_pb2


class AdmissionAborted(Exception):
    pass


class FakeContext:
    async def abort(self, _code, _details):
        raise AdmissionAborted()


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        model_id="test",
        engine="llama.cpp",
        execution_mode="CPU_ONLY",
        score=0,
        context_tokens=128,
        max_output_tokens=16,
        n_gpu_layers=0,
        cpu_threads=1,
        gpu_index=None,
        runtime_profile=None,
        engine_env={},
        reasons=[],
        rejected={},
        estimate=None,
    )


async def _run() -> None:
    servicer = AdaptiveLlmServicer(
        _plan(), "127.0.0.1:1", dry_run=True, max_inflight_requests=1, max_queued_requests=0
    )
    request = llm_pb2.GenerateRequest(session_id="test", text="hello")
    try:
        response = await servicer.Generate(request, FakeContext())
        assert response.text.startswith("[adaptive-llm dry-run]")
        assert (servicer._active, servicer._queued, servicer._admitted) == (0, 0, 0)

        # Hold the one execution slot so the first request occupies the complete
        # admission budget while waiting; the second request must be rejected.
        await servicer._slots.acquire()
        waiting = asyncio.create_task(servicer.Generate(request, FakeContext()))
        await asyncio.sleep(0)
        try:
            await servicer.Generate(request, FakeContext())
        except AdmissionAborted:
            pass
        else:
            raise AssertionError("expected a full admission queue to reject the request")
        waiting.cancel()
        try:
            await waiting
        except asyncio.CancelledError:
            pass
        servicer._slots.release()
        assert (servicer._active, servicer._queued, servicer._admitted) == (0, 0, 0)
    finally:
        await servicer.channel.close()


async def _consume_stream(servicer, request):
    return [chunk async for chunk in servicer.GenerateStream(request, FakeContext())]


async def _run_stream_admission() -> None:
    servicer = AdaptiveLlmServicer(
        _plan(), "127.0.0.1:1", dry_run=True, max_inflight_requests=1, max_queued_requests=0
    )
    request = llm_pb2.GenerateRequest(session_id="stream", text="hello")
    try:
        await servicer._slots.acquire()
        waiting = asyncio.create_task(_consume_stream(servicer, request))
        await asyncio.sleep(0)
        stream = servicer.GenerateStream(request, FakeContext())
        try:
            await anext(stream)
        except AdmissionAborted:
            pass
        else:
            raise AssertionError("expected a full streaming admission queue to reject the request")
        waiting.cancel()
        try:
            await waiting
        except asyncio.CancelledError:
            pass
        servicer._slots.release()
        assert (servicer._active, servicer._queued, servicer._admitted) == (0, 0, 0)
    finally:
        await servicer.channel.close()


async def _run_oldest_queued_age() -> None:
    servicer = AdaptiveLlmServicer(
        _plan(), "127.0.0.1:1", dry_run=True, max_inflight_requests=1, max_queued_requests=1
    )
    request = llm_pb2.GenerateRequest(session_id="age", text="hello")
    try:
        assert servicer._oldest_queued_age() == 0.0
        await servicer._slots.acquire()  # occupy the one execution slot
        waiting = asyncio.create_task(servicer.Generate(request, FakeContext()))
        await asyncio.sleep(0.05)  # let it admit and start waiting for a slot
        age = servicer._oldest_queued_age()
        assert age > 0, f"expected a positive queued age while a request waits, got {age}"
        servicer._slots.release()
        await waiting
        assert servicer._oldest_queued_age() == 0.0, "queued-age tracking leaked after the request was served"
    finally:
        await servicer.channel.close()


class _FakeProcess:
    def __init__(self):
        self.returncode = None


class _FakeWorker:
    def __init__(self, process):
        self.process = process


class _FakeHealthServicer:
    def __init__(self):
        self.statuses = []

    async def set(self, service, status):
        self.statuses.append(status)


async def _run_worker_watchdog() -> None:
    """Phase 3: the facade must notice its local worker subprocess dying and
    flip gRPC health to NOT_SERVING, since the facade's own gRPC server
    otherwise stays up and answers with per-request errors forever."""
    process = _FakeProcess()
    worker = _FakeWorker(process)
    health = _FakeHealthServicer()
    task = asyncio.create_task(_worker_watchdog(worker, servicer=None, health_servicer=health, interval_s=0.02))
    try:
        await asyncio.sleep(0.05)
        assert health.statuses == [], "must not report unhealthy while the worker process is alive"

        process.returncode = 1  # simulate the worker process exiting
        for _ in range(50):
            if health.statuses:
                break
            await asyncio.sleep(0.02)
        assert health.statuses and health.statuses[-1] == health_pb2.HealthCheckResponse.NOT_SERVING, \
            f"expected NOT_SERVING once the worker process died, got {health.statuses}"

        process.returncode = None  # simulate recovery (a real restart hands the watchdog a fresh WorkerHandle)
        for _ in range(50):
            if health.statuses[-1] == health_pb2.HealthCheckResponse.SERVING:
                break
            await asyncio.sleep(0.02)
        assert health.statuses[-1] == health_pb2.HealthCheckResponse.SERVING, \
            f"expected SERVING again after recovery, got {health.statuses}"
    finally:
        task.cancel()


def test_worker_watchdog_reports_unready_on_process_death() -> None:
    asyncio.run(_run_worker_watchdog())


def test_facade_admission_control() -> None:
    asyncio.run(_run())


def test_streaming_uses_the_same_admission_limit() -> None:
    asyncio.run(_run_stream_admission())


def test_oldest_queued_age_tracks_waiting_requests() -> None:
    asyncio.run(_run_oldest_queued_age())
