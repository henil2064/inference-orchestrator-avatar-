"""Adaptive LLM facade using the existing llm.v1 gRPC contract."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import logging
import os
import signal
import time

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server

from .core.config import ROOT, load_config
from .core.logging_utils import configure_logging, exchange_event, profile_event, startup_banner
from .core.runtime_manager import start_worker, stop_worker, wait_for_worker_ready
from .planning.hardware import DynamicSnapshot, collect_dynamic_snapshot, collect_snapshot
from .planning.model_store import resolve_huggingface_model
from .planning.planner import ExecutionPlan, plan as build_plan, write_plan
from .planning.registry import load_registry, resolve_model
from .proto import llm_pb2, llm_pb2_grpc


def _load_dotenv() -> None:
    """Load KEY=VALUE lines from adaptive_llm/.env. Real env wins (CLI/shell
    exports override the file) -- matches stt/tts/orchestrator's own loaders.

    Bug fix (Phase 3): no such loader existed anywhere in this package before
    now. Docker deployments never noticed because Compose's own `env_file:`
    loads adaptive_llm/.env before the container's Python ever starts; a bare
    `python -m adaptive_llm.main` (the only way to run this service per the
    project's "no Docker for the 5 pipeline services" requirement) silently
    ignored the .env file entirely and fell back to hardcoded defaults
    (LLM_MODEL_ID=medgemma_1_5_4b_q4 via llama.cpp) regardless of
    LLM_MODEL_REF/LLM_ENGINE -- with no error, just the wrong model/engine."""
    path = ROOT / "adaptive_llm" / ".env"
    if not path.is_file():
        return
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()  # must run before configure_logging() and any other os.environ read below
configure_logging()
log = logging.getLogger("adaptive-llm")

_LAT_BUCKETS = (.1, .25, .5, .75, 1, 2, 3, 5, 8, 12, 20, 30, 60)
M_PROC = Histogram("adaptive_llm_proxy_seconds", "Facade dispatch time per Generate", buckets=_LAT_BUCKETS)
M_REPLIES = Counter("adaptive_llm_replies_total", "Replies proxied by facade")
M_ERRORS = Counter("adaptive_llm_errors_total", "Facade errors")
M_REJECTED = Counter("adaptive_llm_requests_rejected_total", "Requests rejected by admission control")
M_PLAN_SCORE = Gauge("adaptive_llm_plan_score", "Selected engine score")
M_PLAN_INFO = Info("adaptive_llm_plan", "Selected LLM execution plan")
M_QUEUE = Histogram("adaptive_llm_queue_seconds", "Time admitted requests wait for a worker slot", buckets=_LAT_BUCKETS)
M_ACTIVE = Gauge("adaptive_llm_active_requests", "Requests currently executing through the facade")
M_QUEUED = Gauge("adaptive_llm_queued_requests", "Admitted requests waiting for a worker slot")
M_ADMISSION_CAPACITY = Gauge("adaptive_llm_admission_capacity",
                             "Configured admission bound (max_inflight_requests + max_queued_requests)")
M_QUEUE_OLDEST = Gauge("adaptive_llm_queue_oldest_age_seconds",
                       "Age of the oldest request still waiting for a worker slot")
# Event-loop scheduling drift (see the orchestrator's identical monitor for
# the full rationale): rising values mean the loop is falling behind its own
# scheduled work, independent of which coroutine is responsible.
M_LOOP_LAG = Gauge("adaptive_llm_event_loop_lag_seconds",
                   "Event-loop scheduling drift; rising values mean the loop is falling behind")


async def _loop_lag_monitor(interval_s: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    target = loop.time() + interval_s
    while True:
        await asyncio.sleep(interval_s)
        now = loop.time()
        M_LOOP_LAG.set(max(0.0, now - target))
        target = now + interval_s


async def _worker_watchdog(worker, servicer, health_servicer, interval_s: float = 2.0) -> None:
    """Phase 3: "real backend readiness that detects worker death behind a
    live facade" -- the facade's own gRPC server can stay up and answer RPCs
    with errors indefinitely after the local vLLM/llama.cpp worker subprocess
    dies; nothing upstream of _serve() otherwise notices. Prefers an exact
    process-exit check (auto-started worker); falls back to channel
    connectivity for an externally managed worker (auto_start_worker=0)."""
    reported_unhealthy = False
    while True:
        await asyncio.sleep(interval_s)
        dead = worker.process is not None and worker.process.returncode is not None
        if not dead and worker.process is None:
            try:
                state = servicer.channel.get_state(try_to_connect=True)
                dead = state == grpc.ChannelConnectivity.TRANSIENT_FAILURE
            except Exception:
                dead = False
        if dead and not reported_unhealthy:
            reported_unhealthy = True
            log.error("Local LLM worker appears dead; reporting NOT_SERVING")
            await health_servicer.set("", health_pb2.HealthCheckResponse.NOT_SERVING)
        elif not dead and reported_unhealthy:
            reported_unhealthy = False
            log.info("Local LLM worker reachable again; reporting SERVING")
            await health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)


M_INPUT_CHARS = Histogram(
    "adaptive_llm_input_chars", "Input request size in characters", buckets=(64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
)
M_AVAILABLE_RAM = Gauge("adaptive_llm_available_ram_mib", "Live available host RAM in MiB")
M_CPU_UTIL = Gauge("adaptive_llm_cpu_utilization_pct", "Live host CPU utilization percentage")
M_GPU_FREE = Gauge("adaptive_llm_gpu_free_vram_mib", "Live free GPU VRAM in MiB", ["gpu_index"])
M_GPU_USED = Gauge("adaptive_llm_gpu_used_vram_mib", "Live used GPU VRAM in MiB", ["gpu_index"])
M_GPU_UTIL = Gauge("adaptive_llm_gpu_utilization_pct", "Live GPU utilization percentage", ["gpu_index"])
M_CPU_BANDWIDTH = Gauge("adaptive_llm_cpu_memory_bandwidth_gbps", "Profiled host memory copy bandwidth in GB/s")
M_PCIE_BANDWIDTH = Gauge(
    "adaptive_llm_pcie_h2d_bandwidth_gbps", "Profiled host-to-device PCIe bandwidth in GB/s", ["gpu_index"]
)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _mib(value: int | float) -> str:
    """Format memory consistently without hiding useful sub-GiB values."""
    value = int(value)
    return f"{value / 1024:.1f} GiB" if value >= 1024 else f"{value} MiB"


def _memory_summary_lines(plan_value: ExecutionPlan, model, hardware, live_snapshot, config) -> list[str]:
    """Separate observed system use from planner estimates in the readiness UI."""
    estimate = plan_value.estimate or {}
    weight = int(estimate.get("weight_mib", 0))
    kv_cache = int(estimate.get("kv_cache_mib", 0))
    runtime = int(estimate.get("runtime_mib", 0))
    total = int(estimate.get("total_mib", weight + kv_cache + runtime))
    confidence = str(estimate.get("confidence", "unknown"))
    lines = [
        f"Model     {model.display_name} ({model.format})",
        f"Runtime   {plan_value.engine} / {plan_value.execution_mode}",
        f"Model est* weights {_mib(weight)} | KV {_mib(kv_cache)} | runtime {_mib(runtime)} = {_mib(total)} ({confidence})",
    ]

    host_used = max(0, hardware.static.total_ram_mib - live_snapshot.available_ram_mib)
    host_reserve = max(config.min_headroom_mib, int(hardware.static.total_ram_mib * config.ram_headroom_fraction))
    lines.append(
        f"Host RAM  used {_mib(host_used)} | free {_mib(live_snapshot.available_ram_mib)} | reserve {_mib(host_reserve)}"
    )

    if plan_value.gpu_index is None:
        threads = f" | CPU threads {plan_value.cpu_threads}" if plan_value.cpu_threads else ""
        lines.append("Placement CPU / host RAM" + threads)
        return lines

    gpu_static = next((gpu for gpu in hardware.static.gpus if gpu.index == plan_value.gpu_index), None)
    gpu_live = next((gpu for gpu in live_snapshot.gpus if gpu.index == plan_value.gpu_index), None)
    if gpu_live is None:
        lines.append(f"GPU {plan_value.gpu_index}   selected; live VRAM data unavailable")
        return lines
    total_vram = gpu_static.total_vram_mib if gpu_static else gpu_live.used_vram_mib + gpu_live.free_vram_mib
    reserve = max(config.min_headroom_mib, int(total_vram * config.vram_headroom_fraction))
    layers = f" | layers {plan_value.n_gpu_layers}" if plan_value.n_gpu_layers is not None else ""
    lines.append(
        f"GPU {plan_value.gpu_index}    used {_mib(gpu_live.used_vram_mib)} | free {_mib(gpu_live.free_vram_mib)} | reserve {_mib(reserve)}{layers}"
    )
    return lines


class AdaptiveLlmServicer(llm_pb2_grpc.LlmServicer):
    def __init__(
        self,
        plan_value: ExecutionPlan,
        worker_addr: str,
        dry_run: bool,
        max_inflight_requests: int,
        max_queued_requests: int,
    ):
        self.plan = plan_value
        self.dry_run = dry_run
        self.channel = grpc.aio.insecure_channel(worker_addr)
        self.worker = llm_pb2_grpc.LlmStub(self.channel)
        self._slots = asyncio.Semaphore(max_inflight_requests)
        self._max_admitted = max_inflight_requests + max_queued_requests
        self._state_lock = asyncio.Lock()
        self._active = 0
        self._queued = 0
        self._admitted = 0
        # perf_counter() timestamps of requests currently waiting for a worker
        # slot (admitted but not yet acquired), for the oldest-age gauge.
        self._queued_since: list[float] = []

    def _discard_queued_since(self, t0: float) -> None:
        try:
            self._queued_since.remove(t0)
        except ValueError:
            pass

    def _oldest_queued_age(self) -> float:
        if not self._queued_since:
            return 0.0
        return max(0.0, time.perf_counter() - min(self._queued_since))

    @asynccontextmanager
    async def _admission(self, context):
        """Bound both unary and streaming work before it reaches the worker."""
        admitted = False
        acquired = False
        t0 = time.perf_counter()
        try:
            async with self._state_lock:
                if self._admitted >= self._max_admitted:
                    M_REJECTED.inc()
                    await context.abort(
                        grpc.StatusCode.RESOURCE_EXHAUSTED,
                        "adaptive LLM request queue is full; retry later",
                    )
                self._admitted += 1
                self._queued += 1
                self._queued_since.append(t0)
                admitted = True
                M_QUEUED.set(self._queued)

            await self._slots.acquire()
            acquired = True
            queue_s = time.perf_counter() - t0
            M_QUEUE.observe(queue_s)
            async with self._state_lock:
                self._queued -= 1
                self._discard_queued_since(t0)
                self._active += 1
                M_QUEUED.set(self._queued)
                M_ACTIVE.set(self._active)
            yield queue_s
        finally:
            if admitted:
                async with self._state_lock:
                    self._admitted -= 1
                    if acquired:
                        self._active -= 1
                        M_ACTIVE.set(self._active)
                    else:
                        self._queued -= 1
                        self._discard_queued_since(t0)
                        M_QUEUED.set(self._queued)
                if acquired:
                    self._slots.release()

    async def Generate(self, request, context):
        t0 = time.perf_counter()
        M_INPUT_CHARS.observe(len(request.text))
        async with self._admission(context) as queue_s:
            if self.dry_run:
                text = (
                    f"[adaptive-llm dry-run] selected {self.plan.engine} "
                    f"{self.plan.execution_mode} for {self.plan.model_id}"
                )
                reply = llm_pb2.GenerateResponse(
                    session_id=request.session_id,
                    text=text,
                    ts_ms=_now_ms(),
                    proc_ms=int((time.perf_counter() - t0) * 1000),
                )
            else:
                try:
                    reply = await self.worker.Generate(request)
                except Exception:
                    M_ERRORS.inc()
                    log.exception("worker Generate failed for %s", request.session_id)
                    raise
            M_PROC.observe(time.perf_counter() - t0)
            M_REPLIES.inc()
            total_s = time.perf_counter() - t0
            reply.profile.queue_ms = queue_s * 1000.0
            reply.profile.attributes["facade_total_ms"] = f"{total_s * 1000.0:.3f}"
            exchange_event(request.session_id, request.text, reply.text)
            profile_event(
                "adaptive-llm facade",
                request.session_id,
                f"queue={queue_s * 1000:.0f}ms | total={total_s * 1000:.0f}ms | input={len(request.text)} chars",
            )
            return reply

    async def GenerateStream(self, request, context):
        """Relay worker deltas with the same bounded admission as unary calls."""
        M_INPUT_CHARS.observe(len(request.text))
        async with self._admission(context):
            if self.dry_run:
                yield llm_pb2.GenerateChunk(session_id=request.session_id,
                    text_delta=f"[adaptive-llm dry-run] selected {self.plan.engine} ")
                yield llm_pb2.GenerateChunk(session_id=request.session_id, is_final=True,
                    ts_ms=_now_ms())
                return
            try:
                async for chunk in self.worker.GenerateStream(request):
                    yield chunk
            except grpc.aio.AioRpcError as exc:
                # Older worker engines retain unary compatibility while the
                # vLLM path provides true deltas.
                if exc.code() != grpc.StatusCode.UNIMPLEMENTED:
                    raise
                reply = await self.worker.Generate(request)
                yield llm_pb2.GenerateChunk(session_id=reply.session_id, text_delta=reply.text,
                    ts_ms=reply.ts_ms)
                yield llm_pb2.GenerateChunk(session_id=reply.session_id, is_final=True,
                    ts_ms=_now_ms(), profile=reply.profile)


def _record_live_resources(snapshot: DynamicSnapshot) -> None:
    M_AVAILABLE_RAM.set(snapshot.available_ram_mib)
    if snapshot.cpu_utilization_pct is not None:
        M_CPU_UTIL.set(snapshot.cpu_utilization_pct)
    for gpu in snapshot.gpus:
        labels = M_GPU_FREE.labels(gpu_index=str(gpu.index))
        labels.set(gpu.free_vram_mib)
        M_GPU_USED.labels(gpu_index=str(gpu.index)).set(gpu.used_vram_mib)
        if gpu.utilization_pct is not None:
            M_GPU_UTIL.labels(gpu_index=str(gpu.index)).set(gpu.utilization_pct)


async def _sample_live_resources(stop_event: asyncio.Event, interval_s: float) -> None:
    while not stop_event.is_set():
        try:
            _record_live_resources(await asyncio.to_thread(collect_dynamic_snapshot))
        except Exception:
            log.exception("live hardware metric collection failed")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
        except asyncio.TimeoutError:
            pass


async def serve():
    """Run the facade, turning any startup failure into one clear banner and a
    clean process exit instead of a raw traceback. _serve()'s own try/finally
    already stops a partially-started worker before an exception reaches here,
    so by the time this catches anything the service has nothing left running."""
    try:
        await _serve()
    except SystemExit as exc:
        startup_banner("ADAPTIVE LLM  |  STARTUP FAILED", [
            *str(exc.code if exc.code is not None else exc).split("\n"),
            "",
            "The adaptive LLM facade will not start.",
        ])
        raise
    except Exception as exc:
        startup_banner("ADAPTIVE LLM  |  STARTUP FAILED", [
            f"{exc.__class__.__name__}: {exc}",
            "",
            "The adaptive LLM facade will not start. See above for the worker's own output, if any.",
        ])
        raise SystemExit(1) from exc


async def _serve():
    config = load_config()
    registry = load_registry(config.model_profiles_path)
    # Collected before model resolution (not after, as before) so a Hugging
    # Face model ref can be size-checked against real VRAM/RAM and rejected
    # *before* downloading it -- see model_store.py's
    # _reject_if_remote_size_cannot_fit for why that previously downloaded
    # multi-GiB models the planner was always going to reject afterward.
    hardware = collect_snapshot(profile_bandwidth=config.profile_bandwidth)
    model = (
        resolve_huggingface_model(config, registry, hardware)
        if config.model_ref
        else resolve_model(registry, config.model_id)
    )
    plan_value = build_plan(model, hardware, config)
    write_plan(plan_value, ROOT / "adaptive_llm" / "run" / "current-plan.json")
    log.info("selected plan:\n%s", plan_value.to_json())

    M_PLAN_SCORE.set(plan_value.score)
    M_PLAN_INFO.info(
        {
            "model_id": plan_value.model_id,
            "engine": plan_value.engine,
            "execution_mode": plan_value.execution_mode,
        }
    )
    _record_live_resources(hardware.dynamic)
    if hardware.static.bandwidth.cpu_memory_gbps is not None:
        M_CPU_BANDWIDTH.set(hardware.static.bandwidth.cpu_memory_gbps)
    for gpu_index, bandwidth in hardware.static.bandwidth.pcie_h2d_gbps.items():
        M_PCIE_BANDWIDTH.labels(gpu_index=str(gpu_index)).set(bandwidth)
    metrics_host = os.environ.get("LLM_METRICS_HOST", "0.0.0.0")
    start_http_server(config.metrics_port, addr=metrics_host)
    log.info("metrics on http://%s:%d/metrics", metrics_host, config.metrics_port)

    worker = None
    server = None
    servicer = None
    server_started = False
    loop_lag_task = None
    worker_watchdog_task = None
    metrics_stop = asyncio.Event()
    sampler_task = asyncio.create_task(
        _sample_live_resources(metrics_stop, config.resource_metrics_interval_s)
    )
    try:
        worker = await start_worker(plan_value, model, config)
        await wait_for_worker_ready(worker, config)
        # The initial hardware snapshot is taken before model load for planning.
        # Re-sample after the worker is ready so the banner reports observed RAM/
        # VRAM occupancy, not stale pre-load values.
        live_snapshot = await asyncio.to_thread(collect_dynamic_snapshot)
        _record_live_resources(live_snapshot)
        server = grpc.aio.server()
        servicer = AdaptiveLlmServicer(
            plan_value,
            worker.address,
            config.dry_run,
            config.max_inflight_requests,
            config.max_queued_requests,
        )
        llm_pb2_grpc.add_LlmServicer_to_server(servicer, server)
        health_servicer = health.aio.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
        await health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
        if server.add_insecure_port(config.public_addr) == 0:
            raise RuntimeError(f"could not bind adaptive LLM gRPC address {config.public_addr}")
        await server.start()
        server_started = True
        M_ADMISSION_CAPACITY.set(servicer._max_admitted)
        M_QUEUE_OLDEST.set_function(servicer._oldest_queued_age)
        loop_lag_task = asyncio.create_task(_loop_lag_monitor())
        worker_watchdog_task = asyncio.create_task(_worker_watchdog(worker, servicer, health_servicer))
        startup_banner("ADAPTIVE LLM  |  READY", [
            f"gRPC      {config.public_addr}",
            *_memory_summary_lines(plan_value, model, hardware, live_snapshot, config),
            f"Worker    {worker.address}",
            f"Metrics   http://{metrics_host}:{config.metrics_port}/metrics",
            "* Planner estimate; RAM/VRAM lines are observed after model load",
        ])
        log.info("adaptive LLM gRPC listening on %s", config.public_addr)
        # Bug fix (Phase 3 graceful drain): asyncio installs no default SIGTERM
        # handler, so a supervisor's plain SIGTERM previously killed this
        # process via the OS default action -- skipping the finally block
        # below entirely -- and orphaned the local worker subprocess (observed
        # directly: llamacpp_worker.py/vllm_worker.py processes surviving,
        # reparented, still holding the worker port, after the facade that
        # spawned them was gone). Route both signals through server.stop() so
        # wait_for_termination() returns normally and finally always runs.
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(server.stop(grace=5)))
            except NotImplementedError:
                pass  # non-POSIX platforms; not a target here
        await server.wait_for_termination()
    finally:
        metrics_stop.set()
        await sampler_task
        if loop_lag_task is not None:
            loop_lag_task.cancel()
        if worker_watchdog_task is not None:
            worker_watchdog_task.cancel()
        if server_started:
            await server.stop(grace=0)
        if servicer is not None:
            await servicer.channel.close()
        if worker is not None:
            await stop_worker(worker)


if __name__ == "__main__":
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass
