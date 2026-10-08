"""DhVaani gRPC TTS service.

Loads one TTS backend (bare PyTorch, or llama.cpp for a .gguf model) and one
configured reference voice at startup. Requests are serialized through a single
worker because inference is not thread-safe and this avoids multiple voice
generations exhausting device memory.
"""
import asyncio
import concurrent.futures
import logging
import os
import time
from pathlib import Path

import grpc
import numpy as np
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from proto import tts_pb2, tts_pb2_grpc
from service.backends import load_backend, pcm16_bytes


def _load_dotenv():
    path = Path(__file__).resolve().parents[1] / ".env"
    if not path.is_file(): return
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()
logging.basicConfig(level=getattr(logging, os.environ.get("LOG_LEVEL", "ERROR").upper(), logging.ERROR),
                    format="%(levelname)s %(name)s | %(message)s", force=True)
log = logging.getLogger("tts")
MODEL_DIR = Path(os.environ.get("TTS_MODEL_DIR", Path(__file__).resolve().parents[1] / "models/DhVaani-0.5-fp16")).resolve()
PROMPT_WAV = Path(os.environ.get("TTS_PROMPT_WAV", ""))
MAX_TEXT_CHARS = max(1, int(os.environ.get("TTS_MAX_TEXT_CHARS", "2000")))
MAX_QUEUE = max(0, int(os.environ.get("TTS_MAX_QUEUE", "4")))
QUEUE_TIMEOUT_S = max(0.1, float(os.environ.get("TTS_QUEUE_TIMEOUT_S", "15")))
WARMUP = os.environ.get("TTS_WARMUP", "1") != "0"
POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1)
M_CALLS = Counter("tts_syntheses_total", "TTS syntheses completed")
M_ERRORS = Counter("tts_errors_total", "TTS synthesis errors")
M_LATENCY = Histogram("tts_synthesis_seconds", "TTS request wall time", buckets=(.25, .5, 1, 2, 4, 8, 15, 30))
M_QUEUE_WAIT = Histogram("tts_queue_wait_seconds", "Time admitted TTS requests waited for the GPU worker",
                         buckets=(.01, .05, .1, .25, .5, 1, 2, 4, 8, 15))
M_POOL_WAIT = Histogram("tts_pool_wait_seconds",
                        "Time an admitted sentence waited for a free model worker thread",
                        buckets=(.001, .01, .05, .1, .25, .5, 1, 2, 4, 8, 15))
M_QUEUE_REJECTED = Counter("tts_queue_rejected_total", "TTS requests rejected because the queue was full or timed out")
M_TEXT_REJECTED = Counter("tts_text_rejected_total", "TTS requests rejected for exceeding the text limit")
# Explicit occupancy: how many of the 1+MAX_QUEUE reservation slots are held
# right now (admitted or waiting to be), vs the configured bound -- the pair
# a saturation dashboard actually needs, distinct from the reject counter.
M_OCCUPANCY = Gauge("tts_admission_occupancy", "Reservation slots currently held (admitted or waiting)")
M_CAPACITY = Gauge("tts_admission_capacity", "Configured reservation bound (1 + TTS_MAX_QUEUE)")
M_CAPACITY.set_function(lambda: float(1 + MAX_QUEUE))
# Admission (above) only bounds how many callers may be admitted into POOL's
# own internal queue; POOL itself has a single worker thread, so "admitted"
# is not "running" -- mirrors the identical STT ambiguity. This is set from
# inside the worker thread itself, at the moment a synthesis actually starts
# and ends, so it is never > 1 and never counts a merely-admitted caller.
M_POOL_EXECUTING = Gauge("tts_pool_executing", "Synthesis calls actually running on the worker thread right now")
# Event-loop scheduling drift (see the orchestrator's identical monitor for
# the full rationale): rising values mean the loop is falling behind its own
# scheduled work, independent of which coroutine is responsible.
M_LOOP_LAG = Gauge("tts_event_loop_lag_seconds",
                   "Event-loop scheduling drift; rising values mean the loop is falling behind")


async def _loop_lag_monitor(interval_s: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    target = loop.time() + interval_s
    while True:
        await asyncio.sleep(interval_s)
        now = loop.time()
        M_LOOP_LAG.set(max(0.0, now - target))
        target = now + interval_s


# Wedge detection (Phase 3, completes Phase 1's "a wedged worker must report
# itself unready"): synthesis isn't preemptible, so this is the only signal
# available before the process gets killed and restarted. Set/cleared from
# inside the POOL worker thread at the same points as M_POOL_EXECUTING
# inc/dec (see run() closures below) -- a single scalar assignment is safe to
# read from the event-loop thread under the GIL; POOL has one worker, so at
# most one synthesis is ever "current".
_current_synth_started_at: float | None = None
TTS_WEDGE_TIMEOUT_S = max(1.0, float(os.environ.get("TTS_WEDGE_TIMEOUT_S", "30")))


def _mark_synth_started() -> None:
    global _current_synth_started_at
    _current_synth_started_at = time.monotonic()


def _mark_synth_done() -> None:
    global _current_synth_started_at
    _current_synth_started_at = None


async def _wedge_watchdog(health_servicer: "health.aio.HealthServicer", interval_s: float = 1.0) -> None:
    reported_unhealthy = False
    while True:
        await asyncio.sleep(interval_s)
        started_at = _current_synth_started_at
        wedged = started_at is not None and (time.monotonic() - started_at) > TTS_WEDGE_TIMEOUT_S
        if wedged and not reported_unhealthy:
            reported_unhealthy = True
            log.error("Synthesis worker has run >%.1fs; reporting NOT_SERVING", TTS_WEDGE_TIMEOUT_S)
            await health_servicer.set("", health_pb2.HealthCheckResponse.NOT_SERVING)
        elif not wedged and reported_unhealthy:
            reported_unhealthy = False
            log.info("Synthesis worker recovered; reporting SERVING again")
            await health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)


def load_backend_and_warmup():
    backend = load_backend()
    if WARMUP:
        # Make readiness meaningful: exercise the complete render path (vocoder
        # included) before accepting user traffic.
        backend.warmup()
    return backend


class Tts(tts_pb2_grpc.TtsServicer):
    def __init__(self, backend):
        self.backend = backend
        # One worker plus a small, explicitly bounded waiting room.
        self.admission = asyncio.BoundedSemaphore(1 + MAX_QUEUE)
        # asyncio.Semaphore's own waiter list is unbounded: with only the
        # semaphore, any number of callers can pile up calling acquire() and
        # each ride out the full QUEUE_TIMEOUT_S before rejection, even though
        # only 1+MAX_QUEUE of them could ever be admitted. This counter is the
        # actual bounded gate -- checked synchronously (no await), so it is
        # atomic on the single-threaded event loop -- and rejects overflow
        # immediately instead of making it wait just to find out it lost.
        self._capacity = 1 + MAX_QUEUE
        self._reserved = 0

    def _try_reserve(self) -> bool:
        if self._reserved >= self._capacity:
            return False
        self._reserved += 1
        M_OCCUPANCY.set(self._reserved)
        return True

    def _release_reservation(self) -> None:
        self._reserved -= 1
        M_OCCUPANCY.set(self._reserved)

    def _release(self) -> None:
        """Release both the reservation and the admission slot exactly once,
        for work that was actually admitted."""
        self.admission.release()
        self._release_reservation()

    def _profile(self, *, queue_ms, compute_ms, audio_seconds, extra=None):
        b = self.backend
        return tts_pb2.InferenceProfile(
            model=b.model_name, engine=b.engine, device=b.device, compute_type=b.compute_type,
            queue_ms=queue_ms, compute_ms=compute_ms, audio_seconds=audio_seconds,
            realtime_factor=compute_ms / 1000 / audio_seconds if audio_seconds else 0,
            attributes={**b.profile_attributes, **(extra or {})})

    async def Synthesize(self, request, context):
        text = request.text.strip()
        if not request.session_id or not text:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "session_id and text are required")
        if len(text) > MAX_TEXT_CHARS:
            M_TEXT_REJECTED.inc()
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                                f"text exceeds TTS_MAX_TEXT_CHARS ({MAX_TEXT_CHARS})")
        if not self._try_reserve():
            M_QUEUE_REJECTED.inc()
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED,
                                "TTS queue is full; retry the latest turn")
            return
        queued = time.perf_counter()
        loop = asyncio.get_running_loop()
        admitted = False
        def run():
            M_POOL_EXECUTING.inc()
            _mark_synth_started()
            try:
                started = time.perf_counter()
                audio = self.backend.synthesize(text)
                return np.asarray(audio, dtype=np.float32), (time.perf_counter() - started) * 1000
            finally:
                _mark_synth_done()
                M_POOL_EXECUTING.dec()
        try:
            await asyncio.wait_for(self.admission.acquire(), timeout=QUEUE_TIMEOUT_S)
            admitted = True
        except asyncio.TimeoutError:
            M_QUEUE_REJECTED.inc()
            self._release_reservation()
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED,
                                "TTS queue is full; retry the latest turn")
            return
        queue_ms = (time.perf_counter() - queued) * 1000
        M_QUEUE_WAIT.observe(queue_ms / 1000)
        try:
            work = loop.run_in_executor(POOL, run)
            try:
                audio, compute_ms = await asyncio.shield(work)
            except asyncio.CancelledError:
                # GPU work cannot be safely preempted. Keep its admission slot
                # until the executor task really exits, rather than admitting
                # more work and silently exceeding the queue bound.
                work.add_done_callback(lambda _: self._release())
                admitted = False
                raise
        except Exception as exc:
            M_ERRORS.inc(); log.exception("synthesis failed")
            await context.abort(grpc.StatusCode.INTERNAL, f"TTS synthesis failed: {exc}")
        finally:
            if admitted:
                self._release()
        pcm16 = pcm16_bytes(audio)
        proc_ms = (time.perf_counter() - queued) * 1000
        rate = self.backend.sampling_rate
        M_CALLS.inc(); M_LATENCY.observe(proc_ms / 1000)
        return tts_pb2.SynthesizeResponse(session_id=request.session_id, pcm16=pcm16, sample_rate_hz=rate,
            ts_ms=int(time.time() * 1000), proc_ms=round(proc_ms),
            profile=self._profile(queue_ms=queue_ms, compute_ms=compute_ms,
                                  audio_seconds=len(audio) / rate))

    async def SynthesizeStream(self, request, context):
        """Synthesize sentence-by-sentence and stream each one's audio as it is ready,
        so the caller hears the first sentence without waiting for the whole reply."""
        text = request.text.strip()
        if not request.session_id or not text:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "session_id and text are required")
        if len(text) > MAX_TEXT_CHARS:
            M_TEXT_REJECTED.inc()
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                                f"text exceeds TTS_MAX_TEXT_CHARS ({MAX_TEXT_CHARS})")
        if not self._try_reserve():
            M_QUEUE_REJECTED.inc()
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "TTS queue is full; retry the latest turn")
            return
        queued = time.perf_counter()
        try:
            await asyncio.wait_for(self.admission.acquire(), timeout=QUEUE_TIMEOUT_S)
        except asyncio.TimeoutError:
            M_QUEUE_REJECTED.inc()
            self._release_reservation()
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "TTS queue is full; retry the latest turn")
            return
        queue_ms = (time.perf_counter() - queued) * 1000
        M_QUEUE_WAIT.observe(queue_ms / 1000)
        loop = asyncio.get_running_loop()
        rate = self.backend.sampling_rate
        frame_bytes = rate // 50 * 2  # 20 ms PCM16 transport frames
        slot_held = True
        chunk_index = 0
        try:
            for sentence in self.backend.split_sentences(text):
                # The clock must start INSIDE the worker thread, as the unary path
                # already does. Timing from submission instead folds the wait for a
                # free pool thread into compute_ms, so under concurrency every
                # queued request reports an inflated realtime_factor — which is
                # exactly the metric capacity planning reads. Pool wait is real
                # latency, but it is queue time, not model time.
                timing: dict[str, float] = {}

                def run(s=sentence, t=timing):
                    M_POOL_EXECUTING.inc()
                    _mark_synth_started()
                    try:
                        t["started"] = time.perf_counter()
                        return self.backend.synthesize(s)
                    finally:
                        _mark_synth_done()
                        M_POOL_EXECUTING.dec()

                submitted = time.perf_counter()
                work = loop.run_in_executor(POOL, run)
                try:
                    audio = await asyncio.shield(work)
                except asyncio.CancelledError:
                    # Barge-in: GPU work can't be preempted, so hold the admission
                    # slot until the executor task really exits rather than admitting
                    # more work and silently exceeding the queue bound.
                    work.add_done_callback(lambda _: self._release())
                    slot_held = False
                    raise
                # `started` is always set here: the worker ran to completion.
                pool_wait_ms = (timing["started"] - submitted) * 1000
                compute_ms = (time.perf_counter() - timing["started"]) * 1000
                M_POOL_WAIT.observe(pool_wait_ms / 1000)
                pcm16 = pcm16_bytes(audio)
                audio_seconds = len(audio) / rate
                # One profile per sentence (on its first frame): per-sentence
                # compute_ms/RTF. queue_ms is every kind of waiting this sentence
                # did — admission (first sentence only) plus pool wait — so
                # compute_ms + queue_ms still accounts for the whole request.
                profile = self._profile(
                    queue_ms=(queue_ms if chunk_index == 0 else 0) + pool_wait_ms,
                    compute_ms=compute_ms, audio_seconds=audio_seconds,
                    extra={"streaming": "per_sentence",
                           "pool_wait_ms": f"{pool_wait_ms:.1f}"})
                for offset in range(0, len(pcm16), frame_bytes):
                    yield tts_pb2.SynthesizeChunk(session_id=request.session_id,
                        pcm16=pcm16[offset:offset + frame_bytes], sample_rate_hz=rate,
                        chunk_index=chunk_index, ts_ms=int(time.time() * 1000),
                        profile=profile if offset == 0 else tts_pb2.InferenceProfile())
                    chunk_index += 1
            yield tts_pb2.SynthesizeChunk(session_id=request.session_id, sample_rate_hz=rate,
                chunk_index=chunk_index, is_last=True, ts_ms=int(time.time() * 1000))
            M_CALLS.inc(); M_LATENCY.observe(time.perf_counter() - queued)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            M_ERRORS.inc(); log.exception("streaming synthesis failed")
            await context.abort(grpc.StatusCode.INTERNAL, f"TTS synthesis failed: {exc}")
        finally:
            if slot_held:
                self._release()


async def serve():
    backend = load_backend_and_warmup()
    server = grpc.aio.server()
    tts_pb2_grpc.add_TtsServicer_to_server(Tts(backend), server)
    health_servicer = health.aio.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    await health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    addr = os.environ.get("TTS_ADDR", "[::]:50054")
    server.add_insecure_port(addr); await server.start()
    port = int(os.environ.get("TTS_METRICS_PORT", "9104"))
    metrics_host = os.environ.get("TTS_METRICS_HOST", "0.0.0.0")
    start_http_server(port, addr=metrics_host)
    asyncio.create_task(_loop_lag_monitor())
    asyncio.create_task(_wedge_watchdog(health_servicer))
    print(f"TTS SERVICE | READY | gRPC {addr} | engine {backend.engine} | model {backend.model_name} "
          f"| voice {PROMPT_WAV.name}", flush=True)
    await server.wait_for_termination()


if __name__ == "__main__":
    try: asyncio.run(serve())
    except KeyboardInterrupt: pass
