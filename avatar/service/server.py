"""Avatar gRPC service.

Loads one renderer backend at startup and serves ``avatar.v1.Avatar/Stream``:
the media edge streams a session's agent and microphone audio in, this renders
a talking head from it and streams H.264 back. All GPU work runs on a single
worker thread, so concurrent sessions take turns one chunk at a time; that is
also what makes the render time per chunk the service's whole capacity story.
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

from proto import avatar_pb2, avatar_pb2_grpc
from service.chunker import SAMPLE_RATE, ChunkAssembler
from service.encoder import H264Encoder
from service.renderer import load_renderer


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
log = logging.getLogger("avatar")
OUT_WIDTH = max(16, int(os.environ.get("AVATAR_OUT_WIDTH", "1280")))
OUT_HEIGHT = max(16, int(os.environ.get("AVATAR_OUT_HEIGHT", "720")))
# One chunk takes ~84 ms of a 200 ms budget on an L40-class GPU, and sessions
# share the single GPU worker, so two is what fits with headroom. Measure
# avatar_chunk_render_seconds on the real box before raising it.
MAX_SESSIONS = max(1, int(os.environ.get("AVATAR_MAX_SESSIONS", "2")))
MAX_BACKLOG_CHUNKS = max(0, int(os.environ.get("AVATAR_MAX_BACKLOG_CHUNKS", "2")))
VIDEO_CODECS = [c.strip() for c in os.environ.get("AVATAR_VIDEO_CODECS", "h264_nvenc,libx264").split(",") if c.strip()]
VIDEO_BITRATE = max(100_000, int(os.environ.get("AVATAR_VIDEO_BITRATE", "1500000")))
GOP_SECONDS = max(0.2, float(os.environ.get("AVATAR_GOP_SECONDS", "2")))
# Frames the edge has not collected yet. Past this the edge is not keeping up
# and more queued video only arrives later still.
MAX_PENDING_FRAMES = max(5, int(os.environ.get("AVATAR_MAX_PENDING_FRAMES", "50")))
WARMUP = os.environ.get("AVATAR_WARMUP", "1") != "0"
POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1)
M_SESSIONS = Gauge("avatar_active_sessions", "Avatar sessions currently streaming")
M_CAPACITY = Gauge("avatar_session_capacity", "Configured session bound (AVATAR_MAX_SESSIONS)")
M_CAPACITY.set_function(lambda: float(MAX_SESSIONS))
M_REJECTED = Counter("avatar_sessions_rejected_total", "Sessions rejected because the service was at capacity")
M_CHUNKS = Counter("avatar_chunks_total", "Audio chunks rendered")
M_CHUNK_SECONDS = Histogram("avatar_chunk_render_seconds",
                            "Wall time to render and encode one chunk; real time needs this under the chunk duration",
                            buckets=(.025, .05, .075, .1, .15, .2, .3, .5, 1, 2))
M_POOL_WAIT = Histogram("avatar_pool_wait_seconds", "Time a ready chunk waited for the GPU worker thread",
                        buckets=(.001, .01, .025, .05, .1, .2, .5, 1))
M_CHUNKS_DROPPED = Counter("avatar_chunks_dropped_total", "Chunks skipped because rendering fell behind real time")
M_FRAMES = Counter("avatar_frames_total", "Encoded video frames sent to the edge")
M_FRAMES_DROPPED = Counter("avatar_frames_dropped_total", "Encoded frames dropped because the edge was not reading")
M_ERRORS = Counter("avatar_errors_total", "Sessions ended by a render or encode failure")


class _Session:
    def __init__(self, session_id: str, avatar_id: str, background_id: str, assembler: ChunkAssembler):
        self.session_id = session_id
        self.avatar_id = avatar_id
        self.background_id = background_id
        self.assembler = assembler
        self.encoder: H264Encoder | None = None
        self.state = None            # opaque motion state; touched only on the worker thread
        self.want_keyframe = True    # the first frame must be decodable on its own
        self.out: asyncio.Queue = asyncio.Queue()
        self.wake = asyncio.Event()
        self.reported_drops = 0


class Avatar(avatar_pb2_grpc.AvatarServicer):
    def __init__(self, renderer):
        self.renderer = renderer
        self.fps = SAMPLE_RATE // renderer.frame_samples
        self.default_avatar = self._default("AVATAR_DEFAULT_ID", "maria", renderer.avatars, "avatar")
        self.default_background = self._default("AVATAR_DEFAULT_BG", "plain_white", renderer.backgrounds, "background")
        self._active = 0

    @staticmethod
    def _default(env_key: str, fallback: str, available: list[str], kind: str) -> str:
        wanted = os.environ.get(env_key, fallback)
        if wanted in available:
            return wanted
        log.warning("%s=%s is not an installed %s; using %s (installed: %s)",
                    env_key, wanted, kind, available[0], ", ".join(available))
        return available[0]

    def _new_encoder(self) -> H264Encoder:
        return H264Encoder(width=self.renderer.width, height=self.renderer.height, fps=self.fps,
                           bitrate=VIDEO_BITRATE, gop_frames=max(1, int(self.fps * GOP_SECONDS)),
                           codecs=VIDEO_CODECS)

    async def Stream(self, request_iterator, context):
        if self._active >= MAX_SESSIONS:
            M_REJECTED.inc()
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED,
                                f"avatar service is at capacity ({MAX_SESSIONS} sessions)")
        self._active += 1
        M_SESSIONS.inc()
        session = None
        tasks: list[asyncio.Task] = []
        try:
            first = await anext(aiter(request_iterator), None)
            if first is None or first.WhichOneof("msg") != "open":
                await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "first message must be SessionOpen")
            avatar_id = first.open.avatar_id or self.default_avatar
            background_id = first.open.background_id or self.default_background
            if avatar_id not in self.renderer.avatars:
                await context.abort(grpc.StatusCode.NOT_FOUND,
                                    f"unknown avatar_id {avatar_id!r}; installed: {self.renderer.avatars}")
            if background_id not in self.renderer.backgrounds:
                await context.abort(grpc.StatusCode.NOT_FOUND,
                                    f"unknown background_id {background_id!r}; installed: {self.renderer.backgrounds}")
            session = _Session(first.open.session_id, avatar_id, background_id, ChunkAssembler(
                window=self.renderer.window, step=self.renderer.step, max_backlog_steps=MAX_BACKLOG_CHUNKS))
            loop = asyncio.get_running_loop()
            # Opened on the GPU worker so an NVENC session is created, used and
            # closed from the one thread that owns the CUDA context.
            session.encoder = await loop.run_in_executor(POOL, self._new_encoder)
            yield avatar_pb2.ServerEvent(ready=avatar_pb2.SessionReady(
                session_id=session.session_id, codec="h264", width=self.renderer.width,
                height=self.renderer.height, fps=self.fps,
                lookahead_ms=self.renderer.window * 1000 // SAMPLE_RATE,
                avatar_id=avatar_id, background_id=background_id, encoder=session.encoder.name))
            log.info("session %s open: avatar=%s background=%s encoder=%s",
                     session.session_id, avatar_id, background_id, session.encoder.name)
            tasks = [asyncio.create_task(self._read(request_iterator, session)),
                     asyncio.create_task(self._render_loop(session))]
            for task in tasks:
                # Either half ending ends the session: input closing is the
                # normal path, a render failure is the other.
                task.add_done_callback(lambda _t, s=session: s.out.put_nowait(None))
            while (event := await session.out.get()) is not None:
                yield event
            for task in tasks:
                if task.done() and not task.cancelled() and task.exception() is not None:
                    M_ERRORS.inc()
                    log.error("session %s failed", session.session_id, exc_info=task.exception())
                    await context.abort(grpc.StatusCode.INTERNAL, f"avatar session failed: {task.exception()}")
        finally:
            for task in tasks:
                task.cancel()
            if session is not None and session.encoder is not None:
                # Queued behind any chunk still rendering for this session, so
                # the encoder is never closed under the worker's feet.
                POOL.submit(session.encoder.close)
            self._active -= 1
            M_SESSIONS.dec()

    async def _read(self, request_iterator, session: _Session) -> None:
        async for event in request_iterator:
            kind = event.WhichOneof("msg")
            if kind == "audio":
                session.assembler.push(event.audio.speech_pcm16, event.audio.listen_pcm16, event.audio.pts_samples)
                session.wake.set()
            elif kind == "control":
                if event.control.type == avatar_pb2.Control.FLUSH_SPEECH:
                    session.assembler.flush_speech()
                elif event.control.type == avatar_pb2.Control.KEYFRAME:
                    session.want_keyframe = True
                elif event.control.type == avatar_pb2.Control.SESSION_CLOSE:
                    return

    async def _render_loop(self, session: _Session) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await session.wake.wait()
            session.wake.clear()
            while (chunk := session.assembler.pop()) is not None:
                dropped = session.assembler.dropped_chunks - session.reported_drops
                if dropped:
                    session.reported_drops += dropped
                    M_CHUNKS_DROPPED.inc(dropped)
                    # The encoder never saw the skipped frames, but the viewer's
                    # picture has been frozen meanwhile; restart cleanly.
                    session.want_keyframe = True
                    log.warning("session %s: rendering behind real time, skipped %d chunk(s)",
                                session.session_id, dropped)
                await loop.run_in_executor(POOL, self._render_chunk, loop, session, time.perf_counter(), *chunk)

    def _render_chunk(self, loop, session: _Session, submitted: float, pts: int,
                      speech: np.ndarray, listen: np.ndarray) -> None:
        """Runs on the GPU worker thread: render, encode and hand off one chunk."""
        started = time.perf_counter()
        M_POOL_WAIT.observe(started - submitted)
        session.state, frames = self.renderer.render_chunk(
            session.avatar_id, session.background_id, speech, listen, session.state)
        for index, i420 in enumerate(frames):
            force_key, session.want_keyframe = session.want_keyframe, False
            data, keyframe = session.encoder.encode(i420, force_key=force_key)
            if not data:
                continue
            # Sent frame by frame rather than as a finished chunk: the first
            # frame is the one the edge's audio delay has to wait for.
            loop.call_soon_threadsafe(self._emit, session, avatar_pb2.ServerEvent(video=avatar_pb2.VideoFrame(
                data=data, pts_samples=pts + index * self.renderer.frame_samples, keyframe=keyframe,
                chunk_render_ms=(time.perf_counter() - started) * 1000)))
        M_CHUNKS.inc()
        M_CHUNK_SECONDS.observe(time.perf_counter() - started)

    @staticmethod
    def _emit(session: _Session, event) -> None:
        if session.out.qsize() >= MAX_PENDING_FRAMES:
            M_FRAMES_DROPPED.inc()
            session.want_keyframe = True  # later frames reference the one just dropped
            return
        session.out.put_nowait(event)
        M_FRAMES.inc()


def load_renderer_and_warmup():
    renderer = load_renderer(width=OUT_WIDTH, height=OUT_HEIGHT)
    servicer = Avatar(renderer)
    encoder = servicer._new_encoder()
    chunk_ms = 0.0
    if WARMUP:
        # TensorRT allocates on first use; pay that here, not in a caller's
        # first second. The second pass is the one worth reporting.
        silence = np.zeros(renderer.window, dtype=np.float32)
        state = None
        for _ in range(2):
            started = time.perf_counter()
            state, frames = renderer.render_chunk(
                servicer.default_avatar, servicer.default_background, silence, silence, state)
            for i420 in frames:
                encoder.encode(i420)
            chunk_ms = (time.perf_counter() - started) * 1000
        budget_ms = renderer.step * 1000 / SAMPLE_RATE
        if chunk_ms > budget_ms:
            print(f"AVATAR SERVICE | WARNING | one chunk takes {chunk_ms:.0f} ms but covers {budget_ms:.0f} ms "
                  f"of video; this GPU cannot keep up in real time and chunks will be skipped", flush=True)
    encoder_name = encoder.name
    encoder.close()
    return servicer, encoder_name, chunk_ms


async def serve():
    loop = asyncio.get_running_loop()
    # Loaded on the GPU worker thread so the CUDA context, the engines and
    # every later render call all live on the same thread.
    servicer, encoder_name, chunk_ms = await loop.run_in_executor(POOL, load_renderer_and_warmup)
    renderer = servicer.renderer
    server = grpc.aio.server()
    avatar_pb2_grpc.add_AvatarServicer_to_server(servicer, server)
    health_servicer = health.aio.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    await health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    addr = os.environ.get("AVATAR_ADDR", "[::]:50056")
    server.add_insecure_port(addr); await server.start()
    port = int(os.environ.get("AVATAR_METRICS_PORT", "9105"))
    metrics_host = os.environ.get("AVATAR_METRICS_HOST", "0.0.0.0")
    start_http_server(port, addr=metrics_host)
    print(f"AVATAR SERVICE | READY | gRPC {addr} | engine {renderer.engine} | {renderer.width}x{renderer.height} "
          f"@{servicer.fps} fps | encoder {encoder_name} | avatar {servicer.default_avatar} "
          f"| warmup {chunk_ms:.0f} ms/chunk | max sessions {MAX_SESSIONS}", flush=True)
    await server.wait_for_termination()


if __name__ == "__main__":
    try: asyncio.run(serve())
    except KeyboardInterrupt: pass
