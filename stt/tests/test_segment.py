"""Self-check for bounded-segment cutting. Run in the STT venv (needs torch/
transformers importable, same as server.py):
    python -m tests.test_segment

Covers: (1) the quiet-point RMS scan in isolation, (2) an end-to-end
Transcribe() run with a fake model proving segment_index ordering/reset and
turn_complete placement, including the forced-cut fallback when no pause
exists (the "someone hums continuously" case) — the concrete proof that
worst-case per-call compute stays bounded regardless of audio content.
"""
import asyncio
import struct
import threading

import grpc
import numpy as np

import service.server as server_module
from proto import asr_pb2
from service.server import (AsrServicer, _find_quiet_cut, QUIET_FRAME_BYTES,
                     SEG_GRACE_BYTES, SEG_THRESHOLD_BYTES, SR)

WINDOW_BYTES = int(1.0 * SR) * 2  # matches default STT_QUIET_SCAN_SEC


def test_quiet_cut_finds_engineered_dip():
    loud = np.full(SR, 5000, dtype="<i2")
    quiet = np.full(int(0.12 * SR), 50, dtype="<i2")  # sustained 120ms dip well under the RMS floor
    buf = np.concatenate([loud[: int(0.4 * SR)], quiet, loud[int(0.4 * SR):]]).astype("<i2").tobytes()
    dip_start, dip_end = int(0.4 * SR) * 2, int(0.4 * SR) * 2 + len(quiet) * 2
    cut = _find_quiet_cut(buf, WINDOW_BYTES, QUIET_FRAME_BYTES, quiet_rms=400)
    assert cut is not None and dip_start <= cut <= dip_end, (cut, dip_start, dip_end)


def test_quiet_cut_no_dip_returns_none():
    # Continuous loud tone, no gap anywhere — the "keeps humming" case that must
    # fall through to the forced-cut fallback rather than hang forever.
    buf = np.full(SR, 5000, dtype="<i2").tobytes()
    assert _find_quiet_cut(buf, WINDOW_BYTES, QUIET_FRAME_BYTES, quiet_rms=400) is None


def test_quiet_cut_rms_matches_hand_computation():
    pcm = struct.pack("<4h", 1000, -1000, 1000, -1000)
    arr = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    hand = (sum(x * x for x in arr) / len(arr)) ** 0.5
    # A single-frame buffer just exercises the same reshape/mean/sqrt path _find_quiet_cut uses.
    cut = _find_quiet_cut(pcm, len(pcm), len(pcm), quiet_rms=hand + 1, min_quiet_frames=1)
    assert cut == len(pcm) // 2, cut  # quiet enough (threshold above the frame's RMS) -> cuts at midpoint
    assert _find_quiet_cut(pcm, len(pcm), len(pcm), quiet_rms=hand - 1, min_quiet_frames=1) is None  # not quiet enough


class _Hyp:
    def __init__(self, text):
        self.text = text


class FakeModel:
    """Stands in for SraVaani's AutoModel: records each call's window length and
    returns a canned hypothesis naming which call it was."""
    def __init__(self):
        self.calls = []

    def transcribe(self, waveforms, return_hypotheses=True):
        arr = waveforms[0]
        self.calls.append(arr.size)
        return [_Hyp(f"seg{len(self.calls)}")]


def _pcm(n_samples, amplitude=5000):
    return np.full(n_samples, amplitude, dtype="<i2").tobytes()


async def _drive(model, events):
    async def request_iterator():
        for ev in events:
            yield ev

    servicer = AsrServicer(model)
    out = []
    async for t in servicer.Transcribe(request_iterator(), None):
        out.append(t)
    return out


def _audio(sid, pcm, ts=0):
    return asr_pb2.AsrClientEvent(audio=asr_pb2.AudioChunk(session_id=sid, pcm16=pcm, ts_ms=ts))


def _control(sid, ctype, ts=0):
    return asr_pb2.AsrClientEvent(control=asr_pb2.Control(session_id=sid, type=ctype, ts_ms=ts))


CHUNK_BYTES = int(0.02 * SR) * 2  # 20ms frames — real audio arrives in small chunks, not one
                                   # multi-second blob; the segment-cut check runs per chunk,
                                   # so tests must deliver audio the same way the edge does.


def _audio_events(sid, pcm_bytes, chunk_bytes=CHUNK_BYTES):
    for i in range(0, len(pcm_bytes), chunk_bytes):
        yield _audio(sid, pcm_bytes[i:i + chunk_bytes])


def test_mid_turn_cut_on_quiet_point_then_turn_end():
    # One turn: enough loud audio to cross SEG_THRESHOLD_BYTES with an engineered
    # quiet dip near the end (so a real quiet-point cut fires, not the forced
    # fallback), then more audio, then END_TURN.
    sid = "s1"
    seg_threshold_samples = SEG_THRESHOLD_BYTES // 2
    lead = seg_threshold_samples + int(0.2 * SR)  # cross the threshold before the dip
    dip = int(0.12 * SR)
    seg1 = np.concatenate([
        np.full(lead, 5000, dtype="<i2"),
        np.full(dip, 50, dtype="<i2"),
        np.full(int(0.1 * SR), 5000, dtype="<i2"),  # a little more so the dip isn't the last frame
    ]).astype("<i2").tobytes()

    events = (
        [_control(sid, asr_pb2.Control.SESSION_OPEN)]
        + list(_audio_events(sid, seg1))
        + list(_audio_events(sid, _pcm(int(0.5 * SR))))  # start of the next segment, still mid-turn
        + [_control(sid, asr_pb2.Control.END_TURN), _control(sid, asr_pb2.Control.SESSION_CLOSE)]
    )
    model = FakeModel()
    out = asyncio.run(_drive(model, events))

    assert len(out) >= 2, out
    assert [t.segment_index for t in out] == list(range(len(out))), [t.segment_index for t in out]
    turn_complete_flags = [t.turn_complete for t in out]
    assert turn_complete_flags.count(True) == 1
    assert turn_complete_flags[-1] is True  # the turn-ending segment is last
    assert all(not f for f in turn_complete_flags[:-1])


def test_forced_cut_when_no_quiet_point_within_grace():
    # Continuous loud audio, delivered in small realistic chunks, well past
    # SEG_THRESHOLD_BYTES + SEG_GRACE_BYTES with no gap anywhere: proves the hard
    # fallback fires and TIGHTLY bounds worst-case segment size (to about one
    # chunk past threshold+grace) — this is the "someone hums for 60s" case the
    # whole feature targets.
    sid = "s2"
    total_bytes = SEG_THRESHOLD_BYTES + SEG_GRACE_BYTES + int(1.0 * SR) * 2
    pcm = _pcm(total_bytes // 2)

    events = (
        [_control(sid, asr_pb2.Control.SESSION_OPEN)]
        + list(_audio_events(sid, pcm))
        + [_control(sid, asr_pb2.Control.SESSION_CLOSE)]
    )
    model = FakeModel()
    out = asyncio.run(_drive(model, events))

    assert len(out) >= 1, out
    # First emitted segment must be the forced cut: bounded at threshold+grace
    # (plus at most one chunk of overshoot, since the check runs per chunk), not
    # the full buffered audio.
    first_window_samples = model.calls[0]
    max_forced_samples = (SEG_THRESHOLD_BYTES + SEG_GRACE_BYTES) // 2
    chunk_samples = CHUNK_BYTES // 2
    assert first_window_samples <= max_forced_samples + chunk_samples, \
        (first_window_samples, max_forced_samples, chunk_samples)


def test_end_turn_after_full_mid_turn_cut_emits_empty_final_marker():
    """A cut can consume every buffered sample just before the endpoint.

    The marker has no text of its own, but lets the orchestrator commit the
    previous immutable segment as one completed user turn.
    """
    sid = "s3"
    pcm = _pcm((SEG_THRESHOLD_BYTES + SEG_GRACE_BYTES) // 2)
    events = (
        [_control(sid, asr_pb2.Control.SESSION_OPEN)]
        + list(_audio_events(sid, pcm))
        + [_control(sid, asr_pb2.Control.END_TURN), _control(sid, asr_pb2.Control.SESSION_CLOSE)]
    )
    out = asyncio.run(_drive(FakeModel(), events))

    assert len(out) == 2, out
    assert out[0].turn_complete is False and out[0].text == "seg1"
    assert out[1].turn_complete is True and out[1].text == ""
    assert [t.segment_index for t in out] == [0, 1]


class _OverflowAbort(Exception):
    def __init__(self, code, details):
        super().__init__(details)
        self.code = code
        self.details = details


class _FakeContext:
    def __init__(self):
        self.aborted = None

    async def abort(self, code, details):
        self.aborted = (code, details)
        raise _OverflowAbort(code, details)


class _BlockingModel:
    """The GPU forward for the first segment blocks until the test releases
    it. Real GPU work runs in a background thread (see transcribe_pcm /
    GPU_POOL), so blocking here does not block the event loop -- the reader
    task keeps racing ahead and can build up a real backlog deterministically."""
    def __init__(self, release_event):
        self.calls = []
        self._release = release_event

    def transcribe(self, waveforms, return_hypotheses=True):
        arr = waveforms[0]
        self.calls.append(arr.size)
        if len(self.calls) == 1:
            self._release.wait(timeout=5)
        return [_Hyp(f"seg{len(self.calls)}")]


def test_backlog_overflow_aborts_and_preserves_accepted_audio():
    """A stalled consumer must not let the per-stream backlog grow without
    bound. Once the (here, test-shrunk) depth is exceeded the stream aborts
    explicitly with RESOURCE_EXHAUSTED -- but segments already accepted before
    the overflow are still transcribed and yielded, and teardown (the final
    sentinel put) does not hang even though the queue is momentarily full."""
    sid = "overflow"
    orig = (server_module.SEG_THRESHOLD_BYTES, server_module.SEG_GRACE_BYTES,
            server_module.SEGMENT_QUEUE_MAX, server_module.SEGMENT_QUEUE_MAX_AGE_SEC)
    server_module.SEG_THRESHOLD_BYTES = 4000
    server_module.SEG_GRACE_BYTES = 0
    server_module.SEGMENT_QUEUE_MAX = 2
    server_module.SEGMENT_QUEUE_MAX_AGE_SEC = 1000
    try:
        release = threading.Event()
        model = _BlockingModel(release)
        context = _FakeContext()
        # Continuous loud audio, no gaps: every SEG_THRESHOLD_BYTES forces a
        # cut, so ~5 segments come out of this much audio -- comfortably more
        # than SEGMENT_QUEUE_MAX(=2) can hold while the first sits "processing".
        pcm = _pcm(10000)
        events = (
            [_control(sid, asr_pb2.Control.SESSION_OPEN)]
            + list(_audio_events(sid, pcm))
            + [_control(sid, asr_pb2.Control.SESSION_CLOSE)]
        )

        async def request_iterator():
            for ev in events:
                yield ev

        async def go():
            servicer = AsrServicer(model)
            out = []
            try:
                async for t in servicer.Transcribe(request_iterator(), context):
                    out.append(t)
            except _OverflowAbort:
                pass
            return out

        async def run_test():
            task = asyncio.create_task(go())
            # Let the reader task race ahead and overflow the tiny queue while
            # the first segment sits "stuck" in _BlockingModel.
            await asyncio.sleep(0.2)
            release.set()  # let the stalled forward -- and everything queued behind it -- drain
            return await asyncio.wait_for(task, timeout=5)

        out = asyncio.run(run_test())
        assert context.aborted is not None and context.aborted[0] == grpc.StatusCode.RESOURCE_EXHAUSTED, \
            context.aborted
        assert out, "accepted audio before the overflow must still be transcribed, not silently dropped"
    finally:
        (server_module.SEG_THRESHOLD_BYTES, server_module.SEG_GRACE_BYTES,
         server_module.SEGMENT_QUEUE_MAX, server_module.SEGMENT_QUEUE_MAX_AGE_SEC) = orig
    print("ok test_backlog_overflow_aborts_and_preserves_accepted_audio")


def test_cancel_during_forward_holds_gpu_slot_until_it_exits():
    """A native forward can't be preempted: cancelling the coroutine awaiting
    it must not free the GPU slot until the forward actually finishes, or a
    second forward could run concurrently against the same non-thread-safe
    model. Mirrors the equivalent TTS admission-slot guarantee."""
    release = threading.Event()
    started = threading.Event()

    class SlowModel:
        def transcribe(self, waveforms, return_hypotheses=True):
            started.set()
            release.wait(timeout=5)
            return [_Hyp("done")]

    async def scenario():
        sid = "cancel-test"
        events = (
            [_control(sid, asr_pb2.Control.SESSION_OPEN)]
            + list(_audio_events(sid, _pcm(int(0.2 * SR))))
            + [_control(sid, asr_pb2.Control.END_TURN), _control(sid, asr_pb2.Control.SESSION_CLOSE)]
        )

        async def request_iterator():
            for ev in events:
                yield ev

        servicer = AsrServicer(SlowModel())
        # The async generator does nothing until first advanced -- this single
        # __anext__() both starts it (reader() enqueues the one segment, the
        # consumer dequeues it and calls run()) and is what gets cancelled once
        # the forward is blocked inside the executor thread.
        gen = servicer.Transcribe(request_iterator(), None)

        try:
            await asyncio.wait_for(gen.__anext__(), timeout=0.5)
            assert False, "expected the forward to still be blocked"
        except asyncio.TimeoutError:
            pass  # wait_for cancelled the pending __anext__() -- this is the cancellation under test

        assert started.is_set(), "forward never started within the timeout"
        occupied = server_module.MAX_INFLIGHT - server_module.GPU_SLOTS._value
        assert occupied >= 1, "the slot must still be held while the forward keeps running"
        release.set()
        for _ in range(200):
            if server_module.GPU_SLOTS._value == server_module.MAX_INFLIGHT:
                break
            await asyncio.sleep(0.01)
        assert server_module.GPU_SLOTS._value == server_module.MAX_INFLIGHT, "GPU slot leaked after cancellation"
        await gen.aclose()

    asyncio.run(scenario())
    print("ok test_cancel_during_forward_holds_gpu_slot_until_it_exits")


def test_gpu_executing_gauge_reflects_true_worker_occupancy_not_admission():
    """GPU_SLOTS (STT_MAX_INFLIGHT) only bounds admission into GPU_POOL's own
    internal queue; GPU_POOL itself has a single worker thread, so two
    admitted forwards are never both truly running at once. stt_gpu_executing
    must reflect that (stays at 1) even while stt_gpu_inflight (admission)
    would show 2 -- this is the exact ambiguity Phase 2 calls out."""
    release = threading.Event()
    started = threading.Event()

    class BlockOnFirstCallModel:
        def __init__(self):
            self.calls = 0

        def transcribe(self, waveforms, return_hypotheses=True):
            self.calls += 1
            if self.calls == 1:
                started.set()
                release.wait(timeout=5)
            return [_Hyp(f"seg{self.calls}")]

    async def scenario():
        model = BlockOnFirstCallModel()
        servicer = server_module.AsrServicer(model)

        def events_for(sid):
            return (
                [_control(sid, asr_pb2.Control.SESSION_OPEN)]
                + list(_audio_events(sid, _pcm(int(0.2 * server_module.SR))))
                + [_control(sid, asr_pb2.Control.END_TURN), _control(sid, asr_pb2.Control.SESSION_CLOSE)]
            )

        async def request_iterator(events):
            for ev in events:
                yield ev

        async def drive(sid):
            async for _ in servicer.Transcribe(request_iterator(events_for(sid)), None):
                pass

        task_a = asyncio.create_task(drive("a"))
        for _ in range(200):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set(), "stream a's forward never started"

        task_b = asyncio.create_task(drive("b"))
        # Give b a chance to admit into GPU_SLOTS and queue behind a in the
        # single-threaded GPU_POOL, without b's own forward actually running.
        await asyncio.sleep(0.1)

        executing = server_module.M_EXECUTING._value.get()
        assert executing == 1, f"expected exactly 1 truly-executing forward, got {executing}"

        release.set()
        await asyncio.wait_for(asyncio.gather(task_a, task_b), timeout=5)
        assert server_module.M_EXECUTING._value.get() == 0, "stt_gpu_executing leaked after both forwards finished"

    asyncio.run(scenario())
    print("ok gpu_executing_gauge_reflects_true_worker_occupancy_not_admission")


def test_wedge_watchdog_reports_unready_then_recovers():
    """Phase 3: a forward stuck past STT_WEDGE_TIMEOUT_S must flip gRPC health
    to NOT_SERVING (the only signal available since a native forward can't be
    preempted -- completes Phase 1's "report itself unready"), and flip back
    once the forward completes."""
    release = threading.Event()
    started = threading.Event()

    class SlowModel:
        def transcribe(self, waveforms, return_hypotheses=True):
            started.set()
            release.wait(timeout=5)
            return [_Hyp("done")]

    class FakeHealthServicer:
        def __init__(self):
            self.statuses = []

        async def set(self, service, status):
            self.statuses.append(status)

    async def scenario():
        orig_timeout = server_module.STT_WEDGE_TIMEOUT_S
        server_module.STT_WEDGE_TIMEOUT_S = 0.1
        health = FakeHealthServicer()
        watchdog = asyncio.create_task(server_module._wedge_watchdog(health, interval_s=0.02))
        try:
            sid = "wedge-test"
            events = (
                [_control(sid, asr_pb2.Control.SESSION_OPEN)]
                + list(_audio_events(sid, _pcm(int(0.2 * server_module.SR))))
                + [_control(sid, asr_pb2.Control.END_TURN), _control(sid, asr_pb2.Control.SESSION_CLOSE)]
            )

            async def request_iterator():
                for ev in events:
                    yield ev

            servicer = AsrServicer(SlowModel())
            gen = servicer.Transcribe(request_iterator(), None)
            drive = asyncio.create_task(gen.__anext__())
            for _ in range(200):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert started.is_set(), "forward never started"

            for _ in range(200):
                if server_module.health_pb2.HealthCheckResponse.NOT_SERVING in health.statuses:
                    break
                await asyncio.sleep(0.02)
            assert health.statuses and health.statuses[-1] == server_module.health_pb2.HealthCheckResponse.NOT_SERVING, \
                f"expected NOT_SERVING once wedged, got {health.statuses}"

            release.set()
            drive.cancel()
            try:
                await drive
            except (asyncio.CancelledError, StopAsyncIteration):
                pass
            await gen.aclose()

            for _ in range(200):
                if health.statuses[-1] == server_module.health_pb2.HealthCheckResponse.SERVING:
                    break
                await asyncio.sleep(0.02)
            assert health.statuses[-1] == server_module.health_pb2.HealthCheckResponse.SERVING, \
                f"expected SERVING again after recovery, got {health.statuses}"
        finally:
            watchdog.cancel()
            server_module.STT_WEDGE_TIMEOUT_S = orig_timeout

    asyncio.run(scenario())
    print("ok test_wedge_watchdog_reports_unready_then_recovers")
