"""Avatar servicer against the synthetic renderer and a real H.264 encoder."""
import asyncio

import grpc
import numpy as np
import pytest

from proto import avatar_pb2
from service.renderer import SyntheticRenderer

FRAME = 320  # 20 ms at 16 kHz


class Aborted(Exception):
    def __init__(self, code, details):
        super().__init__(details)
        self.code = code


class FakeContext:
    async def abort(self, code, details):
        raise Aborted(code, details)


@pytest.fixture
def srv(monkeypatch):
    import service.server as srv
    monkeypatch.setattr(srv, "VIDEO_CODECS", ["libx264"])
    return srv


def make_servicer(srv):
    return srv.Avatar(SyntheticRenderer(width=320, height=180))


def audio(i: int, level: int = 8000) -> avatar_pb2.ClientEvent:
    pcm = np.full(FRAME, level, dtype="<i2").tobytes()
    return avatar_pb2.ClientEvent(audio=avatar_pb2.AudioFrame(
        speech_pcm16=pcm, listen_pcm16=pcm, pts_samples=i * FRAME))


def control(kind) -> avatar_pb2.ClientEvent:
    return avatar_pb2.ClientEvent(control=avatar_pb2.Control(type=kind))


async def requests(events, *, hold_open: asyncio.Event | None = None):
    for event in events:
        yield event
        await asyncio.sleep(0)
    if hold_open is not None:
        await hold_open.wait()


async def collect(servicer, events, *, until_frames: int):
    """Run a session until until_frames video frames have arrived, then close it."""
    done = asyncio.Event()
    ready, frames = None, []
    async for event in servicer.Stream(requests(events, hold_open=done), FakeContext()):
        if event.WhichOneof("msg") == "ready":
            ready = event.ready
        else:
            frames.append(event.video)
            if len(frames) >= until_frames:
                done.set()
    return ready, frames


OPEN = avatar_pb2.ClientEvent(open=avatar_pb2.SessionOpen(session_id="s1"))


async def test_session_renders_h264_frames_on_the_audio_timeline(srv):
    servicer = make_servicer(srv)
    # 41 ticks = 13120 samples: enough for the windows starting at 0 and 3200.
    ready, frames = await collect(servicer, [OPEN] + [audio(i) for i in range(41)], until_frames=10)

    assert (ready.codec, ready.width, ready.height, ready.fps) == ("h264", 320, 180, 25)
    assert ready.lookahead_ms == 405
    assert ready.encoder == "libx264"
    assert ready.avatar_id == "synthetic"  # default fell back to what is installed
    assert [f.pts_samples for f in frames] == [i * 640 for i in range(10)]
    assert frames[0].keyframe
    assert all(f.data.startswith(b"\x00\x00\x00\x01") or f.data.startswith(b"\x00\x00\x01") for f in frames)
    assert servicer._active == 0


async def test_keyframe_request_forces_the_next_frame_to_be_a_keyframe(srv):
    servicer = make_servicer(srv)
    inbox: asyncio.Queue = asyncio.Queue()

    async def driven():
        while (event := await inbox.get()) is not None:
            yield event

    for event in [OPEN] + [audio(i) for i in range(21)]:        # first chunk: frames 0-4
        inbox.put_nowait(event)
    frames = []
    async for event in servicer.Stream(driven(), FakeContext()):
        if event.WhichOneof("msg") != "video":
            continue
        frames.append(event.video)
        if len(frames) == 5:
            # Only once the first chunk is out, so the request cannot be
            # absorbed by the session's own opening keyframe.
            inbox.put_nowait(control(avatar_pb2.Control.KEYFRAME))
            for i in range(21, 31):                             # second chunk: frames 5-9
                inbox.put_nowait(audio(i))
        elif len(frames) == 10:
            inbox.put_nowait(None)
    assert [f.keyframe for f in frames[:5]] == [True, False, False, False, False]
    assert frames[5].keyframe


async def test_stream_must_start_with_session_open(srv):
    servicer = make_servicer(srv)
    with pytest.raises(Aborted) as exc:
        async for _ in servicer.Stream(requests([audio(0)]), FakeContext()):
            pass
    assert exc.value.code == grpc.StatusCode.INVALID_ARGUMENT
    assert servicer._active == 0


async def test_unknown_avatar_is_rejected(srv):
    servicer = make_servicer(srv)
    bad = avatar_pb2.ClientEvent(open=avatar_pb2.SessionOpen(session_id="s1", avatar_id="nobody"))
    with pytest.raises(Aborted) as exc:
        async for _ in servicer.Stream(requests([bad]), FakeContext()):
            pass
    assert exc.value.code == grpc.StatusCode.NOT_FOUND


async def test_sessions_beyond_capacity_are_rejected(srv, monkeypatch):
    monkeypatch.setattr(srv, "MAX_SESSIONS", 1)
    servicer = make_servicer(srv)
    hold = asyncio.Event()
    first = servicer.Stream(requests([OPEN], hold_open=hold), FakeContext())
    assert (await anext(first)).WhichOneof("msg") == "ready"
    with pytest.raises(Aborted) as exc:
        async for _ in servicer.Stream(requests([OPEN]), FakeContext()):
            pass
    assert exc.value.code == grpc.StatusCode.RESOURCE_EXHAUSTED
    hold.set()
    async for _ in first:
        pass
    assert servicer._active == 0
