"""Self-check for VAD gating and endpointer translation. Run: python -m tests.test_endpoint

Guards both boundary-sensitive paths: edge END_TURN forwarded as STT END_TURN,
and audio observed just before USER_SPEECH_STARTED must be prepended to the live
utterance instead of being lost to the causal VAD attack window.
"""
import asyncio

from proto import asr_pb2, edge_pb2
from service.server import MediaBridge


class FakeStub:
    """Stands in for the STT AsrStub: records every event the bridge forwards."""
    def __init__(self):
        self.events = []

    async def Transcribe(self, request_iter):
        async for ev in request_iter:
            self.events.append(ev)
        if False:
            yield  # make this an async generator (matches grpc streaming stub)


async def endpoint_events():
    sid = "test-sess"
    yield edge_pb2.ClientEvent(control=edge_pb2.Control(
        session_id=sid, type=edge_pb2.Control.SESSION_OPEN, ts_ms=1))
    yield edge_pb2.ClientEvent(audio=edge_pb2.AudioFrame(
        session_id=sid, pcm16=b"\x00\x01" * 160, ts_ms=2))
    yield edge_pb2.ClientEvent(control=edge_pb2.Control(
        session_id=sid, type=edge_pb2.Control.END_TURN, ts_ms=3))  # endpointer
    yield edge_pb2.ClientEvent(control=edge_pb2.Control(
        session_id=sid, type=edge_pb2.Control.SESSION_CLOSE, ts_ms=4))


async def test_endpoint_translation():
    stub = FakeStub()
    # llm=None: this FakeStub's Transcribe() never yields a Transcript, so
    # from_stt() never reaches the turn_complete branch that would touch it.
    async for _ in MediaBridge(stub, None, vad_gate=True).Stream(endpoint_events(), None):
        pass

    ctrl = [e.control.type for e in stub.events if e.WhichOneof("msg") == "control"]
    assert asr_pb2.Control.END_TURN in ctrl, f"endpointer dropped: {ctrl}"
    assert ctrl.index(asr_pb2.Control.END_TURN) < ctrl.index(asr_pb2.Control.SESSION_CLOSE), \
        f"END_TURN must precede SESSION_CLOSE: {ctrl}"


async def preroll_events():
    sid = "pre-roll-sess"
    yield edge_pb2.ClientEvent(control=edge_pb2.Control(
        session_id=sid, type=edge_pb2.Control.SESSION_OPEN, ts_ms=0))
    # Five 10 ms PCM16 frames before VAD confirmation. A 30 ms pre-roll should
    # retain only C/D/E; A/B are old silence and must remain gated.
    for i, marker in enumerate((b"A", b"B", b"C", b"D", b"E"), start=1):
        yield edge_pb2.ClientEvent(audio=edge_pb2.AudioFrame(
            session_id=sid, pcm16=marker * 320, ts_ms=i * 10))
    yield edge_pb2.ClientEvent(control=edge_pb2.Control(
        session_id=sid, type=edge_pb2.Control.USER_SPEECH_STARTED, ts_ms=60))
    yield edge_pb2.ClientEvent(audio=edge_pb2.AudioFrame(
        session_id=sid, pcm16=b"F" * 320, ts_ms=60))
    yield edge_pb2.ClientEvent(control=edge_pb2.Control(
        session_id=sid, type=edge_pb2.Control.END_TURN, ts_ms=100))

    # The next turn gets its own prefix; no frame from turn one may reappear.
    for i, marker in enumerate((b"W", b"X", b"Y", b"Z"), start=11):
        yield edge_pb2.ClientEvent(audio=edge_pb2.AudioFrame(
            session_id=sid, pcm16=marker * 320, ts_ms=i * 10))
    yield edge_pb2.ClientEvent(control=edge_pb2.Control(
        session_id=sid, type=edge_pb2.Control.USER_SPEECH_STARTED, ts_ms=150))
    yield edge_pb2.ClientEvent(audio=edge_pb2.AudioFrame(
        session_id=sid, pcm16=b"Q" * 320, ts_ms=150))
    yield edge_pb2.ClientEvent(control=edge_pb2.Control(
        session_id=sid, type=edge_pb2.Control.SESSION_CLOSE, ts_ms=200))


async def test_vad_preroll_preserves_first_word_and_resets():
    stub = FakeStub()
    async for _ in MediaBridge(
            stub, None, vad_gate=True, vad_preroll_ms=30).Stream(preroll_events(), None):
        pass

    audio = [e.audio for e in stub.events if e.WhichOneof("msg") == "audio"]
    markers = [a.pcm16[:1] for a in audio]
    assert markers == [b"C", b"D", b"E", b"F", b"X", b"Y", b"Z", b"Q"], markers
    assert [a.ts_ms for a in audio] == [30, 40, 50, 60, 120, 130, 140, 150]


async def test_disabled_gate_still_forwards_without_speech_event():
    stub = FakeStub()

    async def events():
        sid = "ungated-sess"
        yield edge_pb2.ClientEvent(control=edge_pb2.Control(
            session_id=sid, type=edge_pb2.Control.SESSION_OPEN, ts_ms=0))
        yield edge_pb2.ClientEvent(audio=edge_pb2.AudioFrame(
            session_id=sid, pcm16=b"A" * 320, ts_ms=10))
        yield edge_pb2.ClientEvent(control=edge_pb2.Control(
            session_id=sid, type=edge_pb2.Control.SESSION_CLOSE, ts_ms=20))

    async for _ in MediaBridge(stub, None, vad_gate=False).Stream(events(), None):
        pass
    audio = [e.audio for e in stub.events if e.WhichOneof("msg") == "audio"]
    assert [a.pcm16[:1] for a in audio] == [b"A"]
