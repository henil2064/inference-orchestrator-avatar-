"""Self-checks for the agent-turn lifecycle signal.
Run: python -m tests.test_turn_signals

The client needs to know when the agent still owes it work. Per-service "I am
idle" broadcasts cannot answer that — TTS is genuinely idle in the gap between
two clauses of the SAME turn — so the orchestrator, which is the only component
that sees the whole turn, emits the signal itself:

  1. AGENT_TURN_STARTED precedes any audio or text for a turn.
  2. AGENT_TURN_ENDED closes it on EVERY exit path — including a turn that
     produced no audio at all — so a client can never be left waiting.
  3. Every downstream event is stamped with its turn_id, so a barged-in turn's
     trailing signal is distinguishable from the next turn's.

Fakes for STT/LLM/TTS are driven through MediaBridge.Stream, the same seam the
routing and interrupt tests use.
"""
import asyncio

import service.server as server_module
from proto import asr_pb2, edge_pb2, llm_pb2, tts_pb2
from service.server import MediaBridge

from tests.test_interrupt import FakeStt, FakeTts, RecordingSessions, _kind


class StreamingLlm:
    """Streams one complete clause and finishes — the ordinary turn."""
    async def GenerateStream(self, request, timeout=None):
        yield llm_pb2.GenerateChunk(session_id=request.session_id, text_delta="All done. ")
        yield llm_pb2.GenerateChunk(session_id=request.session_id, is_final=True,
                                    profile=llm_pb2.InferenceProfile(model="m", engine="e", device="cpu"))


class BrokenLlm:
    """Fails before emitting anything, so the turn yields no reply and no audio."""
    async def GenerateStream(self, request, timeout=None):
        raise RuntimeError("llm exploded")
        yield  # pragma: no cover - makes this an async generator


class FailingTts:
    """Every clause's TTS call fails outright before any audio is produced --
    the LLM succeeds but nothing is ever actually delivered to the user."""
    def __init__(self):
        self.requests = []

    async def SynthesizeStream(self, request, timeout=None):
        self.requests.append(request)
        raise RuntimeError("tts exploded")
        yield  # pragma: no cover - makes this an async generator


def _one_chunk(session_id, rate=24000):
    frame = (rate // 50) * 2
    pcm = b"\x01\x00" * (frame // 2)
    return tts_pb2.SynthesizeChunk(session_id=session_id, pcm16=pcm, sample_rate_hz=rate,
        chunk_index=0, ts_ms=0,
        profile=tts_pb2.InferenceProfile(model="tts", engine="e", device="cpu", compute_ms=1, audio_seconds=0.02))


class PartialFailTts:
    """Sends one real audio chunk, then the RPC itself fails mid-stream --
    some audio reached the relay even though this clause never completed."""
    def __init__(self):
        self.requests = []

    async def SynthesizeStream(self, request, timeout=None):
        self.requests.append(request)
        yield _one_chunk(request.session_id)
        raise RuntimeError("tts dropped mid-stream")


class AllMalformedTts:
    """The RPC completes normally, but every chunk it returns is malformed
    (wrong session_id) and dropped -- zero audio ever reaches the relay."""
    def __init__(self):
        self.requests = []

    async def SynthesizeStream(self, request, timeout=None):
        self.requests.append(request)
        yield tts_pb2.SynthesizeChunk(session_id="not-" + request.session_id,
            pcm16=b"\x01\x00" * 10, sample_rate_hz=24000, chunk_index=0, ts_ms=0)
        yield tts_pb2.SynthesizeChunk(session_id=request.session_id, sample_rate_hz=24000,
            chunk_index=1, is_last=True, ts_ms=0)


class SlowFirstClauseTts:
    """Blocks every SynthesizeStream call on a gate before yielding anything,
    simulating a contended/slow TTS backend that stalls the consumer."""
    def __init__(self, gate):
        self._gate = gate
        self.requests = []

    async def SynthesizeStream(self, request, timeout=None):
        self.requests.append(request)
        await self._gate.wait()
        yield _one_chunk(request.session_id)
        yield tts_pb2.SynthesizeChunk(session_id=request.session_id, sample_rate_hz=24000,
            chunk_index=1, is_last=True, ts_ms=0)


class ManyClausesLlm:
    """Streams several complete clauses fast (near-zero real elapsed time),
    then finishes -- enough clauses to overflow the bounded clause queue."""
    async def GenerateStream(self, request, timeout=None):
        for i in range(6):
            yield llm_pb2.GenerateChunk(session_id=request.session_id, text_delta=f"Clause {i}. ")
        yield llm_pb2.GenerateChunk(session_id=request.session_id, is_final=True,
                                    profile=llm_pb2.InferenceProfile(model="m", engine="e", device="cpu"))


class DelayedTurnCompleteStt:
    """Like FakeStt, but only emits the turn_complete transcript once `gate`
    is set -- lets a test control exactly how much of the turn deadline
    elapses while STT is "processing"."""
    def __init__(self, text, gate):
        self._text = text
        self._gate = gate

    async def Transcribe(self, request_iter):
        got_end = asyncio.Event()

        async def drain():
            async for ev in request_iter:
                if ev.WhichOneof("msg") == "control" and ev.control.type == asr_pb2.Control.END_TURN:
                    got_end.set()
        task = asyncio.create_task(drain())
        try:
            await got_end.wait()
            await self._gate.wait()
            yield asr_pb2.Transcript(session_id="s", text=self._text, turn_complete=True,
                                     segment_index=0, ts_ms=10, proc_ms=20)
        finally:
            task.cancel()


class TimeoutRecordingLlm:
    """Records the timeout each GenerateStream call actually received, so a
    test can assert the deadline budget was really propagated to the RPC."""
    def __init__(self):
        self.timeouts = []

    async def GenerateStream(self, request, timeout=None):
        self.timeouts.append(timeout)
        yield llm_pb2.GenerateChunk(session_id=request.session_id, text_delta="Done. ")
        yield llm_pb2.GenerateChunk(session_id=request.session_id, is_final=True,
                                    profile=llm_pb2.InferenceProfile(model="m", engine="e", device="cpu"))


class TimeoutRecordingTts:
    """Records the timeout each SynthesizeStream call actually received."""
    def __init__(self):
        self.timeouts = []

    async def SynthesizeStream(self, request, timeout=None):
        self.timeouts.append(timeout)
        yield _one_chunk(request.session_id)
        yield tts_pb2.SynthesizeChunk(session_id=request.session_id, sample_rate_hz=24000,
            chunk_index=1, is_last=True, ts_ms=0)


class DyingStt:
    """Ends the Transcribe stream on its own (e.g. STT aborted after a
    backlog overflow) without the edge ever sending SESSION_CLOSE."""
    async def Transcribe(self, request_iter):
        async def drain():
            async for _ in request_iter:
                pass
        task = asyncio.create_task(drain())
        try:
            await asyncio.sleep(0.05)
            raise RuntimeError("stt backlog overflow")
        finally:
            task.cancel()
        yield  # pragma: no cover - makes this an async generator


def _controls(events, ctype):
    return [e.control for e in events
            if _kind(e) == "control" and e.control.type == ctype]


async def _run_one_turn(llm, tts, stt=None):
    """Drive one complete turn end to end and return every ServerEvent the edge
    would have seen, in order."""
    sid = "s"
    inbox: asyncio.Queue = asyncio.Queue()

    async def edge_events():
        while True:
            ev = await inbox.get()
            if ev is None:
                return
            yield ev

    def ctl(t):
        return edge_pb2.ClientEvent(control=edge_pb2.Control(session_id=sid, type=t, ts_ms=1))

    sessions = RecordingSessions()
    stream = MediaBridge(stt or FakeStt("hello"), llm, sessions, tts=tts).Stream(edge_events(), None)
    await inbox.put(ctl(edge_pb2.Control.SESSION_OPEN))
    await inbox.put(ctl(edge_pb2.Control.END_TURN))

    events = []

    async def collect():
        while True:
            try:
                ev = await stream.__anext__()
            except StopAsyncIteration:
                return
            events.append(ev)
            if _kind(ev) == "control" and ev.control.type == edge_pb2.Control.AGENT_TURN_ENDED:
                await inbox.put(ctl(edge_pb2.Control.SESSION_CLOSE))
                await inbox.put(None)

    await asyncio.wait_for(collect(), timeout=5)
    return events, sessions


async def _run_one_turn_gated(llm, tts, gate, release_after_s):
    """Like _run_one_turn, but the turn is driven concurrently with a delayed
    gate release, so a TTS fake can block on `gate` to simulate backpressure
    without deadlocking the test."""
    sid = "s"
    inbox: asyncio.Queue = asyncio.Queue()

    async def edge_events():
        while True:
            ev = await inbox.get()
            if ev is None:
                return
            yield ev

    def ctl(t):
        return edge_pb2.ClientEvent(control=edge_pb2.Control(session_id=sid, type=t, ts_ms=1))

    sessions = RecordingSessions()
    stream = MediaBridge(FakeStt("hello"), llm, sessions, tts=tts).Stream(edge_events(), None)
    await inbox.put(ctl(edge_pb2.Control.SESSION_OPEN))
    await inbox.put(ctl(edge_pb2.Control.END_TURN))

    events = []

    async def collect():
        while True:
            try:
                ev = await stream.__anext__()
            except StopAsyncIteration:
                return
            events.append(ev)
            if _kind(ev) == "control" and ev.control.type == edge_pb2.Control.AGENT_TURN_ENDED:
                await inbox.put(ctl(edge_pb2.Control.SESSION_CLOSE))
                await inbox.put(None)

    task = asyncio.create_task(collect())
    await asyncio.sleep(release_after_s)
    gate.set()
    await asyncio.wait_for(task, timeout=5)
    return events, sessions


async def test_turn_started_and_ended_bracket_the_reply():
    events, _ = await _run_one_turn(StreamingLlm(), FakeTts())
    kinds = [(_kind(e), e.control.type if _kind(e) == "control" else None) for e in events]

    started = [i for i, k in enumerate(kinds) if k == ("control", edge_pb2.Control.AGENT_TURN_STARTED)]
    ended = [i for i, k in enumerate(kinds) if k == ("control", edge_pb2.Control.AGENT_TURN_ENDED)]
    assert len(started) == 1 and len(ended) == 1, kinds
    audio = [i for i, k in enumerate(kinds) if k[0] == "audio"]
    assert audio, "the turn should have produced audio"
    # The whole turn is bracketed: nothing for it escapes before STARTED or
    # after ENDED, which is exactly the guarantee the client's state machine
    # rests on.
    assert started[0] < min(audio) and max(audio) < ended[0], kinds
    reply = [i for i, k in enumerate(kinds) if k[0] == "reply"]
    assert reply and started[0] < reply[0] < ended[0], kinds

    end = _controls(events, edge_pb2.Control.AGENT_TURN_ENDED)[0]
    assert end.metadata["reason"] == "complete", end.metadata
    print("ok turn_started_and_ended_bracket_the_reply")


async def test_every_turn_event_is_stamped_with_its_turn_id():
    events, _ = await _run_one_turn(StreamingLlm(), FakeTts())
    turn_id = _controls(events, edge_pb2.Control.AGENT_TURN_STARTED)[0].turn_id
    assert turn_id, "AGENT_TURN_STARTED must carry a non-zero turn id"

    assert _controls(events, edge_pb2.Control.AGENT_TURN_ENDED)[0].turn_id == turn_id
    for e in events:
        if _kind(e) == "audio":
            assert e.audio.turn_id == turn_id, (e.audio.turn_id, turn_id)
        elif _kind(e) == "reply":
            assert e.reply.turn_id == turn_id, (e.reply.turn_id, turn_id)
    print("ok every_turn_event_is_stamped_with_its_turn_id")


async def test_failed_turn_still_ends():
    """The failure case is the whole point: a turn that produces no audio and no
    text must still close, or the client waits on a turn that will never speak."""
    events, sessions = await _run_one_turn(BrokenLlm(), FakeTts())
    assert not any(_kind(e) == "audio" for e in events), "no audio expected"
    assert _controls(events, edge_pb2.Control.AGENT_TURN_STARTED), "turn never announced"

    ended = _controls(events, edge_pb2.Control.AGENT_TURN_ENDED)
    assert len(ended) == 1, [e.metadata for e in ended]
    assert ended[0].metadata["reason"] == "error", ended[0].metadata
    # The user turn is still recorded even though the agent had nothing to say.
    assert sessions.turns and sessions.turns[0]["assistant_text"] is None, sessions.turns
    print("ok failed_turn_still_ends")


async def test_turn_with_total_tts_failure_is_not_announced_complete():
    """The LLM succeeding is not enough to call a turn complete: if every TTS
    clause fails, the user heard nothing, so the turn must be announced and
    persisted as a failure -- not silently reported as a heard answer."""
    events, sessions = await _run_one_turn(StreamingLlm(), FailingTts())
    assert not any(_kind(e) == "audio" for e in events), "no audio should have reached the edge"
    # The reply text is still relayed to the client UI -- that channel doesn't
    # depend on TTS -- but the turn's own outcome must reflect the failure.
    assert any(_kind(e) == "reply" for e in events)

    ended = _controls(events, edge_pb2.Control.AGENT_TURN_ENDED)
    assert len(ended) == 1, [e.metadata for e in ended]
    assert ended[0].metadata["reason"] == "error", ended[0].metadata
    assert sessions.turns and sessions.turns[0]["assistant_text"] is None, sessions.turns
    assert sessions.turns[0]["delivery"] == "failed", sessions.turns
    print("ok turn_with_total_tts_failure_is_not_announced_complete")


async def test_partial_tts_delivery_is_not_classified_as_total_failure():
    """A clause that streamed some audio before its RPC failed mid-stream
    still reached the user -- it must not be folded into "nothing was
    delivered" just because no single clause completed cleanly."""
    tts = PartialFailTts()
    events, sessions = await _run_one_turn(StreamingLlm(), tts)
    audio = [e for e in events if _kind(e) == "audio"]
    assert audio, "the partially-streamed clause's audio must still reach the edge"

    ended = _controls(events, edge_pb2.Control.AGENT_TURN_ENDED)
    assert len(ended) == 1, [e.metadata for e in ended]
    assert ended[0].metadata["reason"] == "complete", ended[0].metadata
    assert sessions.turns and sessions.turns[0]["delivery"] == "partial", sessions.turns
    print("ok partial_tts_delivery_is_not_classified_as_total_failure")


async def test_malformed_only_tts_stream_is_not_marked_success():
    """A stream that completes normally but never returns a single valid
    chunk must not be reported as "success", and the attempt must still be
    counted in orch_tts_calls_total even though nothing was ever sent."""
    tts = AllMalformedTts()
    calls_before = server_module.M_TTS_CALLS._value.get()
    events, sessions = await _run_one_turn(StreamingLlm(), tts)
    assert not any(_kind(e) == "audio" for e in events), "malformed chunks must never be relayed"

    ended = _controls(events, edge_pb2.Control.AGENT_TURN_ENDED)
    assert ended[0].metadata["reason"] == "error", ended[0].metadata
    assert sessions.turns and sessions.turns[0]["assistant_text"] is None, sessions.turns
    assert sessions.turns[0]["delivery"] == "failed", sessions.turns
    assert server_module.M_TTS_CALLS._value.get() == calls_before + 1, (
        "the RPC attempt must be counted even though it produced no audio")
    print("ok malformed_only_tts_stream_is_not_marked_success")


async def test_speech_started_and_barge_in_labels_are_distinct():
    """Every USER_SPEECH_STARTED counts as a speech-start; only one that
    actually cuts off a live agent turn counts as a barge-in."""
    sid = "s"
    inbox: asyncio.Queue = asyncio.Queue()

    async def edge_events():
        while True:
            ev = await inbox.get()
            if ev is None:
                return
            yield ev

    def ctl(t):
        return edge_pb2.ClientEvent(control=edge_pb2.Control(session_id=sid, type=t, ts_ms=1))

    speech_before = server_module.M_CONTROL.labels("speech_started")._value.get()
    barge_before = server_module.M_CONTROL.labels("barge_in")._value.get()

    stream = MediaBridge(FakeStt("hi"), StreamingLlm(), RecordingSessions(),
                         tts=FakeTts()).Stream(edge_events(), None)
    await inbox.put(ctl(edge_pb2.Control.SESSION_OPEN))
    # No agent turn is active yet -- this must count as a speech-start but
    # must NOT count as a barge-in.
    await inbox.put(ctl(edge_pb2.Control.USER_SPEECH_STARTED))
    await inbox.put(ctl(edge_pb2.Control.SESSION_CLOSE))
    await inbox.put(None)
    async for _ in stream:
        pass

    assert server_module.M_CONTROL.labels("speech_started")._value.get() == speech_before + 1
    assert server_module.M_CONTROL.labels("barge_in")._value.get() == barge_before, (
        "speech-start with no active agent turn must not count as a barge-in")
    print("ok speech_started_and_barge_in_labels_are_distinct")


async def test_llm_latency_excludes_time_blocked_on_tts_backpressure():
    """produce() can stall on the bounded clause queue when TTS is slow. The
    recorded llm_proc_ms must reflect only the LLM's own streaming time, not
    however long it sat blocked handing clauses to a contended TTS consumer."""
    gate = asyncio.Event()
    tts = SlowFirstClauseTts(gate)
    events, sessions = await _run_one_turn_gated(ManyClausesLlm(), tts, gate, release_after_s=0.2)
    assert sessions.turns, "turn must have been recorded"
    llm_proc_ms = sessions.turns[-1]["llm_proc_ms"]
    assert llm_proc_ms is not None and llm_proc_ms < 100, (
        f"llm_proc_ms={llm_proc_ms} should reflect only LLM stream time, "
        "not the ~200ms the TTS consumer was gated")
    print("ok llm_latency_excludes_time_blocked_on_tts_backpressure")


async def test_session_closes_when_stt_stream_dies_unexpectedly():
    """If STT ends the stream on its own (e.g. it aborted the RPC after a
    backlog overflow) without the edge ever sending SESSION_CLOSE, the
    orchestrator must tear the session down itself instead of continuing to
    accept audio into a dead STT pipe."""
    sid = "s"
    inbox: asyncio.Queue = asyncio.Queue()

    async def edge_events():
        while True:
            ev = await inbox.get()
            if ev is None:
                return
            yield ev

    def ctl(t):
        return edge_pb2.ClientEvent(control=edge_pb2.Control(session_id=sid, type=t, ts_ms=1))

    stream = MediaBridge(DyingStt(), StreamingLlm(), RecordingSessions(),
                         tts=FakeTts()).Stream(edge_events(), None)
    await inbox.put(ctl(edge_pb2.Control.SESSION_OPEN))
    # Deliberately no SESSION_CLOSE -- the session must still end on its own.

    async def collect():
        out = []
        async for ev in stream:
            out.append(ev)
        return out

    await asyncio.wait_for(collect(), timeout=5)
    print("ok session_closes_when_stt_stream_dies_unexpectedly")


async def test_turn_deadline_expired_during_stt_rejects_without_calling_llm():
    """The deadline clock starts at END_TURN (turn commitment) and never
    resets, so time STT itself spends counts against it. If STT takes long
    enough to exhaust the whole budget, the turn must be rejected before ever
    reaching the LLM -- "reject aged STT results before starting the LLM" --
    not silently processed late."""
    orig = server_module.ORCH_TURN_DEADLINE_S
    server_module.ORCH_TURN_DEADLINE_S = 0.1
    try:
        gate = asyncio.Event()
        stt = DelayedTurnCompleteStt("hello", gate)
        llm = StreamingLlm()

        task = asyncio.create_task(_run_one_turn(llm, FakeTts(), stt=stt))
        await asyncio.sleep(0.3)  # exceed the 0.1s deadline while STT is still "processing"
        gate.set()
        events, sessions = await asyncio.wait_for(task, timeout=5)

        assert not any(_kind(e) == "audio" for e in events), "an expired turn must produce no audio"
        assert not any(_kind(e) == "reply" for e in events), "an expired turn must never reach the LLM"
        ended = _controls(events, edge_pb2.Control.AGENT_TURN_ENDED)
        assert len(ended) == 1 and ended[0].metadata["reason"] == "error", ended
        assert sessions.turns and sessions.turns[-1]["delivery"] == "none", sessions.turns
        assert sessions.turns[-1]["assistant_text"] is None, sessions.turns
        assert server_module.M_TURNS_ACTIVE.collect()[0].samples[0].value == 0, "no leaked turn slot"
    finally:
        server_module.ORCH_TURN_DEADLINE_S = orig
    print("ok turn_deadline_expired_during_stt_rejects_without_calling_llm")


async def test_llm_and_tts_calls_receive_the_remaining_deadline_budget():
    """Each downstream RPC's own timeout must be capped by the turn's
    remaining deadline budget, recomputed fresh per call -- not the full
    static TTS_TIMEOUT_S/unbounded LLM call every time."""
    orig = server_module.ORCH_TURN_DEADLINE_S
    server_module.ORCH_TURN_DEADLINE_S = 5.0
    try:
        llm = TimeoutRecordingLlm()
        tts = TimeoutRecordingTts()
        events, _ = await _run_one_turn(llm, tts)
        assert any(_kind(e) == "audio" for e in events), "the turn should still complete normally"

        assert llm.timeouts and llm.timeouts[0] is not None, llm.timeouts
        assert 0 < llm.timeouts[0] <= 5.0, llm.timeouts

        assert tts.timeouts and all(t is not None for t in tts.timeouts), tts.timeouts
        # Capped by the remaining budget, never by more than the static
        # per-clause ceiling (TTS_TIMEOUT_S).
        assert all(0 < t <= server_module.TTS_TIMEOUT_S for t in tts.timeouts), tts.timeouts
        assert all(t <= 5.0 for t in tts.timeouts), tts.timeouts
    finally:
        server_module.ORCH_TURN_DEADLINE_S = orig
    print("ok llm_and_tts_calls_receive_the_remaining_deadline_budget")


async def test_barge_in_ends_the_turn_as_interrupted():
    """A cancelled turn's AUDIO is dropped by the epoch guard, but its ENDED must
    not be: it is the client's only cue that the turn is over."""
    sid = "s"
    gate = asyncio.Event()
    from tests.test_interrupt import FakeLlm

    inbox: asyncio.Queue = asyncio.Queue()

    async def edge_events():
        while True:
            ev = await inbox.get()
            if ev is None:
                return
            yield ev

    def ctl(t):
        return edge_pb2.ClientEvent(control=edge_pb2.Control(session_id=sid, type=t, ts_ms=1))

    stream = MediaBridge(FakeStt("story"), FakeLlm(gate), RecordingSessions(),
                         tts=FakeTts()).Stream(edge_events(), None)
    await inbox.put(ctl(edge_pb2.Control.SESSION_OPEN))
    await inbox.put(ctl(edge_pb2.Control.END_TURN))

    events = []

    async def pull_until_audio():
        while True:
            ev = await stream.__anext__()
            events.append(ev)
            if _kind(ev) == "audio":
                return
    await asyncio.wait_for(pull_until_audio(), timeout=5)

    await inbox.put(ctl(edge_pb2.Control.USER_SPEECH_STARTED))
    gate.set()

    async def collect():
        while True:
            try:
                ev = await stream.__anext__()
            except StopAsyncIteration:
                return
            events.append(ev)
            if _kind(ev) == "control" and ev.control.type == edge_pb2.Control.AGENT_TURN_ENDED:
                await inbox.put(ctl(edge_pb2.Control.SESSION_CLOSE))
                await inbox.put(None)
    await asyncio.wait_for(collect(), timeout=5)

    turn_id = _controls(events, edge_pb2.Control.AGENT_TURN_STARTED)[0].turn_id
    stop = _controls(events, edge_pb2.Control.STOP_PLAYBACK)
    assert stop and stop[0].turn_id == turn_id, "STOP_PLAYBACK must name the turn it cut"

    ended = _controls(events, edge_pb2.Control.AGENT_TURN_ENDED)
    assert len(ended) == 1, [e.metadata for e in ended]
    assert ended[0].metadata["reason"] == "interrupted", ended[0].metadata
    assert ended[0].turn_id == turn_id, (ended[0].turn_id, turn_id)
    print("ok barge_in_ends_the_turn_as_interrupted")
