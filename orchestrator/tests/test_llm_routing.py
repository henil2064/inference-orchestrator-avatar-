"""Self-check for STT-segment -> LLM routing. Run: python -m tests.test_llm_routing

Covers: (1) mid-turn segment texts are joined, in order, into the full turn text
sent to the LLM exactly once when turn_complete=True arrives; (2) the reply and
its proc_ms land in the completed Redis turn, never in mid-turn segment events;
(3) Redis and active-session history is attached as structured messages; (4)
the environment turn cap is honored; (5) LLM_ENABLED=0 skips the call entirely;
(6) a failing LLM call doesn't crash the session or drop the completed Redis
turn — only the reply is missing; (7) a stream is pinned to one session and
foreign control, audio, and STT transcript data is dropped.
"""
import asyncio
import json

from proto import asr_pb2, edge_pb2, llm_pb2
from service import server
from service.server import MediaBridge
from service.session_logger import SessionLogger


class FakeSttStub:
    """Yields a fixed sequence of Transcripts, ignoring what's actually forwarded
    to it (mirrors test_endpoint.py's FakeStub philosophy: this test cares about
    what MediaBridge does with STT's *output*, not what it sends STT)."""
    def __init__(self, transcripts, release=None, end_turn_seen=None):
        self._transcripts = transcripts
        self._release = release
        self._end_turn_seen = end_turn_seen
        self.received = []

    async def Transcribe(self, request_iter):
        request_seen = asyncio.Event()

        async def drain():
            async for event in request_iter:
                self.received.append(event)
                request_seen.set()
                if (self._end_turn_seen is not None and event.WhichOneof("msg") == "control"
                        and event.control.type == asr_pb2.Control.END_TURN):
                    self._end_turn_seen.set()
        drain_task = asyncio.create_task(drain())
        try:
            # Real STT cannot emit before it receives SESSION_OPEN/audio. Keeping
            # the fake ordered prevents a test-only race with Redis history load.
            await request_seen.wait()
            if self._release is not None:
                await self._release.wait()
            for t in self._transcripts:
                yield t
        finally:
            drain_task.cancel()


class FakeLlmStub:
    def __init__(self, reply_text="ack", proc_ms=42, fail=False):
        self.calls = []
        self.reply_text = reply_text
        self.proc_ms = proc_ms
        self.fail = fail

    async def Generate(self, request):
        self.calls.append(request)
        if self.fail:
            raise RuntimeError("boom")
        return llm_pb2.GenerateResponse(session_id=request.session_id, text=self.reply_text,
                                        ts_ms=0, proc_ms=self.proc_ms,
                                        profile=llm_pb2.InferenceProfile(
                                            model="test-model", engine="test-engine", device="cpu",
                                            input_tokens=3, output_tokens=2, ttft_ms=4, decode_ms=5))

    async def GenerateStream(self, request, timeout=None):
        self.calls.append(request)
        if self.fail:
            raise RuntimeError("boom")
        yield llm_pb2.GenerateChunk(session_id=request.session_id, text_delta=self.reply_text)
        yield llm_pb2.GenerateChunk(session_id=request.session_id, is_final=True,
            profile=llm_pb2.InferenceProfile(model="test-model", engine="test-engine", device="cpu",
                input_tokens=3, output_tokens=2, ttft_ms=4, decode_ms=5))


class FakeSessions:
    """Records the SessionStore calls that production sends to Redis."""
    def __init__(self, history=None):
        self.opens = []
        self.events = []
        self.turns = []
        self.closes = []
        self.history = list(history or [])
        self.history_limits = []

    async def open(self, session_id, ts_ms):
        self.opens.append((session_id, ts_ms))

    async def event(self, session_id, event_type, **fields):
        self.events.append((session_id, event_type, fields))

    async def record_turn(self, session_id, user_text, assistant_text,
                          stt_proc_ms, llm_proc_ms, interrupted=False, delivery="unknown"):
        self.turns.append({
            "session_id": session_id,
            "user_text": user_text,
            "assistant_text": assistant_text,
            "stt_proc_ms": stt_proc_ms,
            "llm_proc_ms": llm_proc_ms,
            "interrupted": interrupted,
            "delivery": delivery,
        })

    async def recent_turns(self, session_id, limit):
        self.history_limits.append((session_id, limit))
        return self.history[-limit:]

    async def close(self, session_id):
        self.closes.append(session_id)


async def edge_events(sid="t1"):
    yield edge_pb2.ClientEvent(control=edge_pb2.Control(
        session_id=sid, type=edge_pb2.Control.SESSION_OPEN, ts_ms=1))
    yield edge_pb2.ClientEvent(control=edge_pb2.Control(
        session_id=sid, type=edge_pb2.Control.SESSION_CLOSE, ts_ms=2))


async def overlapping_turn_events(sid, end_turn_seen, release):
    """Start turn 2 before STT returns turn 1's final transcript."""
    yield edge_pb2.ClientEvent(control=edge_pb2.Control(
        session_id=sid, type=edge_pb2.Control.SESSION_OPEN, ts_ms=1))
    yield edge_pb2.ClientEvent(control=edge_pb2.Control(
        session_id=sid, type=edge_pb2.Control.USER_SPEECH_STARTED, ts_ms=2))
    yield edge_pb2.ClientEvent(control=edge_pb2.Control(
        session_id=sid, type=edge_pb2.Control.END_TURN, ts_ms=3))
    await end_turn_seen.wait()
    yield edge_pb2.ClientEvent(control=edge_pb2.Control(
        session_id=sid, type=edge_pb2.Control.USER_SPEECH_STARTED, ts_ms=4))
    # Let the bridge consume the queued controls while FakeStt remains blocked.
    for _ in range(3):
        await asyncio.sleep(0)
    release.set()
    yield edge_pb2.ClientEvent(control=edge_pb2.Control(
        session_id=sid, type=edge_pb2.Control.SESSION_CLOSE, ts_ms=5))


def _transcripts(sid):
    return [
        asr_pb2.Transcript(session_id=sid, text="hel", turn_complete=False,
                           segment_index=0, ts_ms=10, proc_ms=50),
        asr_pb2.Transcript(session_id=sid, text="lo, how are you", turn_complete=True,
                           segment_index=1, ts_ms=20, proc_ms=80),
    ]


async def _run(bridge, events):
    """Drain one session and return only the AgentReply events. The stream also
    carries agent-turn lifecycle controls (AGENT_TURN_STARTED/ENDED); those are
    covered in tests/test_turn_signals.py and are noise here."""
    return [event async for event in bridge.Stream(events, None)
            if event.WhichOneof("msg") == "reply"]


async def test_joins_segments_and_persists_llm_reply():
    sid = "t1"
    stt = FakeSttStub(_transcripts(sid))
    llm = FakeLlmStub(reply_text="ack", proc_ms=42)
    sessions = FakeSessions(history=[{
        "user_text": "What is my destination?",
        "assistant_text": "Pune.",
    }])
    replies = await _run(MediaBridge(stt, llm, sessions), edge_events(sid))

    assert len(llm.calls) == 1, llm.calls
    assert llm.calls[0].text == "hel lo, how are you", llm.calls[0].text
    assert [(message.role, message.content) for message in llm.calls[0].messages] == [
        (llm_pb2.ChatMessage.USER, "What is my destination?"),
        (llm_pb2.ChatMessage.ASSISTANT, "Pune."),
        (llm_pb2.ChatMessage.USER, "hel lo, how are you"),
    ]
    assert sessions.history_limits == [(sid, server.LLM_HISTORY_MAX_TURNS)]

    segments = [fields for _, kind, fields in sessions.events
                if kind == "transcript_segment"]
    assert len(segments) == 2
    assert segments[0]["turn_complete"] is False
    assert segments[1]["turn_complete"] is True
    assert len(sessions.turns) == 1, sessions.turns
    turn = sessions.turns[0]
    assert turn["session_id"] == sid
    assert turn["user_text"] == "hel lo, how are you"
    assert turn["assistant_text"] == "ack"
    assert turn["stt_proc_ms"] == 80
    assert turn["interrupted"] is False
    # llm_proc_ms is the orchestrator's measured round-trip (streaming carries no
    # single service proc_ms), so assert its type, not an exact wall-clock value.
    assert isinstance(turn["llm_proc_ms"], int) and turn["llm_proc_ms"] >= 0
    assert sessions.closes == [sid]
    assert len(replies) == 1 and replies[0].reply.text == "ack"


async def test_llm_disabled_skips_the_call():
    sid = "t2"
    stt = FakeSttStub(_transcripts(sid))
    llm = FakeLlmStub()
    sessions = FakeSessions()
    server.LLM_ENABLED = False
    try:
        replies = await _run(MediaBridge(stt, llm, sessions), edge_events(sid))
    finally:
        server.LLM_ENABLED = True  # restore for later tests/importers
    assert llm.calls == []
    assert replies == []
    assert sessions.turns[0]["assistant_text"] is None
    assert sessions.turns[0]["llm_proc_ms"] is None


async def test_llm_failure_does_not_drop_the_transcript():
    sid = "t3"
    stt = FakeSttStub(_transcripts(sid))
    llm = FakeLlmStub(fail=True)
    sessions = FakeSessions()
    replies = await _run(MediaBridge(stt, llm, sessions), edge_events(sid))  # must not raise

    assert len(llm.calls) == 1  # it was attempted
    # The completed user turn is still persisted; only the LLM fields are absent.
    assert sessions.turns[0]["user_text"] == "hel lo, how are you"
    assert sessions.turns[0]["assistant_text"] is None
    assert sessions.turns[0]["llm_proc_ms"] is None
    assert replies == []


async def test_completed_turn_is_cached_for_the_next_llm_call():
    sid = "t4"
    transcripts = [
        asr_pb2.Transcript(session_id=sid, text="first question", turn_complete=True,
                           segment_index=0, ts_ms=10, proc_ms=20),
        asr_pb2.Transcript(session_id=sid, text="second question", turn_complete=True,
                           segment_index=0, ts_ms=20, proc_ms=20),
    ]
    stt = FakeSttStub(transcripts)
    llm = FakeLlmStub(reply_text="answer", proc_ms=10)
    sessions = FakeSessions()
    await _run(MediaBridge(stt, llm, sessions), edge_events(sid))

    assert len(llm.calls) == 2
    assert [(message.role, message.content) for message in llm.calls[1].messages] == [
        (llm_pb2.ChatMessage.USER, "first question"),
        (llm_pb2.ChatMessage.ASSISTANT, "answer"),
        (llm_pb2.ChatMessage.USER, "second question"),
    ]


async def test_history_turn_cap_limits_the_redis_context():
    sid = "t5"
    sessions = FakeSessions(history=[
        {"user_text": "too old", "assistant_text": "old answer"},
        {"user_text": "most recent", "assistant_text": "recent answer"},
    ])
    llm = FakeLlmStub()
    previous_limit = server.LLM_HISTORY_MAX_TURNS
    server.LLM_HISTORY_MAX_TURNS = 1
    try:
        await _run(MediaBridge(FakeSttStub(_transcripts(sid)), llm, sessions), edge_events(sid))
    finally:
        server.LLM_HISTORY_MAX_TURNS = previous_limit

    assert sessions.history_limits == [(sid, 1)]
    assert [message.content for message in llm.calls[0].messages] == [
        "most recent", "recent answer", "hel lo, how are you",
    ]


async def test_foreign_session_data_is_never_attached_to_this_stream():
    sid = "trusted"
    foreign_sid = "foreign"

    async def events():
        yield edge_pb2.ClientEvent(control=edge_pb2.Control(
            session_id=sid, type=edge_pb2.Control.SESSION_OPEN, ts_ms=1))
        # A second open used to overwrite the stream's session_id and reload
        # another session's Redis history. It must now be ignored.
        yield edge_pb2.ClientEvent(control=edge_pb2.Control(
            session_id=foreign_sid, type=edge_pb2.Control.SESSION_OPEN, ts_ms=2))
        yield edge_pb2.ClientEvent(audio=edge_pb2.AudioFrame(
            session_id=foreign_sid, pcm16=b"foreign audio", ts_ms=3))
        yield edge_pb2.ClientEvent(control=edge_pb2.Control(
            session_id=sid, type=edge_pb2.Control.SESSION_CLOSE, ts_ms=4))

    stt = FakeSttStub(_transcripts(foreign_sid))
    llm = FakeLlmStub()
    sessions = FakeSessions(history=[{"user_text": "private", "assistant_text": "reply"}])
    replies = await _run(MediaBridge(stt, llm, sessions), events())

    assert sessions.opens == [(sid, 1)]
    assert sessions.history_limits == [(sid, server.LLM_HISTORY_MAX_TURNS)]
    assert not llm.calls
    assert not sessions.turns
    assert not replies
    assert all(event.WhichOneof("msg") != "audio" for event in stt.received)


async def test_clickhouse_trace_contains_final_text_and_service_profiles():
    sid = "telemetry"
    inserted = []

    def sink(table, rows):
        inserted.extend((table, row) for row in rows)

    telemetry = SessionLogger("http://unused", batch_size=100, insert_rows=sink)
    await telemetry.start()
    stt = FakeSttStub(_transcripts(sid))
    await _run(MediaBridge(stt, FakeLlmStub(), FakeSessions(), telemetry=telemetry), edge_events(sid))
    await telemetry.close()

    events = [row for table, row in inserted if table == "pipeline_events"]
    by_type = {row["event_type"]: row for row in events}
    assert json.loads(by_type["stt.completed"]["payload_json"])["transcript"] == "hel lo, how are you"
    assert "transcript" not in json.loads(by_type["stt.segment_profile"]["payload_json"])
    llm_profile = json.loads(by_type["llm.completed"]["payload_json"])["profile"]
    assert llm_profile["model"] == "test-model" and llm_profile["ttft_ms"] == 4
    typed_llm = by_type["llm.completed"]
    assert typed_llm["model"] == "test-model" and typed_llm["ttft_ms"] == 4
    assert typed_llm["input_tokens"] == 3 and typed_llm["output_tokens"] == 2
    typed_stt = by_type["stt.segment_profile"]
    assert typed_stt["segment_index"] == 1 and typed_stt["turn_complete"] is True
    assert any(table == "session_summaries" and row["session_id"] == sid for table, row in inserted)


async def test_stt_results_keep_their_original_turn_when_next_turn_starts():
    sid = "turn-attribution"
    inserted = []
    release = asyncio.Event()
    end_turn_seen = asyncio.Event()

    def sink(table, rows):
        inserted.extend((table, row) for row in rows)

    telemetry = SessionLogger("http://unused", batch_size=100, insert_rows=sink)
    await telemetry.start()
    transcripts = [
        asr_pb2.Transcript(session_id=sid, text="first", turn_complete=True,
                           segment_index=0, ts_ms=10, proc_ms=10),
        asr_pb2.Transcript(session_id=sid, text="second", turn_complete=True,
                           segment_index=0, ts_ms=20, proc_ms=10),
    ]
    llm = FakeLlmStub()
    sessions = FakeSessions()
    bridge = MediaBridge(FakeSttStub(transcripts, release, end_turn_seen), llm, sessions,
                         telemetry=telemetry)
    await _run(bridge, overlapping_turn_events(sid, end_turn_seen, release))
    await telemetry.close()

    events = [row for table, row in inserted if table == "pipeline_events"]
    stt_turns = [row["turn_id"] for row in events if row["event_type"] == "stt.completed"]
    llm_turns = [row["turn_id"] for row in events if row["event_type"] == "llm.completed"]
    # Both utterances are transcribed and attributed to their own turn...
    assert stt_turns == [1, 2], stt_turns
    # ...but turn 1's transcript only lands after turn 2's speech has already
    # started, so the user has moved on: it is recorded, never answered.
    superseded = [row["turn_id"] for row in events if row["event_type"] == "turn.superseded"]
    assert superseded == [1], superseded
    assert llm_turns == [2], llm_turns
    # Nothing the user said is lost: turn 1 is persisted with no reply, and its
    # text is carried into the prompt of the turn that actually gets answered.
    assert [(t["user_text"], t["assistant_text"]) for t in sessions.turns] == [
        ("first", None), ("second", "ack"),
    ], sessions.turns
    assert [message.content for message in llm.calls[0].messages] == ["first\nsecond"]


class FlakyLlmStub(FakeLlmStub):
    """Fails only the calls whose index is in ``fail_calls`` (0-based)."""
    def __init__(self, fail_calls, **kwargs):
        super().__init__(**kwargs)
        self._fail_calls = set(fail_calls)

    async def GenerateStream(self, request, timeout=None):
        index = len(self.calls)
        self.calls.append(request)
        if index in self._fail_calls:
            raise RuntimeError("boom")
        yield llm_pb2.GenerateChunk(session_id=request.session_id, text_delta=self.reply_text)
        yield llm_pb2.GenerateChunk(
            session_id=request.session_id, is_final=True,
            profile=llm_pb2.InferenceProfile(model="test-model", engine="test-engine",
                                             device="cpu", input_tokens=3, output_tokens=2,
                                             ttft_ms=4, decode_ms=5))


async def test_unanswered_turn_is_merged_forward_not_dropped():
    """A turn the agent never answered (barge-in cut it before any audio, or the
    LLM failed) still holds something the user said. It must reach the next
    prompt, merged into the following user message so the alternating
    USER/ASSISTANT contract the LLM worker validates still holds."""
    sid = "t6"
    transcripts = [
        asr_pb2.Transcript(session_id=sid, text="first question", turn_complete=True,
                           segment_index=0, ts_ms=10, proc_ms=20),
        asr_pb2.Transcript(session_id=sid, text="second question", turn_complete=True,
                           segment_index=0, ts_ms=20, proc_ms=20),
    ]
    llm = FlakyLlmStub([0], reply_text="answer")
    sessions = FakeSessions()
    await _run(MediaBridge(FakeSttStub(transcripts), llm, sessions), edge_events(sid))

    assert len(llm.calls) == 2
    # Turn 1 produced no reply, so Redis holds it with assistant_text=None...
    assert sessions.turns[0]["user_text"] == "first question"
    assert sessions.turns[0]["assistant_text"] is None
    # ...and turn 2's prompt still carries what the user said in turn 1.
    assert [(message.role, message.content) for message in llm.calls[1].messages] == [
        (llm_pb2.ChatMessage.USER, "first question\nsecond question"),
    ]


async def test_unanswered_redis_turn_is_merged_forward():
    """Same rule for history loaded from Redis at session open: an interrupted
    turn persisted with a null assistant_text is context, not garbage."""
    sid = "t7"
    sessions = FakeSessions(history=[
        {"user_text": "book me a cab", "assistant_text": "Sure, from where?"},
        {"user_text": "actually wait", "assistant_text": None},  # barged in before any audio
    ])
    llm = FakeLlmStub()
    await _run(MediaBridge(FakeSttStub(_transcripts(sid)), llm, sessions), edge_events(sid))

    assert [(message.role, message.content) for message in llm.calls[0].messages] == [
        (llm_pb2.ChatMessage.USER, "book me a cab"),
        (llm_pb2.ChatMessage.ASSISTANT, "Sure, from where?"),
        (llm_pb2.ChatMessage.USER, "actually wait\nhel lo, how are you"),
    ]
