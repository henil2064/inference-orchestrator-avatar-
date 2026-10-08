"""Self-checks for the interrupt controller and LLM->TTS clause pipelining.
Run: python -m tests.test_interrupt

Covers the barge-in correctness contract (PRD R4):
  1. extract_clauses splits sentences and force-flushes long punctuation-free runs.
  2. On USER_SPEECH_STARTED mid-reply the in-flight turn is cancelled, a
     STOP_PLAYBACK is sent to the edge, and NO further audio from the cancelled
     turn is relayed (epoch drop).
  3. The interrupted turn is persisted with interrupted=True and the spoken prefix.

These use in-memory fakes for STT/LLM/TTS driven through MediaBridge.Stream, the
same seam the routing tests use.
"""
import asyncio

from proto import asr_pb2, edge_pb2, llm_pb2, tts_pb2
from service.server import MediaBridge, extract_clauses


# ---------------------------------------------------------------------------
# 1. Pure clause splitter
# ---------------------------------------------------------------------------
def test_extract_clauses():
    clauses, rest = extract_clauses("Hello there. How are", 240)
    assert clauses == ["Hello there."], clauses
    assert rest == "How are", rest

    # Decimals/abbreviations don't split (terminal punctuation needs trailing space).
    clauses, rest = extract_clauses("Pi is 3.14 and", 240)
    assert clauses == [] and rest == "Pi is 3.14 and", (clauses, rest)

    clauses, rest = extract_clauses("One. Two! Three? Four", 240)
    assert clauses == ["One.", "Two!", "Three?"] and rest == "Four", (clauses, rest)

    # Long punctuation-free run force-flushes at spaces under max_chars.
    clauses, rest = extract_clauses("aaaa bbbb cccc dddd eeee", 12)
    assert clauses and all(len(c) <= 12 for c in clauses), clauses
    assert " ".join(clauses + [rest]).split() == "aaaa bbbb cccc dddd eeee".split()
    print("ok extract_clauses")


def _stream_clauses(reply: str, max_chars: int = 240, first_clause_chars: int | None = 50) -> list[str]:
    """Simulate the orchestrator's own per-turn wiring: `first_clause_chars`
    is only passed while no clause has been extracted yet this turn, exactly
    like `produce()`'s `first_clause_pending` flag."""
    buf, out = "", []
    pending = first_clause_chars is not None
    for word in reply.split(" "):
        buf += (" " if buf else "") + word
        clauses, buf = extract_clauses(buf, max_chars, first_clause_chars=first_clause_chars if pending else None)
        if clauses:
            pending = False
        out.extend(clauses)
    if buf.strip():
        out.append(buf.strip())
    return out


def test_extract_clauses_first_clause_overlap():
    """`first_clause_chars` lets TTS start before the LLM stream ends, without
    fragmenting replies that never needed it."""
    # A short reply with no sentence punctuation until the very end still
    # arrives as a single clause -- the comma/50-char rule never fires below
    # its own floor, so nothing is lost versus the pre-overlap behavior.
    assert _stream_clauses("Sure, I can help.") == ["Sure, I can help."]

    # A longer reply with no early punctuation gets its first clause force-cut
    # near first_clause_chars (at the last space), well before the full
    # sentence would otherwise arrive as one late clause.
    clauses = _stream_clauses("Fever and throat pain can happen with a viral infection. Drink fluids and rest.")
    assert clauses[0] == "Fever and throat pain can happen with a viral", clauses
    assert " ".join(clauses).split() == (
        "Fever and throat pain can happen with a viral infection. Drink fluids and rest.").split()

    # Only the FIRST clause gets the early-flush treatment; a comma appearing
    # well after it (in what is now the second clause) is not treated specially.
    clauses = _stream_clauses("Your appointment is confirmed for tomorrow morning, please bring your reports.")
    assert clauses[0] == "Your appointment is confirmed for tomorrow", clauses
    assert clauses[1] == "morning, please bring your reports.", clauses

    # first_clause_chars=None reproduces the exact pre-overlap behavior.
    assert _stream_clauses("Sure, I can help.", first_clause_chars=None) == ["Sure, I can help."]
    print("ok extract_clauses_first_clause_overlap")


# ---------------------------------------------------------------------------
# Fakes for the barge-in integration test
# ---------------------------------------------------------------------------
class FakeStt:
    """Emits one turn_complete transcript once the END_TURN control reaches it —
    which only happens after the orchestrator has processed SESSION_OPEN, so the
    stream's session is pinned and the transcript isn't dropped as foreign."""
    def __init__(self, text):
        self._text = text

    async def Transcribe(self, request_iter):
        got_end = asyncio.Event()
        closed = asyncio.Event()

        async def drain():
            async for ev in request_iter:
                if ev.WhichOneof("msg") == "control":
                    if ev.control.type == asr_pb2.Control.END_TURN:
                        got_end.set()
                    elif ev.control.type == asr_pb2.Control.SESSION_CLOSE:
                        closed.set()
        drain_task = asyncio.create_task(drain())
        try:
            await got_end.wait()
            yield asr_pb2.Transcript(session_id="s", text=self._text, turn_complete=True,
                                     segment_index=0, ts_ms=10, proc_ms=20)
            await closed.wait()  # real STT holds the stream open until SESSION_CLOSE
        finally:
            drain_task.cancel()


class FakeLlm:
    """Streams a first clause, then blocks on `gate_second` so the turn stays
    in-flight (waiting on the LLM) after its first clause has been synthesized."""
    def __init__(self, gate_second):
        self.calls = []
        self._gate_second = gate_second

    async def GenerateStream(self, request, timeout=None):
        self.calls.append(request)
        yield llm_pb2.GenerateChunk(session_id=request.session_id, text_delta="First clause. ")
        await self._gate_second.wait()
        yield llm_pb2.GenerateChunk(session_id=request.session_id, text_delta="Second clause.")
        yield llm_pb2.GenerateChunk(session_id=request.session_id, is_final=True,
                                    profile=llm_pb2.InferenceProfile(model="m", engine="e", device="cpu"))


class FakeTts:
    """Yields three 24 kHz PCM frames per clause, then is_last. No holds — the
    clause fully synthesizes so its text counts as spoken."""
    def __init__(self):
        self.requests = []

    async def SynthesizeStream(self, request, timeout=None):
        self.requests.append(request)
        rate = 24000
        frame = (rate // 50) * 2  # 20 ms PCM16
        pcm = b"\x01\x00" * (frame // 2)
        for i in range(3):
            yield tts_pb2.SynthesizeChunk(session_id=request.session_id, pcm16=pcm,
                sample_rate_hz=rate, chunk_index=i, ts_ms=0,
                profile=tts_pb2.InferenceProfile(model="tts", engine="e", device="cpu",
                                                 compute_ms=1, audio_seconds=0.02))
        yield tts_pb2.SynthesizeChunk(session_id=request.session_id, sample_rate_hz=rate,
                                      chunk_index=99, is_last=True, ts_ms=0)


class RecordingSessions:
    def __init__(self):
        self.turns = []
    async def open(self, *a, **k): pass
    async def event(self, *a, **k): pass
    async def record_turn(self, session_id, user_text, assistant_text, stt_proc_ms,
                          llm_proc_ms, interrupted=False, delivery="unknown"):
        self.turns.append({"user_text": user_text, "assistant_text": assistant_text,
                           "interrupted": interrupted, "delivery": delivery,
                           "llm_proc_ms": llm_proc_ms})
    async def recent_turns(self, *a, **k): return []
    async def close(self, *a, **k): pass


def _kind(ev):
    return ev.WhichOneof("msg")


async def test_barge_in_cuts_reply_and_persists_prefix():
    sid = "s"
    gate_second = asyncio.Event()

    stt = FakeStt("tell me a story")
    llm = FakeLlm(gate_second)
    tts = FakeTts()
    sessions = RecordingSessions()
    inbox: asyncio.Queue = asyncio.Queue()

    async def edge_events():
        while True:
            ev = await inbox.get()
            if ev is None:
                return
            yield ev

    stream = MediaBridge(stt, llm, sessions, tts=tts).Stream(edge_events(), None)

    def ctl(t):
        return edge_pb2.ClientEvent(control=edge_pb2.Control(session_id=sid, type=t, ts_ms=1))

    async def nxt():
        return await stream.__anext__()

    await inbox.put(ctl(edge_pb2.Control.SESSION_OPEN))
    await inbox.put(ctl(edge_pb2.Control.END_TURN))
    # END_TURN reaches STT after SESSION_OPEN is processed -> STT emits the turn ->
    # LLM's first clause -> TTS audio.

    # Pull until turn audio is flowing to the edge.
    async def pull_until_audio():
        while True:
            ev = await nxt()
            if _kind(ev) == "audio":
                return
    await asyncio.wait_for(pull_until_audio(), timeout=5)

    # Barge-in mid-reply, then release the LLM's second clause: a naive (uncancelled)
    # implementation would keep streaming turn audio; a correct one drops it.
    await inbox.put(ctl(edge_pb2.Control.USER_SPEECH_STARTED))
    gate_second.set()

    # Collect edge output after the barge-in until the stream ends.
    post = []
    async def collect_post():
        stopped = False
        while True:
            try:
                ev = await nxt()
            except StopAsyncIteration:
                return
            post.append(ev)
            if not stopped and _kind(ev) == "control" and ev.control.type == edge_pb2.Control.STOP_PLAYBACK:
                stopped = True
                await inbox.put(ctl(edge_pb2.Control.SESSION_CLOSE))
                await inbox.put(None)
    await asyncio.wait_for(collect_post(), timeout=5)

    kinds = [(_kind(e), e.control.type if _kind(e) == "control" else None) for e in post]
    assert ("control", edge_pb2.Control.STOP_PLAYBACK) in kinds, kinds
    # Epoch drop: no cancelled-turn audio after STOP_PLAYBACK.
    stop_i = next(i for i, k in enumerate(kinds) if k == ("control", edge_pb2.Control.STOP_PLAYBACK))
    assert all(k[0] != "audio" for k in kinds[stop_i:]), kinds

    interrupted = [t for t in sessions.turns if t["interrupted"]]
    assert len(interrupted) == 1, sessions.turns
    assert interrupted[0]["user_text"] == "tell me a story"
    assert interrupted[0]["assistant_text"] and "First clause." in interrupted[0]["assistant_text"], interrupted
    assert llm.calls, "LLM was called for the turn"
    print("ok barge_in_cuts_reply_and_persists_prefix")


class SlowSessions(RecordingSessions):
    """record_turn blocks until released, so a test can land a second barge-in
    inside the first cancellation's persistence window."""
    def __init__(self):
        super().__init__()
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def record_turn(self, *a, **k):
        self.entered.set()
        await self.release.wait()
        await super().record_turn(*a, **k)


async def test_repeat_barge_in_does_not_lose_the_interrupted_turn():
    """A second USER_SPEECH_STARTED arriving while the first cancellation is
    still persisting must not re-cancel the turn task: that would raise
    CancelledError at the Redis await and lose the user's transcript from both
    Redis and the prompt history."""
    sid = "s"
    stt = FakeStt("tell me a story")
    llm = FakeLlm(asyncio.Event())  # never released; the turn stays in-flight
    sessions = SlowSessions()
    inbox: asyncio.Queue = asyncio.Queue()

    async def edge_events():
        while True:
            ev = await inbox.get()
            if ev is None:
                return
            yield ev

    stream = MediaBridge(stt, llm, sessions, tts=FakeTts()).Stream(edge_events(), None)
    seen = []

    async def drain():
        try:
            async for ev in stream:
                seen.append(ev)
        except StopAsyncIteration:
            pass
    drainer = asyncio.create_task(drain())

    def ctl(t):
        return edge_pb2.ClientEvent(control=edge_pb2.Control(session_id=sid, type=t, ts_ms=1))

    await inbox.put(ctl(edge_pb2.Control.SESSION_OPEN))
    await inbox.put(ctl(edge_pb2.Control.END_TURN))
    async def wait_for_audio():
        while not any(_kind(e) == "audio" for e in seen):
            await asyncio.sleep(0.01)
    await asyncio.wait_for(wait_for_audio(), timeout=5)

    await inbox.put(ctl(edge_pb2.Control.USER_SPEECH_STARTED))       # barge-in #1
    await asyncio.wait_for(sessions.entered.wait(), timeout=5)       # now mid-persist
    await inbox.put(ctl(edge_pb2.Control.USER_SPEECH_STARTED))       # barge-in #2
    await asyncio.sleep(0.05)                                       # let it be processed
    sessions.release.set()

    await inbox.put(ctl(edge_pb2.Control.SESSION_CLOSE))
    await inbox.put(None)
    await asyncio.wait_for(drainer, timeout=10)

    assert len(sessions.turns) == 1, sessions.turns
    assert sessions.turns[0]["user_text"] == "tell me a story"
    assert sessions.turns[0]["interrupted"] is True
    print("ok repeat_barge_in_does_not_lose_the_interrupted_turn")


# ---------------------------------------------------------------------------
# Transcript ingestion must never be blocked by turn execution
# ---------------------------------------------------------------------------
class ScriptedStt:
    """STT whose transcripts the test emits by hand, so it can prove a
    transcript is ingested while a turn is still in flight. Emission waits for
    SESSION_OPEN to reach STT, which the orchestrator only forwards after it has
    pinned the stream's session -- otherwise transcripts drop as foreign."""
    def __init__(self, outbox):
        self.outbox = outbox
        self.received = []

    async def Transcribe(self, request_iter):
        opened = asyncio.Event()

        async def drain():
            async for ev in request_iter:
                self.received.append(ev)
                if (ev.WhichOneof("msg") == "control"
                        and ev.control.type == asr_pb2.Control.SESSION_OPEN):
                    opened.set()
        drain_task = asyncio.create_task(drain())
        try:
            await opened.wait()
            while True:
                t = await self.outbox.get()
                if t is None:
                    return
                yield t
        finally:
            drain_task.cancel()


class GatedLlm:
    """Blocks inside the first GenerateStream until released, so a turn can be
    held mid-inference while the test pushes more transcripts at the bridge."""
    def __init__(self):
        self.calls = []
        self.started = asyncio.Event()
        self.gate = asyncio.Event()

    async def GenerateStream(self, request, timeout=None):
        self.calls.append(request)
        if len(self.calls) == 1:
            self.started.set()
            await self.gate.wait()
        yield llm_pb2.GenerateChunk(session_id=request.session_id, text_delta="ack")
        yield llm_pb2.GenerateChunk(session_id=request.session_id, is_final=True,
                                    profile=llm_pb2.InferenceProfile(model="m", engine="e", device="cpu"))


class EventSessions(RecordingSessions):
    def __init__(self):
        super().__init__()
        self.events = []

    async def event(self, session_id, event_type, **fields):
        self.events.append((event_type, fields))

    def count(self, event_type):
        return len([e for e in self.events if e[0] == event_type])


def _transcript(sid, text):
    return asr_pb2.Transcript(session_id=sid, text=text, turn_complete=True,
                              segment_index=0, ts_ms=10, proc_ms=20)


async def _until(predicate, timeout=5):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.01)
    await asyncio.wait_for(poll(), timeout=timeout)


class Harness:
    """Drives one MediaBridge session with hand-fed edge controls and STT."""
    def __init__(self, sid="s", llm=None):
        self.sid = sid
        self.outbox: asyncio.Queue = asyncio.Queue()
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.stt = ScriptedStt(self.outbox)
        self.llm = llm or GatedLlm()
        self.sessions = EventSessions()
        self.seen = []

    async def start(self):
        async def edge_events():
            while True:
                ev = await self.inbox.get()
                if ev is None:
                    return
                yield ev
        stream = MediaBridge(self.stt, self.llm, self.sessions,
                             tts=FakeTts()).Stream(edge_events(), None)

        async def drain():
            async for ev in stream:
                self.seen.append(ev)
        self.drainer = asyncio.create_task(drain())
        await self.ctl(edge_pb2.Control.SESSION_OPEN)

    async def ctl(self, t):
        await self.inbox.put(edge_pb2.ClientEvent(
            control=edge_pb2.Control(session_id=self.sid, type=t, ts_ms=1)))

    async def close(self):
        await self.outbox.put(None)
        await self.ctl(edge_pb2.Control.SESSION_CLOSE)
        await self.inbox.put(None)
        await asyncio.wait_for(self.drainer, timeout=10)


async def test_transcripts_are_ingested_while_a_turn_is_in_flight():
    """The regression that lost user speech outright: from_stt() used to await
    the running turn, so nothing drained STT's response stream. grpc flow
    control then backed up into the ASR, whose bounded segment queue overflowed
    and made it stop accepting audio -- speech that was never transcribed at
    all. Ingestion must proceed while the agent is still thinking."""
    h = Harness()
    await h.start()
    await h.ctl(edge_pb2.Control.USER_SPEECH_STARTED)
    await h.ctl(edge_pb2.Control.END_TURN)
    # The transcript must not overtake the controls that open and endpoint the
    # turn, or it would be attributed to turn 0 and read as already superseded.
    await _until(lambda: h.sessions.count("end_turn") == 1)
    await h.outbox.put(_transcript(h.sid, "first question"))
    await asyncio.wait_for(h.llm.started.wait(), timeout=5)  # turn 1 is mid-inference

    # Push a second transcript with turn 1 still held inside the LLM. Under the
    # old inline await this was never read; it must be ingested immediately.
    await h.outbox.put(_transcript(h.sid, "second question"))
    await _until(lambda: h.sessions.count("transcript_segment") >= 2)
    assert not h.llm.gate.is_set(), "turn 1 must still be in flight"

    h.llm.gate.set()
    await _until(lambda: len(h.sessions.turns) == 2)
    await h.close()
    assert [t["user_text"] for t in h.sessions.turns] == ["first question", "second question"]
    print("ok transcripts_are_ingested_while_a_turn_is_in_flight")


async def test_pending_transcript_superseded_by_new_speech_is_recorded_not_answered():
    """User endpoints turn 1, then starts talking again before STT returns it.
    Turn 1 must still be transcribed and persisted, must NOT burn LLM/TTS, and
    must still appear in the prompt of the turn that does get answered."""
    h = Harness()
    h.llm.gate.set()  # never hold a turn in this test
    await h.start()
    await h.ctl(edge_pb2.Control.USER_SPEECH_STARTED)   # turn 1
    await h.ctl(edge_pb2.Control.END_TURN)
    await h.ctl(edge_pb2.Control.USER_SPEECH_STARTED)   # turn 2 begins first
    await _until(lambda: h.sessions.count("user_speech_started") == 2)

    await h.outbox.put(_transcript(h.sid, "what is the weather"))  # turn 1, now stale
    await _until(lambda: len(h.sessions.turns) == 1)

    await h.ctl(edge_pb2.Control.END_TURN)                          # turn 2 endpointed
    await _until(lambda: h.sessions.count("end_turn") == 2)
    await h.outbox.put(_transcript(h.sid, "actually book a cab"))
    await _until(lambda: len(h.sessions.turns) == 2)
    await h.close()

    # Both transcribed and persisted; only the one the user still cared about
    # was answered.
    assert [(t["user_text"], t["assistant_text"]) for t in h.sessions.turns] == [
        ("what is the weather", None), ("actually book a cab", "ack")], h.sessions.turns
    assert len(h.llm.calls) == 1, h.llm.calls
    # ...and the superseded utterance is still inside that single prompt.
    assert [m.content for m in h.llm.calls[0].messages] == [
        "what is the weather\nactually book a cab"], h.llm.calls[0].messages
    print("ok pending_transcript_superseded_is_recorded_not_answered")
