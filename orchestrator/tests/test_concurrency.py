"""Self-checks for cross-session concurrency.
Run: python -m tests.test_concurrency

The orchestrator exists to keep three backends busy: while one session waits on
the LLM, another should be able to use STT. These assert the ORCHESTRATOR layer
imposes no cross-session serialization — one MediaBridge and one set of stubs are
shared by every session, exactly as in production (see serve()).

What they deliberately do NOT cover is backend parallelism: STT and TTS each run
their model in a ThreadPoolExecutor(max_workers=1), so real concurrency there is
1 regardless of what the orchestrator does. That is a backend property, measured
against the real services, not this seam.
"""
import asyncio

from proto import asr_pb2, edge_pb2, llm_pb2, tts_pb2
import service.server as server_module
from service.server import MediaBridge


class SharedFakeStt:
    """Serves many concurrent Transcribe streams, one per session, emitting one
    turn_complete transcript per END_TURN it sees on that stream."""
    def __init__(self):
        self.live = 0
        self.peak = 0

    async def Transcribe(self, request_iter):
        self.live += 1
        self.peak = max(self.peak, self.live)
        sid = None
        try:
            async for ev in request_iter:
                if ev.WhichOneof("msg") != "control":
                    sid = sid or ev.audio.session_id
                    continue
                sid = ev.control.session_id or sid
                if ev.control.type == asr_pb2.Control.END_TURN:
                    yield asr_pb2.Transcript(session_id=sid, text=f"turn for {sid}",
                                             turn_complete=True, segment_index=0,
                                             ts_ms=10, proc_ms=5)
                elif ev.control.type == asr_pb2.Control.SESSION_CLOSE:
                    return
        finally:
            self.live -= 1


class GatedLlm:
    """Holds `blocked_sid` inside its stream until released, so that session sits
    mid-turn while the others run. Records how many turns overlap."""
    def __init__(self, blocked_sid, release: asyncio.Event):
        self.blocked_sid = blocked_sid
        self.release = release
        self.live = 0
        self.peak = 0
        self.entered = asyncio.Event()

    async def GenerateStream(self, request, timeout=None):
        self.live += 1
        self.peak = max(self.peak, self.live)
        try:
            yield llm_pb2.GenerateChunk(session_id=request.session_id, text_delta="Reply. ")
            if request.session_id == self.blocked_sid:
                self.entered.set()
                await self.release.wait()
            yield llm_pb2.GenerateChunk(session_id=request.session_id, is_final=True,
                                        profile=llm_pb2.InferenceProfile(model="m"))
        finally:
            self.live -= 1


class SharedFakeTts:
    def __init__(self):
        self.live = 0
        self.peak = 0

    async def SynthesizeStream(self, request, timeout=None):
        self.live += 1
        self.peak = max(self.peak, self.live)
        try:
            rate = 24000
            await asyncio.sleep(0)  # a real RPC yields; make overlap observable
            yield tts_pb2.SynthesizeChunk(session_id=request.session_id,
                pcm16=b"\x01\x00" * (rate // 50), sample_rate_hz=rate, chunk_index=0,
                ts_ms=0, profile=tts_pb2.InferenceProfile(model="t", compute_ms=1,
                                                          audio_seconds=0.02))
            yield tts_pb2.SynthesizeChunk(session_id=request.session_id, sample_rate_hz=rate,
                                          chunk_index=1, is_last=True, ts_ms=0)
        finally:
            self.live -= 1


class NullSessions:
    async def open(self, *a, **k): pass
    async def event(self, *a, **k): pass
    async def record_turn(self, *a, **k): pass
    async def recent_turns(self, *a, **k): return []
    async def close(self, *a, **k): pass


async def _drive(bridge, sid, done: asyncio.Event):
    """Run one whole session against the shared bridge and return its events."""
    inbox: asyncio.Queue = asyncio.Queue()

    async def edge_events():
        while True:
            ev = await inbox.get()
            if ev is None:
                return
            yield ev

    def ctl(t):
        return edge_pb2.ClientEvent(control=edge_pb2.Control(session_id=sid, type=t, ts_ms=1))

    await inbox.put(ctl(edge_pb2.Control.SESSION_OPEN))
    await inbox.put(ctl(edge_pb2.Control.END_TURN))
    events = []
    async for ev in bridge.Stream(edge_events(), None):
        events.append(ev)
        if ev.WhichOneof("msg") == "control" and ev.control.type == edge_pb2.Control.AGENT_TURN_ENDED:
            done.set()
            await inbox.put(ctl(edge_pb2.Control.SESSION_CLOSE))
            await inbox.put(None)
    return events


async def test_a_blocked_session_does_not_stall_the_others():
    """The core claim: one session parked inside the LLM must not prevent other
    sessions from reaching STT, LLM and TTS and finishing their own turns."""
    orig = (server_module.ORCH_MAX_SESSIONS, server_module.ORCH_MAX_ACTIVE_TURNS,
            server_module.ORCH_MAX_QUEUED_TURNS)
    server_module.ORCH_MAX_SESSIONS = 0
    server_module.ORCH_MAX_ACTIVE_TURNS = 0
    server_module.ORCH_MAX_QUEUED_TURNS = 0
    try:
        release = asyncio.Event()
        stt, tts = SharedFakeStt(), SharedFakeTts()
        llm = GatedLlm("slow", release)
        # ONE bridge and ONE set of stubs for every session, exactly as serve() wires it.
        bridge = MediaBridge(stt, llm, NullSessions(), tts=tts)

        done = {sid: asyncio.Event() for sid in ("slow", "fast-1", "fast-2")}
        tasks = {sid: asyncio.create_task(_drive(bridge, sid, done[sid])) for sid in done}

        # Park "slow" inside its LLM stream, then require the other two to finish
        # while it is still parked. If any cross-session lock existed, this times out.
        await asyncio.wait_for(llm.entered.wait(), timeout=5)
        await asyncio.wait_for(asyncio.gather(done["fast-1"].wait(), done["fast-2"].wait()), timeout=5)
        assert not done["slow"].is_set(), "the blocked session should still be mid-turn"

        release.set()
        await asyncio.wait_for(asyncio.gather(*tasks.values()), timeout=5)
    finally:
        release.set()
        (server_module.ORCH_MAX_SESSIONS, server_module.ORCH_MAX_ACTIVE_TURNS,
         server_module.ORCH_MAX_QUEUED_TURNS) = orig

    assert stt.peak == 3, f"STT streams never overlapped (peak={stt.peak})"
    assert llm.peak >= 2, f"LLM calls never overlapped (peak={llm.peak})"
    print(f"ok blocked session did not stall the others (stt peak={stt.peak}, llm peak={llm.peak})")


async def test_every_session_gets_its_own_turn_and_audio():
    """Concurrency must not cross-contaminate: each session's audio, reply and
    lifecycle controls carry only its own session id."""
    orig = (server_module.ORCH_MAX_SESSIONS, server_module.ORCH_MAX_ACTIVE_TURNS,
            server_module.ORCH_MAX_QUEUED_TURNS)
    server_module.ORCH_MAX_SESSIONS = 0
    server_module.ORCH_MAX_ACTIVE_TURNS = 0
    server_module.ORCH_MAX_QUEUED_TURNS = 0
    try:
        release = asyncio.Event()
        release.set()
        stt, tts = SharedFakeStt(), SharedFakeTts()
        llm = GatedLlm("none", release)
        bridge = MediaBridge(stt, llm, NullSessions(), tts=tts)

        sids = [f"s{i}" for i in range(5)]
        done = {sid: asyncio.Event() for sid in sids}
        results = await asyncio.wait_for(
            asyncio.gather(*(_drive(bridge, sid, done[sid]) for sid in sids)), timeout=10)
    finally:
        (server_module.ORCH_MAX_SESSIONS, server_module.ORCH_MAX_ACTIVE_TURNS,
         server_module.ORCH_MAX_QUEUED_TURNS) = orig

    for sid, events in zip(sids, results):
        for ev in events:
            got = getattr(ev, ev.WhichOneof("msg")).session_id
            assert got == sid, f"session {sid} received an event for {got}"
        ended = [e for e in events if e.WhichOneof("msg") == "control"
                 and e.control.type == edge_pb2.Control.AGENT_TURN_ENDED]
        assert len(ended) == 1 and ended[0].control.metadata["reason"] == "complete", (sid, ended)
        assert any(e.WhichOneof("msg") == "audio" for e in events), f"{sid} got no audio"
    assert tts.peak >= 2, f"TTS calls never overlapped (peak={tts.peak})"
    print(f"ok 5 concurrent sessions stayed isolated (tts peak={tts.peak})")


class RejectAbort(Exception):
    def __init__(self, code, details):
        super().__init__(details)
        self.code = code


class RejectContext:
    def __init__(self):
        self.aborted = None

    async def abort(self, code, details):
        self.aborted = (code, details)
        raise RejectAbort(code, details)


async def _drive_with_context(bridge, sid, done: asyncio.Event, context):
    """Like _drive, but with a real context so admission rejection (which
    calls context.abort()) can be observed instead of crashing on None."""
    inbox: asyncio.Queue = asyncio.Queue()

    async def edge_events():
        while True:
            ev = await inbox.get()
            if ev is None:
                return
            yield ev

    def ctl(t):
        return edge_pb2.ClientEvent(control=edge_pb2.Control(session_id=sid, type=t, ts_ms=1))

    await inbox.put(ctl(edge_pb2.Control.SESSION_OPEN))
    await inbox.put(ctl(edge_pb2.Control.END_TURN))
    events = []
    try:
        async for ev in bridge.Stream(edge_events(), context):
            events.append(ev)
            if ev.WhichOneof("msg") == "control" and ev.control.type == edge_pb2.Control.AGENT_TURN_ENDED:
                done.set()
                await inbox.put(ctl(edge_pb2.Control.SESSION_CLOSE))
                await inbox.put(None)
    except RejectAbort:
        pass
    return events


async def test_session_admission_rejects_the_nplus1th_arrival():
    """ORCH_MAX_SESSIONS is the authoritative session cap: N+1 simultaneous
    arrivals must never admit more than N, and a rejected session must never
    open an STT stream -- no downstream work for a call about to be aborted."""
    orig = server_module.ORCH_MAX_SESSIONS
    n = 3
    server_module.ORCH_MAX_SESSIONS = n
    try:
        release = asyncio.Event()
        release.set()
        stt, tts = SharedFakeStt(), SharedFakeTts()
        llm = GatedLlm("none", release)
        bridge = MediaBridge(stt, llm, NullSessions(), tts=tts)

        sids = [f"s{i}" for i in range(n + 1)]
        done = {sid: asyncio.Event() for sid in sids}
        contexts = {sid: RejectContext() for sid in sids}
        tasks = {sid: asyncio.create_task(_drive_with_context(bridge, sid, done[sid], contexts[sid]))
                 for sid in sids}
        await asyncio.wait_for(asyncio.gather(*tasks.values()), timeout=5)

        admitted = [sid for sid in sids if contexts[sid].aborted is None]
        rejected = [sid for sid in sids if contexts[sid].aborted is not None]
        assert len(admitted) == n, (admitted, rejected)
        assert len(rejected) == 1, (admitted, rejected)
        assert contexts[rejected[0]].aborted[0] == server_module.grpc.StatusCode.RESOURCE_EXHAUSTED
        assert stt.peak <= n, f"a rejected session must never open an STT stream (stt.peak={stt.peak})"
    finally:
        server_module.ORCH_MAX_SESSIONS = orig
    print("ok session_admission_rejects_the_nplus1th_arrival")


async def test_turn_admission_rejects_overflow_and_persists_truthfully():
    """ORCH_MAX_ACTIVE_TURNS+ORCH_MAX_QUEUED_TURNS bounds concurrent inference
    launches across all sessions, independent of how many are merely
    connected. A turn beyond capacity is rejected -- not silently dropped,
    not queued forever -- with a truthful persisted outcome."""
    from tests.test_interrupt import RecordingSessions

    orig = (server_module.ORCH_MAX_ACTIVE_TURNS, server_module.ORCH_MAX_QUEUED_TURNS,
            server_module.ORCH_TURN_ADMISSION_TIMEOUT_S)
    server_module.ORCH_MAX_ACTIVE_TURNS = 1
    server_module.ORCH_MAX_QUEUED_TURNS = 0
    server_module.ORCH_TURN_ADMISSION_TIMEOUT_S = 0.3
    try:
        release = asyncio.Event()
        stt, tts = SharedFakeStt(), SharedFakeTts()
        llm = GatedLlm("slow", release)
        sessions = RecordingSessions()
        # MediaBridge reads the turn-admission constants at __init__ time, so
        # it must be constructed after patching them.
        bridge = MediaBridge(stt, llm, sessions, tts=tts)

        done_slow = asyncio.Event()
        task_slow = asyncio.create_task(_drive(bridge, "slow", done_slow))
        await asyncio.wait_for(llm.entered.wait(), timeout=5)  # "slow" now holds the one active turn slot

        done_extra = asyncio.Event()
        events_extra = await asyncio.wait_for(_drive(bridge, "extra", done_extra), timeout=5)

        ended = [e.control for e in events_extra if e.WhichOneof("msg") == "control"
                 and e.control.type == edge_pb2.Control.AGENT_TURN_ENDED]
        assert len(ended) == 1 and ended[0].metadata["reason"] == "error", ended
        assert llm.live == 1 and llm.peak == 1, "the rejected turn must never have called the LLM"
        assert sessions.turns, "the rejected turn must still be persisted"
        rejected_turn = sessions.turns[-1]
        assert rejected_turn["interrupted"] is False, rejected_turn
        assert rejected_turn["delivery"] == "none", rejected_turn
        assert rejected_turn["assistant_text"] is None, rejected_turn

        release.set()
        await asyncio.wait_for(task_slow, timeout=5)
    finally:
        (server_module.ORCH_MAX_ACTIVE_TURNS, server_module.ORCH_MAX_QUEUED_TURNS,
         server_module.ORCH_TURN_ADMISSION_TIMEOUT_S) = orig
    print("ok turn_admission_rejects_overflow_and_persists_truthfully")
