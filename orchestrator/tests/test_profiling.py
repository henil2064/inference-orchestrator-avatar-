"""Self-checks for the telemetry the latency dashboards are built on.

Every Grafana panel that reports a latency derives it by subtracting one
recorded timestamp from another. That is only meaningful if all of them come
from ONE clock, so these tests pin the two rules that make the arithmetic
valid:

  1. event_time_ms is always this process's wall clock. Peer timestamps are
     recorded as payload, never as the event's own time.
  2. time-to-first-audio is endpoint -> first PCM measured on that one clock,
     and a nonsensical result is reported as absent rather than as a number.

The regression they guard: the media edge stamps controls off a sample-counting
media cursor, which drifts from wall time (up to 7% over a call was observed in
ClickHouse) and in older builds was session-relative. Subtracting it from the
orchestrator's wall clock went negative, a max(0, ...) clamp turned that into a
plausible-looking 0 ms, and the fleet time-to-first-audio p50 read 0.
"""
import asyncio

from proto import edge_pb2
from service.server import MediaBridge, _now_ms
from service.session_logger import SessionLogger

from proto import asr_pb2
from tests.test_interrupt import FakeStt, FakeTts, RecordingSessions, _kind
from tests.test_turn_signals import StreamingLlm


class FlushingStt:
    """Finalizes a turn on its own -- STT's session-close flush -- without the
    edge ever having sent END_TURN, so the turn has no caller endpoint."""

    def __init__(self, text):
        self._text = text

    async def Transcribe(self, request_iter):
        async def drain():
            async for _ in request_iter:
                pass
        task = asyncio.ensure_future(drain())
        await asyncio.sleep(0.05)
        yield asr_pb2.Transcript(session_id="s", text=self._text, turn_complete=True,
                                 segment_index=0, ts_ms=0, proc_ms=20)
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout=5)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass


def _recorder() -> SessionLogger:
    """A logger that builds real rows but never starts its flush task, so the
    queue is the exact set of rows ClickHouse would have received."""
    return SessionLogger("http://unused", enabled=True)


def _rows(logger: SessionLogger) -> list[dict]:
    """Just the pipeline_events rows; session_summaries has its own shape."""
    out = []
    while not logger.queue.empty():
        queued = logger.queue.get_nowait()
        if queued.table == "pipeline_events":
            out.append(queued.row)
    return out


async def _run_turn(*, endpoint_ts_ms: int):
    """One full turn where the edge stamps its controls with endpoint_ts_ms."""
    sid = "s"
    inbox: asyncio.Queue = asyncio.Queue()
    logger = _recorder()

    async def edge_events():
        while True:
            ev = await inbox.get()
            if ev is None:
                return
            yield ev

    def ctl(t, ts_ms, **meta):
        return edge_pb2.ClientEvent(control=edge_pb2.Control(
            session_id=sid, type=t, ts_ms=ts_ms, metadata=meta))

    bridge = MediaBridge(FakeStt("hello"), StreamingLlm(), RecordingSessions(),
                         tts=FakeTts(), telemetry=logger)
    stream = bridge.Stream(edge_events(), None)
    await inbox.put(ctl(edge_pb2.Control.SESSION_OPEN, endpoint_ts_ms,
                        vad_process_ms="5"))
    await inbox.put(ctl(edge_pb2.Control.END_TURN, endpoint_ts_ms,
                        vad_process_ms="17"))

    async def collect():
        while True:
            try:
                ev = await stream.__anext__()
            except StopAsyncIteration:
                return
            if _kind(ev) == "control" and ev.control.type == edge_pb2.Control.AGENT_TURN_ENDED:
                await inbox.put(ctl(edge_pb2.Control.SESSION_CLOSE, endpoint_ts_ms))
                await inbox.put(None)

    await asyncio.wait_for(collect(), timeout=10)
    return _rows(logger)


async def test_ttfa_survives_an_edge_clock_running_ahead():
    """The edge's media cursor is 3 s ahead of wall time -- the exact shape that
    used to clamp time-to-first-audio to a fabricated 0 ms."""
    rows = await _run_turn(endpoint_ts_ms=_now_ms() + 3_000)
    ttfa = [r for r in rows if r["event_type"] == "tts.first_audio_queued"]
    assert ttfa, "the turn produced no first-audio event"
    value = ttfa[0]["time_to_first_audio_ms"]
    assert value is not None, "time-to-first-audio went missing"
    assert value >= 0, f"negative time-to-first-audio: {value}"
    # The whole point: a peer clock 3 s ahead must not collapse this to zero.
    # It is a real elapsed measurement, bounded by the test's own runtime.
    assert value < 3_000, f"time-to-first-audio absorbed the peer's clock skew: {value}"


async def test_event_time_is_always_this_process_clock():
    """A stream-relative peer stamp (an older edge sent 19520 for 19.5 s into
    the call) must never become event_time_ms -- those rows landed in a 1970
    partition, invisible to every dashboard time range."""
    before = _now_ms()
    rows = await _run_turn(endpoint_ts_ms=19_520)
    after = _now_ms()

    for row in rows:
        assert before <= row["event_time_ms"] <= after, (
            f"{row['event_type']} recorded event_time_ms={row['event_time_ms']} "
            f"outside this run's window [{before}, {after}]")

    # And the peer's own stamp is still available for audio-timeline debugging.
    vad = [r for r in rows if r["event_type"] == "vad.completed"][0]
    assert '"edge_ts_ms": null' in vad["payload_json"], (
        "a degenerate peer stamp should be recorded as null, not as an epoch")


async def test_stage_timeline_is_ordered_within_a_turn():
    """The session dashboard's stage breakdown subtracts consecutive stage
    timestamps. Mixing clocks put stt.completed BEFORE the stt.started that
    caused it, so the breakdown read negative and was floored to 0."""
    rows = await _run_turn(endpoint_ts_ms=_now_ms() + 3_000)
    stages = {}
    for row in rows:
        if row["turn_id"] == 1:
            stages.setdefault(row["event_type"], row["event_time_ms"])

    order = ["vad.completed", "stt.started", "stt.completed",
             "llm.started", "tts.started", "tts.first_audio_queued"]
    seen = [(name, stages[name]) for name in order if name in stages]
    assert len(seen) >= 4, f"turn did not reach enough stages: {seen}"
    for (earlier, t0), (later, t1) in zip(seen, seen[1:]):
        assert t0 <= t1, f"{later} ({t1}) is timestamped before {earlier} ({t0})"


async def test_vad_duration_is_per_event_not_cumulative():
    """duration_ms is a per-event duration everywhere else in the schema. The
    edge's counter is session-cumulative, so it is differenced on arrival."""
    rows = await _run_turn(endpoint_ts_ms=_now_ms())
    vad = [r for r in rows if r["event_type"] == "vad.completed"][0]
    # SESSION_OPEN reported 5 ms cumulative, END_TURN reported 17 ms cumulative.
    assert vad["duration_ms"] == 12, (
        f"expected the 12 ms delta, got {vad['duration_ms']} "
        "(the raw cumulative counter)")


async def test_a_turn_with_no_endpoint_records_why_it_is_unmeasured():
    """A turn STT finalized on its own (no edge END_TURN) has no caller
    endpoint to measure from. It must record NULL with a reason, so the
    dashboard can count it rather than let it vanish from the KPI -- 75 such
    turns were silently absent from the fleet numbers."""
    rows = await _run_turn_without_endpoint()
    first_audio = [r for r in rows if r["event_type"] == "tts.first_audio_queued"]
    assert first_audio, "the turn produced no first-audio event"
    assert first_audio[0]["time_to_first_audio_ms"] is None, (
        "a turn with no endpoint must not report a latency")
    assert '"ttfa_missing_reason": "no_endpoint"' in first_audio[0]["payload_json"]


async def _run_turn_without_endpoint():
    """Drive a turn that reaches TTS without the edge ever sending END_TURN."""
    sid = "s"
    inbox: asyncio.Queue = asyncio.Queue()
    logger = _recorder()

    async def edge_events():
        while True:
            ev = await inbox.get()
            if ev is None:
                return
            yield ev

    def ctl(t):
        return edge_pb2.ClientEvent(control=edge_pb2.Control(
            session_id=sid, type=t, ts_ms=_now_ms()))

    bridge = MediaBridge(FlushingStt("hello"), StreamingLlm(), RecordingSessions(),
                         tts=FakeTts(), telemetry=logger)
    stream = bridge.Stream(edge_events(), None)
    await inbox.put(ctl(edge_pb2.Control.SESSION_OPEN))

    async def collect():
        while True:
            try:
                ev = await stream.__anext__()
            except StopAsyncIteration:
                return
            if _kind(ev) == "control" and ev.control.type == edge_pb2.Control.AGENT_TURN_ENDED:
                await inbox.put(ctl(edge_pb2.Control.SESSION_CLOSE))
                await inbox.put(None)

    await asyncio.wait_for(collect(), timeout=10)
    return _rows(logger)


async def test_wall_clock_waterfall_reconciles_with_monotonic_ttfa():
    """The dashboards read the stage breakdown off wall-clock event_time_ms but
    read ttfa off the monotonic clock. Those are two independent measurements of
    the same interval, so they must agree to rounding -- otherwise the waterfall
    and the headline number describe different things and the 'segments sum to
    ttfa' claim on the session dashboard is false."""
    rows = await _run_turn(endpoint_ts_ms=_now_ms() + 3_000)
    stages, ttfa = {}, None
    for row in rows:
        if row["turn_id"] != 1:
            continue
        stages.setdefault(row["event_type"], row["event_time_ms"])
        if row["event_type"] == "tts.first_audio_queued":
            ttfa = row["time_to_first_audio_ms"]

    assert ttfa is not None
    wall_span = stages["tts.first_audio_queued"] - stages["vad.completed"]
    assert abs(wall_span - ttfa) <= 2, (
        f"wall-clock waterfall spans {wall_span} ms but monotonic ttfa is {ttfa} ms")

    # And the four segments the breakdown panel draws must tile that span with
    # no gap and no overlap -- that is what makes the stack sum to ttfa.
    segments = [
        stages["stt.completed"] - stages["vad.completed"],
        stages["llm.started"] - stages["stt.completed"],
        stages["tts.started"] - stages["llm.started"],
        stages["tts.first_audio_queued"] - stages["tts.started"],
    ]
    assert all(seg >= 0 for seg in segments), f"a segment ran backwards: {segments}"
    assert sum(segments) == wall_span, (
        f"segments {segments} sum to {sum(segments)}, not the {wall_span} ms span")
