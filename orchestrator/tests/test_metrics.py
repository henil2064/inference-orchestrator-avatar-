"""Self-checks for Phase 2's loop-lag and per-queue visibility metrics.
Run: python -m tests.test_metrics
"""
import asyncio

import service.server as server_module
from proto import asr_pb2, edge_pb2, llm_pb2, tts_pb2
from service.server import MediaBridge

from tests.test_interrupt import FakeStt, RecordingSessions
from tests.test_turn_signals import ManyClausesLlm, SlowFirstClauseTts, _kind


def _gauge_value(gauge):
    return gauge.collect()[0].samples[0].value


def test_queue_capacity_gauges_report_the_configured_bounds():
    assert _gauge_value(server_module.M_QCAP) == server_module.STT_QUEUE_MAX
    assert _gauge_value(server_module.M_INBOUND_CAP) == server_module.INBOUND_QUEUE_MAX
    assert _gauge_value(server_module.M_OUTBOUND_CAP) == server_module.OUTBOUND_QUEUE_MAX
    assert _gauge_value(server_module.M_CLAUSE_CAP) == server_module.CLAUSE_QUEUE_MAX
    print("ok queue_capacity_gauges_report_the_configured_bounds")


def test_loop_lag_monitor_reports_a_value():
    """A healthy loop should report a small, non-negative drift shortly after
    the monitor starts -- proves the gauge is actually wired to a live task,
    not just defined and never updated."""
    async def scenario():
        task = asyncio.create_task(server_module._loop_lag_monitor(interval_s=0.05))
        try:
            await asyncio.sleep(0.2)
            value = _gauge_value(server_module.M_LOOP_LAG)
            assert value is not None and value >= 0.0, value
            assert value < 1.0, f"unexpectedly large lag on an idle loop: {value}"
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    asyncio.run(scenario())
    print("ok loop_lag_monitor_reports_a_value")


def test_clause_queue_depth_tracks_an_in_flight_turn_and_clears_after():
    """orch_clause_queue_depth must reflect real backlog while a turn is
    stalled on a slow TTS consumer, and drop back to 0 once the turn ends --
    proving the registry is not leaking entries across turns."""
    async def scenario():
        depth_before = _gauge_value(server_module.M_CLAUSE_DEPTH)
        assert depth_before == 0, "no turn should be in flight before this test starts"

        gate = asyncio.Event()
        tts = SlowFirstClauseTts(gate)
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

        stream = MediaBridge(FakeStt("hello"), ManyClausesLlm(), RecordingSessions(),
                             tts=tts).Stream(edge_events(), None)
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
        # ManyClausesLlm produces 6 clauses fast; SlowFirstClauseTts blocks the
        # consumer, so the bounded clause_q should fill while we wait here.
        await asyncio.sleep(0.2)
        depth_during = _gauge_value(server_module.M_CLAUSE_DEPTH)
        assert depth_during > 0, "clause queue should show backlog while TTS is gated"

        gate.set()
        await asyncio.wait_for(task, timeout=5)

        depth_after = _gauge_value(server_module.M_CLAUSE_DEPTH)
        assert depth_after == 0, f"clause queue registry leaked after the turn ended: {depth_after}"

    asyncio.run(scenario())
    print("ok clause_queue_depth_tracks_an_in_flight_turn_and_clears_after")
