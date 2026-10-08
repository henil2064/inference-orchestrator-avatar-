"""Self-check for the non-blocking ClickHouse session logger.

Run: python -m tests.test_session_logger. No ClickHouse daemon is required; the sink
is injected so this verifies queueing, batching, ordering, and failure isolation.
"""

import asyncio

from service.session_logger import SessionLogger


async def test_batches_events_and_summary_in_order():
    inserted = []

    def sink(table, rows):
        inserted.append((table, rows))

    logger = SessionLogger("http://unused", enabled=True, batch_size=2,
                           flush_interval_s=10, insert_rows=sink)
    await logger.start()
    assert logger.event("s1", "session.started", source="edge")
    assert logger.event("s1", "vad.started", source="edge", turn_id=1)
    assert logger.event("s1", "stt.completed", source="stt", turn_id=1, transcript="hello")
    assert logger.event("s1", "tts.first_audio_queued", source="orchestrator", turn_id=1,
                        time_to_first_audio_ms=321, sample_rate_hz=24000, audio_bytes=960)
    assert logger.summary("s1", started_at_ms=1, closed_at_ms=2, outcome="closed",
                          metadata={"vad": {"engine": "silero"}}, totals={"turns": 1})
    await logger.close()

    event_rows = [row for table, rows in inserted if table == "pipeline_events" for row in rows]
    summary_rows = [row for table, rows in inserted if table == "session_summaries" for row in rows]
    assert [row["event_seq"] for row in event_rows] == [1, 2, 3, 4]
    tts_row = event_rows[-1]
    assert tts_row["schema_version"] == 3
    assert tts_row["time_to_first_audio_ms"] == 321
    assert tts_row["sample_rate_hz"] == 24000 and tts_row["audio_bytes"] == 960
    assert {row["pipeline_version"] for row in event_rows} == {logger.pipeline_version}
    assert summary_rows[0]["pipeline_version"] == logger.pipeline_version
    assert len(summary_rows) == 1 and summary_rows[0]["session_id"] == "s1"


async def test_insert_failure_does_not_crash_writer():
    def broken_sink(table, rows):
        raise OSError("ClickHouse down")

    logger = SessionLogger("http://unused", enabled=True, batch_size=1, insert_rows=broken_sink)
    await logger.start()
    assert logger.event("s2", "session.started", source="edge")
    await asyncio.sleep(0)
    await logger.close()
    assert logger.dropped_events == 1


async def test_queue_is_non_blocking_when_full():
    logger = SessionLogger("http://unused", enabled=True, queue_maxsize=1)
    assert logger.event("s3", "one", source="edge")
    assert not logger.event("s3", "two", source="edge")
    assert logger.dropped_events == 1


async def test_llm_prefix_cache_attributes_are_typed():
    inserted = []
    logger = SessionLogger("http://unused", enabled=True, batch_size=1,
                           insert_rows=lambda table, rows: inserted.append((table, rows)))
    await logger.start()
    assert logger.event("cache-session", "llm.completed", source="llm", profile={
        "input_tokens": 200,
        "output_tokens": 12,
        "attributes": {"cached_prompt_tokens": "160"},
    })
    await logger.close()
    row = inserted[0][1][0]
    assert row["schema_version"] == 3
    assert row["cached_prompt_tokens"] == 160
    assert row["prefix_cache_hit"] is True
    assert row["prefix_cache_hit_pct"] == 80.0


async def test_turn_latency_fields_are_typed():
    inserted = []
    logger = SessionLogger("http://unused", enabled=True, batch_size=1,
                           insert_rows=lambda table, rows: inserted.append((table, rows)))
    await logger.start()
    assert logger.event("latency-session", "turn.latency", source="orchestrator", turn_id=4,
                        stt_paid_ms=11.5, llm_queue_ms=22, llm_encode_ms=33,
                        llm_decode_ms=44, phrase_generation_ms=5,
                        tts_queue_ms=6, tts_first_pcm_ms=7, media_edge_queue_ms=8,
                        stt_endpoint_to_final_ms=9, vad_endpoint_to_first_audio_ms=99)
    await logger.close()
    row = inserted[0][1][0]
    assert row["event_type"] == "turn.latency"
    assert row["stt_paid_ms"] == 11.5 and row["media_edge_queue_ms"] == 8
    assert row["stt_endpoint_to_final_ms"] == 9
    assert row["vad_endpoint_to_first_audio_ms"] == 99


async def test_forget_drops_sequence_state():
    """Phase 3: `_sequences` grows one entry per session_id forever unless the
    caller drops it on session close -- this is the fix, not a hypothetical."""
    logger = SessionLogger("http://unused", enabled=True)
    logger.event("leaky-session", "session.started", source="edge")
    logger.event("leaky-session", "session.started", source="edge")
    assert logger._sequences["leaky-session"] == 2
    logger.forget("leaky-session")
    assert "leaky-session" not in logger._sequences
    # forgetting an unknown/never-seen session is a no-op, not an error
    logger.forget("never-seen")


async def test_last_flush_monotonic_tracks_success_only():
    calls = {"n": 0}

    def flaky_sink(table, rows):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("ClickHouse down")

    # A short real flush_interval_s + explicit sleeps (longer than it) make
    # the two flushes deterministically separate calls, so this doesn't
    # depend on asyncio scheduling luck.
    logger = SessionLogger("http://unused", enabled=True, batch_size=1,
                           flush_interval_s=0.01, insert_rows=flaky_sink)
    assert logger.last_flush_monotonic is None  # unknown, not a fabricated zero
    await logger.start()
    assert logger.event("s4", "one", source="edge")
    await asyncio.sleep(0.05)
    assert calls["n"] == 1
    assert logger.last_flush_monotonic is None  # first flush failed
    assert logger.event("s4", "two", source="edge")
    await asyncio.sleep(0.05)
    await logger.close()
    assert calls["n"] == 2
    assert logger.last_flush_monotonic is not None  # second flush succeeded
