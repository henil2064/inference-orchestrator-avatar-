"""Phase-2 orchestrator: bridge the edge audio stream into the STT service, route
each completed turn's text to the LLM service, and persist session state in Redis.

Terminates the edge<->orchestrator gRPC bidi stream (PRD §8.2), and for each
session opens a stream to the STT service (asr.proto), forwarding the decoded
16 kHz PCM16. STT streams back one Transcript per bounded segment; segments are
joined in order as they arrive, and once one arrives with turn_complete=True the
joined text is the full turn — sent once to the LLM service (llm.proto) for a
reply. Recent complete user/assistant turns are loaded from Redis and sent as
structured context; the LLM worker owns final tokenizer-aware budget trimming.
Redis is the only persistent source of session state (no WebSocket yet — PRD
§8.1).
"""

import asyncio
from collections import deque
import json
import logging
import os
import re
import threading
import time
import weakref

import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from proto import (asr_pb2, asr_pb2_grpc, edge_pb2, edge_pb2_grpc, llm_pb2,
                   llm_pb2_grpc, tts_pb2, tts_pb2_grpc)
from service.session_store import SessionStore
from service.session_logger import SessionLogger
from service.audio import rms16 as _rms
from prometheus_client import Counter, Gauge, Histogram, start_http_server


def _load_dotenv():
    """Load KEY=VALUE lines from a .env next to this file. Real env wins (CLI overrides
    the file), so `STT_VAD_GATE=0 python server.py` still beats a file default."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_load_dotenv()  # must run before the module-level os.environ reads below

# Python 3.10 has no Task.cancelling(). Track locally requested cancellation
# for those runtimes, so a second barge-in cannot interrupt the task's
# CancelledError cleanup while it persists its partial turn.
_COMPAT_CANCELLING_TASKS = weakref.WeakSet()

def _cancel_task_once(task: asyncio.Task) -> None:
    cancelling = getattr(task, "cancelling", None)
    already_cancelling = bool(cancelling()) if callable(cancelling) else task in _COMPAT_CANCELLING_TASKS
    if task.done() or already_cancelling:
        return
    if not callable(cancelling):
        _COMPAT_CANCELLING_TASKS.add(task)
    task.cancel()

SAMPLE_RATE = 16000  # matches the edge's opus.NewDecoderWithOutput(16000, 1)


def _resolve_targets(multi_env_var: str, single_env_var: str, default_single: str) -> list[str]:
    """Phase 3 replica fan-out: STT_TARGETS/TTS_TARGETS (comma-separated)
    take priority when set; otherwise this is exactly the old single-target
    behavior via STT_TARGET/TTS_TARGET, so an unscaled deployment is
    unaffected."""
    multi = os.environ.get(multi_env_var, "").strip()
    if multi:
        addresses = [addr.strip() for addr in multi.split(",") if addr.strip()]
        if addresses:
            return addresses
    return [os.environ.get(single_env_var, default_single)]


def _replica_channel(addresses: list[str]) -> grpc.aio.Channel:
    """A single address is today's exact channel (no LB/health-check
    overhead for one backend). Multiple addresses use gRPC's built-in
    multi-address "ipv4:" resolver with client-side round_robin + health
    checking, so a NOT_SERVING replica (Phase 3 wedge self-report) is
    excluded from picks automatically -- verified against a live multi-
    backend channel, not just read from docs. STT sticks per session and TTS
    round-robins per clause "for free": a streaming RPC pins to the
    subchannel it picked for its whole lifetime, and TTS already opens one
    stream per clause."""
    if len(addresses) == 1:
        return grpc.aio.insecure_channel(addresses[0])
    service_config = json.dumps({
        "loadBalancingConfig": [{"round_robin": {}}],
        "healthCheckConfig": {"serviceName": ""},
    })
    options = [
        ("grpc.service_config", service_config),
        ("grpc.keepalive_time_ms", 20000),
        ("grpc.keepalive_timeout_ms", 10000),
        ("grpc.keepalive_permit_without_calls", 1),
    ]
    return grpc.aio.insecure_channel("ipv4:" + ",".join(addresses), options=options)


async def _health_status(channel: grpc.aio.Channel, timeout_s: float = 1.0) -> int | None:
    try:
        stub = health_pb2_grpc.HealthStub(channel)
        resp = await asyncio.wait_for(
            stub.Check(health_pb2.HealthCheckRequest(service="")),
            timeout=timeout_s,
        )
        return resp.status
    except Exception:
        return None


async def _targets_serving(addresses: list[str], *, timeout_s: float = 1.0) -> bool:
    for address in addresses:
        async with grpc.aio.insecure_channel(address) as channel:
            status = await _health_status(channel, timeout_s=timeout_s)
        if status != health_pb2.HealthCheckResponse.SERVING:
            return False
    return True


async def _downstreams_serving(*, timeout_s: float = 1.0) -> bool:
    if not await _targets_serving(STT_TARGETS, timeout_s=timeout_s):
        return False
    if LLM_ENABLED:
        async with grpc.aio.insecure_channel(LLM_TARGET) as channel:
            status = await _health_status(channel, timeout_s=timeout_s)
        if status != health_pb2.HealthCheckResponse.SERVING:
            return False
    if TTS_ENABLED and not await _targets_serving(TTS_TARGETS, timeout_s=timeout_s):
        return False
    return True


async def _downstream_health_gate(
    health_servicer: "health.aio.HealthServicer",
    *,
    interval_s: float = 1.0,
    timeout_s: float = 1.0,
) -> None:
    reported_serving = False
    while True:
        serving = await _downstreams_serving(timeout_s=timeout_s)
        if serving and not reported_serving:
            reported_serving = True
            log.info("all downstream services are SERVING; reporting SERVING")
            await health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
        elif not serving and reported_serving:
            reported_serving = False
            log.error("one or more downstream services are not SERVING; reporting NOT_SERVING")
            await health_servicer.set("", health_pb2.HealthCheckResponse.NOT_SERVING)
        await asyncio.sleep(interval_s)


STT_TARGET = os.environ.get("STT_TARGET", "localhost:50052")
# Forward the configured pre-roll plus audio between the edge VAD's speech onset
# and end-of-turn, so STT gets full words without running on continuous silence
# (mitigates silence hallucination + wasted compute, PRD R-STT-3). ON by default;
# set STT_VAD_GATE=0 to force-forward every frame during diagnostics.
VAD_GATE = os.environ.get("STT_VAD_GATE", "1") != "0"
# A causal VAD can only announce speech after it has observed part of the
# utterance. Retain a small rolling prefix while the gate is closed, then flush
# it when speech starts so STT receives quiet lead-in and the complete first
# word rather than audio beginning at the VAD decision point.
VAD_PREROLL_MS = max(0, int(os.environ.get("STT_VAD_PREROLL_MS", "300")))
LLM_TARGET = os.environ.get("LLM_TARGET", "localhost:50053")
TTS_TARGET = os.environ.get("TTS_TARGET", "localhost:50054")
# Replica fan-out (Phase 3): resolved address lists for the STT/TTS channels.
# LLM stays single-target -- one vLLM instance on the GPU; the facade's own
# admission is what scales, never duplicate weights (see guardrails).
STT_TARGETS = _resolve_targets("STT_TARGETS", "STT_TARGET", "localhost:50052")
TTS_TARGETS = _resolve_targets("TTS_TARGETS", "TTS_TARGET", "localhost:50054")
# Off switch for local dev without an LLM box up (e.g. testing STT/VAD alone).
# A connection/call failure alone does NOT need this — that's already isolated
# per-turn (see call_llm below) so one bad LLM call can't take down a session.
LLM_ENABLED = os.environ.get("LLM_ENABLED", "1") != "0"
TTS_ENABLED = os.environ.get("TTS_ENABLED", "1") != "0"
TTS_TIMEOUT_S = max(0.1, float(os.environ.get("TTS_TIMEOUT_S", "45")))
# After the edge stops sending (SESSION_CLOSE / stream end), how long to keep
# relaying so an in-flight reply's audio can finish reaching the edge. The edge
# half-closes and waits ~12 s for the final reply (media-edge finalReplyWait),
# so this bounds a hung backend without cutting a normal trailing reply.
FINAL_TURN_WAIT_S = max(0.0, float(os.environ.get("ORCH_FINAL_TURN_WAIT_S", "12")))
REDIS_URL = os.environ.get("REDIS_URL", "redis://127.0.0.1:6379/0")
REDIS_ACTIVE_TTL_S = int(os.environ.get("REDIS_ACTIVE_TTL_S", "86400"))
REDIS_CLOSED_TTL_S = int(os.environ.get("REDIS_CLOSED_TTL_S", "604800"))
REDIS_EVENT_MAXLEN = int(os.environ.get("REDIS_EVENT_MAXLEN", "2000"))
# Phase 3 storage health: a sized connection pool (redis-py's async default is
# effectively unbounded) and bounded per-op waits so a slow -- not just
# refused -- Redis can never hang a live call. 128 comfortably covers a few
# hundred sessions' worth of concurrent ops on one Redis instance; raise it
# alongside ORCH_MAX_SESSIONS on a bigger box.
REDIS_POOL_SIZE = max(8, int(os.environ.get("REDIS_POOL_SIZE", "128")))
REDIS_OP_TIMEOUT_S = max(0.1, float(os.environ.get("REDIS_OP_TIMEOUT_S", "3")))
REDIS_CONNECT_TIMEOUT_S = max(0.1, float(os.environ.get("REDIS_CONNECT_TIMEOUT_S", "3")))
# How many completed turns Redis retains per session (storage retention),
# deliberately decoupled from LLM_HISTORY_MAX_TURNS (the much smaller prompt
# window below) -- see docs/LOAD_SCALING_PLAN.md Phase 3 item 1.
REDIS_TURN_RETENTION = max(1, int(os.environ.get("REDIS_TURN_RETENTION", "50")))
# Candidate turn bound for the one-time Redis load at session open. The LLM
# worker applies the authoritative token budget and may retain fewer turns.
LLM_HISTORY_MAX_TURNS = max(0, int(os.environ.get("LLM_HISTORY_MAX_TURNS", "6")))
# Bounded per-session buffers. These are the orchestrator's backpressure knobs:
# a full queue suspends the edge reader, and grpc-aio HTTP/2 flow control then
# backpressures the media edge instead of letting memory grow without bound.
# GPU admission itself lives in the backends (STT_MAX_INFLIGHT, TTS_MAX_QUEUE).
# Frames are ~20 ms each, so the defaults hold a few seconds of audio.
INBOUND_QUEUE_MAX = max(1, int(os.environ.get("ORCH_INBOUND_QUEUE_MAX", "400")))
STT_QUEUE_MAX = max(1, int(os.environ.get("ORCH_STT_QUEUE_MAX", "400")))
OUTBOUND_QUEUE_MAX = max(1, int(os.environ.get("ORCH_OUTBOUND_QUEUE_MAX", "400")))
# Clauses buffered between the LLM token loop (producer) and the per-clause TTS
# consumer inside a turn. Small on purpose: it exists to overlap the next
# sentence's synthesis with the current one, not to run ahead of playback.
CLAUSE_QUEUE_MAX = max(1, int(os.environ.get("ORCH_CLAUSE_QUEUE_MAX", "4")))
# Finalized utterances waiting for turn_runner(). One entry per user utterance
# (not per audio frame), and all but the newest are resolved without inference,
# so this only ever holds a real backlog if a single turn wedges past its
# deadline. Bounded anyway; dispatch_turn() persists inline if it ever fills.
TURN_QUEUE_MAX = max(1, int(os.environ.get("ORCH_TURN_QUEUE_MAX", "32")))
# How many recent turns keep their endpoint/deadline timestamps. Comfortably
# more than can be in flight at once (the turn queue plus the running turn),
# so pruning can never drop a timestamp a live turn still needs to read.
TURN_TIMING_RETAIN = TURN_QUEUE_MAX + 8
# Force-flush an in-progress clause with no sentence punctuation once it grows
# past this, so a long comma-spliced reply still streams to TTS.
MAX_CLAUSE_CHARS = max(20, int(os.environ.get("ORCH_MAX_CLAUSE_CHARS", "240")))
# Let the turn's first clause flush early (see extract_clauses' docstring) so
# TTS overlaps the LLM's own decode instead of waiting for it to finish. Unset
# (0) disables early flushing and restores the old sentence-only behavior.
FIRST_CLAUSE_CHARS = max(0, int(os.environ.get("ORCH_FIRST_CLAUSE_CHARS", "50")))
# Phase 2 bounded overload: the authoritative session cap (0 = unlimited).
# EDGE_MAX_SESSIONS is a defensive backstop above this, not the primary gate.
ORCH_MAX_SESSIONS = max(0, int(os.environ.get("ORCH_MAX_SESSIONS", "0")))
# Bounds concurrent *inference* (LLM+TTS) launches across all sessions, so a
# burst of many connected-but-idle sessions ending their turn at once can't
# all hit the backends simultaneously -- independent of how many sessions are
# merely connected (ORCH_MAX_SESSIONS, above). 0 active = the gate is off.
ORCH_MAX_ACTIVE_TURNS = max(0, int(os.environ.get("ORCH_MAX_ACTIVE_TURNS", "4")))
ORCH_MAX_QUEUED_TURNS = max(0, int(os.environ.get("ORCH_MAX_QUEUED_TURNS", "12")))
ORCH_TURN_ADMISSION_TIMEOUT_S = max(0.1, float(os.environ.get("ORCH_TURN_ADMISSION_TIMEOUT_S", "10")))
# Turn deadline: the clock starts at turn commitment (the edge's END_TURN,
# i.e. the moment the user's turn ends) and never resets, so time already
# spent waiting on STT counts against it -- "already-aged audio" is handled
# for free by starting the clock this early rather than after STT finishes.
# 0 disables the deadline entirely (unlimited, matches the other ORCH_MAX_*
# knobs' convention). Monotonic-clock based throughout.
ORCH_TURN_DEADLINE_S = max(0.0, float(os.environ.get("ORCH_TURN_DEADLINE_S", "30")))
CLICKHOUSE_ENABLED = os.environ.get("CLICKHOUSE_ENABLED", "1") != "0"
CLICKHOUSE_URL = os.environ.get("CLICKHOUSE_URL", "http://127.0.0.1:8123")
CLICKHOUSE_DATABASE = os.environ.get("CLICKHOUSE_DATABASE", "pipeline")
CLICKHOUSE_QUEUE_MAX = int(os.environ.get("CLICKHOUSE_QUEUE_MAX", "10000"))
CLICKHOUSE_BATCH_SIZE = int(os.environ.get("CLICKHOUSE_BATCH_SIZE", "100"))
CLICKHOUSE_FLUSH_INTERVAL_S = float(os.environ.get("CLICKHOUSE_FLUSH_INTERVAL_S", "0.25"))

def _configure_logging() -> None:
    """Keep service terminals quiet by default; use LOG_LEVEL=INFO/DEBUG when needed."""
    level_name = os.environ.get("LOG_LEVEL", "ERROR").upper()
    level = getattr(logging, level_name, logging.ERROR)
    if not isinstance(level, int):
        level = logging.ERROR
    logging.basicConfig(level=level, format="%(levelname)s %(name)s | %(message)s", force=True)


_configure_logging()
log = logging.getLogger("orchestrator")


def _startup_banner(lines: list[str]) -> None:
    """One always-visible readiness summary; LOG_LEVEL controls all later chatter."""
    if os.environ.get("STARTUP_BANNER", "1") == "0":
        return
    width = 60
    print("\n+" + "=" * width + "+")
    print(f"|  {'ORCHESTRATOR  |  READY':<58}|")
    print("+" + "-" * width + "+")
    for line in lines:
        print(f"|  {line:<58.58}|")
    print("+" + "=" * width + "+\n", flush=True)

# Prometheus metrics (scraped at ORCH_METRICS_PORT).
M_SESSIONS = Gauge("orch_active_sessions", "Active edge<->orchestrator sessions")
M_SESSIONS_REJECTED = Counter("orch_sessions_rejected_total",
                              "Sessions rejected because ORCH_MAX_SESSIONS was reached")
M_TURNS_ACTIVE = Gauge("orch_turns_active", "Turns currently running inference (LLM+TTS)")
M_TURNS_REJECTED = Counter("orch_turns_rejected_total",
                           "Turns rejected because the orchestrator's turn-admission capacity was reached")
M_TURNS_EXPIRED = Counter("orch_turn_deadline_expired_total",
                          "Turns or downstream calls skipped because the turn deadline (ORCH_TURN_DEADLINE_S) "
                          "was already exceeded", ["stage"])  # stage: admission | stt | llm | tts
M_TURNS_SUPERSEDED = Counter("orch_turns_superseded_total",
                             "Turns transcribed and persisted but never sent to the LLM, because the "
                             "user had already moved on by the time they could be answered",
                             ["reason"])  # newer_turn | user_speaking | queue_full | session_closed
M_CONTROL = Counter("orch_control_events_total", "Edge control events handled", ["type"])
M_FWD = Counter("orch_frames_forwarded_total", "Audio frames forwarded to STT")
M_DROP = Counter("orch_frames_dropped_total", "Audio frames dropped by the VAD gate")
M_STT_ERR = Counter("orch_stt_stream_errors_total", "Errors on the STT stream")
M_TRANSCRIPTS = Counter("orch_transcripts_total", "Transcripts received from STT", ["kind"])
M_LLM_CALLS = Counter("orch_llm_calls_total", "Turns sent to the LLM service")
M_LLM_ERR = Counter("orch_llm_errors_total", "Errors calling the LLM service")
M_TTS_CALLS = Counter("orch_tts_calls_total", "Replies sent to the TTS service")
M_TTS_ERR = Counter("orch_tts_errors_total", "Errors calling the TTS service")
M_TTS_FRAMES = Counter("orch_tts_frames_total", "PCM frames queued to media edge")
M_TTS_FRAMES_DROPPED = Counter("orch_tts_frames_dropped_total",
                               "PCM frames dropped at relay because their turn was barged-in/superseded")
M_STORE_ERR = Counter("orch_session_store_errors_total", "Redis session-store write errors")
M_SESSION_MISMATCH = Counter(
    "orch_session_id_mismatches_total",
    "Events or STT transcripts dropped because they do not belong to this stream's session",
    ["source"],
)
M_LLM_LATENCY = Histogram("orch_llm_latency_seconds", "orchestrator->LLM->orchestrator round trip",
                          buckets=(.1, .25, .5, .75, 1, 2, 3, 5, 8, 12, 20))
M_TTS_LATENCY = Histogram("orch_tts_latency_seconds", "orchestrator->TTS->orchestrator round trip",
                          buckets=(.1, .25, .5, .75, 1, 2, 3, 5, 8, 12, 20))
# Backpressure signal: summed depth of the per-session queues, evaluated at scrape
# time so an abnormal teardown can't leave the gauge drifted (unlike inc/dec).
_ACTIVE_QUEUES: set = set()
_QLOCK = threading.Lock()  # guards the _ACTIVE_* registries: the scrape thread reads them while the loop mutates them
M_QDEPTH = Gauge("orch_stt_queue_depth", "Audio/control items queued to STT, summed across sessions")
M_QCAP = Gauge("orch_stt_queue_capacity", "Configured per-session STT-forward queue bound (ORCH_STT_QUEUE_MAX)")
# The STT-forward queue is the one item Phase 1's assessment specifically calls
# out as the offered-load bottleneck signal, so it alone gets an oldest-age
# gauge: a parallel per-session deque of put() timestamps (items themselves
# carry no timestamp, and retrofitting one would touch the audio hot path).
# deque has no __hash__, so this is id(deque) -> deque rather than a set.
_ACTIVE_QUEUE_TIMES: dict = {}
M_QOLDEST = Gauge("orch_stt_queue_oldest_age_seconds",
                  "Age of the oldest item still waiting in any session's STT-forward queue")


def _queue_depth() -> float:
    with _QLOCK:
        return float(sum(q.qsize() for q in _ACTIVE_QUEUES))


def _queue_oldest_age() -> float:
    now = time.monotonic()
    with _QLOCK:
        oldest = min((times[0] for times in _ACTIVE_QUEUE_TIMES.values() if times), default=None)
    return max(0.0, now - oldest) if oldest is not None else 0.0


M_QDEPTH.set_function(_queue_depth)
M_QCAP.set_function(lambda: float(STT_QUEUE_MAX))
M_QOLDEST.set_function(_queue_oldest_age)

# Aggregate count/capacity for the other bounded per-session/per-turn queues.
# These don't carry per-item timestamps (see above), so they get count and
# capacity but not oldest-age -- a deliberate, smaller scope than the STT
# queue above, not an oversight.
_ACTIVE_INBOUND: set = set()
_ACTIVE_OUTBOUND: set = set()
_ACTIVE_CLAUSE_QUEUES: set = set()
M_INBOUND_DEPTH = Gauge("orch_inbound_queue_depth", "Edge events queued for processing, summed across sessions")
M_INBOUND_CAP = Gauge("orch_inbound_queue_capacity", "Configured per-session inbound queue bound (ORCH_INBOUND_QUEUE_MAX)")
M_OUTBOUND_DEPTH = Gauge("orch_outbound_queue_depth", "ServerEvents queued for relay to the edge, summed across sessions")
M_OUTBOUND_CAP = Gauge("orch_outbound_queue_capacity", "Configured per-session outbound queue bound (ORCH_OUTBOUND_QUEUE_MAX)")
M_CLAUSE_DEPTH = Gauge("orch_clause_queue_depth", "LLM clauses queued for TTS, summed across in-flight turns")
M_CLAUSE_CAP = Gauge("orch_clause_queue_capacity", "Configured per-turn clause queue bound (ORCH_CLAUSE_QUEUE_MAX)")
M_INBOUND_DEPTH.set_function(lambda: _summed_qsize(_ACTIVE_INBOUND))
M_OUTBOUND_DEPTH.set_function(lambda: _summed_qsize(_ACTIVE_OUTBOUND))
M_CLAUSE_DEPTH.set_function(lambda: _summed_qsize(_ACTIVE_CLAUSE_QUEUES))
M_INBOUND_CAP.set_function(lambda: float(INBOUND_QUEUE_MAX))
M_OUTBOUND_CAP.set_function(lambda: float(OUTBOUND_QUEUE_MAX))
M_CLAUSE_CAP.set_function(lambda: float(CLAUSE_QUEUE_MAX))


def _summed_qsize(registry: set) -> float:
    with _QLOCK:
        return float(sum(q.qsize() for q in registry))


# Event-loop scheduling drift: schedule a wakeup at now+interval and measure
# how much later than that it actually fires. A healthy loop reports ~0;
# rising drift means the loop is falling behind its own scheduled work,
# regardless of which coroutine is responsible.
M_LOOP_LAG = Gauge("orch_event_loop_lag_seconds",
                   "Event-loop scheduling drift; rising values mean the loop is falling behind")

# Telemetry (ClickHouse) health, Phase 3 storage/telemetry item: inserts are
# already off the request path (SessionLogger's own background task); these
# make the previously-invisible parts of that path visible. set_function
# reads live state off the SessionLogger instance set at startup below, same
# pattern as the queue gauges above.
_TELEMETRY_REF: SessionLogger | None = None
M_TELEMETRY_QUEUE_DEPTH = Gauge("orch_telemetry_queue_depth",
                                "Rows queued for the ClickHouse writer, not yet flushed")
M_TELEMETRY_DROPPED = Gauge("orch_telemetry_dropped_total",
                            "Telemetry rows dropped (queue full or insert failure) since startup")
M_TELEMETRY_LAST_FLUSH_AGE = Gauge("orch_telemetry_last_flush_age_seconds",
                                   "Seconds since the last successful ClickHouse flush; "
                                   "NaN if telemetry is disabled or none has landed yet")


def _telemetry_queue_depth() -> float:
    return float(_TELEMETRY_REF.queue.qsize()) if _TELEMETRY_REF is not None else 0.0


def _telemetry_dropped() -> float:
    return float(_TELEMETRY_REF.dropped_events) if _TELEMETRY_REF is not None else 0.0


def _telemetry_last_flush_age() -> float:
    if _TELEMETRY_REF is None or _TELEMETRY_REF.last_flush_monotonic is None:
        return float("nan")  # never coerce "unknown" into a healthy-looking zero
    return max(0.0, time.monotonic() - _TELEMETRY_REF.last_flush_monotonic)


M_TELEMETRY_QUEUE_DEPTH.set_function(_telemetry_queue_depth)
M_TELEMETRY_DROPPED.set_function(_telemetry_dropped)
M_TELEMETRY_LAST_FLUSH_AGE.set_function(_telemetry_last_flush_age)


async def _loop_lag_monitor(interval_s: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    target = loop.time() + interval_s
    while True:
        await asyncio.sleep(interval_s)
        now = loop.time()
        M_LOOP_LAG.set(max(0.0, now - target))
        target = now + interval_s

def _now_ms() -> int:
    return int(time.time() * 1000)


# Timestamps that arrive from another process are never used as this
# orchestrator's own clock. The media edge stamps controls off its MEDIA clock
# (a sample-counting cursor: correct for VAD windows, but it drifts from wall
# time -- measured up to 7% over a call -- and older builds sent it as a
# stream-relative offset, so values like 19520 have reached ClickHouse). STT
# and TTS stamp off their own wall clocks on their own hosts.
#
# Every telemetry timeline in Grafana subtracts one event's event_time_ms from
# another's, so all of them MUST be on one clock: this process's. A foreign
# timestamp is kept in the event payload for audio-timeline debugging, never in
# event_time_ms. See _foreign_ts().
def _foreign_ts(ts_ms: int | None) -> int | None:
    """Normalize a peer-supplied timestamp for payload-only recording.

    Returns None for absent/degenerate values so a stream-relative offset is
    never mistaken for an epoch by anything reading the raw audit trace.
    """
    if not ts_ms or ts_ms < 0:
        return None
    # Anything below this is not a wall-clock epoch (2001-09-09); it is a
    # stream-relative offset from a client that predates the wall-clock
    # contract on Control.ts_ms.
    return int(ts_ms) if ts_ms >= 1_000_000_000_000 else None


def _remaining_budget(deadline: float | None) -> float | None:
    """Seconds left before `deadline` (a time.monotonic() timestamp), or None
    if no deadline is configured (unlimited). Never negative -- a turn whose
    budget already ran out reports 0, not a negative number, since 0 is what
    callers check for "already expired" and a negative grpc timeout is not
    meaningful."""
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


# A sentence end: terminal punctuation, optional closing quote/bracket, then
# whitespace (the whitespace guarantees we don't cut inside "3.14" or "U.S.").
_SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*\s")
# A softer, mid-sentence break usable only for the first clause of a turn (see
# `first_clause_chars` below): comma/semicolon/colon then whitespace.
_SOFT_BREAK = re.compile(r"[,;:]\s")


def extract_clauses(buf: str, max_chars: int, *, first_clause_chars: int | None = None) -> tuple[list[str], str]:
    """Pull complete clauses out of a growing LLM text buffer.

    Returns (clauses_ready_for_tts, remainder_still_accumulating). Splits on
    sentence boundaries; if the buffer runs past max_chars with no boundary
    (a long punctuation-free run), force-cut at the last space so TTS still
    gets fed. Pure and incremental: call it after each token delta.

    `first_clause_chars`, when given, lets the *first* clause pulled out of
    `buf` end early at a comma/semicolon/colon or at `first_clause_chars`
    (whichever comes first), instead of waiting for real sentence punctuation.
    Short replies (the common case at low LLM_MAX_TOKENS) often have no
    sentence boundary until the very last token, so without this the LLM
    stream and TTS never overlap -- the whole reply arrives as one clause
    only after generation finishes. This lets TTS start on the opening words
    while the LLM is still generating the rest. Every later clause still
    waits for a real sentence boundary or the max_chars force-cut, so a
    single short sentence is never needlessly split more than once, and
    prose that reads more naturally with full sentences is unaffected past
    the first clause.

    The comma/semicolon/colon break only fires once `buf` has reached half of
    `first_clause_chars` -- without that floor, a comma a few characters into
    a short reply (e.g. "Sure, I can help.") would fragment it into two tiny
    TTS calls, trading one RPC's overhead for a saving that was never there:
    a reply that short finishes generating almost immediately anyway.
    """
    clauses: list[str] = []
    first = first_clause_chars is not None
    soft_break_floor = max(1, first_clause_chars // 2) if first_clause_chars else 0
    while buf:
        m = _SENTENCE_END.search(buf)
        cut_at = m.end() if m else None
        if first and (cut_at is None or cut_at > first_clause_chars):
            soft = _SOFT_BREAK.search(buf[:first_clause_chars]) if len(buf) >= soft_break_floor else None
            if soft:
                cut_at = soft.end()
            elif len(buf) >= first_clause_chars:
                cut = buf.rfind(" ", 0, first_clause_chars)
                cut_at = cut if cut > 0 else first_clause_chars
        if cut_at is not None:
            clause = buf[:cut_at].strip()
            buf = buf[cut_at:].lstrip()
            if clause:
                clauses.append(clause)
            first = False
            continue
        if len(buf) >= max_chars:
            cut = buf.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            clause = buf[:cut].strip()
            buf = buf[cut:].lstrip()
            if clause:
                clauses.append(clause)
            first = False
            continue
        break
    return clauses, buf


class MediaBridge(edge_pb2_grpc.MediaBridgeServicer):
    def __init__(self, stt: asr_pb2_grpc.AsrStub, llm: llm_pb2_grpc.LlmStub,
                 sessions: SessionStore = None, *, vad_gate: bool = None,
                 vad_preroll_ms: int = None, telemetry: SessionLogger | None = None,
                 tts: tts_pb2_grpc.TtsStub | None = None):
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.sessions = sessions
        self.vad_gate = VAD_GATE if vad_gate is None else vad_gate
        preroll_ms = VAD_PREROLL_MS if vad_preroll_ms is None else max(0, vad_preroll_ms)
        self.vad_preroll_bytes = int(preroll_ms / 1000 * SAMPLE_RATE) * 2
        self.telemetry = telemetry
        # Session admission: an atomic reservation counter (not a read of the
        # M_SESSIONS Prometheus gauge -- that would race against concurrent
        # Stream() calls). Checked synchronously with no intervening await, so
        # it is atomic on the single-threaded event loop.
        self._session_capacity = ORCH_MAX_SESSIONS
        self._sessions_reserved = 0
        # Turn admission: bounds concurrent LLM+TTS launches across all
        # sessions, independent of how many sessions are merely connected.
        # Mirrors the TTS admission gate exactly: an atomic reservation check
        # rejects overflow instantly; a bounded semaphore covers the (small)
        # queued room so a queued turn still waits its turn instead of racing.
        self._turn_capacity = ORCH_MAX_ACTIVE_TURNS + ORCH_MAX_QUEUED_TURNS if ORCH_MAX_ACTIVE_TURNS > 0 else 0
        self._turns_reserved = 0
        self._turn_slots = asyncio.Semaphore(ORCH_MAX_ACTIVE_TURNS) if ORCH_MAX_ACTIVE_TURNS > 0 else None

    def _try_reserve_session(self) -> bool:
        if self._session_capacity <= 0:
            return True  # ORCH_MAX_SESSIONS=0 means unlimited
        if self._sessions_reserved >= self._session_capacity:
            return False
        self._sessions_reserved += 1
        return True

    def _release_session(self) -> None:
        if self._session_capacity > 0:
            self._sessions_reserved -= 1

    def _try_reserve_turn(self) -> bool:
        if self._turn_slots is None:
            return True  # ORCH_MAX_ACTIVE_TURNS=0 means the gate is off
        if self._turns_reserved >= self._turn_capacity:
            return False
        self._turns_reserved += 1
        return True

    def _release_turn_reservation(self) -> None:
        if self._turn_slots is not None:
            self._turns_reserved -= 1

    async def _admit_turn(self, deadline: float | None = None) -> bool:
        """True once genuinely admitted (holding a turn slot); False if the
        turn-admission capacity (active+queued) was already full. The
        admission wait itself is capped by the turn's remaining deadline
        budget (Phase 2 "propagate the remaining budget through admission"),
        never by more than ORCH_TURN_ADMISSION_TIMEOUT_S."""
        if self._turn_slots is None:
            return True
        if not self._try_reserve_turn():
            return False
        timeout = ORCH_TURN_ADMISSION_TIMEOUT_S
        remaining = _remaining_budget(deadline)
        if remaining is not None:
            timeout = min(timeout, remaining)
        try:
            await asyncio.wait_for(self._turn_slots.acquire(), timeout=timeout)
        except asyncio.TimeoutError:
            self._release_turn_reservation()
            return False
        M_TURNS_ACTIVE.inc()
        return True

    def _release_turn(self) -> None:
        if self._turn_slots is not None:
            M_TURNS_ACTIVE.dec()
            self._turn_slots.release()
            self._release_turn_reservation()

    async def _store(self, operation: str, session_id: str, *args, **kwargs) -> bool:
        """A Redis outage is visible, but cannot tear down a live voice call."""
        if self.sessions is None or not session_id:
            return False
        try:
            await getattr(self.sessions, operation)(session_id, *args, **kwargs)
            return True
        except Exception:
            M_STORE_ERR.inc()
            log.exception("session-store update failed for %s", session_id)
            return False

    async def _recent_turns(self, session_id: str) -> list[dict]:
        if self.sessions is None or not session_id or LLM_HISTORY_MAX_TURNS == 0:
            return []
        try:
            return await self.sessions.recent_turns(session_id, LLM_HISTORY_MAX_TURNS)
        except Exception:
            M_STORE_ERR.inc()
            log.exception("session history load failed for %s", session_id)
            return []

    async def Stream(self, request_iterator, context):
        """Thin admission gate in front of the real session handler: reject
        immediately (no queues, no STT stream, nothing) once ORCH_MAX_SESSIONS
        is reached, rather than opening downstream work for a session that's
        about to be torn down anyway."""
        if not self._try_reserve_session():
            M_SESSIONS_REJECTED.inc()
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED,
                                "orchestrator session capacity reached; retry later")
            return
        try:
            async for ev in self._stream_admitted(request_iterator, context):
                yield ev
        finally:
            self._release_session()

    async def _stream_admitted(self, request_iterator, context):
        # One RPC == one voice session. We open one STT stream for it, feed audio
        # through a queue (so the STT request generator is decoupled from edge
        # arrival), and log transcripts as they stream back.
        # Bounded so a slow STT/TTS backend or slow edge applies backpressure
        # (grpc-aio flow control) instead of growing memory without bound.
        q: asyncio.Queue = asyncio.Queue(maxsize=STT_QUEUE_MAX)
        # Parallel put()-timestamp deque for q (see M_QOLDEST above): items
        # queued to STT carry no timestamp of their own, so age is tracked here
        # instead of inside the queue payload, to keep the audio hot path
        # unchanged. Popped in lockstep with q in to_stt() below.
        q_times: deque = deque()
        # outbound holds (epoch, ServerEvent) tuples. epoch=None always relays
        # (replies, control); a tagged audio frame relays only while its turn is
        # the live one, so a barged-in turn's queued frames are dropped, not sent.
        outbound: asyncio.Queue = asyncio.Queue(maxsize=OUTBOUND_QUEUE_MAX)
        with _QLOCK:
            _ACTIVE_QUEUES.add(q)
            _ACTIVE_QUEUE_TIMES[id(q_times)] = q_times
            _ACTIVE_OUTBOUND.add(outbound)

        async def q_put(item) -> None:
            q_times.append(time.monotonic())
            await q.put(item)
        # Interrupt controller state. current_turn_task runs one turn's LLM+TTS
        # as a cancellable unit; live_epoch is the only turn whose audio may be
        # relayed to the edge. Both mutate across the closures below.
        current_turn_task: asyncio.Task | None = None
        live_epoch: int | None = None
        # Finalized utterances awaiting a turn. from_stt() only ever produces
        # into this; turn_runner() consumes. That hand-off is what keeps STT
        # ingestion running while the agent is thinking or speaking.
        turn_requests: asyncio.Queue = asyncio.Queue(maxsize=TURN_QUEUE_MAX)
        # True from USER_SPEECH_STARTED until the edge endpoints that utterance.
        # A turn is never dispatched to the LLM while this is set: the user is
        # mid-sentence, so anything we generated would already be stale.
        user_speaking = False
        M_SESSIONS.inc()
        session_id = None
        session_started_ms = 0
        session_metadata: dict = {}
        session_totals = {"audio_frames": 0, "audio_bytes": 0, "turns": 0,
                          "stt_segments": 0, "llm_calls": 0}
        active_turn = 0
        # Edge endpointer timestamp per turn. It is the user-facing start point
        # for server-side time-to-first-audio (TTFA) measurement.
        # Edge endpoint per turn on the MONOTONIC clock: the start point for
        # measuring elapsed time-to-first-audio, immune to wall-clock steps.
        # The wall-clock instant itself needs no map -- it is recorded directly
        # as vad.completed's event_time_ms.
        turn_endpoint_mono: dict[int, float] = {}
        # Per-turn phase measurements are finalized when the edge reports that
        # the first audio frame reached the WebRTC track.
        turn_phase: dict[int, dict] = {}
        # Turn deadline (monotonic clock), started at the same moment as
        # turn_endpoint_mono above -- turn commitment -- so it keeps ticking
        # through however long STT itself takes on this turn.
        turn_deadline_at: dict[int, float] = {}
        # Edge control and STT results are concurrent. Keep the turn currently
        # being transcribed separate from the most recently started VAD turn;
        # otherwise a fast next speech-start can mislabel the prior final result.
        stt_turn_id: int | None = None
        pending_stt_turn_ids: deque[int] = deque()
        speech_seen = False
        frames = 0
        # Complete AudioFrame messages are retained so their original timestamps
        # survive the flush. The byte-bound loop keeps at least the configured
        # duration and at most one decoded frame beyond it (normally only 10 ms).
        pre_roll = deque()
        pre_roll_bytes = 0
        history: list[dict] = []  # Redis-backed completed turns; local active-session cache
        turn_parts: list = []  # this turn's segment texts so far, in arrival order

        def emit(event_type: str, *, source: str, turn_id: int | None = None,
                 duration_ms: float | None = None, status: str = "ok",
                 occurred_at_ms: int | None = None, **payload) -> None:
            if self.telemetry and session_id:
                self.telemetry.event(session_id, event_type, source=source, turn_id=turn_id,
                                     duration_ms=duration_ms, status=status,
                                     occurred_at_ms=occurred_at_ms, **payload)

        def emit_turn_latency(turn_id: int, media_edge_queue_ms: float) -> None:
            phase = turn_phase.pop(turn_id, {})
            emit("turn.latency", source="orchestrator", turn_id=turn_id,
                 duration_ms=media_edge_queue_ms,
                 stt_paid_ms=phase.get("stt_paid_ms"),
                 llm_queue_ms=phase.get("llm_queue_ms"),
                 llm_encode_ms=phase.get("llm_encode_ms"),
                 llm_decode_ms=phase.get("llm_decode_ms"),
                 phrase_generation_ms=phase.get("phrase_generation_ms"),
                 tts_queue_ms=phase.get("tts_queue_ms"),
                 tts_first_pcm_ms=phase.get("tts_first_pcm_ms"),
                 media_edge_queue_ms=media_edge_queue_ms,
                 stt_endpoint_to_final_ms=phase.get("stt_endpoint_to_final_ms"),
                 vad_endpoint_to_first_audio_ms=phase.get("vad_endpoint_to_first_audio_ms"))

        def maybe_emit_turn_latency(turn_id: int) -> None:
            phase = turn_phase.get(turn_id, {})
            media_edge_queue_ms = phase.get("media_edge_queue_ms")
            if media_edge_queue_ms is None or "llm_decode_ms" not in phase:
                return
            emit_turn_latency(turn_id, float(media_edge_queue_ms))

        def profile_dict(profile) -> dict:
            return {
                "model": profile.model, "engine": profile.engine, "device": profile.device,
                "compute_type": profile.compute_type, "queue_ms": profile.queue_ms,
                "compute_ms": profile.compute_ms, "encode_ms": profile.encode_ms,
                "decode_ms": profile.decode_ms, "audio_seconds": profile.audio_seconds,
                "padded_seconds": profile.padded_seconds,
                "realtime_factor": profile.realtime_factor,
                "attributes": dict(profile.attributes),
            }

        # Running total of edge VAD compute, for the cumulative-to-delta
        # conversion in vad_proc_ms() below.
        vad_proc_total_ms = 0

        def vad_proc_ms(c) -> float:
            """Per-control VAD compute in ms, from either edge generation.

            duration_ms is a PER-EVENT duration everywhere else in the schema,
            but the edge historically filled Control.proc_ms with `v.processNS`
            -- a counter accumulated over the whole session. Stored as-is it
            produced a monotonically climbing "duration" (observed 1 -> 295 ms
            across one call), making every average or quantile over VAD
            duration_ms meaningless.

            A current edge sends the delta in proc_ms and advertises itself by
            also publishing vad_process_total_ms. An older one sends only the
            cumulative value, which is differenced here so an un-rebuilt edge
            binary still yields correct per-event numbers.
            """
            nonlocal vad_proc_total_ms
            meta = c.metadata
            if "vad_process_total_ms" in meta:
                return float(c.proc_ms)
            try:
                total = float(meta.get("vad_process_ms", c.proc_ms))
            except (TypeError, ValueError):
                return float(c.proc_ms)
            # A counter that went backwards is not cumulative; pass it through.
            delta = total - vad_proc_total_ms if total >= vad_proc_total_ms else total
            vad_proc_total_ms = max(vad_proc_total_ms, total)
            return delta

        def queue_stt_turn(turn_id: int) -> None:
            """Register a VAD turn once, preserving STT stream order."""
            if turn_id and turn_id != stt_turn_id and turn_id not in pending_stt_turn_ids:
                pending_stt_turn_ids.append(turn_id)

        async def forward_audio(a):
            M_FWD.inc()
            session_totals["audio_bytes"] += len(a.pcm16)
            await q_put(asr_pb2.AsrClientEvent(audio=asr_pb2.AudioChunk(
                session_id=a.session_id or session_id, pcm16=a.pcm16, ts_ms=a.ts_ms)))

        def buffer_pre_roll(a):
            nonlocal pre_roll_bytes
            if self.vad_preroll_bytes == 0:
                M_DROP.inc()
                return
            pre_roll.append(a)
            pre_roll_bytes += len(a.pcm16)
            # Removing the oldest complete frame must not take us below the
            # configured prefix. This handles variable Opus packet durations
            # without slicing PCM or falsifying the retained frame's timestamp.
            while (pre_roll and
                   pre_roll_bytes - len(pre_roll[0].pcm16) >= self.vad_preroll_bytes):
                old = pre_roll.popleft()
                pre_roll_bytes -= len(old.pcm16)
                M_DROP.inc()

        def discard_pre_roll():
            nonlocal pre_roll_bytes
            if pre_roll:
                M_DROP.inc(len(pre_roll))
                pre_roll.clear()
                pre_roll_bytes = 0

        async def flush_pre_roll():
            nonlocal pre_roll_bytes
            while pre_roll:
                await forward_audio(pre_roll.popleft())
            pre_roll_bytes = 0

        async def to_stt():
            while True:
                item = await q.get()
                if q_times:
                    q_times.popleft()
                if item is None:
                    return
                yield item

        async def put_outbound(ev, epoch: int | None = None):
            """Queue a ServerEvent for relay. epoch=None always relays (replies,
            control); a tagged audio frame relays only while its turn is live."""
            await outbound.put((epoch, ev))

        def control_event(ctype, turn_id: int = 0, reason: str | None = None):
            """Build a downstream Control. Agent-turn lifecycle controls are always
            queued untagged (epoch None) so a cancelled turn's own AGENT_TURN_ENDED
            still reaches the edge after its audio has been dropped."""
            c = edge_pb2.Control(session_id=session_id, type=ctype, ts_ms=_now_ms(),
                                 turn_id=turn_id)
            if reason:
                c.metadata["reason"] = reason
            return edge_pb2.ServerEvent(control=c)

        async def interrupt(reason: str):
            """Barge-in / turn supersede. Cancels the in-flight turn task, drops
            its still-queued audio (via live_epoch), and signals the edge to stop
            playback. The edge also cuts playback locally on its own VAD, so the
            STOP_PLAYBACK below is the belt to the edge's suspenders."""
            nonlocal current_turn_task, live_epoch
            cancelled = live_epoch
            live_epoch = None  # no queued audio may relay until a new turn opens
            # cancelling() guard: run_turn's CancelledError handler awaits Redis
            # to persist the interrupted turn. A second barge-in landing inside
            # that window would re-raise CancelledError at the Redis await and
            # lose the turn from BOTH Redis and history. One cancel is enough --
            # the turn is already unwinding.
            if current_turn_task is not None:
                _cancel_task_once(current_turn_task)
            if session_id:
                # Best-effort: the edge-local cut is the real guarantee, so a full
                # outbound queue is not worth blocking the event loop over.
                try:
                    outbound.put_nowait((None, control_event(
                        edge_pb2.Control.STOP_PLAYBACK, cancelled or 0, reason)))
                except asyncio.QueueFull:
                    log.warning("outbound full; STOP_PLAYBACK relies on edge-local cut for %s", session_id)
            if cancelled is not None:
                # A true barge-in: speech overlapped an active agent turn.
                # Every speech-start is counted separately, unconditionally,
                # where USER_SPEECH_STARTED is handled below.
                M_CONTROL.labels("barge_in").inc()
                emit("turn.interrupted", source="orchestrator", turn_id=cancelled, reason=reason)

        def remember_user_turn(user_text: str) -> dict:
            """Create this utterance's prompt-history record the moment STT
            finalizes it -- before the LLM is even consulted, and regardless of
            whether the agent will ever answer it.

            What the user said is conversation history unconditionally. Building
            the record here (rather than at reply time) is what guarantees a
            barged-in, superseded, failed or rejected turn still reaches the next
            prompt, and keeps utterances in spoken order even when a later turn
            finishes first. finalize_turn() fills in ``assistant_text`` later.
            """
            record = {"user_text": user_text, "assistant_text": None}
            if LLM_HISTORY_MAX_TURNS:
                history.append(record)
                # LLM_HISTORY_MAX_TURNS is the PROMPT window, applied when the
                # prompt is built. The cache itself must additionally hold every
                # in-flight record (this turn's own, plus anything queued behind
                # it), or a turn would evict the very context its own prompt
                # still needs. Cheap: a few dozen short strings.
                cache_max = LLM_HISTORY_MAX_TURNS + TURN_QUEUE_MAX + 1
                if len(history) > cache_max:
                    del history[:-cache_max]
            return record

        async def finalize_turn(sid: str, user_text: str, assistant_text: str | None,
                                turn_id: int, stt_proc_ms, llm_proc_ms, interrupted: bool,
                                delivery: str = "unknown", record: dict | None = None):
            """Persist one turn to Redis exactly once and attach the reply (if any)
            to the history record remember_user_turn() already created.

            Deliberately NOT gated on the Redis write succeeding: a Redis blip
            must not erase the agent's own in-call memory of what was said."""
            if record is not None:
                if assistant_text:
                    record["assistant_text"] = assistant_text
                # Claimed before the await: the recovery paths below persist a
                # turn only if no one else already did, and re-entering here
                # mid-await must not produce a second Redis row for one turn.
                record["persisted"] = True
            await self._store("record_turn", sid, user_text, assistant_text,
                              stt_proc_ms, llm_proc_ms, interrupted, delivery)

        async def call_tts_clause(sid: str, text: str, turn_id: int, state: dict,
                                  deadline: float | None = None) -> str:
            """Synthesize ONE clause and fan its PCM into epoch-tagged 20-ms frames.
            `state` carries per-turn aggregates across clauses so telemetry stays
            turn-level. A CancelledError (barge-in) propagates to abort the RPC.

            Returns the clause's own outcome so the caller never has to guess
            whether the text it fed in was actually delivered anywhere:
              "empty"   - nothing to do (TTS off, or no text)
              "success" - the RPC streamed to completion AND produced audio
              "partial" - some audio was sent before the RPC failed
              "failed"  - no audio was ever sent (RPC failed immediately, or
                          every chunk it did return was malformed)
            """
            if not TTS_ENABLED or self.tts is None or not text:
                return "empty"
            # Recomputed fresh per clause (not once for the whole turn): the
            # budget keeps shrinking as earlier clauses stream, so a later
            # clause in a long reply gets a tighter cap than the first.
            tts_remaining = _remaining_budget(deadline)
            if tts_remaining is not None and tts_remaining <= 0:
                M_TURNS_EXPIRED.labels("tts").inc()
                emit("tts.error", source="orchestrator", turn_id=turn_id, status="error",
                     error="turn deadline already exceeded", error_class="deadline")
                return "failed"
            tts_timeout = min(TTS_TIMEOUT_S, tts_remaining) if tts_remaining is not None else TTS_TIMEOUT_S
            clause_t0 = time.perf_counter()
            sent_any = False
            # Once per clause RPC attempt, regardless of outcome: a failed,
            # canceled, or all-malformed stream is still a real TTS call, and
            # this is the count of calls the orchestrator actually made.
            M_TTS_CALLS.inc()
            try:
                stream = self.tts.SynthesizeStream(tts_pb2.SynthesizeRequest(
                    session_id=sid, text=text, ts_ms=_now_ms()), timeout=tts_timeout)
                async for response in stream:
                    if response.is_last:
                        continue
                    if response.session_id != sid or response.sample_rate_hz <= 0 or len(response.pcm16) % 2:
                        M_TTS_ERR.inc(); log.warning("dropping malformed TTS chunk for %s", sid)
                        continue
                    state["sample_rate"] = response.sample_rate_hz
                    state["total_bytes"] += len(response.pcm16)
                    if response.profile.model:
                        # One profile per synthesized sentence; sum their compute
                        # so tts.completed reports TOTAL TTS work, not the last
                        # sentence's. (Sentences stream interleaved with the LLM.)
                        state["last_profile"] = response.profile
                        state["tts_compute_ms"] += response.profile.compute_ms
                        # Sum waiting the same way as compute: one sentence's
                        # queue_ms is not the turn's. Under TTS contention this is
                        # where the latency actually went.
                        state["tts_queue_ms"] += response.profile.queue_ms
                    frame_bytes = max(2, response.sample_rate_hz // 50 * 2)
                    if state["first_chunk"]:
                        state["tts_first_queue_ms"] = response.profile.queue_ms
                        state["tts_first_pcm_ms"] = response.profile.compute_ms
                        phase = turn_phase.setdefault(turn_id, {})
                        phase["tts_queue_ms"] = state["tts_first_queue_ms"]
                        phase["tts_first_pcm_ms"] = state["tts_first_pcm_ms"]
                        state["first_chunk"] = False
                        # TTS's own clock starts at the first frame dispatched, so
                        # tts.completed.duration_ms is the TTS active window, not
                        # the whole turn (which starts before the LLM even runs).
                        state["tts_t0"] = time.perf_counter()
                        first_audio_at_ms = _now_ms()
                        # Both ends are this process's wall clock (see the
                        # END_TURN handler), so the difference is a real
                        # elapsed time. It is NOT clamped with max(0, ...):
                        # that clamp is what silently laundered every
                        # cross-clock subtraction into a plausible-looking 0 ms
                        # and dragged the fleet TTFA p50 to zero. A negative
                        # value now means a genuine bug, so surface it as a
                        # warning and record NULL (excluded from quantiles)
                        # rather than a fabricated number.
                        endpoint_mono = turn_endpoint_mono.get(turn_id)
                        ttfa_ms = None
                        ttfa_missing = None
                        if endpoint_mono is None:
                            # No END_TURN was ever seen for this turn -- it came
                            # from STT's own close-flush or the active_turn==0
                            # fallback, so there is no caller endpoint to
                            # measure from. Naming the reason keeps these turns
                            # countable on the dashboard instead of silently
                            # absent from the KPI.
                            ttfa_missing = "no_endpoint"
                        else:
                            elapsed = round((time.monotonic() - endpoint_mono) * 1000)
                            if elapsed >= 0:
                                ttfa_ms = elapsed
                            else:
                                ttfa_missing = "negative_elapsed"
                                log.warning(
                                    "turn %s for %s: first audio %d ms BEFORE its endpoint; "
                                    "recording NULL time-to-first-audio", turn_id, sid, -elapsed)
                        emit("tts.first_audio_queued", source="orchestrator", turn_id=turn_id,
                             occurred_at_ms=first_audio_at_ms, duration_ms=ttfa_ms,
                             time_to_first_audio_ms=ttfa_ms, ttfa_missing_reason=ttfa_missing,
                             sample_rate_hz=response.sample_rate_hz,
                             audio_bytes=len(response.pcm16), service_proc_ms=round(response.profile.compute_ms),
                             profile={"model": response.profile.model, "engine": response.profile.engine,
                                      "device": response.profile.device, "compute_type": response.profile.compute_type,
                                      "queue_ms": response.profile.queue_ms, "compute_ms": response.profile.compute_ms,
                                      "audio_seconds": response.profile.audio_seconds,
                                      "realtime_factor": response.profile.realtime_factor,
                                      "attributes": dict(response.profile.attributes)})
                    for offset in range(0, len(response.pcm16), frame_bytes):
                        await put_outbound(edge_pb2.ServerEvent(audio=edge_pb2.AudioFrame(
                            session_id=sid, pcm16=response.pcm16[offset:offset + frame_bytes],
                            sample_rate_hz=response.sample_rate_hz, ts_ms=_now_ms(),
                            turn_id=turn_id)), epoch=turn_id)
                        M_TTS_FRAMES.inc()
                        sent_any = True
                # The latency is this one call's round trip, not turn-elapsed-
                # so-far, and is observed regardless of outcome (below too) so
                # it reflects every attempt, not just the fast successful ones.
                M_TTS_LATENCY.observe(time.perf_counter() - clause_t0)
                # Reaching stream completion does not by itself mean anything
                # was delivered -- every chunk could have been malformed and
                # dropped above, in which case this was really a no-op call.
                return "success" if sent_any else "failed"
            except grpc.aio.AioRpcError as e:
                M_TTS_LATENCY.observe(time.perf_counter() - clause_t0)
                M_TTS_ERR.inc()
                log.warning("TTS clause failed for %s: %s", sid, e.details())
                emit("tts.error", source="orchestrator", turn_id=turn_id, status="error", error=e.details())
                return "partial" if sent_any else "failed"
            except Exception:
                # A malformed chunk or unexpected fault on one clause must not
                # crash the turn task; barge-in (CancelledError) still propagates.
                M_TTS_LATENCY.observe(time.perf_counter() - clause_t0)
                M_TTS_ERR.inc()
                log.exception("TTS clause failed unexpectedly for %s", sid)
                emit("tts.error", source="orchestrator", turn_id=turn_id, status="error",
                     error_class="unexpected")
                return "partial" if sent_any else "failed"

        async def run_turn(sid: str, text: str, turn_id: int, stt_proc_ms,
                           deadline: float | None = None, record: dict | None = None):
            """One cancellable turn. The LLM token stream is split into clauses and
            each is synthesized as it is ready, so first audio lands after the
            first sentence, not the whole reply. On barge-in the task is cancelled
            and the prefix actually spoken is persisted as interrupted."""
            t0 = time.perf_counter()

            async def _reject_before_inference(counter, log_reason: str, error_text: str):
                """STT already ran for this turn (that's how we got here), so a
                turn rejected before ever reaching the LLM is still persisted
                truthfully as attempted-but-rejected, never silently dropped."""
                counter.inc()
                log.warning("turn %s rejected for %s: %s", turn_id, sid, log_reason)
                await put_outbound(control_event(edge_pb2.Control.AGENT_TURN_STARTED, turn_id))
                emit("turn.agent_started", source="orchestrator", turn_id=turn_id)
                emit("turn.rejected", source="orchestrator", turn_id=turn_id, status="error", error=error_text)
                await finalize_turn(sid, text, None, turn_id, stt_proc_ms, None,
                                    interrupted=False, delivery="none", record=record)
                await put_outbound(control_event(edge_pb2.Control.AGENT_TURN_ENDED, turn_id, "error"))
                emit("turn.agent_ended", source="orchestrator", turn_id=turn_id, reason="error", status="error")

            # Turn deadline: the clock started at END_TURN (turn commitment),
            # so if STT itself took long enough to exhaust the whole budget,
            # never even attempt admission on an answer that can no longer be
            # timely -- "reject aged STT results before starting the LLM."
            if deadline is not None and time.monotonic() >= deadline:
                await _reject_before_inference(M_TURNS_EXPIRED.labels("admission"),
                                               "turn deadline already exceeded before admission",
                                               "turn deadline exceeded before admission")
                return
            # Turn admission, before any downstream work (even prompt-building):
            # bounds how many turns across ALL sessions can be running inference
            # at once, so a burst of sessions ending their turn simultaneously
            # can't all hit the LLM/TTS backends together.
            if not await self._admit_turn(deadline):
                await _reject_before_inference(M_TURNS_REJECTED,
                                               "orchestrator turn-admission capacity reached",
                                               "orchestrator turn-admission capacity reached")
                return
            # Utterances strictly BEFORE this turn's own record. Anything after
            # it is something the user said while this turn was queued or
            # running; that belongs to a later prompt, never this one.
            prior_history = history
            for index, entry in enumerate(history):
                if entry is record:
                    prior_history = history[:index]
                    break
            candidate_history = prior_history[-LLM_HISTORY_MAX_TURNS:] if LLM_HISTORY_MAX_TURNS else []
            messages = []
            # A turn the agent never answered (barge-in before any audio, LLM
            # error, rejected admission) still contains something the user
            # actually said, so it must reach the prompt. The LLM contract wants
            # strictly alternating USER/ASSISTANT history pairs, so an unanswered
            # user turn is carried forward and merged into the next real user
            # message -- ultimately this turn's own text. Nothing the user said
            # is dropped; no assistant reply is invented to pad the pair.
            pending_user: list[str] = []
            for turn in candidate_history:
                u, a = turn.get("user_text"), turn.get("assistant_text")
                if u:
                    pending_user.append(u)
                if not a:
                    continue  # nothing was delivered; this user text stays pending
                merged_user = "\n".join(pending_user)
                pending_user = []
                if not merged_user:
                    continue  # assistant text with no user text is unusable alone
                messages.extend((
                    llm_pb2.ChatMessage(role=llm_pb2.ChatMessage.USER, content=merged_user),
                    llm_pb2.ChatMessage(role=llm_pb2.ChatMessage.ASSISTANT, content=a),
                ))
            merged_count = len(pending_user)  # unanswered earlier turns folded in
            pending_user.append(text)
            messages.append(llm_pb2.ChatMessage(
                role=llm_pb2.ChatMessage.USER, content="\n".join(pending_user)))
            if merged_count:
                # Visible in the worker's own prompt dump: the current user
                # message intentionally carries earlier barged-in utterances.
                log.info("turn %s: merged %d unanswered user turn(s) into the prompt for %s",
                         turn_id, merged_count, sid)
            phase = turn_phase.setdefault(turn_id, {})
            phase["llm_started_mono"] = time.monotonic()
            stt_completed_mono = phase.get("stt_completed_mono")
            if stt_completed_mono is not None:
                phase["llm_queue_ms"] = round(
                    (phase["llm_started_mono"] - stt_completed_mono) * 1000, 3)
            emit("llm.started", source="orchestrator", turn_id=turn_id, input_text=text,
                 history_turns=len(messages) // 2, message_count=len(messages))

            turn_ended_sent = False

            async def announce_turn_ended(reason: str):
                """Tell the edge this turn will produce nothing further. Idempotent:
                the first reason wins, so the cancelled path's "interrupted" is not
                overwritten by the outer error handler."""
                nonlocal turn_ended_sent
                if turn_ended_sent:
                    return
                turn_ended_sent = True
                await put_outbound(control_event(
                    edge_pb2.Control.AGENT_TURN_ENDED, turn_id, reason))
                emit("turn.agent_ended", source="orchestrator", turn_id=turn_id,
                     reason=reason, status="error" if reason == "error" else "ok")

            clause_q: asyncio.Queue = asyncio.Queue(maxsize=CLAUSE_QUEUE_MAX)
            with _QLOCK:
                _ACTIVE_CLAUSE_QUEUES.add(clause_q)
            reply_parts: list[str] = []
            spoken_parts: list[str] = []  # only clauses call_tts_clause fully delivered
            clauses_attempted = 0
            clauses_failed = 0
            final_profile = llm_pb2.InferenceProfile()
            state = {"t0": t0, "llm_ok": False, "llm_proc_ms": None, "first_chunk": True,
                     "total_bytes": 0, "sample_rate": 0, "last_profile": None,
                     "tts_t0": None, "tts_compute_ms": 0.0, "tts_queue_ms": 0.0,
                     "llm_first_token_mono": None, "phrase_generation_ms": None,
                     "tts_first_pcm_ms": None, "tts_first_queue_ms": None}

            async def produce():
                """Stream LLM deltas, split into clauses, hand off to the consumer.
                Never raises for a backend failure — the turn just yields no reply."""
                buf = ""
                # True until the turn's first clause has been extracted, so
                # only that one gets the early-flush treatment (see
                # extract_clauses' docstring) -- every clause after it still
                # waits for a real sentence boundary or the max-chars cut.
                first_clause_pending = FIRST_CLAUSE_CHARS > 0
                # Time spent inside clause_q.put() waiting for the bounded
                # queue to free a slot -- i.e. waiting on the TTS consumer,
                # not on the LLM. Subtracted below so a slow/contended TTS
                # stage can never inflate the measured LLM time, even though
                # produce() only ever measures its own single elapsed clock.
                blocked_s = 0.0
                try:
                    if not LLM_ENABLED:
                        return
                    llm_remaining = _remaining_budget(deadline)
                    if llm_remaining is not None and llm_remaining <= 0:
                        M_TURNS_EXPIRED.labels("llm").inc()
                        emit("llm.completed", source="orchestrator", turn_id=turn_id, duration_ms=0,
                             status="error", error="turn deadline already exceeded", error_class="deadline")
                        return
                    stream = self.llm.GenerateStream(llm_pb2.GenerateRequest(
                        session_id=sid, text=text, ts_ms=_now_ms(), messages=messages),
                        timeout=llm_remaining)
                    async for chunk in stream:
                        if chunk.is_final:
                            final_profile.CopyFrom(chunk.profile); continue
                        if chunk.session_id != sid or not chunk.text_delta:
                            continue
                        if state["llm_first_token_mono"] is None:
                            state["llm_first_token_mono"] = time.monotonic()
                        reply_parts.append(chunk.text_delta)
                        buf += chunk.text_delta
                        clauses, buf = extract_clauses(
                            buf, MAX_CLAUSE_CHARS,
                            first_clause_chars=FIRST_CLAUSE_CHARS if first_clause_pending else None)
                        if clauses:
                            first_clause_pending = False
                        for c in clauses:
                            wait_t0 = time.perf_counter()
                            await clause_q.put(c)
                            blocked_s += time.perf_counter() - wait_t0
                    leftover = buf.strip()
                    state["llm_ok"] = True
                    # Measured at the LLM stream's own end, net of time spent
                    # blocked handing clauses to TTS above -- that wait is TTS
                    # backpressure, not LLM processing time.
                    state["llm_proc_ms"] = round((time.perf_counter() - t0 - blocked_s) * 1000)
                    if leftover:
                        await clause_q.put(leftover)
                except grpc.aio.AioRpcError as e:
                    M_LLM_ERR.inc()
                    log.warning("LLM call failed for %s: %s", sid, e.details())
                    emit("llm.completed", source="orchestrator", turn_id=turn_id,
                         duration_ms=(time.perf_counter() - t0) * 1000, status="error",
                         error=e.details(), error_class="grpc")
                except Exception:
                    # A bad/slow/down LLM must never crash the turn task: the user
                    # turn is still recorded (assistant=None), only the reply is
                    # lost. Barge-in (CancelledError) still propagates.
                    M_LLM_ERR.inc()
                    log.exception("LLM call failed unexpectedly for %s", sid)
                    emit("llm.completed", source="orchestrator", turn_id=turn_id,
                         duration_ms=(time.perf_counter() - t0) * 1000, status="error",
                         error_class="unexpected")
                finally:
                    await clause_q.put(None)  # close the clause stream for the consumer

            async def consume():
                nonlocal clauses_attempted, clauses_failed
                while True:
                    clause = await clause_q.get()
                    if clause is None:
                        return
                    if state["first_chunk"]:
                        first_token_mono = state["llm_first_token_mono"]
                        if first_token_mono is not None:
                            state["phrase_generation_ms"] = round(
                                (time.monotonic() - first_token_mono) * 1000, 3)
                            turn_phase.setdefault(turn_id, {})["phrase_generation_ms"] = state["phrase_generation_ms"]
                        emit("tts.started", source="orchestrator", turn_id=turn_id, input_text=text)
                    outcome = await call_tts_clause(sid, clause, turn_id, state, deadline)
                    if outcome == "empty":
                        continue  # TTS disabled or nothing to say; not an attempt
                    clauses_attempted += 1
                    if outcome == "success":
                        # Never append a clause TTS did not fully deliver: a
                        # half-failed clause must not be recorded as heard.
                        spoken_parts.append(clause)
                    else:
                        clauses_failed += 1

            def tts_completed(status: str):
                if state["first_chunk"]:
                    return  # no audio was produced
                sr, lp = state["sample_rate"], state["last_profile"]
                audio_s = state["total_bytes"] / 2 / sr if sr else 0
                compute_ms = state["tts_compute_ms"]           # total TTS synthesis work
                rtf = (compute_ms / 1000 / audio_s) if audio_s else 0
                # duration_ms is the TTS active window (first frame -> last frame),
                # NOT the whole turn; service_proc_ms/compute_ms is total synthesis.
                tts_wall_ms = (time.perf_counter() - state["tts_t0"]) * 1000 if state["tts_t0"] else compute_ms
                emit("tts.completed", source="tts", turn_id=turn_id, status=status,
                     duration_ms=tts_wall_ms, service_proc_ms=round(compute_ms),
                     audio_seconds=audio_s, sample_rate_hz=sr, audio_bytes=state["total_bytes"],
                     profile=({"model": lp.model, "engine": lp.engine, "device": lp.device,
                               "compute_type": lp.compute_type, "queue_ms": state["tts_queue_ms"],
                               "compute_ms": compute_ms, "audio_seconds": audio_s,
                               "realtime_factor": rtf, "attributes": dict(lp.attributes)} if lp else {}))

            # The agent-turn lifecycle signal. The orchestrator is the only
            # component that knows whether a turn still has work pending: TTS is
            # idle in the gap between two clauses of the SAME turn, so a
            # per-service "idle" broadcast would be a lie about the turn. ENDED is
            # emitted on every exit path — including a failed LLM that produced no
            # audio at all — so a client can always close its turn state machine.
            await put_outbound(control_event(edge_pb2.Control.AGENT_TURN_STARTED, turn_id))
            emit("turn.agent_started", source="orchestrator", turn_id=turn_id)
            try:
                try:
                    await asyncio.gather(produce(), consume())
                except asyncio.CancelledError:
                    spoken = " ".join(spoken_parts).strip()
                    # total_bytes accumulates across every clause this turn,
                    # including one that was still mid-stream when cancelled --
                    # so it catches audio actually queued even when no single
                    # clause finished cleanly enough to land in spoken_parts.
                    any_audio_sent = state["total_bytes"] > 0
                    tts_completed("interrupted")
                    await finalize_turn(sid, text, spoken or None, turn_id, stt_proc_ms,
                                        state["llm_proc_ms"], interrupted=True,
                                        delivery=("queued" if spoken else
                                                 "partial" if any_audio_sent else "none"),
                                        record=record)
                    await announce_turn_ended("interrupted")
                    raise

                full = "".join(reply_parts).strip()
                delivered = " ".join(spoken_parts).strip()
                if state["llm_ok"] and full:
                    # state["llm_proc_ms"] is fixed at the LLM stream's own end
                    # (inside produce()); measuring here would fold in however
                    # long the concurrently-running consume()/TTS tail took.
                    llm_duration_s = (state["llm_proc_ms"] or 0) / 1000
                    M_LLM_CALLS.inc(); M_LLM_LATENCY.observe(llm_duration_s)
                    session_totals["llm_calls"] += 1
                    p = final_profile
                    phase = turn_phase.setdefault(turn_id, {})
                    phase["llm_encode_ms"] = p.prefill_ms
                    phase["llm_decode_ms"] = p.decode_ms
                    phase["phrase_generation_ms"] = state["phrase_generation_ms"]
                    phase["tts_queue_ms"] = state["tts_first_queue_ms"]
                    phase["tts_first_pcm_ms"] = state["tts_first_pcm_ms"]
                    maybe_emit_turn_latency(turn_id)
                    session_metadata["llm"] = {"model": p.model, "engine": p.engine,
                                               "device": p.device, "compute_type": p.compute_type}
                    emit("llm.completed", source="llm", turn_id=turn_id,
                         duration_ms=state["llm_proc_ms"], response_text=full,
                         service_proc_ms=state["llm_proc_ms"], history_turns=len(messages) // 2,
                         message_count=len(messages),
                         profile={"model": p.model, "engine": p.engine, "device": p.device,
                                  "compute_type": p.compute_type, "queue_ms": p.queue_ms,
                                  "prompt_build_ms": p.prompt_build_ms, "prefill_ms": p.prefill_ms,
                                  "ttft_ms": p.ttft_ms, "decode_ms": p.decode_ms, "compute_ms": p.compute_ms,
                                  "input_tokens": p.input_tokens, "output_tokens": p.output_tokens,
                                  "decode_tokens_per_second": p.decode_tokens_per_second,
                                  "attributes": dict(p.attributes)})
                    log.info('%s LLM reply (%sms): "%s"', sid, state["llm_proc_ms"], full)
                    await put_outbound(edge_pb2.ServerEvent(reply=edge_pb2.AgentReply(
                        session_id=sid, text=full, ts_ms=_now_ms(),
                        proc_ms=state["llm_proc_ms"] or 0, turn_id=turn_id)))
                    emit("reply.queued_for_relay", source="orchestrator", turn_id=turn_id, response_text=full)
                # A turn whose TTS clauses all failed must not be announced or
                # persisted as complete just because the LLM succeeded -- the
                # user never heard anything. Judge "nothing was delivered" by
                # actual bytes queued (state["total_bytes"], which accumulates
                # across every clause including a partial one), NOT by whether
                # any single clause fully completed -- a clause that streamed
                # some audio before failing mid-stream still reached the user.
                tts_active = TTS_ENABLED and self.tts is not None
                any_audio_sent = state["total_bytes"] > 0
                tts_total_failure = tts_active and clauses_attempted > 0 and not any_audio_sent
                tts_completed("error" if tts_total_failure else ("partial" if clauses_failed else "ok"))
                # Persist only what was generated (text-only deployments, where
                # AgentReply IS the delivery) or actually queued to the relay --
                # never claim "heard" without a playback ack from the edge.
                # "queued" needs a complete clause to attribute text to; a
                # turn with only partial clauses still gets an honest "partial"
                # delivery status instead of being folded into "failed".
                assistant_text = (delivered or None) if tts_active else (full or None)
                if delivered:
                    delivery = "queued"
                elif any_audio_sent:
                    delivery = "partial"
                elif tts_active and clauses_attempted:
                    delivery = "failed"
                elif not tts_active and full:
                    delivery = "queued"
                else:
                    delivery = "none"
                await finalize_turn(sid, text, assistant_text, turn_id, stt_proc_ms,
                                    state["llm_proc_ms"], interrupted=False, delivery=delivery,
                                    record=record)
                reason = "error" if (not state["llm_ok"] or tts_total_failure) else "complete"
                await announce_turn_ended(reason)
            except asyncio.CancelledError:
                raise  # already announced as interrupted above
            except BaseException:
                await announce_turn_ended("error")
                raise
            finally:
                with _QLOCK:
                    _ACTIVE_CLAUSE_QUEUES.discard(clause_q)
                self._release_turn()

        async def dispatch_turn(sid: str, text: str, turn_id: int, stt_proc_ms,
                                record: dict) -> None:
            """Hand a finalized utterance to turn_runner(). Never blocks the STT
            reader: on the (unreachable in practice) full queue the utterance is
            persisted inline instead, because losing it is not an option."""
            try:
                turn_requests.put_nowait((sid, text, turn_id, stt_proc_ms, record))
            except asyncio.QueueFull:
                M_TURNS_SUPERSEDED.labels("queue_full").inc()
                log.warning("turn queue full for %s; recording turn %s without a reply",
                            sid, turn_id)
                emit("turn.superseded", source="orchestrator", turn_id=turn_id,
                     reason="queue_full")
                await finalize_turn(sid, text, None, turn_id, stt_proc_ms, None,
                                    interrupted=False, delivery="none", record=record)

        async def turn_runner():
            """Owns turn execution, one at a time, off the STT reader's back.

            A turn is answered only if the user has not already moved past it:
            no newer utterance has begun (`turn_id < active_turn`) and the user
            is not mid-sentence right now (`user_speaking`). If either holds the
            utterance is still transcribed, still persisted and still in the next
            prompt -- it just gets no LLM or TTS spent on it, because whatever we
            generated would be answering something the user has already replaced.
            """
            nonlocal current_turn_task, live_epoch
            while True:
                request = await turn_requests.get()
                if request is None:
                    return
                sid, text, turn_id, stt_proc_ms, record = request
                if turn_id < active_turn or user_speaking:
                    reason = "newer_turn" if turn_id < active_turn else "user_speaking"
                    M_TURNS_SUPERSEDED.labels(reason).inc()
                    log.info("turn %s superseded (%s) for %s; recorded, not answered",
                             turn_id, reason, sid)
                    emit("turn.superseded", source="orchestrator", turn_id=turn_id,
                         reason=reason)
                    await finalize_turn(sid, text, None, turn_id, stt_proc_ms, None,
                                        interrupted=False, delivery="none", record=record)
                    continue
                # live_epoch must be set before the task exists, so a barge-in
                # landing in between still has an epoch to drop audio against.
                live_epoch = turn_id
                current_turn_task = asyncio.create_task(
                    run_turn(sid, text, turn_id, stt_proc_ms,
                             turn_deadline_at.get(turn_id), record))
                try:
                    await current_turn_task
                except asyncio.CancelledError:
                    # Barge-in / supersede cancelled this turn; keep serving later
                    # ones. Re-raise only if turn_runner itself is being torn down.
                    if not current_turn_task.cancelled():
                        raise
                except Exception:
                    log.exception("turn %s failed for %s (runner continues)", turn_id, sid)
                    if not record.get("persisted"):
                        await finalize_turn(sid, text, None, turn_id, stt_proc_ms, None,
                                            interrupted=False, delivery="none", record=record)

        async def from_stt():
            nonlocal turn_parts, stt_turn_id, user_speaking
            try:
                async for t in self.stt.Transcribe(to_stt()):
                    # A bridge stream is pinned to exactly one session. Do not
                    # let a faulty/reused STT stream contaminate its local
                    # history cache or Redis state.
                    if not session_id or t.session_id != session_id:
                        M_SESSION_MISMATCH.labels("stt_transcript").inc()
                        log.warning("dropping STT transcript for foreign session %s on stream %s",
                                    t.session_id, session_id)
                        continue
                    if stt_turn_id is None:
                        # A valid pipeline queues this at speech-start. The
                        # fallback retains compatibility with older edges that
                        # send only an end-turn control.
                        stt_turn_id = (pending_stt_turn_ids.popleft()
                                       if pending_stt_turn_ids else active_turn)
                    transcript_turn_id = stt_turn_id
                    M_TRANSCRIPTS.labels("turn_complete" if t.turn_complete else "mid_turn").inc()
                    log.info('%s seg=%d: "%s" (%s, %dms)', t.session_id, t.segment_index, t.text,
                             "turn-end" if t.turn_complete else "mid-cut", t.proc_ms)
                    turn_parts.append(t.text)
                    session_totals["stt_segments"] += 1
                    # Isolate one transcript's downstream work so an unexpected
                    # failure cannot kill the STT receive loop for later turns.
                    try:
                        llm_text = llm_proc_ms = None
                        # Redis retains its bounded UI feed. ClickHouse receives the
                        # final VAD-bounded transcript, not intermediate segment text.
                        await self._store("event", t.session_id, "transcript_segment",
                                          text=t.text, turn_complete=t.turn_complete,
                                          segment_index=t.segment_index, proc_ms=t.proc_ms,
                                          stt_ts_ms=t.ts_ms)
                        # occurred_at_ms is receipt time on THIS clock, not the
                        # STT host's own stamp: those two clocks differ, and the
                        # gap between them is real (up to 28 s observed when
                        # gRPC flow control backs the transcript stream up).
                        # Recording the peer's stamp put stt.completed BEFORE
                        # the stt.started that caused it, so the session
                        # dashboard's stage breakdown read negative and was
                        # floored to 0. The peer's stamp is kept in the payload.
                        emit("stt.segment_profile", source="stt", turn_id=transcript_turn_id,
                             duration_ms=t.proc_ms, occurred_at_ms=_now_ms(),
                             stt_ts_ms=_foreign_ts(t.ts_ms),
                             turn_complete=t.turn_complete, segment_index=t.segment_index,
                             profile=profile_dict(t.profile))
                        if t.turn_complete:
                            joined = " ".join(p.strip() for p in turn_parts if p.strip())
                            stt_profile = profile_dict(t.profile)
                            session_metadata["stt"] = {k: stt_profile[k] for k in ("model", "engine", "device", "compute_type")}
                            phase = turn_phase.setdefault(transcript_turn_id, {})
                            phase["stt_paid_ms"] = float(t.proc_ms)
                            phase["stt_completed_mono"] = time.monotonic()
                            endpoint_mono = turn_endpoint_mono.get(transcript_turn_id)
                            if endpoint_mono is not None:
                                phase["stt_endpoint_to_final_ms"] = round(
                                    max(0.0, (phase["stt_completed_mono"] - endpoint_mono) * 1000), 3)
                            emit("stt.completed", source="stt", turn_id=transcript_turn_id,
                                 duration_ms=t.proc_ms, occurred_at_ms=_now_ms(),
                                 stt_ts_ms=_foreign_ts(t.ts_ms),
                                 transcript=joined, segment_count=t.segment_index + 1,
                                 service_proc_ms=t.proc_ms, profile=stt_profile)
                            turn_parts = []  # next turn starts fresh
                            stt_turn_id = None
                            # A turn_complete transcript for the newest turn means
                            # that utterance is endpointed and finalized, so the
                            # user is no longer mid-sentence. Normally END_TURN
                            # already cleared this; STT's session-close flush can
                            # also land here when the edge never sent one, and
                            # without this the flag would latch on forever.
                            if transcript_turn_id >= active_turn:
                                user_speaking = False
                            if not joined:
                                # Empty-turn gate: a VAD endpoint that transcribed to
                                # nothing (silence/noise) must not burn an LLM+TTS cycle.
                                emit("turn.skipped_empty", source="orchestrator",
                                     turn_id=transcript_turn_id)
                                continue
                            # Record what the user said FIRST, then hand the turn
                            # off to turn_runner() and go straight back to reading
                            # STT. This loop must never block on inference: while
                            # it was awaiting a turn, nothing drained the STT
                            # response stream, so grpc flow control backed up into
                            # the ASR, whose bounded segment queue then overflowed
                            # and made it stop accepting audio mid-conversation --
                            # user speech that was never transcribed at all.
                            record = remember_user_turn(joined)
                            await dispatch_turn(t.session_id, joined, transcript_turn_id,
                                                t.proc_ms, record)
                    except Exception:
                        log.exception("transcript handling failed for %s (stream continues)", t.session_id)
            except grpc.aio.AioRpcError as e:
                M_STT_ERR.inc()
                log.warning("STT stream error: %s", e.details())
                emit("stt.stream_error", source="stt", turn_id=active_turn, status="error",
                     error=e.details(), error_class="grpc")
            except Exception:
                M_STT_ERR.inc()
                log.exception("from_stt loop ended unexpectedly")  # was silently swallowed before
                emit("stt.stream_error", source="stt", turn_id=active_turn, status="error",
                     error_class="unexpected")

        async def read_edge():
            try:
                async for ev in request_iterator:
                    await inbound.put(ev)
            except grpc.aio.AioRpcError as e:
                log.warning("edge stream ended abnormally for %s: %s", session_id, e.details())
            finally:
                await inbound.put(None)

        # Bounded so a slow edge (or a backed-up event loop) backpressures the
        # request_iterator reader rather than growing memory (see read_edge).
        inbound: asyncio.Queue = asyncio.Queue(maxsize=INBOUND_QUEUE_MAX)
        with _QLOCK:
            _ACTIVE_INBOUND.add(inbound)
        edge_reader = asyncio.create_task(read_edge())
        recv = asyncio.create_task(from_stt())
        runner = asyncio.create_task(turn_runner())
        inbound_get = asyncio.create_task(inbound.get())
        outbound_get = asyncio.create_task(outbound.get())
        # Once the edge stops sending (SESSION_CLOSE or stream end) we stop reading
        # inbound but keep relaying until the in-flight turn finishes and drains —
        # the edge half-closes and waits for the final reply (see runBridge). A
        # deadline bounds a hung turn so the RPC can never wedge open.
        inbound_done = False
        close_deadline = 0.0
        try:
            while True:
                if recv.done() and not inbound_done:
                    # from_stt() only ends during normal operation if the STT
                    # stream itself died (error or unexpected early close) --
                    # its SESSION_CLOSE-triggered exit happens in the teardown
                    # `finally` below, after this loop has already broken out.
                    # Without this, the edge could keep feeding audio into a
                    # dead STT pipe until the inbound queue backpressures it.
                    log.warning("STT stream ended unexpectedly for %s; closing the session", session_id)
                    emit("stt.stream_error", source="orchestrator", status="error",
                         error="stt stream ended while session was still active")
                    inbound_done = True
                    close_deadline = time.monotonic()  # STT is already gone; no grace period
                    if current_turn_task is not None:
                        _cancel_task_once(current_turn_task)
                waiters = [outbound_get] if inbound_done else [inbound_get, outbound_get]
                done, _ = await asyncio.wait(waiters, timeout=0.25,
                                             return_when=asyncio.FIRST_COMPLETED)
                # A reply can be emitted while the edge is idle: this is the
                # reason the inbound iterator runs in its own task.
                if outbound_get in done:
                    epoch, ev = outbound_get.result()
                    outbound_get = asyncio.create_task(outbound.get())
                    # Barge-in / supersede guard: a cancelled turn's audio may
                    # still be queued. Only the live turn's frames reach the edge;
                    # replies and control (epoch None) always pass.
                    if ev.WhichOneof("msg") == "audio" and epoch != live_epoch:
                        M_TTS_FRAMES_DROPPED.inc()
                    else:
                        yield ev
                # Exit only when the edge is done, the turn has finished producing,
                # and everything queued has been relayed — or the close deadline.
                if inbound_done:
                    turn_idle = ((current_turn_task is None or current_turn_task.done())
                                 and turn_requests.empty())
                    if turn_idle and outbound.empty():
                        break
                    if close_deadline and time.monotonic() > close_deadline:
                        if current_turn_task is not None and not current_turn_task.done():
                            current_turn_task.cancel()
                        break
                    continue
                if inbound_get not in done:
                    continue
                ev = inbound_get.result()
                inbound_get = asyncio.create_task(inbound.get())
                if ev is None:
                    inbound_done = True
                    close_deadline = time.monotonic() + FINAL_TURN_WAIT_S
                    continue
                kind = ev.WhichOneof("msg")
                if kind == "control":
                    c = ev.control
                    if c.type == edge_pb2.Control.SESSION_OPEN:
                        # The first open pins this stream. A second open—even
                        # for the same ID—must not reset active turn history.
                        if not c.session_id or session_id is not None:
                            M_SESSION_MISMATCH.labels("edge_control").inc()
                            log.warning("dropping duplicate/foreign session_open %s on stream %s",
                                        c.session_id, session_id)
                            continue
                        session_id = c.session_id
                        # session_summaries.started_at_ms drives the session
                        # dashboard's "Started" and "Duration"; a peer's
                        # stream-relative stamp put those in 1970 with an
                        # epoch-sized duration. Anchor on this process's clock.
                        session_started_ms = _foreign_ts(c.ts_ms) or _now_ms()
                        session_metadata = {"vad": dict(c.metadata), "orchestrator": {
                            "vad_gate": self.vad_gate, "vad_preroll_ms": VAD_PREROLL_MS}}
                        M_CONTROL.labels("session_open").inc()
                        log.info("session_open %s", session_id)
                        await self._store("open", session_id, c.ts_ms)
                        # Prime the VAD compute baseline from the open: an
                        # older edge reports its cumulative counter on every
                        # control, so the first per-event delta must be
                        # measured from what it had already accrued here.
                        vad_proc_ms(c)
                        emit("session.started", source="edge", occurred_at_ms=session_started_ms,
                             edge_ts_ms=_foreign_ts(c.ts_ms), metadata=session_metadata)
                        history[:] = await self._recent_turns(session_id)
                        await q_put(asr_pb2.AsrClientEvent(control=asr_pb2.Control(
                            session_id=session_id, type=asr_pb2.Control.SESSION_OPEN, ts_ms=c.ts_ms)))
                    elif not session_id or c.session_id != session_id:
                        M_SESSION_MISMATCH.labels("edge_control").inc()
                        log.warning("dropping control for foreign session %s on stream %s",
                                    c.session_id, session_id)
                        continue
                    elif c.type == edge_pb2.Control.SESSION_CLOSE:
                        M_CONTROL.labels("session_close").inc()
                        log.info("session_close %s", session_id)
                        # Don't abandon an in-flight reply: stop reading input but
                        # keep relaying until the turn finishes (bounded by the
                        # deadline). The edge is waiting for this final audio.
                        inbound_done = True
                        close_deadline = time.monotonic() + FINAL_TURN_WAIT_S
                        continue
                    elif c.type == edge_pb2.Control.USER_SPEECH_STARTED:
                        active_turn += 1
                        session_totals["turns"] = active_turn
                        queue_stt_turn(active_turn)
                        # Emitted synchronously, right after this turn_id is
                        # queued and before any await below: from_stt() can
                        # pop it and race ahead (dispatching llm.started, even
                        # turn.interrupted) the moment this coroutine yields,
                        # so recording vad.started after an await let a
                        # turn's own start event land in ClickHouse AFTER
                        # events that causally depend on it -- corrupting
                        # event_time_ms ordering for the session-detail
                        # dashboard's per-turn timing panels.
                        emit("vad.started", source="edge", turn_id=active_turn,
                             occurred_at_ms=_now_ms(), duration_ms=vad_proc_ms(c),
                             edge_ts_ms=_foreign_ts(c.ts_ms),
                             metadata=dict(c.metadata))
                        # Unconditional: every speech-start, whether or not it
                        # actually overlapped an active agent turn. interrupt()
                        # below increments the "barge_in" label itself, and only
                        # when there was a live turn to cut off.
                        M_CONTROL.labels("speech_started").inc()
                        speech_seen = True
                        # Held until END_TURN. turn_runner() refuses to answer a
                        # turn while this is set -- the user is still talking.
                        user_speaking = True
                        log.info("barge-in: user_speech_started %s", session_id)
                        # Interrupt controller: the user speaking over the agent
                        # aborts the in-flight LLM+TTS turn, drops its queued audio,
                        # and signals the edge to stop playback (which it also cut
                        # locally on its own VAD).
                        await interrupt("barge_in")
                        await self._store("event", session_id, "user_speech_started",
                                          ts_ms=c.ts_ms)
                        if self.vad_gate:
                            await flush_pre_roll()
                    elif c.type == edge_pb2.Control.END_TURN:
                        # Edge's endpointer fired: the user's turn just ended
                        # (silence past hangover). Forward it to STT so STT
                        # finalizes this turn and resets its growing window —
                        # without this the window grows for the whole call and
                        # per-partial compute climbs without bound.
                        M_CONTROL.labels("end_turn").inc()
                        if active_turn == 0:
                            active_turn = 1
                            session_totals["turns"] = active_turn
                            queue_stt_turn(active_turn)
                        # Receipt time on THIS process's clock -- the same clock
                        # that stamps tts.first_audio_queued, so their difference
                        # is the caller-felt latency. The edge's own stamp is a
                        # drifting media-clock cursor and is kept as payload only.
                        endpoint_at_ms = _now_ms()
                        # The ELAPSED time to first audio is measured on the
                        # monotonic clock, not by subtracting two _now_ms()
                        # readings: time.time() can step (NTP correction,
                        # manual set) mid-turn, which would silently add or
                        # subtract that step from the caller-felt latency.
                        # endpoint_at_ms above stays wall clock because
                        # event_time_ms must remain orderable and partitionable
                        # against every other row in ClickHouse.
                        turn_endpoint_mono[active_turn] = time.monotonic()
                        if ORCH_TURN_DEADLINE_S > 0:
                            turn_deadline_at[active_turn] = time.monotonic() + ORCH_TURN_DEADLINE_S
                        # One entry per turn, never otherwise removed: a long
                        # call would keep every turn's endpoint alive for the
                        # life of the session. Only turns still in flight can
                        # be read, so retain a generous recent window.
                        for stale in [t for t in turn_endpoint_mono if t < active_turn - TURN_TIMING_RETAIN]:
                            turn_endpoint_mono.pop(stale, None)
                            turn_deadline_at.pop(stale, None)
                        log.info("end_of_turn %s", session_id)
                        await self._store("event", session_id, "end_turn", ts_ms=c.ts_ms)
                        emit("vad.completed", source="edge", turn_id=active_turn,
                             occurred_at_ms=endpoint_at_ms, duration_ms=vad_proc_ms(c),
                             edge_ts_ms=_foreign_ts(c.ts_ms),
                             metadata=dict(c.metadata), audio_frames=session_totals["audio_frames"],
                             audio_bytes=session_totals["audio_bytes"])
                        session_metadata.setdefault("vad", {}).update(dict(c.metadata))
                        emit("stt.started", source="orchestrator", turn_id=active_turn,
                             occurred_at_ms=endpoint_at_ms)
                        await q_put(asr_pb2.AsrClientEvent(control=asr_pb2.Control(
                            session_id=session_id, type=asr_pb2.Control.END_TURN, ts_ms=c.ts_ms)))
                        speech_seen = False  # re-arm the VAD gate for the next turn
                        user_speaking = False  # this utterance is closed; it may be answered
                        # Normally empty because speech-start flushed it. This
                        # also makes a stray end event safely reset a candidate.
                        discard_pre_roll()
                    elif c.type == edge_pb2.Control.PLAYBACK_STARTED:
                        # proc_ms is measured entirely on the edge: first PCM
                        # receipt -> first frame emitted to the WebRTC track.
                        # This avoids subtracting clocks from two processes.
                        if c.turn_id > 0 and c.proc_ms >= 0:
                            emit("media_edge.playback_started", source="edge",
                                 turn_id=c.turn_id, duration_ms=c.proc_ms,
                                 media_edge_queue_ms=c.proc_ms,
                                 edge_ts_ms=_foreign_ts(c.ts_ms))
                            turn_phase.setdefault(c.turn_id, {})["media_edge_queue_ms"] = float(c.proc_ms)
                            edge_ttfa = c.metadata.get("vad_endpoint_to_first_audio_ms")
                            if edge_ttfa not in (None, ""):
                                try:
                                    turn_phase.setdefault(c.turn_id, {})["vad_endpoint_to_first_audio_ms"] = float(edge_ttfa)
                                except (TypeError, ValueError):
                                    log.warning("invalid edge TTFA for turn %s: %r", c.turn_id, edge_ttfa)
                            maybe_emit_turn_latency(c.turn_id)
                elif kind == "audio":
                    # The media edge supplies an ID for every frame. Reject an
                    # absent or foreign one rather than silently attaching it to
                    # the stream's pinned session.
                    if not session_id or ev.audio.session_id != session_id:
                        M_SESSION_MISMATCH.labels("edge_audio").inc()
                        log.warning("dropping audio for foreign session %s on stream %s",
                                    ev.audio.session_id, session_id)
                        continue
                    frames += 1
                    session_totals["audio_frames"] += 1
                    if (not self.vad_gate) or speech_seen:
                        await forward_audio(ev.audio)
                    else:
                        buffer_pre_roll(ev.audio)
                    if frames % 200 == 0:
                        log.info("  %s: %d frames forwarded", session_id, frames)
        finally:
            # Buffered audio with no subsequent speech-start is genuine silence
            # (or an incomplete candidate) and must not leak into STT on close.
            discard_pre_roll()
            # Stop any in-flight turn; its cancel handler persists the spoken
            # prefix. Give it a moment to finish that write, then move on.
            if current_turn_task is not None and not current_turn_task.done():
                _cancel_task_once(current_turn_task)
                try:
                    await asyncio.wait_for(asyncio.shield(current_turn_task), timeout=5)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
                except Exception:
                    log.exception("turn task teardown failed for %s", session_id)
            for task in (inbound_get, outbound_get, edge_reader):
                if not task.done():
                    task.cancel()
            M_SESSIONS.dec()
            with _QLOCK:
                _ACTIVE_QUEUES.discard(q)
                _ACTIVE_QUEUE_TIMES.pop(id(q_times), None)
                _ACTIVE_OUTBOUND.discard(outbound)
                _ACTIVE_INBOUND.discard(inbound)
            # The queue is bounded now, so a wedged STT could otherwise block the
            # close forever. Bound the wait; a dropped close-marker only costs the
            # trailing final transcript, never a hung session.
            async def _close_put(item) -> None:
                try:
                    await asyncio.wait_for(q.put(item), timeout=2)
                except asyncio.TimeoutError:
                    log.warning("STT queue full at close for %s; skipping close marker", session_id)
            if session_id:  # tell STT the turn is over and let it emit the final
                await _close_put(asr_pb2.AsrClientEvent(control=asr_pb2.Control(
                    session_id=session_id, type=asr_pb2.Control.SESSION_CLOSE, ts_ms=0)))
            await _close_put(None)
            try:
                await asyncio.wait_for(recv, timeout=10)  # drain trailing final transcript first
            except asyncio.TimeoutError:
                recv.cancel()
            # Only now stop the runner: from_stt() above may have dispatched a
            # final turn while draining STT, and the edge is still waiting for
            # that last reply (see runBridge). The sentinel lets the runner
            # finish what it has; the bound stops a wedged turn wedging close.
            try:
                turn_requests.put_nowait(None)
            except asyncio.QueueFull:
                runner.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(runner), timeout=FINAL_TURN_WAIT_S)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                # Cancel the turn too: cancelling only the runner would orphan a
                # turn task still holding a turn-admission slot and still writing
                # to a queue nobody drains any more.
                runner.cancel()
                if current_turn_task is not None and not current_turn_task.done():
                    current_turn_task.cancel()
            except Exception:
                log.exception("turn runner teardown failed for %s", session_id)
            # Anything still queued was never answered, but the user did say it:
            # persist it so the next session's history is not missing a turn.
            while not turn_requests.empty():
                request = turn_requests.get_nowait()
                if request is None:
                    continue
                sid_q, text_q, turn_q, stt_ms_q, record_q = request
                if record_q.get("persisted"):
                    continue
                M_TURNS_SUPERSEDED.labels("session_closed").inc()
                emit("turn.superseded", source="orchestrator", turn_id=turn_q,
                     reason="session_closed")
                await finalize_turn(sid_q, text_q, None, turn_q, stt_ms_q, None,
                                    interrupted=False, delivery="none", record=record_q)
            if session_id:  # apply closed-session retention after STT drains
                await self._store("close", session_id)
                closed_at_ms = _now_ms()
                emit("session.closed", source="orchestrator", occurred_at_ms=closed_at_ms,
                     duration_ms=max(0, closed_at_ms - session_started_ms), totals=session_totals)
                if self.telemetry:
                    self.telemetry.summary(session_id, started_at_ms=session_started_ms,
                                           closed_at_ms=closed_at_ms, outcome="closed",
                                           metadata=session_metadata, totals=session_totals)
                    # Last telemetry touch for this session_id -- safe to drop
                    # its sequence counter now (Phase 3: _sequences otherwise
                    # grows one entry per session for the life of the process).
                    self.telemetry.forget(session_id)
            turn_phase.clear()
        # from_stt may have queued a final reply while its last transcript was
        # being drained during teardown. Yield it before completing the RPC.
        # (Session is closing, so relay everything remaining regardless of epoch.)
        while not outbound.empty():
            _, ev = await outbound.get()
            yield ev


async def serve(addr: str = "[::]:50051") -> None:
    async with _replica_channel(STT_TARGETS) as stt_chan, \
               grpc.aio.insecure_channel(LLM_TARGET) as llm_chan, \
               _replica_channel(TTS_TARGETS) as tts_chan:
        sessions = await SessionStore.connect(REDIS_URL, REDIS_ACTIVE_TTL_S,
                                              REDIS_CLOSED_TTL_S, REDIS_EVENT_MAXLEN,
                                              pool_size=REDIS_POOL_SIZE, op_timeout_s=REDIS_OP_TIMEOUT_S,
                                              connect_timeout_s=REDIS_CONNECT_TIMEOUT_S,
                                              turn_retention=REDIS_TURN_RETENTION)
        telemetry = SessionLogger(CLICKHOUSE_URL, CLICKHOUSE_DATABASE, enabled=CLICKHOUSE_ENABLED,
                                  queue_maxsize=CLICKHOUSE_QUEUE_MAX,
                                  batch_size=CLICKHOUSE_BATCH_SIZE,
                                  flush_interval_s=CLICKHOUSE_FLUSH_INTERVAL_S)
        global _TELEMETRY_REF
        _TELEMETRY_REF = telemetry
        await telemetry.start()
        stt = asr_pb2_grpc.AsrStub(stt_chan)
        llm = llm_pb2_grpc.LlmStub(llm_chan)
        tts = tts_pb2_grpc.TtsStub(tts_chan)
        # A hard backstop above ORCH_MAX_SESSIONS: our own admission check
        # (Stream()'s _try_reserve_session) is what should normally reject
        # overflow with a clean, informative status; this exists so gRPC
        # itself never lets concurrent RPCs run away unbounded if that logic
        # is ever bypassed. None (unbounded) when ORCH_MAX_SESSIONS is unset.
        max_concurrent_rpcs = ORCH_MAX_SESSIONS * 2 if ORCH_MAX_SESSIONS > 0 else None
        server = grpc.aio.server(maximum_concurrent_rpcs=max_concurrent_rpcs)
        edge_pb2_grpc.add_MediaBridgeServicer_to_server(MediaBridge(stt, llm, sessions, telemetry=telemetry, tts=tts), server)
        health_servicer = health.aio.HealthServicer()
        health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
        # Redis is already verified (SessionStore.connect() pinged it above).
        # Start NOT_SERVING, then flip SERVING only after required downstreams
        # also report gRPC health SERVING. That makes orchestrator readiness a
        # real pipeline-readiness signal instead of just "the local process is up."
        await health_servicer.set("", health_pb2.HealthCheckResponse.NOT_SERVING)
        server.add_insecure_port(addr)
        await server.start()
        metrics_port = int(os.environ.get("ORCH_METRICS_PORT", "9102"))
        metrics_host = os.environ.get("ORCH_METRICS_HOST", "0.0.0.0")
        start_http_server(metrics_port, addr=metrics_host)  # own thread; never touches the event loop
        loop_lag_task = asyncio.create_task(_loop_lag_monitor())
        downstream_health_task = asyncio.create_task(_downstream_health_gate(health_servicer))
        stt_display = ",".join(STT_TARGETS)
        tts_display = ",".join(TTS_TARGETS)
        _startup_banner([
            f"gRPC      {addr}",
            f"STT       {stt_display}  |  VAD gate: {'on' if VAD_GATE else 'off'}"
            f"  |  pre-roll: {VAD_PREROLL_MS}ms",
            f"LLM       {LLM_TARGET}  |  enabled: {'yes' if LLM_ENABLED else 'no'}",
            f"TTS       {tts_display}  |  enabled: {'yes' if TTS_ENABLED else 'no'}",
            f"Metrics   http://{metrics_host}:{metrics_port}/metrics",
            f"Store     Redis  |  TTL active/closed: {REDIS_ACTIVE_TTL_S}/{REDIS_CLOSED_TTL_S}s",
            f"Telemetry ClickHouse {'on' if CLICKHOUSE_ENABLED else 'off'} | {CLICKHOUSE_URL}",
        ])
        log.info("orchestrator gRPC on %s -> STT at %s, LLM at %s (enabled=%s), TTS at %s (enabled=%s) "
                 "(vad_gate=%s, vad_preroll_ms=%d), "
                 "Redis -> %s (active_ttl_s=%d, closed_ttl_s=%d, event_maxlen=%d, "
                 "history_max_turns=%d), "
                 "metrics on http://%s:%d/metrics",
                 addr, stt_display, LLM_TARGET, LLM_ENABLED, tts_display, TTS_ENABLED, VAD_GATE, VAD_PREROLL_MS,
                 REDIS_URL, REDIS_ACTIVE_TTL_S, REDIS_CLOSED_TTL_S,
                 REDIS_EVENT_MAXLEN, LLM_HISTORY_MAX_TURNS, metrics_host, metrics_port)
        try:
            await server.wait_for_termination()
        finally:
            loop_lag_task.cancel()
            downstream_health_task.cancel()
            await telemetry.close()
            await sessions.aclose()


if __name__ == "__main__":
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass
