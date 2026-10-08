"""Non-blocking ClickHouse session telemetry writer.

The orchestrator is the only producer.  Pipeline coroutines enqueue compact
events; a single background task groups them into ClickHouse HTTP inserts so a
slow analytics database never delays media, STT, or LLM work.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import socket
import subprocess
import time
import urllib.parse
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable


log = logging.getLogger("orchestrator.session_logger")


def _pipeline_version() -> str:
    """Return an immutable release identity without making Git a runtime dependency.

    A deployment/CI value wins because production images commonly omit .git.
    Local development automatically falls back to the checked-out commit.
    """
    configured = (os.environ.get("PIPELINE_VERSION") or os.environ.get("GIT_SHA")
                  or os.environ.get("GIT_COMMIT"))
    if configured:
        return configured
    try:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result = subprocess.run(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=repo_root,
            capture_output=True, text=True, timeout=1, check=True,
        )
        commit = result.stdout.strip()
        if commit:
            return commit
    except (OSError, subprocess.SubprocessError):
        pass
    return "unversioned"


@dataclass(frozen=True)
class QueuedRow:
    table: str
    row: dict[str, Any]


class SessionLogger:
    """Batched append-only event sink with a bounded, non-blocking ingress."""

    def __init__(self, url: str, database: str = "pipeline", *, enabled: bool = True,
                 queue_maxsize: int = 10_000, batch_size: int = 100,
                 flush_interval_s: float = 0.25,
                 insert_rows: Callable[[str, list[dict[str, Any]]], None] | None = None):
        self.url = url.rstrip("/")
        self.database = database
        self.enabled = enabled
        self.queue: asyncio.Queue[QueuedRow | None] = asyncio.Queue(maxsize=queue_maxsize)
        self.batch_size = batch_size
        self.flush_interval_s = flush_interval_s
        self._insert_rows = insert_rows
        self._task: asyncio.Task | None = None
        self._closed = False
        self._stop = asyncio.Event()
        self._sequences: defaultdict[str, int] = defaultdict(int)
        self.dropped_events = 0
        # Monotonic timestamp of the last flush attempt that landed at least
        # one table's rows successfully -- None until the first one, so a
        # scrape before any traffic reads as "never", not a fabricated "now".
        # Read by orchestrator/service/server.py to export queue-age/drops/
        # last-flush as Prometheus gauges (Phase 3 telemetry-health item).
        self.last_flush_monotonic: float | None = None
        self.deployment_id = os.environ.get("TELEMETRY_DEPLOYMENT_ID") or None
        self.host_id = os.environ.get("TELEMETRY_HOST_ID") or socket.gethostname()
        self.pipeline_version = _pipeline_version()

    async def start(self) -> None:
        if self.enabled and self._task is None:
            self._task = asyncio.create_task(self._run(), name="clickhouse-session-writer")

    def event(self, session_id: str, event_type: str, *, source: str,
              turn_id: int | None = None, duration_ms: float | None = None,
              status: str = "ok", occurred_at_ms: int | None = None,
              **payload: Any) -> bool:
        """Queue an event without awaiting I/O. Returns false if telemetry is full."""
        if not self.enabled or self._closed or not session_id:
            return False
        self._sequences[session_id] += 1
        profile = payload.get("profile")
        profile = profile if isinstance(profile, dict) else {}
        attributes = profile.get("attributes")
        attributes = attributes if isinstance(attributes, dict) else {}

        def profile_value(name: str):
            value = profile.get(name)
            return value if value not in (None, "") else None

        def payload_value(name: str):
            value = payload.get(name)
            return value if value not in (None, "") else None

        def attribute_value(name: str):
            value = attributes.get(name)
            return value if value not in (None, "") else None

        def uint_value(value: Any) -> int | None:
            try:
                return int(value) if value not in (None, "") else None
            except (TypeError, ValueError):
                return None

        def float_value(value: Any) -> float | None:
            try:
                return float(value) if value not in (None, "") else None
            except (TypeError, ValueError):
                return None

        cached_prompt_tokens = uint_value(attribute_value("cached_prompt_tokens"))
        input_tokens = uint_value(profile_value("input_tokens"))
        prefix_cache_hit_pct = (
            100.0 * cached_prompt_tokens / input_tokens
            if cached_prompt_tokens is not None and input_tokens and input_tokens > 0
            else None
        )

        row = {
            "session_id": session_id,
            "event_seq": self._sequences[session_id],
            "turn_id": turn_id or 0,
            "event_time_ms": occurred_at_ms or int(time.time() * 1000),
            "recorded_at_ms": int(time.time() * 1000),
            "source": source,
            "event_type": event_type,
            "status": status,
            "duration_ms": duration_ms or 0.0,
            "schema_version": 3,
            "pipeline_version": self.pipeline_version,
            "deployment_id": attribute_value("deployment_id") or self.deployment_id,
            "host_id": attribute_value("host_id") or self.host_id,
            "service_instance_id": attribute_value("service_instance_id"),
            "model": profile_value("model"),
            "model_revision": attribute_value("model_revision"),
            "engine": profile_value("engine"),
            "device": profile_value("device"),
            "compute_type": profile_value("compute_type"),
            "gpu_uuid": attribute_value("gpu_uuid"),
            "gpu_index": uint_value(attribute_value("gpu_index")),
            "service_proc_ms": float_value(payload_value("service_proc_ms")),
            "queue_ms": float_value(profile_value("queue_ms")),
            "compute_ms": float_value(profile_value("compute_ms")),
            "prompt_build_ms": float_value(profile_value("prompt_build_ms")),
            "prefill_ms": float_value(profile_value("prefill_ms")),
            "ttft_ms": float_value(profile_value("ttft_ms")),
            "decode_ms": float_value(profile_value("decode_ms")),
            "input_tokens": input_tokens,
            "output_tokens": uint_value(profile_value("output_tokens")),
            "decode_tokens_per_second": float_value(profile_value("decode_tokens_per_second")),
            "history_turns": uint_value(payload_value("history_turns")),
            "message_count": uint_value(payload_value("message_count")),
            "max_output_tokens": uint_value(attribute_value("max_tokens")),
            "batch_size": uint_value(attribute_value("batch_size")),
            "running_requests": uint_value(attribute_value("running_requests")),
            "waiting_requests": uint_value(attribute_value("waiting_requests")),
            "kv_cache_usage_pct": float_value(attribute_value("kv_cache_usage_pct")),
            "cached_prompt_tokens": cached_prompt_tokens,
            "prefix_cache_hit_pct": prefix_cache_hit_pct,
            "prefix_cache_hit": cached_prompt_tokens > 0 if cached_prompt_tokens is not None else None,
            "encode_ms": float_value(profile_value("encode_ms")),
            "audio_seconds": float_value(profile_value("audio_seconds")),
            "padded_seconds": float_value(profile_value("padded_seconds")),
            "realtime_factor": float_value(profile_value("realtime_factor")),
            "sample_rate_hz": uint_value(payload_value("sample_rate_hz")),
            "audio_bytes": uint_value(payload_value("audio_bytes")),
            "time_to_first_audio_ms": float_value(payload_value("time_to_first_audio_ms")),
            "segment_index": uint_value(payload_value("segment_index")),
            "segment_count": uint_value(payload_value("segment_count")),
            "turn_complete": payload_value("turn_complete"),
            "stt_paid_ms": float_value(payload_value("stt_paid_ms")),
            "llm_queue_ms": float_value(payload_value("llm_queue_ms")),
            "llm_encode_ms": float_value(payload_value("llm_encode_ms")),
            "llm_decode_ms": float_value(payload_value("llm_decode_ms")),
            "phrase_generation_ms": float_value(payload_value("phrase_generation_ms")),
            "tts_queue_ms": float_value(payload_value("tts_queue_ms")),
            "tts_first_pcm_ms": float_value(payload_value("tts_first_pcm_ms")),
            "media_edge_queue_ms": float_value(payload_value("media_edge_queue_ms")),
            "stt_endpoint_to_final_ms": float_value(payload_value("stt_endpoint_to_final_ms")),
            "vad_endpoint_to_first_audio_ms": float_value(payload_value("vad_endpoint_to_first_audio_ms")),
            "error_code": payload_value("error_code"),
            "error_class": payload_value("error_class"),
            "payload_json": json.dumps(payload, ensure_ascii=False, default=str),
        }
        try:
            self.queue.put_nowait(QueuedRow("pipeline_events", row))
            return True
        except asyncio.QueueFull:
            self.dropped_events += 1
            return False

    def summary(self, session_id: str, *, started_at_ms: int, closed_at_ms: int,
                outcome: str, metadata: dict[str, Any], totals: dict[str, Any]) -> bool:
        if not self.enabled or self._closed:
            return False
        row = {
            "session_id": session_id,
            "started_at_ms": started_at_ms,
            "closed_at_ms": closed_at_ms,
            "outcome": outcome,
            "pipeline_version": self.pipeline_version,
            "metadata_json": json.dumps(metadata, ensure_ascii=False, default=str),
            "totals_json": json.dumps(totals, ensure_ascii=False, default=str),
        }
        try:
            self.queue.put_nowait(QueuedRow("session_summaries", row))
            return True
        except asyncio.QueueFull:
            self.dropped_events += 1
            return False

    def forget(self, session_id: str) -> None:
        """Drop the per-session sequence counter. Without this, `_sequences`
        (one entry per session_id, never otherwise removed) grows unbounded
        for the life of the process -- call this from the session-close
        teardown path, once no further `event()` calls for the session are
        possible."""
        self._sequences.pop(session_id, None)

    async def close(self, timeout_s: float = 2.0) -> None:
        self._closed = True
        if self._task is None:
            return
        self._stop.set()
        try:
            await asyncio.wait_for(self._task, timeout=timeout_s)
        except asyncio.TimeoutError:
            self._task.cancel()
            log.warning("ClickHouse telemetry writer did not flush before shutdown")

    async def _run(self) -> None:
        pending: list[QueuedRow] = []
        while True:
            if self._stop.is_set():
                while not self.queue.empty():
                    row = self.queue.get_nowait()
                    if row is not None:
                        pending.append(row)
                if pending:
                    await self._flush(pending)
                return
            get_task = asyncio.create_task(self.queue.get())
            stop_task = asyncio.create_task(self._stop.wait())
            done, waiting = await asyncio.wait((get_task, stop_task),
                                               timeout=self.flush_interval_s,
                                               return_when=asyncio.FIRST_COMPLETED)
            for task in waiting:
                task.cancel()
            if waiting:
                await asyncio.gather(*waiting, return_exceptions=True)
            if not done:
                if pending:
                    await self._flush(pending)
                    pending = []
                continue
            if get_task in done:
                row = get_task.result()
                if row is not None:
                    pending.append(row)
            if stop_task in done:
                continue
            if len(pending) >= self.batch_size:
                await self._flush(pending)
                pending = []

    async def _flush(self, rows: list[QueuedRow]) -> None:
        grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[row.table].append(row.row)
        for table, batch in grouped.items():
            try:
                if self._insert_rows is None:
                    await self._http_insert(table, batch)
                else:
                    result = self._insert_rows(table, batch)
                    if inspect.isawaitable(result):
                        await result
                self.last_flush_monotonic = time.monotonic()
            except Exception:
                self.dropped_events += len(batch)
                log.exception("ClickHouse insert failed; dropped %d %s rows", len(batch), table)

    async def _http_insert(self, table: str, rows: list[dict[str, Any]]) -> None:
        query = urllib.parse.urlencode({"database": self.database, "query": f"INSERT INTO {table} FORMAT JSONEachRow"})
        body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        parsed = urllib.parse.urlsplit(self.url)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("CLICKHOUSE_URL must be an http:// URL with a hostname")
        port = parsed.port or 80
        path = (parsed.path.rstrip("/") or "") + "/?" + query
        request = (
            f"POST {path} HTTP/1.1\r\nHost: {parsed.netloc}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n"
        ).encode() + body
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection(parsed.hostname, port), timeout=5)
            writer.write(request)
            await asyncio.wait_for(writer.drain(), timeout=5)
            status = await asyncio.wait_for(reader.readline(), timeout=5)
            writer.close()
            await writer.wait_closed()
        except (OSError, asyncio.TimeoutError) as exc:
            raise RuntimeError(f"ClickHouse unavailable at {self.url}: {exc}") from exc
        if not status.startswith(b"HTTP/1.1 2"):
            raise RuntimeError(f"ClickHouse insert failed: {status.decode(errors='replace').strip()}")
