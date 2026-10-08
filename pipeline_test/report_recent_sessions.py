#!/usr/bin/env python3
"""Print an aggregate voice-agent performance report from ClickHouse.

The report intentionally selects recent sessions internally and returns only
aggregates. It never prints session IDs, transcripts, replies, or event rows.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_URL = "http://127.0.0.1:8123"
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def sql_string(value: str) -> str:
    """Return a ClickHouse string literal for a user-supplied filter."""
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def line(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(values)) + " |"

    return "\n".join((line(headers), "| " + " | ".join("-" * width for width in widths) + " |", *(line(row) for row in rows)))


def number(value: Any, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    return f"{float(value):,.1f}{suffix}"


class ClickHouse:
    def __init__(self, url: str, database: str, timeout_s: float):
        self.url = url.rstrip("/")
        self.database = database
        self.timeout_s = timeout_s

    def query(self, sql: str) -> list[dict[str, Any]]:
        request = Request(
            f"{self.url}/?{urlencode({'database': self.database, 'default_format': 'JSONEachRow'})}",
            data=(sql.rstrip() + " FORMAT JSONEachRow\n").encode(),
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_s) as response:
                return [json.loads(line) for line in response.read().decode().splitlines() if line]
        except HTTPError as exc:
            detail = exc.read().decode(errors="replace").strip()
            raise RuntimeError(f"ClickHouse query failed ({exc.code}): {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach ClickHouse at {self.url}: {exc.reason}") from exc


def recent_sessions_cte(args: argparse.Namespace) -> str:
    time_filter = ""
    time_having = ""
    if args.since_hours is not None:
        seconds = round(args.since_hours * 3600)
        time_having = (
            "HAVING max(session_summaries.closed_at_ms) >= "
            f"toUnixTimestamp64Milli(now64(3) - toIntervalSecond({seconds}))"
        )
    session_limit = "" if args.since_hours is not None else f"LIMIT {args.sessions}"
    cte = f"""recent AS (
    SELECT session_id FROM (
        SELECT session_id, max(closed_at_ms) AS closed_at_ms
        FROM session_summaries
        {time_filter}
        GROUP BY session_id
        {time_having}
        ORDER BY closed_at_ms DESC
        {session_limit}
    )
)"""
    matching = []
    if args.model:
        matching.append(f"model = {sql_string(args.model)}")
    if args.engine:
        matching.append(f"engine = {sql_string(args.engine)}")
    if args.device:
        matching.append(f"device = {sql_string(args.device)}")
    if args.pipeline_version:
        matching.append(f"pipeline_version = {sql_string(args.pipeline_version)}")
    if not matching:
        return cte + ",\nselected AS (SELECT session_id FROM recent)"
    # Filters choose *sessions* that exercised the requested deployment. Apply
    # them neither to VAD/STT rows (which lack an LLM/TTS model) nor to errors
    # (which often lack profile dimensions), otherwise the funnel lies.
    return cte + f""",
selected AS (
    SELECT session_id FROM recent
    WHERE session_id IN (
        SELECT DISTINCT session_id FROM pipeline_events
        WHERE {' AND '.join(matching)}
    )
)"""


def event_filter() -> str:
    return "session_id IN (SELECT session_id FROM selected)"


def fetch_report(db: ClickHouse, args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    cte = recent_sessions_cte(args)
    where = event_filter()
    overview = db.query(f"""WITH {cte}
SELECT
    (SELECT count() FROM selected) AS selected_sessions,
    countDistinct(session_id) AS sessions_with_events,
    countIf(event_type = 'vad.completed') AS caller_endpoints,
    countIf(event_type = 'stt.completed' AND status = 'ok') AS stt_final,
    countIf(event_type = 'llm.completed' AND status = 'ok') AS llm_completed,
    countIf(event_type = 'tts.first_audio_queued' AND status = 'ok') AS agent_starts,
    countIf(event_type = 'turn.interrupted') AS interrupt_signals,
    countIf(event_type = 'tts.completed' AND status = 'interrupted') AS cancelled_replies,
    countIf(event_type = 'turn.skipped_empty') AS skipped_empty_turns,
    countIf(status = 'error') AS errors
FROM pipeline_events WHERE {where}""")[0]
    stages = db.query(f"""WITH {cte}
SELECT
    event_type,
    count() AS samples,
    avg(duration_ms) AS avg_duration_ms,
    avg(service_proc_ms) AS avg_service_proc_ms,
    avg(queue_ms) AS avg_queue_ms,
    avg(compute_ms) AS avg_compute_ms,
    avg(prompt_build_ms) AS avg_prompt_build_ms,
    avg(ttft_ms) AS avg_ttft_ms,
    avg(decode_ms) AS avg_decode_ms,
    avg(input_tokens) AS avg_input_tokens,
    avg(output_tokens) AS avg_output_tokens,
    avg(decode_tokens_per_second) AS avg_decode_tps,
    avg(time_to_first_audio_ms) AS avg_endpoint_to_audio_ms,
    avg(audio_seconds) AS avg_audio_seconds,
    avg(realtime_factor) AS avg_realtime_factor
    ,quantile(0.50)(duration_ms) AS p50_duration_ms
    ,quantile(0.95)(duration_ms) AS p95_duration_ms
    ,quantile(0.50)(time_to_first_audio_ms) AS p50_endpoint_to_audio_ms
    ,quantile(0.95)(time_to_first_audio_ms) AS p95_endpoint_to_audio_ms
    ,sum(cached_prompt_tokens) AS cached_prompt_tokens
    ,sum(input_tokens) AS total_input_tokens
    ,countIf(prefix_cache_hit = 1) AS prefix_cache_hits
FROM pipeline_events
WHERE {where} AND event_type IN ('vad.completed', 'stt.completed', 'llm.completed', 'tts.first_audio_queued', 'tts.completed') AND status = 'ok'
GROUP BY event_type""")
    deployment = db.query(f"""WITH {cte}
SELECT
    event_type,
    arrayStringConcat(arraySort(groupUniqArrayIf(model, model IS NOT NULL)), ', ') AS models,
    arrayStringConcat(arraySort(groupUniqArrayIf(engine, engine IS NOT NULL)), ', ') AS engines,
    arrayStringConcat(arraySort(groupUniqArrayIf(device, device IS NOT NULL)), ', ') AS devices,
    arrayStringConcat(arraySort(groupUniqArrayIf(compute_type, compute_type IS NOT NULL)), ', ') AS compute_types
FROM pipeline_events
WHERE {where} AND event_type IN ('llm.completed', 'tts.completed') AND status = 'ok'
GROUP BY event_type""")
    turn_latency = db.query(f"""WITH {cte}
SELECT
    count() AS turn_samples,
    avgIf(stt_paid_ms, isNotNull(stt_paid_ms)) AS avg_stt_paid_ms,
    quantileIf(0.50)(stt_paid_ms, isNotNull(stt_paid_ms)) AS p50_stt_paid_ms,
    quantileIf(0.95)(stt_paid_ms, isNotNull(stt_paid_ms)) AS p95_stt_paid_ms,
    countIf(isNotNull(stt_paid_ms)) AS samples_stt_paid_ms,
    avgIf(stt_endpoint_to_final_ms, isNotNull(stt_endpoint_to_final_ms)) AS avg_stt_endpoint_to_final_ms,
    quantileIf(0.50)(stt_endpoint_to_final_ms, isNotNull(stt_endpoint_to_final_ms)) AS p50_stt_endpoint_to_final_ms,
    quantileIf(0.95)(stt_endpoint_to_final_ms, isNotNull(stt_endpoint_to_final_ms)) AS p95_stt_endpoint_to_final_ms,
    countIf(isNotNull(stt_endpoint_to_final_ms)) AS samples_stt_endpoint_to_final_ms,
    avgIf(llm_queue_ms, isNotNull(llm_queue_ms)) AS avg_llm_queue_ms,
    quantileIf(0.50)(llm_queue_ms, isNotNull(llm_queue_ms)) AS p50_llm_queue_ms,
    quantileIf(0.95)(llm_queue_ms, isNotNull(llm_queue_ms)) AS p95_llm_queue_ms,
    countIf(isNotNull(llm_queue_ms)) AS samples_llm_queue_ms,
    avgIf(llm_encode_ms, isNotNull(llm_encode_ms)) AS avg_llm_encode_ms,
    quantileIf(0.50)(llm_encode_ms, isNotNull(llm_encode_ms)) AS p50_llm_encode_ms,
    quantileIf(0.95)(llm_encode_ms, isNotNull(llm_encode_ms)) AS p95_llm_encode_ms,
    countIf(isNotNull(llm_encode_ms)) AS samples_llm_encode_ms,
    avgIf(llm_decode_ms, isNotNull(llm_decode_ms)) AS avg_llm_decode_ms,
    quantileIf(0.50)(llm_decode_ms, isNotNull(llm_decode_ms)) AS p50_llm_decode_ms,
    quantileIf(0.95)(llm_decode_ms, isNotNull(llm_decode_ms)) AS p95_llm_decode_ms,
    countIf(isNotNull(llm_decode_ms)) AS samples_llm_decode_ms,
    avgIf(phrase_generation_ms, isNotNull(phrase_generation_ms)) AS avg_phrase_generation_ms,
    quantileIf(0.50)(phrase_generation_ms, isNotNull(phrase_generation_ms)) AS p50_phrase_generation_ms,
    quantileIf(0.95)(phrase_generation_ms, isNotNull(phrase_generation_ms)) AS p95_phrase_generation_ms,
    countIf(isNotNull(phrase_generation_ms)) AS samples_phrase_generation_ms,
    avgIf(tts_queue_ms, isNotNull(tts_queue_ms)) AS avg_tts_queue_ms,
    quantileIf(0.50)(tts_queue_ms, isNotNull(tts_queue_ms)) AS p50_tts_queue_ms,
    quantileIf(0.95)(tts_queue_ms, isNotNull(tts_queue_ms)) AS p95_tts_queue_ms,
    countIf(isNotNull(tts_queue_ms)) AS samples_tts_queue_ms,
    avgIf(tts_first_pcm_ms, isNotNull(tts_first_pcm_ms)) AS avg_tts_first_pcm_ms,
    quantileIf(0.50)(tts_first_pcm_ms, isNotNull(tts_first_pcm_ms)) AS p50_tts_first_pcm_ms,
    quantileIf(0.95)(tts_first_pcm_ms, isNotNull(tts_first_pcm_ms)) AS p95_tts_first_pcm_ms,
    countIf(isNotNull(tts_first_pcm_ms)) AS samples_tts_first_pcm_ms,
    avgIf(media_edge_queue_ms, isNotNull(media_edge_queue_ms)) AS avg_media_edge_queue_ms,
    quantileIf(0.50)(media_edge_queue_ms, isNotNull(media_edge_queue_ms)) AS p50_media_edge_queue_ms,
    quantileIf(0.95)(media_edge_queue_ms, isNotNull(media_edge_queue_ms)) AS p95_media_edge_queue_ms,
    countIf(isNotNull(media_edge_queue_ms)) AS samples_media_edge_queue_ms
    ,avgIf(vad_endpoint_to_first_audio_ms, isNotNull(vad_endpoint_to_first_audio_ms)) AS avg_vad_endpoint_to_first_audio_ms
    ,quantileIf(0.50)(vad_endpoint_to_first_audio_ms, isNotNull(vad_endpoint_to_first_audio_ms)) AS p50_vad_endpoint_to_first_audio_ms
    ,quantileIf(0.95)(vad_endpoint_to_first_audio_ms, isNotNull(vad_endpoint_to_first_audio_ms)) AS p95_vad_endpoint_to_first_audio_ms
    ,countIf(isNotNull(vad_endpoint_to_first_audio_ms)) AS samples_vad_endpoint_to_first_audio_ms
FROM pipeline_events
WHERE {where} AND event_type = 'turn.latency' AND status = 'ok'""")[0]
    return overview, stages, deployment, turn_latency


def render(overview: dict[str, Any], stages: list[dict[str, Any]], deployment: list[dict[str, Any]], turn_latency: dict[str, Any], args: argparse.Namespace) -> str:
    by_type = {row["event_type"]: row for row in stages}
    stt = by_type.get("stt.completed", {})
    llm = by_type.get("llm.completed", {})
    first_audio = by_type.get("tts.first_audio_queued", {})
    tts = by_type.get("tts.completed", {})
    deployment_by_type = {row["event_type"]: row for row in deployment}
    llm_deployment = deployment_by_type.get("llm.completed", {})
    tts_deployment = deployment_by_type.get("tts.completed", {})
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    scope = f"all sessions closed within {args.since_hours:g} hour(s)" if args.since_hours is not None else f"latest {args.sessions} closed session(s)"
    lines = [
        "# Voice-agent performance report",
        "",
        f"Generated {now}; scope: {scope}. Aggregate-only: no session IDs, transcripts, replies, or event rows.",
        "",
        markdown_table(["Coverage", "Value"], [[
            "Sessions selected", str(overview["selected_sessions"])],
            ["Sessions with telemetry", str(overview["sessions_with_events"])],
            ["Caller endpoints", str(overview["caller_endpoints"])],
            ["STT finals / LLM completions / agent starts", f"{overview['stt_final']} / {overview['llm_completed']} / {overview['agent_starts']}"],
            ["Interrupt signals / cancelled replies / empty turns", f"{overview['interrupt_signals']} / {overview['cancelled_replies']} / {overview['skipped_empty_turns']}"],
            ["Recorded errors", str(overview["errors"])],
        ]),
        "",
        markdown_table(["Deployment", "Value"], [
            ["LLM model", llm_deployment.get("models") or "n/a"],
            ["LLM engine / device / compute type", " / ".join(llm_deployment.get(key) or "n/a" for key in ("engines", "devices", "compute_types"))],
            ["TTS model", tts_deployment.get("models") or "n/a"],
            ["TTS engine / device / compute type", " / ".join(tts_deployment.get(key) or "n/a" for key in ("engines", "devices", "compute_types"))],
        ]),
        "",
        "## Caller endpoint → agent speech start",
        "",
        markdown_table(["Stage or computation", "p50 / p95", "Average", "Samples"], [
            ["End-to-first-agent-audio (headline)", f"{number(turn_latency.get('p50_vad_endpoint_to_first_audio_ms'), ' ms')} / {number(turn_latency.get('p95_vad_endpoint_to_first_audio_ms'), ' ms')}", number(turn_latency.get("avg_vad_endpoint_to_first_audio_ms"), " ms"), str(turn_latency.get("samples_vad_endpoint_to_first_audio_ms", 0))],
            ["VAD endpoint-to-final STT", f"{number(turn_latency.get('p50_stt_endpoint_to_final_ms'), ' ms')} / {number(turn_latency.get('p95_stt_endpoint_to_final_ms'), ' ms')}", number(turn_latency.get("avg_stt_endpoint_to_final_ms"), " ms"), str(turn_latency.get("samples_stt_endpoint_to_final_ms", 0))],
            ["LLM total service time", f"{number(llm.get('p50_duration_ms'), ' ms')} / {number(llm.get('p95_duration_ms'), ' ms')}", number(llm.get("avg_service_proc_ms") or llm.get("avg_duration_ms"), " ms"), str(llm.get("samples", 0))],
            ["LLM queue wait", "—", number(llm.get("avg_queue_ms"), " ms"), str(llm.get("samples", 0))],
            ["LLM prompt build", "—", number(llm.get("avg_prompt_build_ms"), " ms"), str(llm.get("samples", 0))],
            ["LLM time to first token", "—", number(llm.get("avg_ttft_ms"), " ms"), str(llm.get("samples", 0))],
            ["LLM decode", "—", number(llm.get("avg_decode_ms"), " ms"), str(llm.get("samples", 0))],
            ["TTS start-to-first-PCM", "—", number(first_audio.get("avg_service_proc_ms") or first_audio.get("avg_compute_ms"), " ms"), str(first_audio.get("samples", 0))],
        ]),
        "",
        "## Turn phase latency",
        "",
        "Each row is aggregated across turns; a metric's sample count includes only turns where that phase was recorded.",
        "",
        markdown_table(["Phase", "Average", "p50", "p95", "Turns"], [
            ["Time to first audio", number(turn_latency.get("avg_vad_endpoint_to_first_audio_ms"), " ms"), number(turn_latency.get("p50_vad_endpoint_to_first_audio_ms"), " ms"), number(turn_latency.get("p95_vad_endpoint_to_first_audio_ms"), " ms"), str(turn_latency.get("samples_vad_endpoint_to_first_audio_ms", 0))],
            ["STT", number(turn_latency.get("avg_stt_endpoint_to_final_ms"), " ms"), number(turn_latency.get("p50_stt_endpoint_to_final_ms"), " ms"), number(turn_latency.get("p95_stt_endpoint_to_final_ms"), " ms"), str(turn_latency.get("samples_stt_endpoint_to_final_ms", 0))],
            ["LLM Queue", number(turn_latency.get("avg_llm_queue_ms"), " ms"), number(turn_latency.get("p50_llm_queue_ms"), " ms"), number(turn_latency.get("p95_llm_queue_ms"), " ms"), str(turn_latency.get("samples_llm_queue_ms", 0))],
            ["LLM Encode", number(turn_latency.get("avg_llm_encode_ms"), " ms"), number(turn_latency.get("p50_llm_encode_ms"), " ms"), number(turn_latency.get("p95_llm_encode_ms"), " ms"), str(turn_latency.get("samples_llm_encode_ms", 0))],
            ["LLM Decode", number(turn_latency.get("avg_llm_decode_ms"), " ms"), number(turn_latency.get("p50_llm_decode_ms"), " ms"), number(turn_latency.get("p95_llm_decode_ms"), " ms"), str(turn_latency.get("samples_llm_decode_ms", 0))],
            ["Phrase Generation", number(turn_latency.get("avg_phrase_generation_ms"), " ms"), number(turn_latency.get("p50_phrase_generation_ms"), " ms"), number(turn_latency.get("p95_phrase_generation_ms"), " ms"), str(turn_latency.get("samples_phrase_generation_ms", 0))],
            ["TTS Queue", number(turn_latency.get("avg_tts_queue_ms"), " ms"), number(turn_latency.get("p50_tts_queue_ms"), " ms"), number(turn_latency.get("p95_tts_queue_ms"), " ms"), str(turn_latency.get("samples_tts_queue_ms", 0))],
            ["TTS First PCM", number(turn_latency.get("avg_tts_first_pcm_ms"), " ms"), number(turn_latency.get("p50_tts_first_pcm_ms"), " ms"), number(turn_latency.get("p95_tts_first_pcm_ms"), " ms"), str(turn_latency.get("samples_tts_first_pcm_ms", 0))],
            ["Media Edge Queue", number(turn_latency.get("avg_media_edge_queue_ms"), " ms"), number(turn_latency.get("p50_media_edge_queue_ms"), " ms"), number(turn_latency.get("p95_media_edge_queue_ms"), " ms"), str(turn_latency.get("samples_media_edge_queue_ms", 0))],
        ]),
        "",
        "## Throughput and full-response work",
        "",
        markdown_table(["Metric", "Average", "Samples"], [
            ["LLM input / output", f"{number(llm.get('avg_input_tokens'))} / {number(llm.get('avg_output_tokens'))} tokens", str(llm.get("samples", 0))],
            ["LLM decode throughput", number(llm.get("avg_decode_tps"), " tokens/s"), str(llm.get("samples", 0))],
            ["TTS full synthesis wall time", number(tts.get("avg_duration_ms"), " ms"), str(tts.get("samples", 0))],
            ["TTS generated audio", number(tts.get("avg_audio_seconds"), " s"), str(tts.get("samples", 0))],
            ["TTS real-time factor", number(tts.get("avg_realtime_factor"), "×"), str(tts.get("samples", 0))],
            ["LLM prefix-cache reuse", f"{number(100 * (llm.get('cached_prompt_tokens') or 0) / llm['total_input_tokens'] if llm.get('total_input_tokens') else None, '%')} ({llm.get('prefix_cache_hits', 0)} hit turns)", str(llm.get("samples", 0))],
        ]),
        "",
        "Notes: Time to first audio is directly measured on the media edge from VAD endpoint to the first WebRTC audio frame emitted. STT is measured from the orchestrator receiving the VAD endpoint to the final transcript. Component timers may overlap and must not be summed into end-to-end latency. An interrupt signal records the interrupt-controller path; a cancelled reply is the stricter measure of a reply that TTS actually ended as interrupted. The test client waits for inbound WebRTC audio to drain, but that client-side playout completion is not stored in ClickHouse. Deployment filters select sessions that used the requested dimensions; all telemetry within those sessions remains visible.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=2, help="number of latest distinct closed sessions when --since-hours is absent (default: 2)")
    parser.add_argument("--since-hours", type=float, help="include every session closed in this lookback window")
    parser.add_argument("--url", default=os.environ.get("CLICKHOUSE_URL", DEFAULT_URL), help=f"ClickHouse HTTP URL (default: $CLICKHOUSE_URL or {DEFAULT_URL})")
    parser.add_argument("--database", default=os.environ.get("CLICKHOUSE_DATABASE", "pipeline"), help="ClickHouse database (default: $CLICKHOUSE_DATABASE or pipeline)")
    parser.add_argument("--model", help="select sessions that used this exact model")
    parser.add_argument("--engine", help="select sessions that used this exact engine")
    parser.add_argument("--device", help="select sessions that used this exact device")
    parser.add_argument("--pipeline-version", help="select sessions that used this exact pipeline version")
    parser.add_argument("--timeout-s", type=float, default=15, help="HTTP query timeout (default: 15)")
    parser.add_argument("--output", help="also write the Markdown report to this file")
    args = parser.parse_args()
    if args.sessions < 1:
        parser.error("--sessions must be at least 1")
    if args.since_hours is not None and args.since_hours <= 0:
        parser.error("--since-hours must be positive")
    if args.timeout_s <= 0:
        parser.error("--timeout-s must be positive")
    if not IDENTIFIER.fullmatch(args.database):
        parser.error("--database must be a simple ClickHouse identifier")
    return args


def main() -> int:
    args = parse_args()
    try:
        overview, stages, deployment, turn_latency = fetch_report(ClickHouse(args.url, args.database, args.timeout_s), args)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    report = render(overview, stages, deployment, turn_latency, args)
    print(report, end="")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
