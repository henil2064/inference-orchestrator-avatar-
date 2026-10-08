CREATE DATABASE IF NOT EXISTS pipeline;

CREATE TABLE IF NOT EXISTS pipeline.pipeline_events
(
    session_id String,
    event_seq UInt64,
    turn_id UInt32,
    event_time_ms Int64,
    recorded_at_ms Int64,
    source LowCardinality(String),
    event_type LowCardinality(String),
    status LowCardinality(String),
    duration_ms Float64,
    -- Typed dimensions: safe to filter and group by in fleet dashboards.
    schema_version UInt16 DEFAULT 1,
    -- Immutable build/version stamp for before-and-after performance comparison.
    pipeline_version LowCardinality(String) DEFAULT 'unversioned',
    deployment_id LowCardinality(Nullable(String)),
    host_id LowCardinality(Nullable(String)),
    service_instance_id LowCardinality(Nullable(String)),
    model LowCardinality(Nullable(String)),
    model_revision LowCardinality(Nullable(String)),
    engine LowCardinality(Nullable(String)),
    device LowCardinality(Nullable(String)),
    compute_type LowCardinality(Nullable(String)),
    gpu_uuid LowCardinality(Nullable(String)),
    gpu_index Nullable(UInt16),
    -- Common service and scheduler timings. NULL means not applicable/not emitted.
    service_proc_ms Nullable(Float32),
    queue_ms Nullable(Float32),
    compute_ms Nullable(Float32),
    -- LLM request metrics.
    prompt_build_ms Nullable(Float32),
    prefill_ms Nullable(Float32),
    ttft_ms Nullable(Float32),
    decode_ms Nullable(Float32),
    input_tokens Nullable(UInt32),
    output_tokens Nullable(UInt32),
    decode_tokens_per_second Nullable(Float32),
    history_turns Nullable(UInt16),
    message_count Nullable(UInt16),
    max_output_tokens Nullable(UInt32),
    batch_size Nullable(UInt16),
    running_requests Nullable(UInt16),
    waiting_requests Nullable(UInt16),
    kv_cache_usage_pct Nullable(Float32),
    -- vLLM prefix-cache telemetry. These are token counts for one LLM turn;
    -- aggregate hit rate is sum(cached_prompt_tokens) / sum(input_tokens).
    cached_prompt_tokens Nullable(UInt32),
    prefix_cache_hit_pct Nullable(Float32),
    prefix_cache_hit Nullable(Bool),
    -- STT segment metrics.
    encode_ms Nullable(Float32),
    audio_seconds Nullable(Float32),
    padded_seconds Nullable(Float32),
    realtime_factor Nullable(Float32),
    -- TTS egress metrics. time_to_first_audio_ms is measured from the edge
    -- end-of-turn signal until the orchestrator queues the first PCM frame.
    sample_rate_hz Nullable(UInt32),
    audio_bytes Nullable(UInt64),
    time_to_first_audio_ms Nullable(Float32),
    segment_index Nullable(UInt32),
    segment_count Nullable(UInt32),
    turn_complete Nullable(Bool),
    -- Finalized per-turn latency breakdown. Populated on turn.latency rows.
    stt_paid_ms Nullable(Float32),
    llm_queue_ms Nullable(Float32),
    llm_encode_ms Nullable(Float32),
    llm_decode_ms Nullable(Float32),
    phrase_generation_ms Nullable(Float32),
    tts_queue_ms Nullable(Float32),
    tts_first_pcm_ms Nullable(Float32),
    media_edge_queue_ms Nullable(Float32),
    stt_endpoint_to_final_ms Nullable(Float32),
    vad_endpoint_to_first_audio_ms Nullable(Float32),
    error_code LowCardinality(Nullable(String)),
    error_class LowCardinality(Nullable(String)),
    -- Raw audit context only; never use this for fleet metric queries.
    payload_json String
)
ENGINE = MergeTree
PARTITION BY toDate(toDateTime(intDiv(event_time_ms, 1000)))
ORDER BY (session_id, event_seq);

-- The primary key above is intentionally session-first for trace lookup. This
-- projection gives Grafana fleet queries a time/model-friendly access path.
ALTER TABLE pipeline.pipeline_events ADD PROJECTION IF NOT EXISTS by_metric_time
(
    SELECT *
    ORDER BY (event_type, event_time_ms, model, engine, device)
);

-- Existing developer volumes are not recreated when this file changes.
-- Keep these idempotent migrations beside the initial schema so upgrading an
-- already-running ClickHouse instance exposes TTS metrics immediately.
ALTER TABLE pipeline.pipeline_events ADD COLUMN IF NOT EXISTS sample_rate_hz Nullable(UInt32);
ALTER TABLE pipeline.pipeline_events ADD COLUMN IF NOT EXISTS audio_bytes Nullable(UInt64);
ALTER TABLE pipeline.pipeline_events ADD COLUMN IF NOT EXISTS time_to_first_audio_ms Nullable(Float32);
ALTER TABLE pipeline.pipeline_events ADD COLUMN IF NOT EXISTS cached_prompt_tokens Nullable(UInt32);
ALTER TABLE pipeline.pipeline_events ADD COLUMN IF NOT EXISTS prefix_cache_hit_pct Nullable(Float32);
ALTER TABLE pipeline.pipeline_events ADD COLUMN IF NOT EXISTS prefix_cache_hit Nullable(Bool);
ALTER TABLE pipeline.pipeline_events ADD COLUMN IF NOT EXISTS stt_paid_ms Nullable(Float32);
ALTER TABLE pipeline.pipeline_events ADD COLUMN IF NOT EXISTS llm_queue_ms Nullable(Float32);
ALTER TABLE pipeline.pipeline_events ADD COLUMN IF NOT EXISTS llm_encode_ms Nullable(Float32);
ALTER TABLE pipeline.pipeline_events ADD COLUMN IF NOT EXISTS llm_decode_ms Nullable(Float32);
ALTER TABLE pipeline.pipeline_events ADD COLUMN IF NOT EXISTS phrase_generation_ms Nullable(Float32);
ALTER TABLE pipeline.pipeline_events ADD COLUMN IF NOT EXISTS tts_queue_ms Nullable(Float32);
ALTER TABLE pipeline.pipeline_events ADD COLUMN IF NOT EXISTS tts_first_pcm_ms Nullable(Float32);
ALTER TABLE pipeline.pipeline_events ADD COLUMN IF NOT EXISTS media_edge_queue_ms Nullable(Float32);
ALTER TABLE pipeline.pipeline_events ADD COLUMN IF NOT EXISTS stt_endpoint_to_final_ms Nullable(Float32);
ALTER TABLE pipeline.pipeline_events ADD COLUMN IF NOT EXISTS vad_endpoint_to_first_audio_ms Nullable(Float32);

CREATE TABLE IF NOT EXISTS pipeline.session_summaries
(
    session_id String,
    started_at_ms Int64,
    closed_at_ms Int64,
    outcome LowCardinality(String),
    pipeline_version LowCardinality(String) DEFAULT 'unversioned',
    metadata_json String,
    totals_json String
)
ENGINE = MergeTree
PARTITION BY toDate(toDateTime(intDiv(closed_at_ms, 1000)))
ORDER BY (closed_at_ms, session_id);
