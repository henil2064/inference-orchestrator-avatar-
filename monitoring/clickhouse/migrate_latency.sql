-- Run once against an existing ClickHouse volume before deploying the
-- per-turn latency telemetry. Fresh volumes receive these from init.sql.
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
