-- Run once against an existing ClickHouse volume before deploying TTS telemetry.
ALTER TABLE pipeline.pipeline_events ADD COLUMN IF NOT EXISTS sample_rate_hz Nullable(UInt32);
ALTER TABLE pipeline.pipeline_events ADD COLUMN IF NOT EXISTS audio_bytes Nullable(UInt64);
ALTER TABLE pipeline.pipeline_events ADD COLUMN IF NOT EXISTS time_to_first_audio_ms Nullable(Float32);
