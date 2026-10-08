-- Run once for an existing ClickHouse data volume before deploying cache telemetry.
ALTER TABLE pipeline.pipeline_events ADD COLUMN IF NOT EXISTS cached_prompt_tokens Nullable(UInt32);
ALTER TABLE pipeline.pipeline_events ADD COLUMN IF NOT EXISTS prefix_cache_hit_pct Nullable(Float32);
ALTER TABLE pipeline.pipeline_events ADD COLUMN IF NOT EXISTS prefix_cache_hit Nullable(Bool);
