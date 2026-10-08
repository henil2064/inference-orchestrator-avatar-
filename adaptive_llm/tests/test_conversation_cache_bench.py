from adaptive_llm.tools.vllm_cache_tune import summarize


def test_cache_summary_reports_token_and_request_hit_rates():
    result = summarize([{"requests": [
        {"ok": True, "input_tokens": 100, "cached_prompt_tokens": 75, "expected_terms": (), "ttft_ms": 1, "elapsed_s": .1, "decode_tps": 10},
        {"ok": True, "input_tokens": 50, "cached_prompt_tokens": 0, "expected_terms": (), "ttft_ms": 2, "elapsed_s": .2, "decode_tps": 20},
    ]}])
    assert result["token_weighted_prefix_cache_hit_rate_pct"] == 50
    assert result["request_prefix_hit_rate_pct"] == 50


def test_cache_summary_prefill_decode_split_falls_back_without_streaming_fields():
    # Rows from the unary `request()` helper predate prefill_ms/decode_ms and
    # must not crash summarize(); prefill falls back to the total ttft_ms.
    result = summarize([{"requests": [
        {"ok": True, "input_tokens": 100, "cached_prompt_tokens": 0, "expected_terms": (), "ttft_ms": 500, "elapsed_s": .5, "decode_tps": 10},
    ]}])
    assert result["p50_prefill_ms"] == 500
    assert result["p50_decode_ms"] == 0


def test_cache_summary_prefill_decode_split_uses_streaming_fields_when_present():
    # Rows from `request_stream()` carry a genuine prefill/decode split.
    result = summarize([{"requests": [
        {"ok": True, "input_tokens": 100, "cached_prompt_tokens": 0, "expected_terms": (),
         "ttft_ms": 500, "prefill_ms": 120, "decode_ms": 380, "elapsed_s": .5, "decode_tps": 10},
    ]}])
    assert result["p50_prefill_ms"] == 120
    assert result["p50_decode_ms"] == 380
