"""GenerateStream is the path actually used in production (the orchestrator
always calls it, never the unary Generate), but it used to record almost none
of the metrics/telemetry the unary path does -- a total blind spot on the LLM
worker actually under load. This proves the streaming path now shares the same
started/completed/tokens/TTFT instrumentation as Generate.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from adaptive_llm.chat.context_window import PreparedContext
from adaptive_llm.engines import vllm_worker


def _prepared(text: str = "hello there") -> PreparedContext:
    return PreparedContext(payload=text, prompt_tokens=7, prompt_chars=len(text),
                            history_turns_used=1, history_turns_trimmed=0,
                            current_truncated=False)


class _FakeOutput:
    def __init__(self, text, token_ids):
        self.text = text
        self.token_ids = token_ids


class _FakeRequestOutput:
    def __init__(self, text, token_ids, num_cached_tokens=0):
        self.outputs = [_FakeOutput(text, token_ids)]
        self.num_cached_tokens = num_cached_tokens


class _FakeEngine:
    """Emits the same cumulative-text shape vLLM's real AsyncLLMEngine.generate()
    does: each output carries the full text so far, not a delta."""
    def __init__(self, steps):
        self.steps = steps  # list of (cumulative_text, token_ids)

    async def generate(self, prompt, sampling, request_id):
        for text, token_ids in self.steps:
            yield _FakeRequestOutput(text, token_ids)


async def _consume(servicer, request):
    return [c async for c in servicer.GenerateStream(request, None)]


def test_generate_stream_records_the_same_metrics_as_unary_generate(monkeypatch):
    monkeypatch.setattr(vllm_worker, "prepare_context", lambda tokenizer, request: _prepared())
    monkeypatch.setattr(vllm_worker, "sampling_params", lambda: object())
    steps = [("Hello", [1]), ("Hello world", [1, 2]), ("Hello world.", [1, 2, 3])]
    engine = _FakeEngine(steps)
    servicer = vllm_worker.LlmServicer(engine=engine, tokenizer=object())
    request = SimpleNamespace(session_id="s1", text="hi")

    replies_before = vllm_worker.M_REPLIES._value.get()
    prompt_tokens_sum_before = vllm_worker.M_PROMPT_TOKENS._sum.get()
    tokens_out_sum_before = vllm_worker.M_TOKENS_OUT._sum.get()
    prefix_prompt_before = vllm_worker.M_PREFIX_PROMPT_TOKENS._value.get()

    chunks = asyncio.run(_consume(servicer, request))

    # Cumulative vLLM output is still forwarded as delta-only text, in order.
    deltas = [c.text_delta for c in chunks if not c.is_final]
    assert deltas == ["Hello", " world", "."]
    finals = [c for c in chunks if c.is_final]
    assert len(finals) == 1
    profile = finals[0].profile
    assert profile.input_tokens == 7
    assert profile.output_tokens == 3
    assert profile.compute_ms >= 0 and profile.ttft_ms >= 0 and profile.decode_ms >= 0

    # These were previously never touched by GenerateStream at all -- a total
    # blind spot on the path production actually calls.
    assert vllm_worker.M_REPLIES._value.get() == replies_before + 1
    assert vllm_worker.M_PROMPT_TOKENS._sum.get() == prompt_tokens_sum_before + 7
    assert vllm_worker.M_TOKENS_OUT._sum.get() == tokens_out_sum_before + 3
    assert vllm_worker.M_PREFIX_PROMPT_TOKENS._value.get() == prefix_prompt_before + 7
