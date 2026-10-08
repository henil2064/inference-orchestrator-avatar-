"""GenerateStream is the path the orchestrator actually calls in production
(see orchestrator/service/server.py's produce()); before this, the llama.cpp
engine had no GenerateStream at all, so every call hit an UNIMPLEMENTED
round trip and fell back to a single unary Generate() wrapped as a fake
one-chunk stream -- functionally fine, but no true streaming, an ERROR-level
traceback logged on every turn, and none of Generate's own metrics recorded
for what was actually happening. This proves the real streaming path now
forwards true per-chunk deltas and records the same metrics as Generate.

Kept in its own module for the same reason as test_llamacpp_interrupt.py:
llamacpp_worker and sglang_worker register identically-named Prometheus
metrics on the same global registry at import time, so they must not both be
imported in one interpreter.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from adaptive_llm.chat.context_window import PreparedContext
from adaptive_llm.engines import llamacpp_worker


def _prepared(text: str = "hello there") -> PreparedContext:
    return PreparedContext(payload=[{"role": "user", "content": text}], prompt_tokens=7,
                            prompt_chars=len(text), history_turns_used=1,
                            history_turns_trimmed=0, current_truncated=False)


class _FakeStreamingLlama:
    """Stands in for llama_cpp.Llama's stream=True chat-completion shape:
    each yielded piece is a NEW delta (unlike vLLM's cumulative-text shape),
    matching what generate_reply's own chunk-parsing already assumes."""

    EOS = 999

    def __init__(self, deltas: list[str]):
        self.deltas = deltas

    def token_eos(self) -> int:
        return self.EOS

    def tokenize(self, data: bytes, add_bos: bool = False):
        return list(data)  # one "token" per byte is close enough for a token count

    def create_chat_completion(self, *, messages, max_tokens, temperature, top_p,
                                repeat_penalty, stream, logits_processor):
        def gen():
            for delta in self.deltas:
                yield {"choices": [{"delta": {"content": delta}}]}
        return gen()


async def _consume(servicer, request):
    return [c async for c in servicer.GenerateStream(request, None)]


def test_generate_stream_forwards_true_deltas_and_records_generate_metrics(monkeypatch):
    monkeypatch.setattr(llamacpp_worker, "prepare_context", lambda model, request: _prepared())
    model = _FakeStreamingLlama(["Hello", " world", "."])
    servicer = llamacpp_worker.LlmServicer(model)
    request = SimpleNamespace(session_id="s1")

    replies_before = llamacpp_worker.M_REPLIES._value.get()
    prompt_tokens_sum_before = llamacpp_worker.M_PROMPT_TOKENS._sum.get()
    tokens_out_sum_before = llamacpp_worker.M_TOKENS_OUT._sum.get()

    chunks = asyncio.run(_consume(servicer, request))

    deltas = [c.text_delta for c in chunks if not c.is_final]
    assert deltas == ["Hello", " world", "."]
    finals = [c for c in chunks if c.is_final]
    assert len(finals) == 1
    profile = finals[0].profile
    assert profile.engine == "llama.cpp"
    assert profile.input_tokens == 7
    assert profile.output_tokens == len("Hello world.")  # _FakeStreamingLlama tokenizes per byte
    assert profile.compute_ms >= 0 and profile.ttft_ms >= 0 and profile.decode_ms >= 0

    # Previously untouched by this path at all -- it never existed.
    assert llamacpp_worker.M_REPLIES._value.get() == replies_before + 1
    assert llamacpp_worker.M_PROMPT_TOKENS._sum.get() == prompt_tokens_sum_before + 7
    assert llamacpp_worker.M_TOKENS_OUT._sum.get() == tokens_out_sum_before + len("Hello world.")


def test_generate_stream_with_no_output_tokens_yields_only_a_final_chunk(monkeypatch):
    monkeypatch.setattr(llamacpp_worker, "prepare_context", lambda model, request: _prepared())
    model = _FakeStreamingLlama([])  # immediate EOS, no content
    servicer = llamacpp_worker.LlmServicer(model)
    request = SimpleNamespace(session_id="s1")

    chunks = asyncio.run(_consume(servicer, request))

    assert [c.text_delta for c in chunks if not c.is_final] == []
    assert len(chunks) == 1 and chunks[0].is_final
    assert chunks[0].profile.output_tokens == 0
