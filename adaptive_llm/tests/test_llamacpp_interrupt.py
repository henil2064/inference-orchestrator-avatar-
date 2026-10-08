"""Barge-in must stop in-flight llama.cpp decode quickly, not just stop
consuming it. See DECISION.md's interrupt-controller entries: cancelling the
orchestrator's turn task cancels the client-side gRPC stream, but llama.cpp
runs synchronous decode in a single-worker thread pool that can't be
preempted -- it needs its own cooperative stop signal to actually free up.

Kept in its own module: llamacpp_worker and sglang_worker register
identically-named Prometheus metrics on the same global registry at import
time (matching how they run as separate processes in production), so they
must not both be imported in one interpreter. See test_sglang_interrupt.py
for the SGLang-side equivalent of these tests.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

import numpy as np

from adaptive_llm.chat.context_window import PreparedContext
from adaptive_llm.engines import llamacpp_worker


def _prepared(text: str = "hi") -> PreparedContext:
    return PreparedContext(payload=[{"role": "user", "content": text}], prompt_tokens=1,
                            prompt_chars=len(text), history_turns_used=0,
                            history_turns_trimmed=0, current_truncated=False)


def test_stop_logits_processor_is_a_passthrough_until_signaled():
    stop_event = threading.Event()
    processor = llamacpp_worker._stop_logits_processor(stop_event, eos_token_id=2)
    scores = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    result = processor(None, scores)

    assert np.array_equal(result, scores)


def test_stop_logits_processor_forces_eos_once_signaled():
    stop_event = threading.Event()
    stop_event.set()
    processor = llamacpp_worker._stop_logits_processor(stop_event, eos_token_id=2)
    scores = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    result = processor(None, scores)

    assert result[2] == 0.0
    assert result[0] == float("-inf")
    assert result[1] == float("-inf")
    assert scores[2] == 3.0  # the caller's array is left untouched


class _SlowFakeLlama:
    """Stands in for llama_cpp.Llama closely enough to exercise generate_reply's
    real stopping path: one simulated decode step per loop iteration, the
    caller's logits_processor applied exactly like llama.cpp applies it, and
    generation stopping the moment the processor forces EOS -- mirroring
    llama.cpp's is_eog check in its C generate loop."""

    EOS = 999

    def __init__(self, step_delay_s: float):
        self.step_delay_s = step_delay_s
        self.tokens_emitted = 0

    def token_eos(self) -> int:
        return self.EOS

    def tokenize(self, data: bytes, add_bos: bool = False):
        return list(data)

    def create_chat_completion(self, *, messages, max_tokens, temperature, top_p,
                                repeat_penalty, stream, logits_processor):
        def gen():
            for _ in range(max_tokens):
                time.sleep(self.step_delay_s)
                scores = np.zeros(self.EOS + 1, dtype=np.float32)
                scores[1] = 1.0  # the "natural" next token, absent interference
                if logits_processor is not None:
                    scores = logits_processor(None, scores)
                picked = int(np.argmax(scores))
                if picked == self.EOS:
                    return
                self.tokens_emitted += 1
                yield {"choices": [{"delta": {"content": "w "}}]}
        return gen()


def test_barge_in_stops_llamacpp_decode_within_a_token_not_max_tokens(monkeypatch):
    monkeypatch.setattr(llamacpp_worker, "prepare_context", lambda model, request: _prepared())
    monkeypatch.setattr(llamacpp_worker, "MAX_TOKENS", 500)
    step_delay = 0.02
    model = _SlowFakeLlama(step_delay_s=step_delay)

    async def scenario():
        servicer = llamacpp_worker.LlmServicer(model)
        request = SimpleNamespace(session_id="s1")
        task = asyncio.create_task(servicer.Generate(request, None))
        await asyncio.sleep(step_delay * 3)  # let a few tokens decode first
        tokens_before_cancel = model.tokens_emitted
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return tokens_before_cancel

    tokens_before_cancel = asyncio.run(scenario())
    assert 0 < tokens_before_cancel < 500

    # The executor thread can't be preempted, but the forced-EOS logits
    # processor should end it within a token or two of the cancel -- not
    # after all 500 simulated tokens (10s away at this step delay).
    deadline = time.perf_counter() + 1.0
    while time.perf_counter() < deadline:
        time.sleep(0.02)
    assert model.tokens_emitted <= tokens_before_cancel + 2


def test_next_llamacpp_turn_is_not_blocked_behind_an_abandoned_generation(monkeypatch):
    monkeypatch.setattr(llamacpp_worker, "prepare_context", lambda model, request: _prepared())
    step_delay = 0.02
    model = _SlowFakeLlama(step_delay_s=step_delay)

    async def scenario():
        servicer = llamacpp_worker.LlmServicer(model)  # one servicer == one single-worker executor

        monkeypatch.setattr(llamacpp_worker, "MAX_TOKENS", 500)
        task = asyncio.create_task(servicer.Generate(SimpleNamespace(session_id="s1"), None))
        await asyncio.sleep(step_delay * 3)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        monkeypatch.setattr(llamacpp_worker, "MAX_TOKENS", 2)
        t0 = time.perf_counter()
        await servicer.Generate(SimpleNamespace(session_id="s2"), None)
        return time.perf_counter() - t0

    elapsed = asyncio.run(scenario())
    # Without the stopping-criteria fix, the next turn queues behind the
    # abandoned generation on the single-worker executor: ~500 * step_delay
    # (~10s) before it can even start. With it, the worker frees up within
    # about one token of the cancel.
    assert elapsed < 2.0


async def _drain(stream) -> None:
    async for _ in stream:
        pass


def test_barge_in_stops_llamacpp_generate_stream_within_a_token_not_max_tokens(monkeypatch):
    """GenerateStream drives the exact same executor thread and stop_event as
    Generate (see its own docstring for why cancelling it never cancels
    `work`), so it must have the identical barge-in guarantee."""
    monkeypatch.setattr(llamacpp_worker, "prepare_context", lambda model, request: _prepared())
    monkeypatch.setattr(llamacpp_worker, "MAX_TOKENS", 500)
    step_delay = 0.02
    model = _SlowFakeLlama(step_delay_s=step_delay)

    async def scenario():
        servicer = llamacpp_worker.LlmServicer(model)
        request = SimpleNamespace(session_id="s1")
        task = asyncio.create_task(_drain(servicer.GenerateStream(request, None)))
        await asyncio.sleep(step_delay * 3)  # let a few tokens decode first
        tokens_before_cancel = model.tokens_emitted
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return tokens_before_cancel

    tokens_before_cancel = asyncio.run(scenario())
    assert 0 < tokens_before_cancel < 500

    deadline = time.perf_counter() + 1.0
    while time.perf_counter() < deadline:
        time.sleep(0.02)
    assert model.tokens_emitted <= tokens_before_cancel + 2


def test_next_llamacpp_turn_is_not_blocked_behind_an_abandoned_generate_stream(monkeypatch):
    monkeypatch.setattr(llamacpp_worker, "prepare_context", lambda model, request: _prepared())
    step_delay = 0.02
    model = _SlowFakeLlama(step_delay_s=step_delay)

    async def scenario():
        servicer = llamacpp_worker.LlmServicer(model)  # one servicer == one single-worker executor

        monkeypatch.setattr(llamacpp_worker, "MAX_TOKENS", 500)
        task = asyncio.create_task(
            _drain(servicer.GenerateStream(SimpleNamespace(session_id="s1"), None)))
        await asyncio.sleep(step_delay * 3)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        monkeypatch.setattr(llamacpp_worker, "MAX_TOKENS", 2)
        t0 = time.perf_counter()
        await _drain(servicer.GenerateStream(SimpleNamespace(session_id="s2"), None))
        return time.perf_counter() - t0

    elapsed = asyncio.run(scenario())
    assert elapsed < 2.0
