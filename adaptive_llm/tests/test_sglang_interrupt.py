"""Barge-in must stop the SGLang worker's blocking SSE read quickly, not just
stop consuming its queue. See DECISION.md's interrupt-controller entries:
cancelling the orchestrator's turn task cancels the client-side gRPC stream,
but the background thread reading SGLang's SSE response can't be interrupted
directly -- it needs its own cooperative stop signal to close the connection
promptly (which also lets SGLang detect the disconnect and abort server-side).

Kept in its own module: sglang_worker and llamacpp_worker register
identically-named Prometheus metrics on the same global registry at import
time (matching how they run as separate processes in production), so they
must not both be imported in one interpreter. See test_llamacpp_interrupt.py
for the llama.cpp-side equivalent of these tests.
"""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace

from adaptive_llm.chat.context_window import PreparedContext
from adaptive_llm.engines import sglang_worker


def _prepared(text: str = "hi") -> PreparedContext:
    return PreparedContext(payload=text, prompt_tokens=1, prompt_chars=len(text),
                            history_turns_used=0, history_turns_trimmed=0,
                            current_truncated=False)


class _FakeSSEResponse:
    """Stands in for the object urllib.request.urlopen returns: a context
    manager that is also a slow, line-oriented iterable, like a real SGLang
    SSE stream emitting one line per generated token."""

    def __init__(self, delay_s: float, n_lines: int = 1000):
        self.delay_s = delay_s
        self.n_lines = n_lines
        self.lines_yielded = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.closed = True
        return False

    def __iter__(self):
        for i in range(self.n_lines):
            time.sleep(self.delay_s)
            self.lines_yielded += 1
            yield ('data: {"text": "w%d "}\n' % i).encode("utf-8")
        yield b"data: [DONE]\n"


def test_stream_events_stops_reading_once_signaled(monkeypatch):
    response = _FakeSSEResponse(delay_s=0.01, n_lines=1000)
    monkeypatch.setattr(sglang_worker.urllib.request, "urlopen",
                         lambda request, timeout=None: response)

    loop = asyncio.new_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop_event = threading.Event()
    thread = threading.Thread(target=sglang_worker._stream_events,
                               args=({}, loop, queue, stop_event))
    try:
        thread.start()
        time.sleep(0.03)  # let a few lines go by
        stop_event.set()
        thread.join(timeout=2.0)

        assert not thread.is_alive()
        assert response.closed
        assert response.lines_yielded < 1000
    finally:
        loop.close()


async def _consume(servicer, request, received):
    async for chunk in servicer.GenerateStream(request, None):
        received.append(chunk)


def test_barge_in_stops_sglang_stream_reader_quickly(monkeypatch):
    monkeypatch.setattr(sglang_worker, "prepare_context", lambda tokenizer, request: _prepared())
    response = _FakeSSEResponse(delay_s=0.01, n_lines=1000)
    monkeypatch.setattr(sglang_worker.urllib.request, "urlopen",
                         lambda request, timeout=None: response)

    async def scenario():
        servicer = sglang_worker.LlmServicer(tokenizer=object())
        request = SimpleNamespace(session_id="s1", text="hi")
        received: list = []
        task = asyncio.create_task(_consume(servicer, request, received))
        while len(received) < 2:
            await asyncio.sleep(0.005)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        return len(received)

    n_received = asyncio.run(scenario())
    assert 0 < n_received < 1000

    # The blocking SSE read can't be interrupted from outside its thread, but
    # it checks stop_event once per line -- prove it actually exits and closes
    # the connection shortly after cancel, instead of reading all 1000 lines.
    deadline = time.perf_counter() + 1.0
    while time.perf_counter() < deadline and not response.closed:
        time.sleep(0.02)
    assert response.closed
    assert response.lines_yielded < 1000
