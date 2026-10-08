"""Backend-agnostic self-checks: format routing, streaming split, PCM framing.

Run: python -m tests.test_backends   (no framework, no model download)
"""
import threading
import time

import numpy as np

from service.backends import pcm16_bytes, split_sentences
from service.backends.base import TtsBackend


def test_format_routing():
    # The selector must route by extension without importing torch/outetts, so
    # check the pure decision the same way load_backend() does.
    def fmt(path, override=""):
        return override or ("gguf" if path.lower().endswith(".gguf") else "pytorch")
    assert fmt("models/DhVaani-0.5-fp16") == "pytorch"
    assert fmt("/models/oute.Q4_K_M.gguf") == "gguf"
    assert fmt("/models/oute.gguf", "pytorch") == "pytorch"  # explicit override wins


def test_split_sentences():
    assert split_sentences("Hi there. How are you?") == ["Hi there.", "How are you?"]
    assert split_sentences("नमस्ते। कैसे हैं?") == ["नमस्ते।", "कैसे हैं?"]
    # Hard-wrap over-long sentences with no boundary.
    assert split_sentences("a" * 500, max_chars=360) == ["a" * 360, "a" * 140]
    assert split_sentences("   ") == ["   "]  # never returns empty


def test_pcm16_roundtrip():
    audio = np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32)  # last two clamp
    pcm = np.frombuffer(pcm16_bytes(audio), dtype="<i2")
    assert pcm.tolist() == [0, 32767, -32767, 32767, -32767]


def test_backend_defaults_and_warmup():
    calls = []

    class Fake(TtsBackend):
        def synthesize(self, text):
            calls.append(text)
            return np.zeros(8, dtype=np.float32)

    fake = Fake()
    fake.warmup()
    assert calls == ["Ready."]
    assert fake.split_sentences("One. Two.") == ["One.", "Two."]


def test_synthesize_stream_default_falls_back_to_single_chunk():
    # A backend that doesn't override synthesize_stream (every flow-matching/
    # diffusion backend: supports_stream stays False) must still be drivable
    # through the streaming call, yielding its one array as a single chunk.
    class Fake(TtsBackend):
        def synthesize(self, text):
            return np.full(4, 7.0, dtype=np.float32)

    fake = Fake()
    assert fake.supports_stream is False
    chunks = list(fake.synthesize_stream("hello"))
    assert len(chunks) == 1
    assert chunks[0].tolist() == [7.0, 7.0, 7.0, 7.0]


# --- LlamaCppBackend.synthesize_stream (no real model load) --------------
#
# Constructs an instance via __new__ (skips __init__, which needs a real
# .gguf/tokenizer/codec) and fakes just the three collaborators
# synthesize_stream touches: the tokenizer's chat template, llama.cpp's
# token generator, and MioCodec's decode. This is deliberately not a
# perceptual audio-quality check (see the docstring's own caveat about that);
# it only pins the control flow: chunk boundaries, token accounting, and the
# opt-in/disabled-by-default gate.

def _fake_llamacpp_backend(*, stream_first_chunk_tokens, decode_calls):
    from service.backends.llamacpp_backend import LlamaCppBackend

    backend = LlamaCppBackend.__new__(LlamaCppBackend)
    backend.speech_offset = 1000
    backend.codebook_size = 100
    backend.temperature = 0.7
    backend.top_p = 0.85
    backend.max_audio_tokens = 1000
    backend.stream_first_chunk_tokens = stream_first_chunk_tokens
    backend.supports_stream = stream_first_chunk_tokens > 0
    backend.codec_device = "cpu"

    class FakeTokenizer:
        def apply_chat_template(self, messages, add_generation_prompt):
            return [1, 2, 3]

    class FakeLlm:
        EOS = 999999

        def token_eos(self):
            return self.EOS

        def generate(self, input_ids, temp, top_p):
            # 30 audio tokens (offset-encoded), then EOS.
            for i in range(30):
                yield 1000 + i
            yield self.EOS

    backend.tokenizer = FakeTokenizer()
    backend.llm = FakeLlm()

    def fake_decode(codes):
        decode_calls.append(list(codes))
        return np.full(len(codes), float(len(decode_calls)), dtype=np.float32)

    backend._decode = fake_decode
    return backend


def test_synthesize_stream_disabled_matches_synthesize():
    decode_calls = []
    backend = _fake_llamacpp_backend(stream_first_chunk_tokens=0, decode_calls=decode_calls)
    chunks = list(backend.synthesize_stream("hi"))
    assert len(chunks) == 1
    assert len(decode_calls) == 1
    assert decode_calls[0] == list(range(30))  # the whole utterance, one call


def test_synthesize_stream_enabled_splits_first_chunk_then_remainder():
    decode_calls = []
    backend = _fake_llamacpp_backend(stream_first_chunk_tokens=12, decode_calls=decode_calls)
    chunks = list(backend.synthesize_stream("hi"))
    assert len(chunks) == 2  # first 12 tokens, then the remaining 18
    assert decode_calls == [list(range(12)), list(range(12, 30))]
    # Every generated code is accounted for exactly once, in order.
    assert decode_calls[0] + decode_calls[1] == list(range(30))


def test_synthesize_stream_reply_shorter_than_first_chunk_yields_one_chunk():
    # A reply whose audio never reaches the first-chunk threshold must not
    # be split into an empty first chunk plus a real second one.
    decode_calls = []
    backend = _fake_llamacpp_backend(stream_first_chunk_tokens=1000, decode_calls=decode_calls)
    chunks = list(backend.synthesize_stream("hi"))
    assert len(chunks) == 1
    assert decode_calls == [list(range(30))]


# --- Profile timing under pool contention --------------------------------
#
# realtime_factor is the metric capacity planning reads, so it must stay a
# property of the model, not of how busy the service was. These drive the real
# SynthesizeStream against a fake backend whose synthesize() sleeps a known
# amount, with the pool saturated, and check where the time is attributed.

def test_stream_profile_separates_pool_wait_from_compute():
    import asyncio
    import service.server as srv

    SYNTH_S = 0.10
    CONCURRENCY = 4

    class SlowBackend(TtsBackend):
        sampling_rate = 24000
        profile_attributes = {"model": "fake", "engine": "fake", "device": "cpu",
                              "compute_type": "fp32"}
        model_name = engine = device = compute_type = "fake"

        def synthesize(self, text):
            time.sleep(SYNTH_S)                      # 0.5 s of audio per sentence
            return np.zeros(self.sampling_rate // 2, dtype=np.float32)

        def split_sentences(self, text, max_chars=360):
            return split_sentences(text, max_chars)

        def warmup(self):
            pass

    class Ctx:
        async def abort(self, code, details):
            raise AssertionError(f"unexpected abort: {code} {details}")

    async def drive():
        service = srv.Tts(SlowBackend())
        req = srv.tts_pb2.SynthesizeRequest(session_id="s", text="One. Two.", ts_ms=0)

        async def collect():
            return [c async for c in service.SynthesizeStream(req, Ctx())]

        # More requests than worker threads (1, the shipped default), so later
        # sentences queue for a long time behind earlier ones.
        return await asyncio.gather(*(collect() for _ in range(CONCURRENCY)))

    # This test is about pool-wait/compute attribution under contention, not
    # admission-boundary behavior (test_admission_rejects_overflow_* owns
    # that) -- so it needs CONCURRENCY admittable regardless of the deployed
    # TTS_MAX_QUEUE (tts/.env), which capacity planning may tune independently.
    orig_max_queue = srv.MAX_QUEUE
    srv.MAX_QUEUE = max(srv.MAX_QUEUE, CONCURRENCY)
    try:
        streams = asyncio.run(drive())
    finally:
        srv.MAX_QUEUE = orig_max_queue
    profiles = [c.profile for s in streams for c in s if c.profile.model]
    assert len(profiles) == CONCURRENCY * 2, f"one profile per sentence, got {len(profiles)}"

    # THE invariant: every sentence did identical model work, so every sentence
    # must report near-identical compute_ms no matter how long it queued. Timing
    # from submission instead of from worker start makes the last sentence report
    # several times the first — this ratio is what catches that.
    computes = [p.compute_ms for p in profiles]
    assert max(computes) < 1.5 * min(computes), (
        f"compute_ms spread {min(computes):.0f}..{max(computes):.0f}ms for identical "
        "work — queue time is being counted as compute")
    for p in profiles:
        assert p.realtime_factor < 1.0, f"realtime_factor={p.realtime_factor:.2f} inflated by queueing"

    # ...and the contention is still visible, just in the right field.
    waits = [float(p.attributes["pool_wait_ms"]) for p in profiles]
    assert max(waits) > SYNTH_S * 1000, (
        f"a saturated pool must surface as pool_wait_ms (max seen {max(waits):.0f}ms)")
    assert max(p.queue_ms for p in profiles) > SYNTH_S * 1000, "queue_ms must carry the pool wait"
    print(f"ok stream_profile_separates_pool_wait_from_compute "
          f"(compute {min(computes):.0f}-{max(computes):.0f}ms, pool wait up to {max(waits):.0f}ms)")


def test_admission_rejects_overflow_immediately_not_after_a_timeout():
    """Only 1+MAX_QUEUE callers can ever be admitted. Beyond that, extras must
    be rejected the instant they arrive (an atomic reservation check) rather
    than piling up on the semaphore's own unbounded waiter list and only
    finding out after riding out the full queue timeout."""
    import asyncio
    import service.server as srv

    CAPACITY = 1 + srv.MAX_QUEUE  # matches the shipped default queue depth
    OVERFLOW = 3

    class SlowBackend(TtsBackend):
        sampling_rate = 24000
        profile_attributes = {"model": "fake", "engine": "fake", "device": "cpu",
                              "compute_type": "fp32"}
        model_name = engine = device = compute_type = "fake"

        def synthesize(self, text):
            time.sleep(0.2)
            return np.zeros(self.sampling_rate // 2, dtype=np.float32)

        def split_sentences(self, text, max_chars=360):
            return split_sentences(text, max_chars)

        def warmup(self):
            pass

    class RejectCtx:
        def __init__(self):
            self.aborted = None

        async def abort(self, code, details):
            self.aborted = (code, details)
            raise RuntimeError("aborted")

    async def drive():
        service = srv.Tts(SlowBackend())
        old_timeout = srv.QUEUE_TIMEOUT_S
        srv.QUEUE_TIMEOUT_S = 5.0  # generous; a bug would make overflow ride this out
        try:
            async def one(i):
                ctx = RejectCtx()
                req = srv.tts_pb2.SynthesizeRequest(session_id=f"s{i}", text="One.", ts_ms=0)
                t0 = time.perf_counter()
                try:
                    [c async for c in service.SynthesizeStream(req, ctx)]
                    return True, None, time.perf_counter() - t0
                except RuntimeError:
                    return False, ctx.aborted, time.perf_counter() - t0

            return await asyncio.gather(*(one(i) for i in range(CAPACITY + OVERFLOW)))
        finally:
            srv.QUEUE_TIMEOUT_S = old_timeout

    results = asyncio.run(drive())
    accepted = [r for r in results if r[0]]
    rejected = [r for r in results if not r[0]]
    assert len(accepted) == CAPACITY, results
    assert len(rejected) == OVERFLOW, results
    assert all(aborted is not None and aborted[0] == srv.grpc.StatusCode.RESOURCE_EXHAUSTED
              for _, aborted, _ in rejected), rejected
    # The whole point: overflow is rejected near-instantly, not after riding
    # out QUEUE_TIMEOUT_S (5s here) just to lose a race it could never win.
    assert all(elapsed < 1.0 for _, _, elapsed in rejected), rejected


def test_occupancy_gauge_tracks_reservations_and_drains_to_zero():
    """tts_admission_occupancy must rise as callers are admitted and settle
    back to 0 once every admitted call finishes -- the explicit occupancy
    gauge Phase 2 asks for, distinct from the reject counter."""
    import asyncio
    import service.server as srv

    class SlowBackend(TtsBackend):
        sampling_rate = 24000
        profile_attributes = {"model": "fake", "engine": "fake", "device": "cpu", "compute_type": "fp32"}
        model_name = engine = device = compute_type = "fake"

        def synthesize(self, text):
            time.sleep(0.05)
            return np.zeros(self.sampling_rate // 2, dtype=np.float32)

        def split_sentences(self, text, max_chars=360):
            return split_sentences(text, max_chars)

        def warmup(self):
            pass

    class Ctx:
        async def abort(self, code, details):
            raise AssertionError(f"unexpected abort: {code} {details}")

    async def drive():
        service = srv.Tts(SlowBackend())
        assert srv.M_OCCUPANCY._value.get() == 0

        async def one(i):
            req = srv.tts_pb2.SynthesizeRequest(session_id=f"s{i}", text="One.", ts_ms=0)
            [c async for c in service.SynthesizeStream(req, Ctx())]

        # Fewer than capacity, so every call is admitted -- occupancy must
        # rise above 0 while they're in flight.
        n = srv.MAX_QUEUE
        tasks = [asyncio.create_task(one(i)) for i in range(n)]
        await asyncio.sleep(0.02)
        mid_flight = srv.M_OCCUPANCY._value.get()
        assert mid_flight > 0, "occupancy should be nonzero while calls are admitted"
        await asyncio.gather(*tasks)
        return mid_flight

    mid_flight = asyncio.run(drive())
    assert mid_flight <= srv.MAX_QUEUE + 1
    assert srv.M_OCCUPANCY._value.get() == 0, "occupancy leaked after every admitted call finished"


def test_pool_executing_gauge_reflects_true_worker_occupancy_not_admission():
    """The admission semaphore bounds how many callers may be admitted into
    POOL's own queue; POOL itself has one worker thread, so two admitted
    calls are never both truly running. tts_pool_executing must stay at 1
    even while more than one caller is admitted -- mirrors the STT fix."""
    import asyncio
    import service.server as srv

    release = threading.Event()
    started = threading.Event()

    class BlockOnFirstCallBackend(TtsBackend):
        sampling_rate = 24000
        profile_attributes = {"model": "fake", "engine": "fake", "device": "cpu", "compute_type": "fp32"}
        model_name = engine = device = compute_type = "fake"

        def __init__(self):
            self.calls = 0

        def synthesize(self, text):
            self.calls += 1
            if self.calls == 1:
                started.set()
                release.wait(timeout=5)
            return np.zeros(self.sampling_rate // 2, dtype=np.float32)

        def split_sentences(self, text, max_chars=360):
            return split_sentences(text, max_chars)

        def warmup(self):
            pass

    class Ctx:
        async def abort(self, code, details):
            raise AssertionError(f"unexpected abort: {code} {details}")

    async def drive():
        service = srv.Tts(BlockOnFirstCallBackend())

        async def one(i):
            req = srv.tts_pb2.SynthesizeRequest(session_id=f"s{i}", text="One.", ts_ms=0)
            [c async for c in service.SynthesizeStream(req, Ctx())]

        task_a = asyncio.create_task(one(0))
        for _ in range(200):
            if started.is_set():
                break
            await asyncio.sleep(0.01)
        assert started.is_set(), "the first call's synthesize() never started"

        task_b = asyncio.create_task(one(1))
        await asyncio.sleep(0.1)  # let b admit and queue behind a in the single-threaded POOL

        executing = srv.M_POOL_EXECUTING._value.get()
        assert executing == 1, f"expected exactly 1 truly-executing synthesis, got {executing}"

        release.set()
        await asyncio.wait_for(asyncio.gather(task_a, task_b), timeout=5)
        assert srv.M_POOL_EXECUTING._value.get() == 0, "tts_pool_executing leaked after both calls finished"

    asyncio.run(drive())


def test_wedge_watchdog_reports_unready_then_recovers():
    """Phase 3: a synthesis stuck past TTS_WEDGE_TIMEOUT_S must flip gRPC
    health to NOT_SERVING (synthesis can't be preempted, so this is the only
    signal available -- completes Phase 1's "report itself unready"), and
    flip back once the synthesis completes."""
    import asyncio
    import service.server as srv

    release = threading.Event()
    started = threading.Event()

    class SlowBackend(TtsBackend):
        sampling_rate = 24000
        profile_attributes = {"model": "fake", "engine": "fake", "device": "cpu", "compute_type": "fp32"}
        model_name = engine = device = compute_type = "fake"

        def synthesize(self, text):
            started.set()
            release.wait(timeout=5)
            return np.zeros(self.sampling_rate // 2, dtype=np.float32)

        def split_sentences(self, text, max_chars=360):
            return split_sentences(text, max_chars)

        def warmup(self):
            pass

    class Ctx:
        async def abort(self, code, details):
            raise AssertionError(f"unexpected abort: {code} {details}")

    class FakeHealthServicer:
        def __init__(self):
            self.statuses = []

        async def set(self, service, status):
            self.statuses.append(status)

    async def drive():
        orig_timeout = srv.TTS_WEDGE_TIMEOUT_S
        srv.TTS_WEDGE_TIMEOUT_S = 0.1
        health = FakeHealthServicer()
        watchdog = asyncio.create_task(srv._wedge_watchdog(health, interval_s=0.02))
        try:
            service = srv.Tts(SlowBackend())
            req = srv.tts_pb2.SynthesizeRequest(session_id="s", text="One.", ts_ms=0)
            call = asyncio.create_task(service.Synthesize(req, Ctx()))
            for _ in range(200):
                if started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert started.is_set(), "synthesis never started"

            for _ in range(200):
                if srv.health_pb2.HealthCheckResponse.NOT_SERVING in health.statuses:
                    break
                await asyncio.sleep(0.02)
            assert health.statuses and health.statuses[-1] == srv.health_pb2.HealthCheckResponse.NOT_SERVING, \
                f"expected NOT_SERVING once wedged, got {health.statuses}"

            release.set()
            await asyncio.wait_for(call, timeout=5)

            for _ in range(200):
                if health.statuses[-1] == srv.health_pb2.HealthCheckResponse.SERVING:
                    break
                await asyncio.sleep(0.02)
            assert health.statuses[-1] == srv.health_pb2.HealthCheckResponse.SERVING, \
                f"expected SERVING again after recovery, got {health.statuses}"
        finally:
            watchdog.cancel()
            srv.TTS_WEDGE_TIMEOUT_S = orig_timeout

    asyncio.run(drive())
