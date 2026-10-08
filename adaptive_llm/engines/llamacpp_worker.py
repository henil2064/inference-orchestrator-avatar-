"""llama.cpp worker for the adaptive LLM runtime."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import grpc
from prometheus_client import Counter, Histogram, start_http_server

from adaptive_llm.chat.context_window import request_dialogue, trim_to_budget
from adaptive_llm.chat.prompt_manager import load_system_prompt
from adaptive_llm.core.logging_utils import configure_logging, context_event, profile_event
from adaptive_llm.engines.common import load_dotenv, now_ms, telemetry_attributes
from adaptive_llm.proto import llm_pb2, llm_pb2_grpc

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

from llama_cpp import Llama, LogitsProcessorList


load_dotenv()
configure_logging()
log = logging.getLogger("adaptive-llm-llamacpp")

MODEL_PATH = os.environ.get("LLM_MODEL_PATH")
REPO_ID = os.environ.get("LLM_HF_REPO_ID")
GGUF_FILE = os.environ.get("LLM_GGUF_FILE")
AUTO_DOWNLOAD = os.environ.get("LLM_AUTO_DOWNLOAD", "1") == "1"
PROMPT_ID, SYSTEM_PROMPT = load_system_prompt()
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "256"))
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.5"))
TOP_P = float(os.environ.get("LLM_TOP_P", "0.9"))
REPEAT_PENALTY = float(os.environ.get("LLM_REPEAT_PENALTY", "1.05"))
N_CTX = int(os.environ.get("LLM_N_CTX", os.environ.get("LLM_MAX_MODEL_LEN", "2048")))
CONTEXT_SAFETY_TOKENS = max(0, int(os.environ.get("LLM_CONTEXT_SAFETY_TOKENS", "128")))
N_BATCH = int(os.environ.get("LLM_N_BATCH", "128"))
N_UBATCH = int(os.environ.get("LLM_N_UBATCH", "64"))
N_GPU_LAYERS = int(os.environ.get("LLM_N_GPU_LAYERS", "0"))
N_THREADS = int(os.environ.get("LLM_N_THREADS", str(max(1, (os.cpu_count() or 4) // 2))))
MAIN_GPU = int(os.environ.get("LLM_MAIN_GPU", "0"))
USE_MLOCK = os.environ.get("LLM_USE_MLOCK", "0") == "1"
VERBOSE = os.environ.get("LLM_VERBOSE", "0") == "1"

_LAT_BUCKETS = (.25, .5, .75, 1, 2, 3, 5, 8, 12, 20, 30, 60)
M_COMPUTE = Histogram("llm_compute_seconds", "llama.cpp completion time per reply", buckets=_LAT_BUCKETS)
M_PROC = Histogram("llm_proc_seconds", "Total dispatch->result time per reply", buckets=_LAT_BUCKETS)
M_TOKENS_OUT = Histogram("llm_tokens_out", "Generated token count per reply",
                         buckets=(8, 16, 32, 64, 128, 256, 512))
M_REPLIES = Counter("llm_replies_total", "Replies generated")
M_ERRORS = Counter("llm_errors_total", "Generation errors")
M_TTFT = Histogram("llm_ttft_seconds", "Time to first generated token", buckets=_LAT_BUCKETS)
M_PREFILL = Histogram("llm_prefill_seconds", "Approximate prefill time (measured as TTFT)", buckets=_LAT_BUCKETS)
M_DECODE = Histogram("llm_decode_seconds", "Decode time after the first generated token", buckets=_LAT_BUCKETS)
M_DECODE_TPS = Histogram(
    "llm_decode_tokens_per_second", "Decode throughput after the first generated token",
    buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256),
)
M_PROMPT_CHARS = Histogram(
    "llm_prompt_chars", "User prompt size in characters",
    buckets=(64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768),
)
M_PROMPT_TOKENS = Histogram(
    "llm_prompt_tokens", "Conservative rendered prompt size in tokens",
    buckets=(32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768),
)
M_HISTORY_USED = Histogram(
    "llm_history_turns_used", "Completed history turns included in the prompt",
    buckets=(0, 1, 2, 4, 8, 12, 16, 24, 32, 64),
)
M_HISTORY_TRIMMED = Counter(
    "llm_history_turns_trimmed_total", "History turns removed to fit the token budget"
)
M_CURRENT_TRUNCATED = Counter(
    "llm_current_input_truncated_total", "Current user inputs truncated as a last resort"
)


def ensure_model() -> str:
    if MODEL_PATH and Path(MODEL_PATH).is_file():
        return MODEL_PATH
    if not (AUTO_DOWNLOAD and REPO_ID and GGUF_FILE):
        raise SystemExit("set LLM_MODEL_PATH to a GGUF file, or set LLM_HF_REPO_ID + LLM_GGUF_FILE")
    if hf_hub_download is None:
        raise SystemExit("huggingface_hub is required for automatic GGUF download")
    target_dir = Path(MODEL_PATH).parent if MODEL_PATH else Path.cwd() / "models" / REPO_ID
    target_dir.mkdir(parents=True, exist_ok=True)
    return hf_hub_download(repo_id=REPO_ID, filename=GGUF_FILE, local_dir=str(target_dir))


def _llama_kwargs(model_path: str) -> dict:
    kwargs = {
        "model_path": model_path,
        "n_ctx": N_CTX,
        "n_batch": N_BATCH,
        "n_ubatch": N_UBATCH,
        "n_gpu_layers": N_GPU_LAYERS,
        "n_threads": N_THREADS,
        "main_gpu": MAIN_GPU,
        "use_mlock": USE_MLOCK,
        "verbose": VERBOSE,
    }
    supported = set(inspect.signature(Llama).parameters)
    return {k: v for k, v in kwargs.items() if k in supported}


def load_model() -> Llama:
    model_path = ensure_model()
    log.info(
        "loading %s with llama.cpp (prompt_id=%s, ctx=%d, batch=%d/%d, gpu_layers=%d, threads=%d) ...",
        model_path,
        PROMPT_ID,
        N_CTX,
        N_BATCH,
        N_UBATCH,
        N_GPU_LAYERS,
        N_THREADS,
    )
    return Llama(**_llama_kwargs(model_path))


def _messages_for_model(model: Llama, candidate: list[dict[str, str]]) -> list[dict[str, str]]:
    """Render system instructions in a form supported by the selected chat format.

    llama.cpp's Gemma formatter ignores ``system`` role messages. MedGemma is a
    Gemma-family model, so place the system instruction at the start of the
    first user turn instead. The request validator guarantees that a dialogue
    begins with a user message.
    """
    architecture = str(model.metadata.get("general.architecture", "")).lower()
    if not architecture.startswith("gemma"):
        return [{"role": "system", "content": SYSTEM_PROMPT}, *candidate]

    messages = [dict(message) for message in candidate]
    messages[0]["content"] = f"{SYSTEM_PROMPT}\n\n{messages[0]['content']}"
    return messages


def prepare_context(model: Llama, request):
    dialogue = request_dialogue(request)
    input_budget = N_CTX - MAX_TOKENS - CONTEXT_SAFETY_TOKENS

    def render_and_count(candidate):
        messages = _messages_for_model(model, candidate)
        # llama-cpp-python does not expose the selected chat handler's rendered
        # prompt. Count every message separately and reserve a conservative 16
        # template tokens per message in addition to CONTEXT_SAFETY_TOKENS.
        content_tokens = sum(
            len(model.tokenize(message["content"].encode("utf-8"), add_bos=False))
            for message in messages
        )
        return messages, content_tokens + 16 * (len(messages) + 1)

    def truncate_current(text, minimum_drop):
        token_ids = model.tokenize(text.encode("utf-8"), add_bos=False)
        remaining = token_ids[min(len(token_ids), minimum_drop):]
        return model.detokenize(remaining).decode("utf-8", errors="ignore")

    return trim_to_budget(dialogue, input_budget, render_and_count, truncate_current)


def _stop_logits_processor(stop_event: threading.Event, eos_token_id: int):
    """Force EOS once ``stop_event`` fires, so a barge-in stops decode after
    the in-flight token instead of running to ``max_tokens``.

    ``create_chat_completion`` doesn't expose ``stopping_criteria`` (only
    ``create_completion`` does), so this hijacks the one hook chat completion
    does forward. llama.cpp's generation loop breaks on any EOG token, and
    EOS always qualifies, so this works across chat formats.
    """
    def processor(input_ids, scores):
        if stop_event.is_set():
            scores = scores.copy()
            scores[:] = float("-inf")
            scores[eos_token_id] = 0.0
        return scores
    return processor


def generate_reply(model: Llama, messages, stop_event: threading.Event | None = None) -> tuple[str, int, float]:
    t0 = time.perf_counter()
    logits_processor = (
        LogitsProcessorList([_stop_logits_processor(stop_event, model.token_eos())])
        if stop_event is not None else None
    )
    result = model.create_chat_completion(
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repeat_penalty=REPEAT_PENALTY,
        stream=True,
        logits_processor=logits_processor,
    )
    if isinstance(result, dict):
        choice = result["choices"][0]
        text = choice.get("message", {}).get("content", "") or choice.get("text", "")
        ttft_s = time.perf_counter() - t0
    else:
        chunks: list[str] = []
        ttft_s = None
        for chunk in result:
            choice = chunk["choices"][0]
            text = choice.get("delta", {}).get("content", "") or choice.get("text", "")
            if text:
                if ttft_s is None:
                    ttft_s = time.perf_counter() - t0
                chunks.append(text)
        text = "".join(chunks)
        if ttft_s is None:
            ttft_s = time.perf_counter() - t0
    token_count = len(model.tokenize(text.encode("utf-8"), add_bos=False)) if text else 0
    return text.strip(), token_count, ttft_s


class LlmServicer(llm_pb2_grpc.LlmServicer):
    def __init__(self, model: Llama):
        self.model = model
        self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llama")

    async def Generate(self, request, context):
        t0 = time.perf_counter()
        tokenize_start = time.perf_counter()
        prepared = prepare_context(self.model, request)
        messages = prepared.payload
        prompt_tokens = prepared.prompt_tokens
        tokenize_s = time.perf_counter() - tokenize_start
        context_event("llama.cpp", request.session_id, messages, prompt_tokens)
        M_PROMPT_CHARS.observe(prepared.prompt_chars)
        M_PROMPT_TOKENS.observe(prompt_tokens)
        M_HISTORY_USED.observe(prepared.history_turns_used)
        M_HISTORY_TRIMMED.inc(prepared.history_turns_trimmed)
        if prepared.current_truncated:
            M_CURRENT_TRUNCATED.inc()
            log.warning("current user input truncated to fit context for %s", request.session_id)
        stop_event = threading.Event()
        try:
            async with self.lock:
                t_gen = time.perf_counter()
                loop = asyncio.get_running_loop()
                work = loop.run_in_executor(
                    self.executor,
                    generate_reply,
                    self.model,
                    messages,
                    stop_event,
                )
                try:
                    text, n_tokens, ttft_s = await asyncio.shield(work)
                except asyncio.CancelledError:
                    # Barge-in: the executor thread can't be preempted, so
                    # signal the logits processor to force EOS on the next
                    # token instead of decoding to max_tokens. The single
                    # worker thread frees up after that one extra token,
                    # not after the whole reply.
                    stop_event.set()
                    raise
                compute_ms = (time.perf_counter() - t_gen) * 1000.0
        except Exception:
            M_ERRORS.inc()
            log.exception("generate failed for %s", request.session_id)
            raise

        proc_ms = (time.perf_counter() - t0) * 1000.0
        M_COMPUTE.observe(compute_ms / 1000.0)
        M_PROC.observe(proc_ms / 1000.0)
        M_TOKENS_OUT.observe(n_tokens)
        decode_s = max(0.0, compute_ms / 1000.0 - ttft_s)
        M_TTFT.observe(ttft_s)
        M_PREFILL.observe(ttft_s)
        M_DECODE.observe(decode_s)
        if n_tokens and decode_s > 0:
            M_DECODE_TPS.observe(n_tokens / decode_s)
        M_REPLIES.inc()
        profile = (
            f"tokenize={tokenize_s * 1000:.0f}ms | prefill/TTFT={ttft_s * 1000:.0f}ms "
            f"| decode={decode_s * 1000:.0f}ms | total={proc_ms:.0f}ms "
            f"| input<={prompt_tokens} tokens | output={n_tokens} tokens "
            f"| history={prepared.history_turns_used} turns "
            f"| trimmed={prepared.history_turns_trimmed} turns"
        )
        if n_tokens and decode_s > 0:
            profile += f" | decode={n_tokens / decode_s:.1f} tok/s"
        profile_event(
            "llama.cpp",
            request.session_id,
            profile,
        )
        return llm_pb2.GenerateResponse(
            session_id=request.session_id,
            text=text,
            ts_ms=now_ms(),
            proc_ms=int(proc_ms),
            profile=llm_pb2.InferenceProfile(
                model=MODEL_PATH or GGUF_FILE or REPO_ID or "unknown",
                engine="llama.cpp",
                device="cuda" if N_GPU_LAYERS else "cpu",
                compute_type="gguf",
                prompt_build_ms=tokenize_s * 1000.0,
                prefill_ms=ttft_s * 1000.0,
                ttft_ms=ttft_s * 1000.0,
                decode_ms=decode_s * 1000.0,
                compute_ms=compute_ms,
                input_tokens=prompt_tokens,
                output_tokens=n_tokens,
                decode_tokens_per_second=(n_tokens / decode_s if n_tokens and decode_s > 0 else 0.0),
                attributes={**telemetry_attributes(gpu_index=MAIN_GPU), "prompt_id": PROMPT_ID,
                            "n_ctx": str(N_CTX), "n_gpu_layers": str(N_GPU_LAYERS),
                            "max_tokens": str(MAX_TOKENS)},
            ),
        )

    async def GenerateStream(self, request, context):
        """Stream llama.cpp's own incremental chat-completion output as
        delta-only gRPC chunks, instead of generating the whole reply before
        returning it (Generate, above). Previously unimplemented: the facade
        (adaptive_llm/main.py) caught the resulting UNIMPLEMENTED and fell
        back to a single unary Generate() call wrapped as a fake one-chunk
        stream -- functionally fine, but no true streaming, and an
        UNIMPLEMENTED round trip (with an ERROR-level traceback logged by
        grpc's own exception handler) on every single turn.

        `create_chat_completion(..., stream=True)` returns a blocking
        generator -- llama.cpp decode is synchronous C code, so it must be
        driven entirely on the executor thread, same as Generate. Each
        chunk it yields is handed to this coroutine via a plain
        `asyncio.Queue`, pushed from the executor thread with
        `loop.call_soon_threadsafe` (the one thread-safe way to touch an
        asyncio.Queue from outside the event loop thread). Same lock +
        single-worker executor + barge-in stop_event as Generate: cancelling
        this async generator (barge-in) only ever interrupts `await
        queue.get()`, never `work` itself (an unrelated awaitable, so
        cancellation here does not touch it) -- `work` keeps running in the
        background exactly as Generate's `asyncio.shield` already achieves,
        stopping within a token or two once `stop_event` forces EOS.
        """
        t0 = time.perf_counter()
        tokenize_start = time.perf_counter()
        prepared = prepare_context(self.model, request)
        messages = prepared.payload
        prompt_tokens = prepared.prompt_tokens
        tokenize_s = time.perf_counter() - tokenize_start
        context_event("llama.cpp", request.session_id, messages, prompt_tokens)
        M_PROMPT_CHARS.observe(prepared.prompt_chars)
        M_PROMPT_TOKENS.observe(prompt_tokens)
        M_HISTORY_USED.observe(prepared.history_turns_used)
        M_HISTORY_TRIMMED.inc(prepared.history_turns_trimmed)
        if prepared.current_truncated:
            M_CURRENT_TRUNCATED.inc()
            log.warning("current user input truncated to fit context for %s", request.session_id)

        stop_event = threading.Event()
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        _DONE, model = object(), self.model

        def produce() -> None:
            logits_processor = LogitsProcessorList([_stop_logits_processor(stop_event, model.token_eos())])
            try:
                result = model.create_chat_completion(
                    messages=messages,
                    max_tokens=MAX_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    repeat_penalty=REPEAT_PENALTY,
                    stream=True,
                    logits_processor=logits_processor,
                )
                for chunk in result:
                    choice = chunk["choices"][0]
                    text = choice.get("delta", {}).get("content", "") or choice.get("text", "")
                    if text:
                        loop.call_soon_threadsafe(queue.put_nowait, text)
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)
            except Exception as exc:
                loop.call_soon_threadsafe(queue.put_nowait, exc)

        parts: list[str] = []
        try:
            async with self.lock:
                t_gen = time.perf_counter()
                work = loop.run_in_executor(self.executor, produce)
                ttft_s = None
                try:
                    while True:
                        item = await queue.get()
                        if item is _DONE:
                            break
                        if isinstance(item, Exception):
                            raise item
                        if ttft_s is None:
                            ttft_s = time.perf_counter() - t_gen
                        parts.append(item)
                        yield llm_pb2.GenerateChunk(session_id=request.session_id, text_delta=item,
                                                    ts_ms=now_ms())
                except asyncio.CancelledError:
                    # Barge-in: see Generate's identical handling above --
                    # `work` is a separate awaitable from `queue.get()`, so
                    # cancelling this coroutine never cancels it; it keeps
                    # running until stop_event takes effect.
                    stop_event.set()
                    raise
                compute_ms = (time.perf_counter() - t_gen) * 1000.0
                text = "".join(parts).strip()
                n_tokens = len(model.tokenize(text.encode("utf-8"), add_bos=False)) if text else 0
                if ttft_s is None:
                    ttft_s = compute_ms / 1000.0
        except Exception:
            M_ERRORS.inc()
            log.exception("stream generate failed for %s", request.session_id)
            raise

        proc_ms = (time.perf_counter() - t0) * 1000.0
        M_COMPUTE.observe(compute_ms / 1000.0)
        M_PROC.observe(proc_ms / 1000.0)
        M_TOKENS_OUT.observe(n_tokens)
        decode_s = max(0.0, compute_ms / 1000.0 - ttft_s)
        M_TTFT.observe(ttft_s)
        M_PREFILL.observe(ttft_s)
        M_DECODE.observe(decode_s)
        if n_tokens and decode_s > 0:
            M_DECODE_TPS.observe(n_tokens / decode_s)
        M_REPLIES.inc()
        profile = (
            f"tokenize={tokenize_s * 1000:.0f}ms | prefill/TTFT={ttft_s * 1000:.0f}ms "
            f"| decode={decode_s * 1000:.0f}ms | total={proc_ms:.0f}ms "
            f"| input<={prompt_tokens} tokens | output={n_tokens} tokens "
            f"| history={prepared.history_turns_used} turns "
            f"| trimmed={prepared.history_turns_trimmed} turns"
        )
        if n_tokens and decode_s > 0:
            profile += f" | decode={n_tokens / decode_s:.1f} tok/s"
        profile_event("llama.cpp", request.session_id, profile)
        yield llm_pb2.GenerateChunk(
            session_id=request.session_id,
            is_final=True,
            ts_ms=now_ms(),
            profile=llm_pb2.InferenceProfile(
                model=MODEL_PATH or GGUF_FILE or REPO_ID or "unknown",
                engine="llama.cpp",
                device="cuda" if N_GPU_LAYERS else "cpu",
                compute_type="gguf",
                prompt_build_ms=tokenize_s * 1000.0,
                prefill_ms=ttft_s * 1000.0,
                ttft_ms=ttft_s * 1000.0,
                decode_ms=decode_s * 1000.0,
                compute_ms=compute_ms,
                input_tokens=prompt_tokens,
                output_tokens=n_tokens,
                decode_tokens_per_second=(n_tokens / decode_s if n_tokens and decode_s > 0 else 0.0),
                attributes={**telemetry_attributes(gpu_index=MAIN_GPU), "prompt_id": PROMPT_ID,
                            "n_ctx": str(N_CTX), "n_gpu_layers": str(N_GPU_LAYERS),
                            "max_tokens": str(MAX_TOKENS)},
            ),
        )


async def serve():
    model = load_model()
    log.info("llama.cpp worker ready")
    metrics_port = int(os.environ.get("LLM_WORKER_METRICS_PORT", "9115"))
    metrics_host = os.environ.get("LLM_WORKER_METRICS_HOST", "0.0.0.0")
    start_http_server(metrics_port, addr=metrics_host)
    addr = os.environ.get("LLM_ADDR", "127.0.0.1:50054")
    server = grpc.aio.server()
    llm_pb2_grpc.add_LlmServicer_to_server(LlmServicer(model), server)
    if server.add_insecure_port(addr) == 0:
        raise RuntimeError(f"could not bind llama.cpp worker address {addr}")
    await server.start()
    log.info("llama.cpp worker listening on %s", addr)
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
