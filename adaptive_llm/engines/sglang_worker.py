"""SGLang-backed worker for the Adaptive LLM gRPC facade.

The public service supplies the complete completed dialogue on every request.
SGLang's radix cache is therefore the session store: turn N+1 has turn N's
rendered prompt as an exact prefix and reuses its KV blocks without making a
second, potentially divergent, client-side history store.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
import uuid

import grpc
from prometheus_client import Counter, Histogram, start_http_server
from transformers import AutoTokenizer

from adaptive_llm.chat.context_window import request_dialogue, trim_to_budget
from adaptive_llm.chat.prompt_manager import load_system_prompt
from adaptive_llm.core.logging_utils import configure_logging, context_event, profile_event
from adaptive_llm.engines.common import load_dotenv, now_ms, telemetry_attributes
from adaptive_llm.proto import llm_pb2, llm_pb2_grpc


load_dotenv()
configure_logging()
log = logging.getLogger("adaptive-llm-sglang")

MODEL_PATH = os.environ.get("LLM_MODEL_PATH")
TOKENIZER = os.environ.get("LLM_TOKENIZER")
PROMPT_ID, SYSTEM_PROMPT = load_system_prompt()
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "512"))
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("LLM_TOP_P", "1.0"))
TOP_K = int(os.environ.get("LLM_TOP_K", "-1"))
MIN_P = float(os.environ.get("LLM_MIN_P", "0.0"))
REPETITION_PENALTY = float(os.environ.get("LLM_REPEAT_PENALTY", "1.0"))
ENABLE_THINKING = os.environ.get("LLM_ENABLE_THINKING", "0") == "1"
MAX_MODEL_LEN = os.environ.get("LLM_MAX_MODEL_LEN")
CONTEXT_SAFETY_TOKENS = max(0, int(os.environ.get("LLM_CONTEXT_SAFETY_TOKENS", "128")))
SGLANG_HOST = os.environ.get("LLM_SGLANG_HOST", "127.0.0.1")
SGLANG_PORT = int(os.environ.get("LLM_SGLANG_PORT", "30000"))
SGLANG_URL = os.environ.get("LLM_SGLANG_URL", f"http://{SGLANG_HOST}:{SGLANG_PORT}").rstrip("/")
SGLANG_TIMEOUT_S = float(os.environ.get("LLM_SGLANG_TIMEOUT_S", "300"))
SGLANG_SERVER_TIMEOUT_S = float(os.environ.get("LLM_SGLANG_SERVER_READY_TIMEOUT_S", "900"))
_READINESS_PROMPT = "ping"

_LAT_BUCKETS = (.1, .25, .5, .75, 1, 2, 3, 5, 8, 12, 20)
M_COMPUTE = Histogram("llm_compute_seconds", "SGLang generate time per reply", buckets=_LAT_BUCKETS)
M_PROC = Histogram("llm_proc_seconds", "Total dispatch->result time per reply", buckets=_LAT_BUCKETS)
M_TOKENS_OUT = Histogram("llm_tokens_out", "Generated token count per reply", buckets=(8, 16, 32, 64, 128, 256, 512, 1024))
M_REPLIES = Counter("llm_replies_total", "Replies generated")
M_ERRORS = Counter("llm_errors_total", "Generation errors")
M_TTFT = Histogram("llm_ttft_seconds", "Time to first generated token", buckets=_LAT_BUCKETS)
M_PREFILL = Histogram("llm_prefill_seconds", "SGLang prefill time (TTFT)", buckets=_LAT_BUCKETS)
M_DECODE = Histogram("llm_decode_seconds", "Decode time after the first generated token", buckets=_LAT_BUCKETS)
M_DECODE_TPS = Histogram("llm_decode_tokens_per_second", "Decode throughput", buckets=(1, 2, 4, 8, 16, 32, 64, 128, 256))
M_PREFIX_CACHED_TOKENS = Counter("llm_prefix_cached_tokens_total", "Prompt tokens reused from SGLang radix cache")
M_PREFIX_PROMPT_TOKENS = Counter("llm_prefix_cache_prompt_tokens_total", "Prompt tokens considered for SGLang radix-cache reuse")
M_HISTORY_USED = Histogram("llm_history_turns_used", "Completed history turns included in the prompt", buckets=(0, 1, 2, 4, 8, 12, 16, 24, 32, 64))
M_HISTORY_TRIMMED = Counter("llm_history_turns_trimmed_total", "History turns removed to fit the token budget")
M_CURRENT_TRUNCATED = Counter("llm_current_input_truncated_total", "Current user inputs truncated as a last resort")


def prepare_context(tokenizer, request):
    dialogue = request_dialogue(request)
    model_len = int(MAX_MODEL_LEN) if MAX_MODEL_LEN else int(tokenizer.model_max_length)
    input_budget = model_len - MAX_TOKENS - CONTEXT_SAFETY_TOKENS

    def render_and_count(candidate):
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, *candidate],
            tokenize=False, add_generation_prompt=True, enable_thinking=ENABLE_THINKING,
        )
        return prompt, len(tokenizer.encode(prompt, add_special_tokens=False))

    def truncate_current(text, minimum_drop):
        ids = tokenizer.encode(text, add_special_tokens=False)
        return tokenizer.decode(ids[min(len(ids), minimum_drop):], skip_special_tokens=True)

    return trim_to_budget(dialogue, input_budget, render_and_count, truncate_current)


def _payload(prompt: str, session_id: str, *, stream: bool) -> dict:
    # ``session_id`` is a stable server-side identity, while the full prompt is
    # still supplied deliberately. This preserves correctness after worker
    # restarts and lets RadixAttention match every retained history prefix.
    return {
        "text": prompt,
        "session_id": session_id,
        "conversation_id": session_id,
        "stream": stream,
        "sampling_params": {
            "temperature": TEMPERATURE, "max_new_tokens": MAX_TOKENS,
            "top_p": TOP_P, "top_k": TOP_K, "min_p": MIN_P,
            "repetition_penalty": REPETITION_PENALTY,
        },
    }


def _post_json(payload: dict) -> dict:
    request = urllib.request.Request(
        f"{SGLANG_URL}/generate", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=SGLANG_TIMEOUT_S) as response:
        return json.loads(response.read().decode("utf-8"))


def _stream_events(payload: dict, loop, queue: asyncio.Queue, stop_event: threading.Event) -> None:
    """Bridge SGLang's SSE response into the gRPC async event loop.

    The blocking HTTP read can't be interrupted from outside this thread, so
    ``stop_event`` is checked once per SSE line (one per generated token) on
    barge-in. Breaking closes the connection via the ``with`` block instead of
    reading the reply to completion, which also signals SGLang to abort
    generation server-side on client disconnect.
    """
    request = urllib.request.Request(
        f"{SGLANG_URL}/generate", data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=SGLANG_TIMEOUT_S) as response:
            for line in response:
                if stop_event.is_set():
                    break
                line = line.decode("utf-8").strip()
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                loop.call_soon_threadsafe(queue.put_nowait, json.loads(data))
    except Exception as exc:
        loop.call_soon_threadsafe(queue.put_nowait, exc)
    finally:
        loop.call_soon_threadsafe(queue.put_nowait, None)


async def _wait_for_server(process: asyncio.subprocess.Process) -> None:
    deadline = time.monotonic() + SGLANG_SERVER_TIMEOUT_S
    loop = asyncio.get_running_loop()
    while time.monotonic() < deadline:
        if process.returncode is not None:
            raise RuntimeError(f"SGLang server exited during startup (exit code {process.returncode})")
        try:
            await loop.run_in_executor(None, _post_json, {"text": _READINESS_PROMPT, "sampling_params": {"max_new_tokens": 1}, "log_metrics": False})
            return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
            await asyncio.sleep(0.5)
    raise TimeoutError(f"timed out waiting for SGLang at {SGLANG_URL}")


def _sglang_env() -> dict:
    # FlashInfer JIT-compiles CUDA kernels on first use via the `nvcc` it finds
    # on PATH. On hosts where a distro `nvidia-cuda-toolkit` package shadows a
    # newer NVIDIA-installed toolkit, that older nvcc's CUDA headers can fail
    # to build FlashInfer's current kernel sources (e.g. ambiguous
    # __nv_bfloat16 constructor overloads), independent of GPU compute
    # capability. Point at a specific toolkit only when the operator opts in.
    env = os.environ.copy()
    if cuda_home := os.environ.get("LLM_SGLANG_CUDA_HOME"):
        env["CUDA_HOME"] = cuda_home
        env["PATH"] = os.pathsep.join([os.path.join(cuda_home, "bin"), env.get("PATH", "")])
    if conv_dtype := os.environ.get("LLM_SGLANG_MAMBA_CONV_DTYPE"):
        # Hybrid conv/SSM architectures (e.g. LFM2) cache their short-conv
        # state in bf16 by default (SGLang's SGLANG_MAMBA_CONV_DTYPE, distinct
        # from --mamba-ssm-dtype which only covers the temporal/SSM state).
        # Pre-Ampere GPUs lack native bf16, so that cache tensor mismatches
        # the fp16 fallback the rest of the model already uses and the worker
        # crashes on first forward pass. Set this to "float16" on such
        # hardware.
        env["SGLANG_MAMBA_CONV_DTYPE"] = conv_dtype
    return env


async def launch_server() -> asyncio.subprocess.Process:
    if not MODEL_PATH:
        raise SystemExit("set LLM_MODEL_PATH for SGLang worker")
    command = [
        os.environ.get("LLM_SGLANG_PYTHON", os.sys.executable), "-m", "sglang.launch_server",
        "--model-path", MODEL_PATH, "--host", SGLANG_HOST, "--port", str(SGLANG_PORT),
    ]
    if MAX_MODEL_LEN:
        command.extend(["--context-length", MAX_MODEL_LEN])
    if os.environ.get("LLM_ENABLE_PREFIX_CACHING", "1").strip().lower() in {"0", "false", "no", "off"}:
        command.append("--disable-radix-cache")
    if memory := os.environ.get("LLM_GPU_MEMORY_UTILIZATION"):
        command.extend(["--mem-fraction-static", memory])
    if max_requests := os.environ.get("LLM_MAX_NUM_SEQS"):
        command.extend(["--max-running-requests", max_requests])
    if max_prefill := os.environ.get("LLM_MAX_NUM_BATCHED_TOKENS"):
        command.extend(["--max-prefill-tokens", max_prefill])
    if os.environ.get("LLM_SGLANG_DISABLE_CUDA_GRAPH", "0") == "1":
        command.append("--disable-cuda-graph")
    if mamba_dtype := os.environ.get("LLM_SGLANG_MAMBA_SSM_DTYPE"):
        # Hybrid conv/SSM architectures (e.g. LFM2) cache their conv/SSM state
        # in bf16 by default. Pre-Ampere GPUs (compute capability < sm80) lack
        # native bf16 support, so that cache tensor mismatches the fp16
        # fallback the rest of the model already uses, and the model crashes
        # on first forward pass. Set this to "float16" on such hardware.
        command.extend(["--mamba-ssm-dtype", mamba_dtype])
    log.info("starting SGLang server at %s (radix cache=%s)", SGLANG_URL, "--disable-radix-cache" not in command)
    process = await asyncio.create_subprocess_exec(*command, env=_sglang_env())
    await _wait_for_server(process)
    return process


async def wait_for_external_server() -> None:
    deadline = time.monotonic() + SGLANG_SERVER_TIMEOUT_S
    loop = asyncio.get_running_loop()
    while time.monotonic() < deadline:
        try:
            await loop.run_in_executor(None, _post_json, {"text": _READINESS_PROMPT, "sampling_params": {"max_new_tokens": 1}, "log_metrics": False})
            return
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
            await asyncio.sleep(0.5)
    raise TimeoutError(f"timed out waiting for externally managed SGLang at {SGLANG_URL}")


class LlmServicer(llm_pb2_grpc.LlmServicer):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    async def Generate(self, request, context):
        t0 = time.perf_counter()
        prepared = prepare_context(self.tokenizer, request)
        context_event("sglang", request.session_id, prepared.payload, prepared.prompt_tokens)
        M_HISTORY_USED.observe(prepared.history_turns_used)
        M_HISTORY_TRIMMED.inc(prepared.history_turns_trimmed)
        if prepared.current_truncated:
            M_CURRENT_TRUNCATED.inc()
        try:
            t_gen = time.perf_counter()
            result = await asyncio.get_running_loop().run_in_executor(None, _post_json, _payload(prepared.payload, request.session_id, stream=False))
            compute_ms = (time.perf_counter() - t_gen) * 1000
        except Exception:
            M_ERRORS.inc()
            log.exception("SGLang generate failed for %s", request.session_id)
            raise
        text = str(result.get("text", ""))
        meta = result.get("meta_info", {}) or {}
        n_tokens = int(meta.get("completion_tokens", len(self.tokenizer.encode(text, add_special_tokens=False))))
        cached = int(meta.get("cached_tokens", 0) or 0)
        proc_ms = (time.perf_counter() - t0) * 1000
        ttft_s = compute_ms / 1000  # non-streaming endpoint cannot expose TTFT
        self._metrics(prepared, compute_ms, proc_ms, n_tokens, cached, ttft_s)
        return self._response(request.session_id, text, prepared, compute_ms, proc_ms, n_tokens, cached, ttft_s)

    def _metrics(self, prepared, compute_ms, proc_ms, n_tokens, cached, ttft_s):
        decode_s = max(0.0, compute_ms / 1000 - ttft_s)
        M_COMPUTE.observe(compute_ms / 1000); M_PROC.observe(proc_ms / 1000); M_TOKENS_OUT.observe(n_tokens)
        M_TTFT.observe(ttft_s); M_PREFILL.observe(ttft_s); M_DECODE.observe(decode_s)
        if n_tokens and decode_s: M_DECODE_TPS.observe(n_tokens / decode_s)
        M_REPLIES.inc(); M_PREFIX_CACHED_TOKENS.inc(cached); M_PREFIX_PROMPT_TOKENS.inc(prepared.prompt_tokens)

    def _profile(self, prepared, compute_ms, n_tokens, cached, ttft_s):
        decode_s = max(0.0, compute_ms / 1000 - ttft_s)
        return llm_pb2.InferenceProfile(
            model=MODEL_PATH or "unknown", engine="sglang", device="cuda", compute_type=os.environ.get("LLM_QUANTIZATION", "auto") or "auto",
            prefill_ms=ttft_s * 1000, ttft_ms=ttft_s * 1000, decode_ms=decode_s * 1000, compute_ms=compute_ms,
            input_tokens=prepared.prompt_tokens, output_tokens=n_tokens,
            decode_tokens_per_second=(n_tokens / decode_s if n_tokens and decode_s else 0),
            attributes={**telemetry_attributes(gpu_index=0), "prompt_id": PROMPT_ID, "max_tokens": str(MAX_TOKENS),
                        "cached_prompt_tokens": str(cached), "history_turns_used": str(prepared.history_turns_used),
                        "history_turns_trimmed": str(prepared.history_turns_trimmed), "cache_strategy": "radix-prefix"},
        )

    def _response(self, session_id, text, prepared, compute_ms, proc_ms, n_tokens, cached, ttft_s):
        profile = self._profile(prepared, compute_ms, n_tokens, cached, ttft_s)
        profile_event("SGLang", session_id, f"total={proc_ms:.0f}ms | input={prepared.prompt_tokens} tokens | output={n_tokens} tokens | cached-prefix={cached} tokens | history={prepared.history_turns_used} turns")
        return llm_pb2.GenerateResponse(session_id=session_id, text=text, ts_ms=now_ms(), proc_ms=int(proc_ms), profile=profile)

    async def GenerateStream(self, request, context):
        t0 = time.perf_counter()
        prepared = prepare_context(self.tokenizer, request)
        context_event("sglang", request.session_id, prepared.payload, prepared.prompt_tokens)
        M_HISTORY_USED.observe(prepared.history_turns_used)
        M_HISTORY_TRIMMED.inc(prepared.history_turns_trimmed)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop_event = threading.Event()
        producer = loop.run_in_executor(
            None, _stream_events, _payload(prepared.payload, request.session_id, stream=True),
            loop, queue, stop_event)
        emitted = ""; final_meta: dict = {}; first_token_s = None
        try:
            while True:
                event = await queue.get()
                if event is None:
                    break
                if isinstance(event, Exception):
                    M_ERRORS.inc(); raise event
                text = str(event.get("text", ""))
                # SGLang can be configured for incremental or cumulative SSE.
                delta = text[len(emitted):] if text.startswith(emitted) else text
                emitted = text if text.startswith(emitted) else emitted + text
                final_meta = event.get("meta_info", final_meta) or final_meta
                if delta:
                    if first_token_s is None: first_token_s = time.perf_counter() - t0
                    yield llm_pb2.GenerateChunk(session_id=request.session_id, text_delta=delta, ts_ms=now_ms())
        except asyncio.CancelledError:
            # Barge-in: signal the producer thread to stop after the current
            # SSE line instead of reading the reply to completion.
            stop_event.set()
            raise
        await producer
        compute_ms = (time.perf_counter() - t0) * 1000
        n_tokens = int(final_meta.get("completion_tokens", len(self.tokenizer.encode(emitted, add_special_tokens=False))))
        cached = int(final_meta.get("cached_tokens", 0) or 0)
        ttft_s = first_token_s if first_token_s is not None else compute_ms / 1000
        self._metrics(prepared, compute_ms, compute_ms, n_tokens, cached, ttft_s)
        yield llm_pb2.GenerateChunk(session_id=request.session_id, is_final=True, ts_ms=now_ms(), profile=self._profile(prepared, compute_ms, n_tokens, cached, ttft_s))


async def serve():
    tokenizer_id = TOKENIZER or MODEL_PATH
    if not tokenizer_id:
        raise SystemExit("set LLM_MODEL_PATH for SGLang worker")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    external_server = bool(os.environ.get("LLM_SGLANG_URL"))
    server_process = None if external_server else await launch_server()
    if external_server:
        await wait_for_external_server()
    metrics_port = int(os.environ.get("LLM_WORKER_METRICS_PORT", "9116"))
    metrics_host = os.environ.get("LLM_WORKER_METRICS_HOST", "0.0.0.0")
    start_http_server(metrics_port, addr=metrics_host)
    server = grpc.aio.server()
    llm_pb2_grpc.add_LlmServicer_to_server(LlmServicer(tokenizer), server)
    address = os.environ.get("LLM_ADDR", "127.0.0.1:50054")
    if server.add_insecure_port(address) == 0:
        raise RuntimeError(f"could not bind SGLang worker address {address}")
    await server.start(); log.info("SGLang worker listening on %s", address)
    try:
        await server.wait_for_termination()
    finally:
        if server_process and server_process.returncode is None:
            server_process.terminate()
            try: await asyncio.wait_for(server_process.wait(), timeout=10)
            except asyncio.TimeoutError: server_process.kill(); await server_process.wait()


if __name__ == "__main__":
    asyncio.run(serve())
