"""LLM gRPC service. Hosts a local MedGemma-4B GGUF model on vLLM's async engine
and generates one reply per COMPLETED turn.

Unlike stt/server.py (a batch model wrapped in a single-worker thread pool because
SraVaani has no native async/concurrency story), vLLM's AsyncLLMEngine natively
does continuous batching — concurrent Generate() calls from multiple sessions are
NOT serialized behind a lock here; the engine interleaves them on the GPU itself.
No manual warm-up either: engine construction (CUDA graph capture etc., unless
LLM_ENFORCE_EAGER=1) already pays that cost before serve() starts accepting RPCs.

The orchestrator supplies bounded structured conversation history. This worker
renders it with the model's chat template and applies the final exact-token
budget guard; RAG remains separate orchestrator work.

NOTE: vLLM's Python API shifts across versions and GGUF loading is one of its
less-traveled paths (limited quantization/parallelism support, historically
Linux-only). This file is written against a version of that API I'm reasonably
confident is broadly correct, but it has NOT been exercised end-to-end in this
sandbox (no GPU / no Windows vLLM support here) — verify once on the real box,
see RUN_MEDGEMMA.md.

Env:
  LLM_MODEL_PATH       path to the local .gguf file, OR a directory containing one
                        + config.json (the directory form works around a vLLM
                        engine-construction bug — see load_engine()) (required)
  LLM_TOKENIZER        HF repo id OR local directory for tokenizer + chat template (default
                        google/medgemma-4b-it — NOTE: that repo is gated; either
                        `huggingface-cli login` with an approved account, or point this at a
                        local dir with tokenizer.json/tokenizer.model/tokenizer_config.json/
                        special_tokens_map.json/added_tokens.json/chat_template.jinja instead)
  LLM_ADDR             listen address (default [::]:50053)
  LLM_SYSTEM_PROMPT    system prompt prepended to every turn (default: minimal voice-assistant prompt)
  LLM_MAX_TOKENS       max reply tokens (default 512)
  LLM_TEMPERATURE      sampling temperature (default 0.7)
  LLM_ENABLE_THINKING=1  emit a <think>...</think> reasoning block before the reply
                        (default off — a voice agent wants a short spoken answer, not a
                        visible reasoning trace, and thinking tokens add latency). Only
                        Qwen3-family chat templates currently act on this; harmless no-op
                        on models whose template doesn't reference it.
  LLM_MAX_MODEL_LEN    vLLM max context length override (default: model's own config, e.g.
                        131072 for Gemma3 — almost certainly too large for a 4GB card's KV
                        cache; set this explicitly, e.g. 4096, on VRAM-constrained boxes)
  LLM_GPU_MEMORY_UTILIZATION  fraction of GPU memory vLLM may claim for the KV cache
                        (default 0.85) — does NOT bound the chunked-prefill scratch
                        buffer, see LLM_MAX_NUM_BATCHED_TOKENS below
  LLM_MAX_NUM_BATCHED_TOKENS  cap on vLLM's chunked-prefill batch (default: vLLM's own
                        heuristic, e.g. 2048) — this buffer isn't governed by
                        LLM_GPU_MEMORY_UTILIZATION and can OOM well before KV cache
                        sizing is reached on a small/fp32-forced card; lower it
                        (e.g. 256-512) if a single large allocation fails at startup
  LLM_QUANTIZATION     vLLM quantization method override (default: unset, auto-detect —
                        GGUF is self-describing; passing "gguf" explicitly is REJECTED)
  LLM_ENFORCE_EAGER=1  skip CUDA graph capture (faster startup, slower steady-state; default off)
  LLM_METRICS_HOST     Prometheus bind host (default 0.0.0.0)
  LLM_METRICS_PORT     Prometheus port (default 9104)
  CUDA_VISIBLE_DEVICES pin to a GPU on a multi-GPU box (see PRD topology)
"""
import asyncio
import logging
import os
import time
import uuid

import grpc
import llm_pb2
import llm_pb2_grpc
from prometheus_client import Counter, Histogram, start_http_server
from transformers import AutoTokenizer
from vllm import AsyncEngineArgs, SamplingParams
from vllm.engine.async_llm_engine import AsyncLLMEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("llm")


def _load_dotenv():
    """Load KEY=VALUE lines from a .env next to this file. Real env wins (CLI overrides
    the file), so `LLM_ADDR=... python server.py` still beats a file default."""
    path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_load_dotenv()  # must run before the module-level os.environ reads below

MODEL_PATH = os.environ.get("LLM_MODEL_PATH")
TOKENIZER = os.environ.get("LLM_TOKENIZER", "google/medgemma-4b-it")
SYSTEM_PROMPT = os.environ.get(
    "LLM_SYSTEM_PROMPT",
    "You are a helpful, concise voice assistant. Keep replies short and natural to speak aloud. "
    "The earlier conversation is completed history: assistant messages were already delivered. "
    "Never repeat, rewrite, or re-answer a historical assistant message. Respond only to the newest "
    "user message; if it is merely an acknowledgement, respond briefly and naturally.")
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "512"))
CONTEXT_SAFETY_TOKENS = max(0, int(os.environ.get("LLM_CONTEXT_SAFETY_TOKENS", "128")))
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
# Off by default: a voice agent wants a short spoken reply, not a visible
# reasoning trace, and thinking tokens add real latency. Only Qwen3-family chat
# templates currently look at this kwarg — apply_chat_template() passes extra
# kwargs into the Jinja template context, so on a model whose template doesn't
# reference `enable_thinking` this is silently ignored, not an error.
ENABLE_THINKING = os.environ.get("LLM_ENABLE_THINKING", "0") == "1"
MAX_MODEL_LEN = os.environ.get("LLM_MAX_MODEL_LEN")
GPU_MEMORY_UTILIZATION = float(os.environ.get("LLM_GPU_MEMORY_UTILIZATION", "0.85"))
# GGUF is self-describing (quant type is embedded in the file), so leave this
# unset by default and let vLLM auto-detect — passing "gguf" explicitly is
# REJECTED by vllm's ModelConfig validator (that field is for formats that
# genuinely need an explicit hint: awq/gptq/fp8/etc). Only set LLM_QUANTIZATION
# if you have a specific reason vLLM isn't auto-detecting correctly.
QUANTIZATION = os.environ.get("LLM_QUANTIZATION", "") or None
ENFORCE_EAGER = os.environ.get("LLM_ENFORCE_EAGER", "0") == "1"
# vLLM's chunked-prefill scratch buffer is sized off this (default: an internal
# heuristic, e.g. 2048 tokens) — NOT bounded by GPU_MEMORY_UTILIZATION, which only
# governs the KV cache reservation. On a small/fp32-forced card (no bf16 support)
# this buffer alone can OOM well before KV cache sizing is even reached. Lower it
# (e.g. 256-512) if you see a single large allocation fail at startup.
MAX_NUM_BATCHED_TOKENS = os.environ.get("LLM_MAX_NUM_BATCHED_TOKENS")

# Prometheus metrics (scraped at LLM_METRICS_PORT).
_LAT_BUCKETS = (.1, .25, .5, .75, 1, 2, 3, 5, 8, 12, 20)
M_COMPUTE = Histogram("llm_compute_seconds", "vLLM generate() time per reply", buckets=_LAT_BUCKETS)
M_PROC = Histogram("llm_proc_seconds", "Total dispatch->result time per reply", buckets=_LAT_BUCKETS)
M_TOKENS_OUT = Histogram("llm_tokens_out", "Generated token count per reply",
                         buckets=(8, 16, 32, 64, 128, 256, 512, 1024))
M_REPLIES = Counter("llm_replies_total", "Replies generated")
M_ERRORS = Counter("llm_errors_total", "Generation errors")
M_HISTORY_USED = Histogram(
    "llm_history_turns_used", "Completed history turns included in the prompt",
    buckets=(0, 1, 2, 4, 8, 12, 16, 24, 32, 64))
M_HISTORY_TRIMMED = Counter(
    "llm_history_turns_trimmed_total", "History turns removed to fit the token budget")
M_CURRENT_TRUNCATED = Counter(
    "llm_current_input_truncated_total", "Current user inputs truncated as a last resort")


def _now_ms() -> int:
    return int(time.time() * 1000)


def load_engine():
    # LLM_MODEL_PATH can be a bare .gguf file OR a directory containing one +
    # config.json — the directory form works around a vLLM bug where engine
    # construction tries to JSON-parse the .gguf itself as an HF config
    # (maybe_override_with_speculators calling PretrainedConfig.get_config_dict
    # on `model=` directly); a real config.json alongside it gives that check
    # something valid to read instead of crashing on the binary file.
    if not MODEL_PATH:
        raise SystemExit("set $LLM_MODEL_PATH to a local .gguf file/directory, a local "
                         "HF-format model directory, or an HF repo id")
    # Only validate when it's actually a local path. An HF repo id (e.g.
    # "Qwen/Qwen3-0.6B") is neither a dir nor a file on this machine — that's
    # not an error, it just means vLLM should resolve it remotely; let vLLM's
    # own error surface handle a genuinely bad repo id rather than second-guess
    # it here. This check exists only to catch the one local-path mistake it's
    # actually useful for: a directory with weights but no loadable model file.
    if os.path.isdir(MODEL_PATH):
        has_weights = any(f.endswith((".gguf", ".safetensors", ".bin")) for f in os.listdir(MODEL_PATH))
        if not has_weights:
            raise SystemExit(f"{MODEL_PATH}: no .gguf/.safetensors/.bin weights found in this directory")
    log.info("loading %s (tokenizer=%s, quantization=%s) ...", MODEL_PATH, TOKENIZER, QUANTIZATION)
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        tokenizer=TOKENIZER,
        quantization=QUANTIZATION,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=int(MAX_MODEL_LEN) if MAX_MODEL_LEN else None,
        max_num_batched_tokens=int(MAX_NUM_BATCHED_TOKENS) if MAX_NUM_BATCHED_TOKENS else None,
        enforce_eager=ENFORCE_EAGER,
        dtype="auto",
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    return engine, tokenizer


def build_prompt(tokenizer, user_text: str) -> str:
    """Compatibility helper used by query_profile.py for a single user turn."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=ENABLE_THINKING)


def prepare_context(tokenizer, request):
    if request.messages:
        dialogue = []
        for message in request.messages:
            if message.role == llm_pb2.ChatMessage.USER:
                role = "user"
            elif message.role == llm_pb2.ChatMessage.ASSISTANT:
                role = "assistant"
            else:
                raise ValueError("LLM request messages may only use USER or ASSISTANT roles")
            dialogue.append({"role": role, "content": message.content})
    else:
        dialogue = [{"role": "user", "content": request.text}]
    if not dialogue or dialogue[-1]["role"] != "user":
        raise ValueError("LLM request must end with the current USER message")
    history, current = dialogue[:-1], dialogue[-1].copy()
    if len(history) % 2 or any(
            message["role"] != ("user" if index % 2 == 0 else "assistant")
            for index, message in enumerate(history)):
        raise ValueError("LLM history must contain alternating USER/ASSISTANT pairs")

    model_len = int(MAX_MODEL_LEN) if MAX_MODEL_LEN else int(tokenizer.model_max_length)
    input_budget = model_len - MAX_TOKENS - CONTEXT_SAFETY_TOKENS
    if input_budget <= 0:
        raise ValueError("LLM input token budget must be positive")
    original_turns = len(history) // 2
    current_truncated = False

    def render():
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history, current]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING)
        return prompt, len(tokenizer.encode(prompt, add_special_tokens=False))

    prompt, prompt_tokens = render()
    while prompt_tokens > input_budget and history:
        del history[:2]
        prompt, prompt_tokens = render()
    while prompt_tokens > input_budget and current["content"]:
        ids = tokenizer.encode(current["content"], add_special_tokens=False)
        drop = min(len(ids), max(1, prompt_tokens - input_budget))
        shortened = tokenizer.decode(ids[drop:], skip_special_tokens=True)
        # Some tokenizers can round-trip a leading whitespace token back to the
        # same visible string. Guarantee progress instead of risking a loop.
        current["content"] = "" if shortened == current["content"] else shortened
        current_truncated = True
        prompt, prompt_tokens = render()
    if prompt_tokens > input_budget:
        raise ValueError("system/chat-template overhead exceeds the LLM input token budget")
    return (prompt, prompt_tokens, sum(len(message["content"]) for message in history + [current]),
            len(history) // 2, original_turns - len(history) // 2, current_truncated)


class LlmServicer(llm_pb2_grpc.LlmServicer):
    def __init__(self, engine, tokenizer):
        self.engine = engine
        self.tokenizer = tokenizer

    async def Generate(self, request, context):
        t0 = time.perf_counter()
        (prompt, prompt_tokens, prompt_chars, history_used,
         history_trimmed, current_truncated) = prepare_context(self.tokenizer, request)
        M_HISTORY_USED.observe(history_used)
        M_HISTORY_TRIMMED.inc(history_trimmed)
        if current_truncated:
            M_CURRENT_TRUNCATED.inc()
            log.warning("current user input truncated to fit context for %s", request.session_id)
        sampling = SamplingParams(temperature=TEMPERATURE, max_tokens=MAX_TOKENS)
        request_id = str(uuid.uuid4())

        try:
            t_gen = time.perf_counter()
            final = None
            async for output in self.engine.generate(prompt, sampling, request_id):
                final = output  # AsyncLLMEngine streams intermediate steps; keep the latest
            compute_ms = (time.perf_counter() - t_gen) * 1000.0
        except Exception:
            M_ERRORS.inc()
            log.exception("generate failed for %s", request.session_id)
            raise

        text = final.outputs[0].text if final and final.outputs else ""
        n_tokens = len(final.outputs[0].token_ids) if final and final.outputs else 0
        proc_ms = (time.perf_counter() - t0) * 1000.0

        M_COMPUTE.observe(compute_ms / 1000.0)
        M_PROC.observe(proc_ms / 1000.0)
        M_TOKENS_OUT.observe(n_tokens)
        M_REPLIES.inc()
        log.info('%s reply (%d tok, compute %.0fms, proc %.0fms): "%s"',
                 request.session_id, n_tokens, compute_ms, proc_ms, text)

        return llm_pb2.GenerateResponse(session_id=request.session_id, text=text,
                                        ts_ms=_now_ms(), proc_ms=int(proc_ms))


async def serve():
    engine, tokenizer = load_engine()
    log.info("engine ready")

    metrics_port = int(os.environ.get("LLM_METRICS_PORT", "9104"))
    metrics_host = os.environ.get("LLM_METRICS_HOST", "0.0.0.0")
    start_http_server(metrics_port, addr=metrics_host)  # own thread; never touches the event loop
    log.info("metrics on http://%s:%d/metrics", metrics_host, metrics_port)

    addr = os.environ.get("LLM_ADDR", "[::]:50053")
    server = grpc.aio.server()
    llm_pb2_grpc.add_LlmServicer_to_server(LlmServicer(engine, tokenizer), server)
    server.add_insecure_port(addr)
    await server.start()
    log.info("LLM gRPC listening on %s (max_tokens=%d, temperature=%.2f)", addr, MAX_TOKENS, TEMPERATURE)
    await server.wait_for_termination()


if __name__ == "__main__":
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass
