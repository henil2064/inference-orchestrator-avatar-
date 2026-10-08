"""vLLM worker for the adaptive LLM runtime."""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import re
import time
import uuid

import grpc
from prometheus_client import Counter, Histogram, start_http_server
from transformers import AutoTokenizer

from adaptive_llm.chat.context_window import request_dialogue, trim_to_budget
from adaptive_llm.chat.prompt_manager import load_system_prompt
from adaptive_llm.engines.common import (
    apply_triton_prefix_prefill_limit,
    attention_head_size,
    cuda_compute_capability,
    disable_float16_safety_blocklist,
    engine_kwargs_from_env,
    ensure_gemma3_sliding_window_pattern,
    ensure_gguf_gemma3_arch_alias,
    ensure_gguf_gemma3_quantized_embedding,
    ensure_gguf_gemma3_sliding_window,
    ensure_gguf_gemma_norm_offset,
    ensure_v0_for_gguf,
    ensure_v1_gpu_compatibility,
    gguf_config_overrides,
    gguf_source_repository,
    load_dotenv,
    now_ms,
    require_gguf_runtime,
    telemetry_attributes,
)
from adaptive_llm.core.logging_utils import configure_logging, context_event, profile_event
from adaptive_llm.proto import llm_pb2, llm_pb2_grpc


load_dotenv()
configure_logging()
log = logging.getLogger("adaptive-llm-vllm")


def _gguf_compute_type(model_path: str) -> str | None:
    """Recover the quantization label a GGUF filename ends with, e.g. Q4_K_M."""
    match = re.search(r"([A-Za-z]*(?:Q\d|BF16|F16|F32)[A-Za-z0-9_]*)\.gguf$", model_path)
    return f"gguf:{match.group(1).upper()}" if match else "gguf"


MODEL_PATH = os.environ.get("LLM_MODEL_PATH")
IS_GGUF = bool(MODEL_PATH) and MODEL_PATH.lower().endswith(".gguf")
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
QUANTIZATION = os.environ.get("LLM_QUANTIZATION", "") or None
# vLLM infers `quantization="gguf"` from the file itself, so LLM_QUANTIZATION
# is normally unset for a GGUF model. Reporting "auto" would make a Q4_K_M run
# indistinguishable from a bf16 one in telemetry, which is exactly the
# comparison this field exists for -- so name the GGUF variant from its file.
COMPUTE_TYPE = QUANTIZATION or (_gguf_compute_type(MODEL_PATH) if IS_GGUF else None) or "auto"
# The public Adaptive LLM gRPC contract currently accepts text only. Disable a
# VLM's image/video encoder unless a future contract explicitly enables it;
# otherwise vLLM reserves/profiles multimodal caches during startup for no use.
LANGUAGE_MODEL_ONLY = os.environ.get("LLM_LANGUAGE_MODEL_ONLY", "1") == "1"

_LAT_BUCKETS = (.1, .25, .5, .75, 1, 2, 3, 5, 8, 12, 20)
M_COMPUTE = Histogram("llm_compute_seconds", "vLLM generate() time per reply", buckets=_LAT_BUCKETS)
M_PROC = Histogram("llm_proc_seconds", "Total dispatch->result time per reply", buckets=_LAT_BUCKETS)
M_TOKENS_OUT = Histogram("llm_tokens_out", "Generated token count per reply",
                         buckets=(8, 16, 32, 64, 128, 256, 512, 1024))
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
    "llm_prompt_tokens", "Rendered prompt size in tokens",
    buckets=(32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768),
)
M_PREFIX_CACHED_TOKENS = Counter(
    "llm_prefix_cached_tokens_total",
    "Prompt tokens reused from vLLM's prefix cache.",
)
M_PREFIX_PROMPT_TOKENS = Counter(
    "llm_prefix_cache_prompt_tokens_total",
    "Prompt tokens considered for vLLM prefix-cache reuse.",
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


def load_engine():
    if not MODEL_PATH:
        raise SystemExit("set LLM_MODEL_PATH for vLLM worker")
    # vLLM 0.8.3 accesses Gemma 3's old public configuration field, while
    # Transformers 4.57.1 keeps that value as ``_sliding_window_pattern``.
    # Install the bidirectional compatibility alias before vLLM reads either
    # a Gemma 3 1B or MedGemma model configuration.
    ensure_gemma3_sliding_window_pattern()
    # Both of these must happen before vLLM is imported: V1 is selected at
    # import/startup time. vLLM 0.8.x only supports V1 on Ampere-or-newer GPUs,
    # so pre-Ampere cards automatically use the compatible V0 engine for either
    # Gemma model -- and V1 cannot run a GGUF checkpoint on any GPU.
    if ensure_v0_for_gguf(os.environ, MODEL_PATH):
        log.info("GGUF checkpoint selected; using the vLLM V0 engine (V1 cannot serve GGUF)")
    v1_fallback_capability = ensure_v1_gpu_compatibility(os.environ)
    if v1_fallback_capability is not None:
        log.warning(
            "CUDA compute capability %s.%s does not support vLLM V1; using V0",
            *v1_fallback_capability,
        )
    if IS_GGUF:
        log.info("serving a GGUF checkpoint with gguf %s", require_gguf_runtime())
    if IS_GGUF and ensure_gguf_gemma3_arch_alias():
        log.info("mapped Transformers' gemma3_text config onto the gguf package's gemma3 architecture")
    from vllm import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    if IS_GGUF:
        ensure_gguf_gemma3_sliding_window()
        ensure_gguf_gemma3_quantized_embedding()
        ensure_gguf_gemma_norm_offset()

    if os.environ.get("LLM_ALLOW_UNSAFE_FLOAT16") == "1":
        # vLLM blocklists float16 for gemma2/gemma3/glm4/plamo2 because deep
        # variants of these models can overflow fp16's range. That blocklist
        # leaves pre-Ampere GPUs (no bf16 tensor cores, so vLLM refuses
        # bfloat16 too) with only float32, which needs 2x the VRAM of the
        # checkpoint. This is an explicit, opt-in escape hatch for exactly
        # that situation -- verify generated text isn't garbage/NaN after
        # enabling it, since the instability risk is real for some models.
        # vLLM has exposed this internal list from different modules across
        # releases.  In particular, 0.8.3 has ``vllm.config`` as a module and
        # has no such blocklist, whereas newer releases may use
        # ``vllm.config.model``.  This is an opt-in compatibility tweak, so a
        # release without the guard should remain a valid no-op rather than
        # preventing the worker from starting.
        if not disable_float16_safety_blocklist():
            log.info(
                "LLM_ALLOW_UNSAFE_FLOAT16=1: installed vLLM has no known "
                "float16 blocklist; no compatibility patch needed"
            )
        else:
            log.warning(
                "LLM_ALLOW_UNSAFE_FLOAT16=1: bypassing vLLM's float16 safety "
                "blocklist (normally required for pre-Ampere GPUs since bfloat16 "
                "is gated to compute capability >= 8.0). Verify output quality."
            )

    # A GGUF file is not a Hugging Face model directory: the chat template and
    # tokenizer must come from the original repository, or prompt rendering
    # silently diverges from what the model was instruction-tuned on. Prefer a
    # configured value, then whatever the file itself names as its source.
    tokenizer_id = TOKENIZER
    if IS_GGUF and not tokenizer_id:
        tokenizer_id = gguf_source_repository(MODEL_PATH)
        if tokenizer_id:
            log.info("no tokenizer configured; using the base model this GGUF records: %s", tokenizer_id)
    if IS_GGUF and not tokenizer_id:
        raise SystemExit(
            "set LLM_TOKENIZER (or a model profile's `tokenizer`) to the Hugging Face repo "
            "this GGUF was converted from; a .gguf file carries no chat template and this one "
            f"records no base model to fall back on: {MODEL_PATH}"
        )
    tokenizer_id = tokenizer_id or MODEL_PATH
    log.info(
        "loading %s with vLLM (V1=%s, tokenizer=%s, prompt_id=%s, text_only=%s) ...",
        MODEL_PATH,
        os.environ.get("VLLM_USE_V1"),
        tokenizer_id,
        PROMPT_ID,
        LANGUAGE_MODEL_ONLY,
    )
    # vLLM 0.25 removed ``swap_space`` from AsyncEngineArgs.  Keep the public
    # environment contract (and its explicit zero value) while avoiding an
    # unsupported constructor argument on newer releases.  Non-zero values are
    # rejected rather than silently changing the requested capacity behavior.
    engine_kwargs = engine_kwargs_from_env(os.environ)
    head_size = attention_head_size(MODEL_PATH)
    if apply_triton_prefix_prefill_limit(engine_kwargs, head_size, cuda_compute_capability()):
        # Prefix caching and chunked prefill are the two features that send a
        # prompt through vLLM's Triton prefix-prefill kernel, which needs more
        # shared memory per block than a pre-Ampere SM has at head_size 256.
        log.warning(
            "head_size %s on a pre-Ampere GPU cannot use vLLM's Triton "
            "prefix-prefill kernel; disabling prefix caching and chunked "
            "prefill (max_num_batched_tokens=%s)",
            head_size,
            engine_kwargs.get("max_num_batched_tokens"),
        )
    supported = set(inspect.signature(AsyncEngineArgs).parameters)
    unsupported = set(engine_kwargs) - supported
    nonzero_unsupported = {
        key: engine_kwargs[key] for key in unsupported if engine_kwargs[key] not in (None, 0, 0.0, False)
    }
    if nonzero_unsupported:
        raise ValueError(
            "installed vLLM does not support requested engine arguments: "
            + ", ".join(sorted(nonzero_unsupported))
        )
    if unsupported:
        log.info("ignoring unsupported neutral vLLM arguments: %s", ", ".join(sorted(unsupported)))
    init_kwargs = {
        "model": MODEL_PATH,
        "tokenizer": tokenizer_id,
        **{key: value for key, value in engine_kwargs.items() if key in supported},
    }
    if IS_GGUF:
        overrides = gguf_config_overrides(tokenizer_id)
        if "hf_overrides" not in supported:
            raise SystemExit(
                "installed vLLM cannot accept hf_overrides, so a GGUF model would run with "
                f"Transformers defaults for {', '.join(overrides)} instead of this checkpoint's values"
            )
        init_kwargs["hf_overrides"] = overrides
        log.info("recovered GGUF-lossy config fields from %s: %s", tokenizer_id, overrides)
    # ``language_model_only`` was added after the CUDA-12.4-compatible vLLM
    # release used on R550 driver hosts.  Do not make an otherwise compatible
    # wheel fail at construction; newer vLLM versions still skip the unused
    # multimodal towers for this text-only gRPC service.
    if "language_model_only" in supported:
        init_kwargs["language_model_only"] = LANGUAGE_MODEL_ONLY
    elif LANGUAGE_MODEL_ONLY:
        log.info("installed vLLM does not support language_model_only; using its default multimodal loading")
    engine_args = AsyncEngineArgs(**init_kwargs)
    return AsyncLLMEngine.from_engine_args(engine_args), AutoTokenizer.from_pretrained(tokenizer_id)


def prepare_context(tokenizer, request):
    dialogue = request_dialogue(request)
    model_len = int(MAX_MODEL_LEN) if MAX_MODEL_LEN else int(tokenizer.model_max_length)
    input_budget = model_len - MAX_TOKENS - CONTEXT_SAFETY_TOKENS

    def render_and_count(candidate):
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, *candidate]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=ENABLE_THINKING,
        )
        return prompt, len(tokenizer.encode(prompt, add_special_tokens=False))

    def truncate_current(text, minimum_drop):
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        return tokenizer.decode(token_ids[min(len(token_ids), minimum_drop):],
                                skip_special_tokens=True)

    return trim_to_budget(dialogue, input_budget, render_and_count, truncate_current)


def sampling_params():
    """Create one user-configurable vLLM sampling configuration per request."""
    from vllm import SamplingParams

    return SamplingParams(
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=TOP_P,
        top_k=TOP_K,
        min_p=MIN_P,
        repetition_penalty=REPETITION_PENALTY,
    )


class LlmServicer(llm_pb2_grpc.LlmServicer):
    def __init__(self, engine, tokenizer):
        self.engine = engine
        self.tokenizer = tokenizer

    async def Generate(self, request, context):
        t0 = time.perf_counter()
        tokenize_start = time.perf_counter()
        prepared = prepare_context(self.tokenizer, request)
        prompt = prepared.payload
        prompt_tokens = prepared.prompt_tokens
        M_PROMPT_CHARS.observe(prepared.prompt_chars)
        tokenize_s = time.perf_counter() - tokenize_start
        context_event("vllm", request.session_id, prompt, prompt_tokens)
        M_PROMPT_TOKENS.observe(prompt_tokens)
        M_HISTORY_USED.observe(prepared.history_turns_used)
        M_HISTORY_TRIMMED.inc(prepared.history_turns_trimmed)
        if prepared.current_truncated:
            M_CURRENT_TRUNCATED.inc()
            log.warning("current user input truncated to fit context for %s", request.session_id)
        sampling = sampling_params()
        request_id = str(uuid.uuid4())
        try:
            t_gen = time.perf_counter()
            final = None
            first_token_s = None
            async for output in self.engine.generate(prompt, sampling, request_id):
                final = output
                if first_token_s is None and output.outputs and output.outputs[0].token_ids:
                    first_token_s = time.perf_counter() - t_gen
            compute_ms = (time.perf_counter() - t_gen) * 1000.0
        except Exception:
            M_ERRORS.inc()
            log.exception("generate failed for %s", request.session_id)
            raise

        text = final.outputs[0].text if final and final.outputs else ""
        n_tokens = len(final.outputs[0].token_ids) if final and final.outputs else 0
        cached_prompt_tokens = int(getattr(final, "num_cached_tokens", 0) or 0)
        proc_ms = (time.perf_counter() - t0) * 1000.0
        M_COMPUTE.observe(compute_ms / 1000.0)
        M_PROC.observe(proc_ms / 1000.0)
        M_TOKENS_OUT.observe(n_tokens)
        ttft_s = first_token_s if first_token_s is not None else compute_ms / 1000.0
        decode_s = max(0.0, compute_ms / 1000.0 - ttft_s)
        M_TTFT.observe(ttft_s)
        M_PREFILL.observe(ttft_s)
        M_DECODE.observe(decode_s)
        if n_tokens and decode_s > 0:
            M_DECODE_TPS.observe(n_tokens / decode_s)
        M_REPLIES.inc()
        M_PREFIX_CACHED_TOKENS.inc(cached_prompt_tokens)
        M_PREFIX_PROMPT_TOKENS.inc(prompt_tokens)
        profile_event(
            "vLLM",
            request.session_id,
            f"tokenize={tokenize_s * 1000:.0f}ms | prefill/TTFT={ttft_s * 1000:.0f}ms "
            f"| decode={decode_s * 1000:.0f}ms | total={proc_ms:.0f}ms "
            f"| input={prompt_tokens} tokens | output={n_tokens} tokens "
            f"| cached-prefix={cached_prompt_tokens} tokens "
            f"| history={prepared.history_turns_used} turns "
            f"| trimmed={prepared.history_turns_trimmed} turns"
            + (f" | decode={n_tokens / decode_s:.1f} tok/s" if n_tokens and decode_s > 0 else ""),
        )
        return llm_pb2.GenerateResponse(
            session_id=request.session_id,
            text=text,
            ts_ms=now_ms(),
            proc_ms=int(proc_ms),
            profile=llm_pb2.InferenceProfile(
                model=MODEL_PATH or "unknown",
                engine="vllm",
                device="cuda",
                compute_type=COMPUTE_TYPE,
                prompt_build_ms=tokenize_s * 1000.0,
                prefill_ms=ttft_s * 1000.0,
                ttft_ms=ttft_s * 1000.0,
                decode_ms=decode_s * 1000.0,
                compute_ms=compute_ms,
                input_tokens=prompt_tokens,
                output_tokens=n_tokens,
                decode_tokens_per_second=(n_tokens / decode_s if n_tokens and decode_s > 0 else 0.0),
                attributes={**telemetry_attributes(gpu_index=0), "prompt_id": PROMPT_ID,
                            "max_tokens": str(MAX_TOKENS),
                            "cached_prompt_tokens": str(cached_prompt_tokens),
                            "history_turns_used": str(prepared.history_turns_used),
                            "history_turns_trimmed": str(prepared.history_turns_trimmed)},
            ),
        )

    async def GenerateStream(self, request, context):
        """Expose vLLM's existing incremental output as delta-only gRPC chunks,
        instrumented with the same started/completed/tokens/TTFT semantics as
        the unary Generate path above -- this stream is the path actually used
        in production, so it must not be a metrics blind spot."""
        t0 = time.perf_counter()
        tokenize_start = time.perf_counter()
        prepared = prepare_context(self.tokenizer, request)
        prompt = prepared.payload
        prompt_tokens = prepared.prompt_tokens
        M_PROMPT_CHARS.observe(prepared.prompt_chars)
        tokenize_s = time.perf_counter() - tokenize_start
        context_event("vllm", request.session_id, prompt, prompt_tokens)
        M_PROMPT_TOKENS.observe(prompt_tokens)
        M_HISTORY_USED.observe(prepared.history_turns_used)
        M_HISTORY_TRIMMED.inc(prepared.history_turns_trimmed)
        if prepared.current_truncated:
            M_CURRENT_TRUNCATED.inc()
            log.warning("current user input truncated to fit context for %s", request.session_id)
        sampling = sampling_params()
        request_id = str(uuid.uuid4())
        emitted = ""
        final = None
        first_token_s = None
        t_gen = time.perf_counter()
        try:
            async for output in self.engine.generate(prompt, sampling, request_id):
                final = output
                text = output.outputs[0].text if output.outputs else ""
                if first_token_s is None and output.outputs and output.outputs[0].token_ids:
                    first_token_s = time.perf_counter() - t_gen
                # vLLM sends the cumulative decoded text; forward only new text.
                delta = text[len(emitted):]
                emitted = text
                if delta:
                    yield llm_pb2.GenerateChunk(session_id=request.session_id, text_delta=delta,
                                                 ts_ms=now_ms())
        except Exception:
            M_ERRORS.inc()
            log.exception("stream generate failed for %s", request.session_id)
            raise
        compute_ms = (time.perf_counter() - t_gen) * 1000.0
        proc_ms = (time.perf_counter() - t0) * 1000.0
        n_tokens = len(final.outputs[0].token_ids) if final and final.outputs else 0
        cached_prompt_tokens = int(getattr(final, "num_cached_tokens", 0) or 0)
        ttft_s = first_token_s if first_token_s is not None else compute_ms / 1000.0
        decode_s = max(0.0, compute_ms / 1000.0 - ttft_s)
        M_COMPUTE.observe(compute_ms / 1000.0)
        M_PROC.observe(proc_ms / 1000.0)
        M_TOKENS_OUT.observe(n_tokens)
        M_TTFT.observe(ttft_s)
        M_PREFILL.observe(ttft_s)
        M_DECODE.observe(decode_s)
        if n_tokens and decode_s > 0:
            M_DECODE_TPS.observe(n_tokens / decode_s)
        M_REPLIES.inc()
        M_PREFIX_CACHED_TOKENS.inc(cached_prompt_tokens)
        M_PREFIX_PROMPT_TOKENS.inc(prompt_tokens)
        profile_event(
            "vLLM",
            request.session_id,
            f"tokenize={tokenize_s * 1000:.0f}ms | prefill/TTFT={ttft_s * 1000:.0f}ms "
            f"| decode={decode_s * 1000:.0f}ms | total={proc_ms:.0f}ms "
            f"| input={prompt_tokens} tokens | output={n_tokens} tokens "
            f"| cached-prefix={cached_prompt_tokens} tokens "
            f"| history={prepared.history_turns_used} turns "
            f"| trimmed={prepared.history_turns_trimmed} turns"
            + (f" | decode={n_tokens / decode_s:.1f} tok/s" if n_tokens and decode_s > 0 else ""),
        )
        yield llm_pb2.GenerateChunk(session_id=request.session_id, is_final=True, ts_ms=now_ms(),
            profile=llm_pb2.InferenceProfile(model=MODEL_PATH or "unknown", engine="vllm", device="cuda",
                compute_type=COMPUTE_TYPE, prompt_build_ms=tokenize_s * 1000.0, prefill_ms=ttft_s * 1000,
                ttft_ms=ttft_s * 1000, decode_ms=decode_s * 1000, compute_ms=compute_ms,
                input_tokens=prompt_tokens, output_tokens=n_tokens,
                decode_tokens_per_second=(n_tokens / decode_s if n_tokens and decode_s else 0),
                attributes={**telemetry_attributes(gpu_index=0), "prompt_id": PROMPT_ID,
                            "max_tokens": str(MAX_TOKENS),
                            "cached_prompt_tokens": str(cached_prompt_tokens),
                            "history_turns_used": str(prepared.history_turns_used),
                            "history_turns_trimmed": str(prepared.history_turns_trimmed)}))


async def serve():
    engine, tokenizer = load_engine()
    log.info("vLLM worker ready")
    metrics_port = int(os.environ.get("LLM_WORKER_METRICS_PORT", "9114"))
    metrics_host = os.environ.get("LLM_WORKER_METRICS_HOST", "0.0.0.0")
    start_http_server(metrics_port, addr=metrics_host)
    addr = os.environ.get("LLM_ADDR", "127.0.0.1:50054")
    server = grpc.aio.server()
    llm_pb2_grpc.add_LlmServicer_to_server(LlmServicer(engine, tokenizer), server)
    if server.add_insecure_port(addr) == 0:
        raise RuntimeError(f"could not bind vLLM worker address {addr}")
    await server.start()
    log.info("vLLM worker listening on %s", addr)
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
