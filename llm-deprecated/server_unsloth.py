"""LLM gRPC service for unsloth/medgemma-1.5-4b-it-GGUF via llama.cpp.

This is the small-VRAM sibling of server.py. It keeps the same llm.v1 gRPC
contract but uses llama-cpp-python instead of vLLM because a 4 GB GTX 1650 is a
better fit for GGUF with partial GPU offload than for vLLM's engine/KV-cache
startup footprint.

Env is loaded from $LLM_ENV_FILE, then .env.unsloth, then .env in this folder.
Real environment variables always win.

Recommended first-run settings for GTX 1650 4 GB:
  LLM_MODEL_DIR=./models/unsloth/medgemma-1.5-4b-it-GGUF
  LLM_GGUF_FILE=medgemma-1.5-4b-it-Q4_K_S.gguf
  LLM_N_GPU_LAYERS=20
  LLM_N_CTX=2048
  LLM_N_BATCH=128
  LLM_N_UBATCH=64
  LLM_MAX_TOKENS=256

If startup OOMs, lower LLM_N_GPU_LAYERS first. If prompt ingestion OOMs, lower
LLM_N_BATCH / LLM_N_UBATCH. If generation is too CPU-bound and VRAM is available,
raise LLM_N_GPU_LAYERS gradually.
"""
import asyncio
import inspect
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor

import grpc
import llm_pb2
import llm_pb2_grpc
from prometheus_client import Counter, Histogram, start_http_server

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # optional when the GGUF is already present
    hf_hub_download = None

try:
    from llama_cpp import Llama
except ImportError as exc:
    raise SystemExit(
        "llama-cpp-python is required for server_unsloth.py. "
        "Install llm/requirements-llama.txt in a dedicated venv."
    ) from exc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("llm-unsloth")


def _load_dotenv():
    base = os.path.dirname(__file__)
    candidates = []
    if os.environ.get("LLM_ENV_FILE"):
        candidates.append(os.environ["LLM_ENV_FILE"])
    candidates.extend([os.path.join(base, ".env.unsloth"), os.path.join(base, ".env")])

    for path in candidates:
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))
        log.info("loaded env defaults from %s", path)
        return


_load_dotenv()

BASE_DIR = os.path.dirname(__file__)

REPO_ID = os.environ.get("LLM_HF_REPO_ID", "unsloth/medgemma-1.5-4b-it-GGUF")
MODEL_DIR = os.environ.get(
    "LLM_MODEL_DIR",
    os.path.join(BASE_DIR, "models", "unsloth", "medgemma-1.5-4b-it-GGUF"),
)
if not os.path.isabs(MODEL_DIR):
    MODEL_DIR = os.path.join(BASE_DIR, MODEL_DIR)
GGUF_FILE = os.environ.get("LLM_GGUF_FILE", "medgemma-1.5-4b-it-Q4_K_S.gguf")
MODEL_PATH = os.environ.get("LLM_MODEL_PATH") or os.path.join(MODEL_DIR, GGUF_FILE)
if not os.path.isabs(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, MODEL_PATH)
AUTO_DOWNLOAD = os.environ.get("LLM_AUTO_DOWNLOAD", "1") == "1"

SYSTEM_PROMPT = os.environ.get(
    "LLM_SYSTEM_PROMPT",
    "You are a helpful, concise voice assistant. Keep replies short and natural to speak aloud.",
)
MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "256"))
TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.5"))
TOP_P = float(os.environ.get("LLM_TOP_P", "0.9"))
REPEAT_PENALTY = float(os.environ.get("LLM_REPEAT_PENALTY", "1.05"))

N_CTX = int(os.environ.get("LLM_N_CTX", "2048"))
N_BATCH = int(os.environ.get("LLM_N_BATCH", "128"))
N_UBATCH = int(os.environ.get("LLM_N_UBATCH", "64"))
N_GPU_LAYERS = int(os.environ.get("LLM_N_GPU_LAYERS", "20"))
N_THREADS = int(os.environ.get("LLM_N_THREADS", str(max(1, (os.cpu_count() or 4) // 2))))
MAIN_GPU = int(os.environ.get("LLM_MAIN_GPU", "0"))
USE_MLOCK = os.environ.get("LLM_USE_MLOCK", "0") == "1"
VERBOSE = os.environ.get("LLM_VERBOSE", "0") == "1"

_LAT_BUCKETS = (.25, .5, .75, 1, 2, 3, 5, 8, 12, 20, 30, 60)
M_COMPUTE = Histogram("llm_compute_seconds", "llama.cpp completion time per reply", buckets=_LAT_BUCKETS)
M_PROC = Histogram("llm_proc_seconds", "Total dispatch->result time per reply", buckets=_LAT_BUCKETS)
M_TOKENS_OUT = Histogram(
    "llm_tokens_out", "Generated token count per reply", buckets=(8, 16, 32, 64, 128, 256, 512)
)
M_REPLIES = Counter("llm_replies_total", "Replies generated")
M_ERRORS = Counter("llm_errors_total", "Generation errors")


def _now_ms() -> int:
    return int(time.time() * 1000)


def ensure_model() -> str:
    if os.path.isfile(MODEL_PATH):
        return MODEL_PATH
    if not AUTO_DOWNLOAD:
        raise SystemExit(f"{MODEL_PATH} not found and LLM_AUTO_DOWNLOAD=0")
    if hf_hub_download is None:
        raise SystemExit(
            f"{MODEL_PATH} not found and huggingface_hub is not installed. "
            "Install llm/requirements-llama.txt or download the GGUF manually."
        )

    os.makedirs(MODEL_DIR, exist_ok=True)
    log.info("downloading %s:%s into %s ...", REPO_ID, GGUF_FILE, MODEL_DIR)
    path = hf_hub_download(repo_id=REPO_ID, filename=GGUF_FILE, local_dir=MODEL_DIR)
    if os.path.abspath(path) != os.path.abspath(MODEL_PATH):
        log.info("downloaded model to %s", path)
    return path


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
        "loading %s (ctx=%d, batch=%d/%d, gpu_layers=%d, threads=%d) ...",
        model_path,
        N_CTX,
        N_BATCH,
        N_UBATCH,
        N_GPU_LAYERS,
        N_THREADS,
    )
    return Llama(**_llama_kwargs(model_path))


def generate_reply(model: Llama, user_text: str) -> tuple[str, int]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]
    result = model.create_chat_completion(
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repeat_penalty=REPEAT_PENALTY,
    )
    choice = result["choices"][0]
    text = choice.get("message", {}).get("content", "") or choice.get("text", "")
    token_count = len(model.tokenize(text.encode("utf-8"), add_bos=False)) if text else 0
    return text.strip(), token_count


class LlmServicer(llm_pb2_grpc.LlmServicer):
    def __init__(self, model: Llama):
        self.model = model
        self.lock = asyncio.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llama")

    async def Generate(self, request, context):
        t0 = time.perf_counter()
        try:
            async with self.lock:
                t_gen = time.perf_counter()
                loop = asyncio.get_running_loop()
                text, n_tokens = await loop.run_in_executor(
                    self.executor, generate_reply, self.model, request.text
                )
                compute_ms = (time.perf_counter() - t_gen) * 1000.0
        except Exception:
            M_ERRORS.inc()
            log.exception("generate failed for %s", request.session_id)
            raise

        proc_ms = (time.perf_counter() - t0) * 1000.0
        M_COMPUTE.observe(compute_ms / 1000.0)
        M_PROC.observe(proc_ms / 1000.0)
        M_TOKENS_OUT.observe(n_tokens)
        M_REPLIES.inc()
        log.info(
            '%s reply (%d tok, compute %.0fms, proc %.0fms): "%s"',
            request.session_id,
            n_tokens,
            compute_ms,
            proc_ms,
            text,
        )
        return llm_pb2.GenerateResponse(
            session_id=request.session_id, text=text, ts_ms=_now_ms(), proc_ms=int(proc_ms)
        )


async def serve():
    model = load_model()
    log.info("model ready")

    metrics_port = int(os.environ.get("LLM_METRICS_PORT", "9105"))
    metrics_host = os.environ.get("LLM_METRICS_HOST", "0.0.0.0")
    start_http_server(metrics_port, addr=metrics_host)
    log.info("metrics on http://%s:%d/metrics", metrics_host, metrics_port)

    addr = os.environ.get("LLM_ADDR", "[::]:50053")
    server = grpc.aio.server()
    llm_pb2_grpc.add_LlmServicer_to_server(LlmServicer(model), server)
    server.add_insecure_port(addr)
    await server.start()
    log.info("LLM gRPC listening on %s (max_tokens=%d, temperature=%.2f)", addr, MAX_TOKENS, TEMPERATURE)
    await server.wait_for_termination()


if __name__ == "__main__":
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass
