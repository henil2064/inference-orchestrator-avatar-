"""STT gRPC service (Phase 1). Hosts the SraVaani model once, GPU-pinned, and
streams transcripts per session.

SraVaani is a batch TorchScript model — it has no native streaming API, and
re-transcribing an ever-growing turn on every tick (the old design) makes
compute grow with turn length: a user who talks for 60s without a long pause
would make the encoder reprocess a bigger window every tick. Instead we buffer
each session's PCM into BOUNDED SEGMENTS: once the in-progress segment exceeds
STT_MAX_SEGMENT_SEC, we scan the trailing STT_QUIET_SCAN_SEC for a natural
low-energy gap (a breath, a word boundary) and cut there instead of mid-word;
if no gap turns up within STT_SEGMENT_GRACE_SEC past the threshold, we force a
cut anyway. Either way, worst-case per-call compute is bounded by
STT_MAX_SEGMENT_SEC + STT_SEGMENT_GRACE_SEC regardless of turn length or how
continuously someone talks. Every emitted Transcript is a complete,
non-revisable segment (no more provisional "partial" — see asr.proto's
turn_complete/segment_index fields); a consumer reconstructs the full turn
text by joining segments in order up to the one with turn_complete=True, which
coincides with the real VAD end-of-turn or session close.
See stt/scripts/sravaani_profile.py, stt/tests/test_segment.py and
stt/docs/RUN_SRAVAANI.md.

Two profiling-driven details:
  - COLD START (~5.5s one-time CUDA/cuDNN init) is paid once at startup via a
    warm-up forward, so the first real session doesn't wear it.
  - Each NEW audio length pays a one-time kernel-autotune stall (~1s). A
    segment can be any length up to the worst case, so we PAD each snapshot up
    to the next whole second (BUCKET_SEC) — that bounds the distinct shapes to
    a handful, and the warm-up primes them. ponytail: 1s buckets; shrink if
    latency granularity matters more than the (small) padded-silence compute.

Env:
  STT_RUNTIME             transformers (default) | onnx
  SRAVAANI_MODEL          legacy Transformers model folder (required for default runtime)
  STT_MODEL_DIR           Transformers model folder (preferred alias for SRAVAANI_MODEL)
  STT_ONNX_MODEL_DIR      ONNX model folder (default depends on variant)
  STT_ONNX_VARIANT        int8 (default) | fp8 | fp16
  STT_ONNX_PROVIDER       auto (default) | cpu | cuda
  STT_ADDR                listen address (default [::]:50052)
  STT_MAX_SEGMENT_SEC     cut a segment once buffered audio exceeds this (default 8.0)
  STT_SEGMENT_GRACE_SEC   force-cut this long past the threshold if no quiet point found (default 2.0)
  STT_QUIET_SCAN_SEC      trailing window scanned for a natural pause to cut on (default 1.0)
  STT_QUIET_RMS_THRESHOLD int16-RMS a frame must be under to count as "quiet" (default 400; calibrate per mic/environment)
  STT_QUIET_MIN_SEC       contiguous quiet audio required for a natural cut (default 0.06)
  STT_MAX_INFLIGHT        cap queued + running GPU forwards across streams (default 8)
  STT_WARM_ALL_BUCKETS    warm every whole-second input shape at startup (default 1)
  STT_DEVICE              auto (default) | cpu (force CPU even with a GPU) | cuda
  STT_CPU_THREADS         cap torch intra-op threads on CPU (default: all physical cores)
  CUDA_VISIBLE_DEVICES    pin to GPU1 on a multi-GPU box (see PRD topology)

Terminal profiling toggles (all off by default — Prometheus always records regardless):
  STT_LOG_INFERENCE=1     per-forward line (window/compute/wait/proc/rtf) — noisy, one per segment
  STT_PROFILE_RES=1       periodic CPU/RAM/GPU-util/GPU-mem summary line
  STT_PROFILE_INTERVAL=5  seconds between resource summaries (with STT_PROFILE_RES)
"""
import asyncio
import concurrent.futures
import logging
import math
import os
import shutil
import socket
import subprocess
import threading
import time
import warnings

import numpy as np
import psutil
import torch
from service.backends import SttBackend, load_backend

from proto import asr_pb2, asr_pb2_grpc
import grpc
from grpc_health.v1 import health, health_pb2, health_pb2_grpc
from prometheus_client import Counter, Gauge, Histogram, start_http_server

warnings.filterwarnings("ignore", message="RNN module weights are not part")


def _configure_logging() -> None:
    """Keep service terminals quiet by default; use LOG_LEVEL=INFO/DEBUG when needed."""
    level_name = os.environ.get("LOG_LEVEL", "ERROR").upper()
    level = getattr(logging, level_name, logging.ERROR)
    if not isinstance(level, int):
        level = logging.ERROR
    logging.basicConfig(level=level, format="%(levelname)s %(name)s | %(message)s", force=True)


def _startup_banner(lines: list[str]) -> None:
    """One always-visible readiness summary; LOG_LEVEL controls all later chatter."""
    if os.environ.get("STARTUP_BANNER", "1") == "0":
        return
    width = 60
    print("\n+" + "=" * width + "+")
    print(f"|  {'STT SERVICE  |  READY':<58}|")
    print("+" + "-" * width + "+")
    for line in lines:
        print(f"|  {line:<58.58}|")
    print("+" + "=" * width + "+\n", flush=True)


def _load_dotenv():
    """Load KEY=VALUE lines from stt/.env (this file lives in stt/service/, one
    level below it). Real env wins (CLI overrides the file), so
    `STT_PROFILE_RES=1 python server.py` still beats a file default.

    Bug fix (Phase 3): this previously computed dirname(__file__) -- i.e.
    stt/service/.env, which has never existed -- so stt/.env silently never
    loaded via this path; every prior run depended on the caller having
    already exported the same variables into the shell environment."""
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if not os.path.isfile(path):
        return
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_load_dotenv()  # must run before the module-level os.environ reads below
_configure_logging()
log = logging.getLogger("stt")

# STT_DEVICE: auto (use GPU if present) | cpu (force CPU even on a GPU box) | cuda.
# CPU path is identical code — just fp32 and torch's own intra-op threads. A CPU-only
# torch venv makes torch.cuda.is_available() False, so `auto` already lands on CPU there.
_DEVICE = os.environ.get("STT_DEVICE", "auto").lower()
if _DEVICE not in {"auto", "cpu", "cuda"}:
    raise RuntimeError("STT_DEVICE must be one of: auto, cpu, cuda")
# STT_DEVICE belongs to the legacy Transformers backend.  The ONNX facade has
# its own STT_ONNX_PROVIDER switch, so do not reject an otherwise-valid CPU
# ONNX deployment merely because an inherited STT_DEVICE says cuda.
if (_DEVICE == "cuda" and os.environ.get("STT_RUNTIME", "transformers").lower() != "onnx"
        and not torch.cuda.is_available()):
    raise RuntimeError("STT_DEVICE=cuda was requested, but PyTorch cannot access CUDA")
CUDA = torch.cuda.is_available() and _DEVICE != "cpu"
DEV = "cuda" if CUDA else "cpu"
if not CUDA:  # optional core cap; default lets torch use all physical cores
    _threads = os.environ.get("STT_CPU_THREADS")
    if _threads:
        torch.set_num_threads(int(_threads))
SR = 16000
BUCKET_SEC = 1.0            # pad each snapshot up to a whole-second multiple (bounds shapes)

# Bounded-segment cut tuning (see module docstring). Byte constants are derived once
# at import time — PCM16 mono, so 2 bytes/sample.
MAX_SEGMENT_SEC = float(os.environ.get("STT_MAX_SEGMENT_SEC", "8.0"))
SEGMENT_GRACE_SEC = float(os.environ.get("STT_SEGMENT_GRACE_SEC", "2.0"))
QUIET_SCAN_SEC = float(os.environ.get("STT_QUIET_SCAN_SEC", "1.0"))
QUIET_RMS_THRESHOLD = float(os.environ.get("STT_QUIET_RMS_THRESHOLD", "400"))
QUIET_FRAME_MS = 30  # algorithm-internal scan granularity, not an operator knob
QUIET_MIN_SEC = float(os.environ.get("STT_QUIET_MIN_SEC", "0.06"))
MAX_INFLIGHT = int(os.environ.get("STT_MAX_INFLIGHT", "8"))
WARM_ALL_BUCKETS = os.environ.get("STT_WARM_ALL_BUCKETS", "1") != "0"
# Per-stream bound on buffered-but-not-yet-transcribed audio segments: a count
# bound alone can still hold minutes of stale speech (each segment can be up to
# MAX_SEGMENT_SEC+SEGMENT_GRACE_SEC long), so age is bounded independently. On
# overflow the stream aborts explicitly rather than growing memory without
# bound or silently splicing segments together.
SEGMENT_QUEUE_MAX = int(os.environ.get("STT_SEGMENT_QUEUE_MAX", "4"))
SEGMENT_QUEUE_MAX_AGE_SEC = float(os.environ.get("STT_SEGMENT_QUEUE_MAX_AGE_SEC", "20"))

if MAX_SEGMENT_SEC <= 0 or SEGMENT_GRACE_SEC < 0 or QUIET_SCAN_SEC <= 0:
    raise RuntimeError("STT_MAX_SEGMENT_SEC and STT_QUIET_SCAN_SEC must be > 0; STT_SEGMENT_GRACE_SEC must be >= 0")
if QUIET_RMS_THRESHOLD < 0 or QUIET_MIN_SEC <= 0 or MAX_INFLIGHT < 1:
    raise RuntimeError("STT_QUIET_RMS_THRESHOLD must be >= 0; STT_QUIET_MIN_SEC and STT_MAX_INFLIGHT must be > 0")
if SEGMENT_QUEUE_MAX < 1 or SEGMENT_QUEUE_MAX_AGE_SEC <= 0:
    raise RuntimeError("STT_SEGMENT_QUEUE_MAX must be >= 1; STT_SEGMENT_QUEUE_MAX_AGE_SEC must be > 0")

SEG_THRESHOLD_BYTES = int(MAX_SEGMENT_SEC * SR) * 2
SEG_GRACE_BYTES = int(SEGMENT_GRACE_SEC * SR) * 2
QUIET_WINDOW_BYTES = int(QUIET_SCAN_SEC * SR) * 2
QUIET_FRAME_BYTES = int(QUIET_FRAME_MS / 1000 * SR) * 2
QUIET_MIN_FRAMES = max(1, math.ceil(QUIET_MIN_SEC * 1000 / QUIET_FRAME_MS))

# Terminal profiling toggles (see module docstring). Prometheus records unconditionally.
LOG_INFERENCE = os.environ.get("STT_LOG_INFERENCE", "0") == "1"
PROFILE_RES = os.environ.get("STT_PROFILE_RES", "0") == "1"
PROFILE_INTERVAL = float(os.environ.get("STT_PROFILE_INTERVAL", "5"))
# Privacy-conscious transcript display for live troubleshooting. A "partial" is
# a bounded mid-turn segment, not a revisable hypothesis (see module docstring).
TRANSCRIPT_LOG_MODE = os.environ.get("STT_TRANSCRIPT_LOG", "none").strip().lower()
if TRANSCRIPT_LOG_MODE not in {"none", "partials", "finals", "both"}:
    TRANSCRIPT_LOG_MODE = "none"

# One shared worker so all GPU forwards serialize — a 4 GB card can't run parallel
# forwards, and SraVaani's TorchScript graphs aren't guaranteed thread-safe anyway.
# ponytail: global GPU lock via a 1-thread pool; add replicas only if throughput bites.
GPU_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=1)
# This bounds work waiting both in asyncio and in ThreadPoolExecutor's otherwise
# unbounded internal queue. It also applies backpressure to the gRPC reader.
GPU_SLOTS = asyncio.Semaphore(MAX_INFLIGHT)

# Wedge detection (Phase 3, completes Phase 1's "a wedged worker must report
# itself unready"): a native forward can't be preempted by a timeout, so this
# is the only signal available before the process gets killed and restarted.
# Set right before dispatch to the single-worker GPU_POOL, cleared when that
# forward's future completes; GPU_POOL's max_workers=1 means at most one
# forward is ever in flight, so a single module-level timestamp is sufficient
# -- mutated only via asyncio-scheduled callbacks, so no lock is needed.
_current_forward_started_at: float | None = None
STT_WEDGE_TIMEOUT_S = max(1.0, float(os.environ.get("STT_WEDGE_TIMEOUT_S",
                                                    str(MAX_SEGMENT_SEC + SEGMENT_GRACE_SEC))))

# Prometheus metrics (scraped at STT_METRICS_PORT). The per-forward timings are
# already computed in run() — we just also record them. Buckets straddle the
# ~0.5s encoder floor (see the sravaani-perf-profile).
_LAT_BUCKETS = (.05, .1, .25, .5, .75, 1, 1.5, 2, 3, 5)
M_COMPUTE = Histogram("stt_compute_seconds", "GPU model forward time per transcription", buckets=_LAT_BUCKETS)
M_WAIT = Histogram("stt_wait_seconds", "Time a request waited for the single GPU worker", buckets=_LAT_BUCKETS)
M_PROC = Histogram("stt_proc_seconds", "Total dispatch->result time per transcription", buckets=_LAT_BUCKETS)
M_RTF = Histogram("stt_rtf", "Real-time factor compute/window; >1 means slower than real-time",
                  buckets=(.02, .05, .1, .25, .5, .75, 1, 1.5, 2))
M_WINDOW = Histogram("stt_window_seconds", "Transcription window length; bounded by MAX_SEGMENT_SEC+SEGMENT_GRACE_SEC",
                     buckets=_LAT_BUCKETS)
M_PAD = Histogram("stt_pad_seconds", "Silence padding added to bound tensor shapes",
                  buckets=(0, .1, .25, .5, .75, 1))
M_INFLIGHT = Gauge("stt_gpu_inflight", "Requests dispatched to the GPU worker, including waiters (>1 = contention)")
# stt_gpu_inflight above counts from the moment a request starts waiting for
# GPU_SLOTS, so it cannot distinguish "queued behind others" from "actually
# running" -- exactly the ambiguity Phase 2 calls out. This one only counts
# time actually spent inside the model forward.
M_EXECUTING = Gauge("stt_gpu_executing", "Requests whose model forward is actually running right now")
M_GPU_SLOTS_CAPACITY = Gauge("stt_gpu_slots_capacity", "Configured GPU worker concurrency bound (STT_MAX_INFLIGHT)")
M_SEGMENT_QUEUE_CAPACITY = Gauge("stt_segment_queue_capacity",
                                 "Configured per-stream segment queue bound (STT_SEGMENT_QUEUE_MAX)")
M_SEGMENT_QUEUE_OLDEST = Gauge("stt_segment_queue_oldest_age_seconds",
                               "Age of the oldest segment still waiting in any active stream's queue")
M_TRANSCRIPTS = Counter("stt_transcriptions_total", "Transcripts emitted", ["kind"])  # kind: turn_complete | mid_turn
M_SEGMENT_CUTS = Counter("stt_segment_cuts_total", "Mid-turn segment cuts by trigger", ["type"])  # type: quiet | forced
M_ERRORS = Counter("stt_model_errors_total", "Model forward errors")
M_BACKLOG_OVERFLOW = Counter("stt_backlog_overflow_total",
                             "Streams aborted because the per-stream segment backlog overflowed",
                             ["reason"])  # reason: depth | age

# Registry of active per-stream segment queues, for the two set_function
# gauges above. Guarded by a lock because the Prometheus scrape runs on its
# own thread (start_http_server), separate from the asyncio event loop.
_ACTIVE_SEGMENT_QUEUES: set = set()
_SEGQ_LOCK = threading.Lock()


def _segment_queue_depth() -> float:
    with _SEGQ_LOCK:
        return float(sum(q.qsize() for q in _ACTIVE_SEGMENT_QUEUES))


def _segment_queue_oldest_age() -> float:
    now = time.monotonic()
    oldest = None
    with _SEGQ_LOCK:
        queues = list(_ACTIVE_SEGMENT_QUEUES)
    for q in queues:
        # Items are (enqueued_at, pcm, turn_complete) or the None sentinel.
        # asyncio.Queue is deque-backed; peek defensively rather than pop, so
        # a scrape never disturbs the stream being scraped.
        try:
            item = q._queue[0]
        except (AttributeError, IndexError):
            continue
        if item is None:
            continue
        ts = item[0]
        if oldest is None or ts < oldest:
            oldest = ts
    return max(0.0, now - oldest) if oldest is not None else 0.0


M_SEGMENT_QUEUE_DEPTH = Gauge("stt_segment_queue_depth",
                              "Segments queued for transcription, summed across active streams")
M_SEGMENT_QUEUE_DEPTH.set_function(_segment_queue_depth)
M_SEGMENT_QUEUE_OLDEST.set_function(_segment_queue_oldest_age)
M_SEGMENT_QUEUE_CAPACITY.set_function(lambda: float(SEGMENT_QUEUE_MAX))
M_GPU_SLOTS_CAPACITY.set_function(lambda: float(MAX_INFLIGHT))

# Event-loop scheduling drift (see the orchestrator's identical monitor for
# the full rationale): rising values mean the loop is falling behind its own
# scheduled work, independent of which coroutine is responsible.
M_LOOP_LAG = Gauge("stt_event_loop_lag_seconds",
                   "Event-loop scheduling drift; rising values mean the loop is falling behind")


async def _loop_lag_monitor(interval_s: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    target = loop.time() + interval_s
    while True:
        await asyncio.sleep(interval_s)
        now = loop.time()
        M_LOOP_LAG.set(max(0.0, now - target))
        target = now + interval_s


def _clear_forward_started_at(_future) -> None:
    # Registered via add_done_callback on an asyncio-wrapped executor Future,
    # so this runs on the event loop thread (call_soon_threadsafe under the
    # hood), not the GPU_POOL worker thread -- safe to touch module state.
    global _current_forward_started_at
    _current_forward_started_at = None


async def _wedge_watchdog(health_servicer: "health.aio.HealthServicer", interval_s: float = 1.0) -> None:
    """Flip gRPC health to NOT_SERVING while the sole GPU worker's current
    forward has run longer than STT_WEDGE_TIMEOUT_S. A native ONNX/torch
    forward can't be preempted, so self-reporting unready is the only signal
    available before the Phase 3 supervisor kills and restarts the process --
    completes Phase 1's "a wedged worker must report itself unready"."""
    reported_unhealthy = False
    while True:
        await asyncio.sleep(interval_s)
        started_at = _current_forward_started_at
        wedged = started_at is not None and (time.monotonic() - started_at) > STT_WEDGE_TIMEOUT_S
        if wedged and not reported_unhealthy:
            reported_unhealthy = True
            log.error("GPU worker forward has run >%.1fs; reporting NOT_SERVING", STT_WEDGE_TIMEOUT_S)
            await health_servicer.set("", health_pb2.HealthCheckResponse.NOT_SERVING)
        elif not wedged and reported_unhealthy:
            reported_unhealthy = False
            log.info("GPU worker forward completed; reporting SERVING again")
            await health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)


def _now_ms() -> int:
    return int(time.time() * 1000)


def _transcript_event(session_id: str, segment_index: int, text: str,
                      turn_complete: bool, proc_ms: float) -> None:
    """Show selected transcript events even while routine logs stay quiet."""
    kind = "final" if turn_complete else "partial"
    if TRANSCRIPT_LOG_MODE not in {"both", f"{kind}s"}:
        return
    compact_text = " ".join(text.split())
    print(
        f'TRANSCRIPT stt | session={session_id} segment={segment_index} '
        f'{kind} | processing={proc_ms:.0f}ms | "{compact_text}"',
        flush=True,
    )


_PROC = psutil.Process()
_HAVE_SMI = CUDA and bool(shutil.which("nvidia-smi"))
_GPU_INDEX = torch.cuda.current_device() if CUDA else None
_GPU_UUID = os.environ.get("TELEMETRY_GPU_UUID")
if CUDA and not _GPU_UUID and _HAVE_SMI:
    try:
        _GPU_UUID = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader",
             "--id", str(_GPU_INDEX)], text=True, timeout=2,
        ).strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        pass


def _parse_smi(out: str):
    """First line of `utilization.gpu,memory.used` csv -> (util%, mem_MiB)."""
    u, m = out.strip().splitlines()[0].split(",")
    return float(u), float(m)


def _gpu_stats():
    """(util%, used_MiB) for the pinned device via nvidia-smi, or (None, None)."""
    if not _HAVE_SMI:
        return None, None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", "-i", str(torch.cuda.current_device())],
            text=True, timeout=2)
        return _parse_smi(out)
    except Exception:
        return None, None


async def resource_logger():
    """Log a CPU/RAM/GPU summary every PROFILE_INTERVAL sec. cpu_percent() is measured
    since the previous call, so the loop cadence IS the sampling window."""
    _PROC.cpu_percent()      # prime: first call always returns 0.0
    psutil.cpu_percent()
    while True:
        await asyncio.sleep(PROFILE_INTERVAL)
        vm = psutil.virtual_memory()
        line = ("res | cpu proc %4.0f%% sys %3.0f%% | ram proc %5.0fMiB sys %4.1f/%.1fGB"
                % (_PROC.cpu_percent(), psutil.cpu_percent(),
                   _PROC.memory_info().rss / 2**20,
                   (vm.total - vm.available) / 2**30, vm.total / 2**30))
        if CUDA:
            u, m = _gpu_stats()
            line += (" | gpu util %3.0f%% mem %5.0fMiB (torch %.0fMiB)"
                     % (u or 0, m or 0, torch.cuda.memory_allocated() / 2**20))
        log.info(line)


def load_model() -> SttBackend:
    """Compatibility name for the STT facade loader."""
    return load_backend()


def _profile(backend: SttBackend, compute_ms: float, wait_ms: float, window_s: float, pad_s: float,
             rtf: float):
    """The STT-side part of a transcript trace, sent to the orchestrator."""
    attributes = {
        "sample_rate_hz": str(SR),
        "segment_max_seconds": str(MAX_SEGMENT_SEC),
        "cuda_available": str(torch.cuda.is_available()).lower(),
        "requested_device": _DEVICE,
        "effective_device": getattr(backend, "device", DEV),
        "host_id": os.environ.get("TELEMETRY_HOST_ID") or socket.gethostname(),
        "torch_version": str(torch.__version__),
    }
    for key, value in {
        "deployment_id": os.environ.get("TELEMETRY_DEPLOYMENT_ID"),
        "service_instance_id": os.environ.get("TELEMETRY_SERVICE_INSTANCE_ID"),
        "model_revision": os.environ.get("STT_MODEL_REVISION"),
        "gpu_uuid": _GPU_UUID,
        "gpu_index": _GPU_INDEX,
    }.items():
        if value not in (None, ""):
            attributes[key] = str(value)
    return asr_pb2.InferenceProfile(
        model=getattr(backend, "model_name", "sravaani"),
        engine=getattr(backend, "engine", "test"),
        device=getattr(backend, "device", DEV),
        compute_type=getattr(backend, "compute_type", "float32"),
        queue_ms=wait_ms,
        compute_ms=compute_ms,
        audio_seconds=max(0.0, window_s - pad_s),
        padded_seconds=pad_s,
        realtime_factor=rtf,
        attributes={**attributes, **getattr(backend, "profile_attributes", {})},
    )


def transcribe_pcm(model, pcm: bytes):
    """PCM16 bytes -> (text, compute_ms, window_sec, pad_sec).

    Pads up to the next BUCKET_SEC to bound shapes. compute_ms is the pure model
    time (transcribe() ends on a greedy decode that .item()s, so CUDA is synced by
    return — no explicit sync needed)."""
    arr = np.frombuffer(pcm[: len(pcm) & ~1], dtype="<i2").astype(np.float32) / 32768.0
    if arr.size < int(0.1 * SR):
        return "", 0.0, 0.0, 0.0
    pad_sec = 0.0
    if not getattr(model, "preserve_input_length", False):
        bucket = int(BUCKET_SEC * SR)
        pad = (-arr.size) % bucket
        pad_sec = pad / SR
        if pad:
            arr = np.concatenate([arr, np.zeros(pad, dtype=np.float32)])
    window_sec = arr.size / SR
    t = time.perf_counter()
    if hasattr(model, "transcribe_waveform"):
        text = model.transcribe_waveform(arr)
    else:
        # Kept for the lightweight segmentation test's FakeModel and external
        # callers that used this helper before the facade existed.
        hyps = model.transcribe([arr], return_hypotheses=True)
        text = hyps[0].text
    return text, (time.perf_counter() - t) * 1000.0, window_sec, pad_sec


def _transcribe_pcm_tracked(model, pcm: bytes):
    """transcribe_pcm, wrapped to mark true GPU-forward occupancy.

    GPU_SLOTS (up to STT_MAX_INFLIGHT) only bounds how many forwards may be
    *admitted* into GPU_POOL's own internal queue; GPU_POOL itself has a
    single worker thread, so "admitted" is not "running" -- a request can
    hold its slot while still queued behind another admitted forward. This
    inc/dec runs on the worker thread itself, at the moment the forward
    actually starts and ends, so stt_gpu_executing is never > 1 and never
    counts a request that is merely admitted-but-waiting.
    """
    M_EXECUTING.inc()
    try:
        return transcribe_pcm(model, pcm)
    finally:
        M_EXECUTING.dec()


def _find_quiet_cut(buf: bytes, window_bytes: int, frame_bytes: int, quiet_rms: float,
                    min_quiet_frames: int = QUIET_MIN_FRAMES):
    """Scan the trailing `window_bytes` of PCM16 mono `buf` in `frame_bytes`-byte
    frames for a contiguous quiet run. Returns its midpoint if the run lasts at
    least `min_quiet_frames`, else None — no clear enough gap in this window,
    caller either waits for more audio or force-cuts. Pure, no I/O."""
    lo = max(0, len(buf) - window_bytes)
    window = buf[lo:]
    n_frames = len(window) // frame_bytes
    if n_frames == 0:
        return None
    spf = frame_bytes // 2  # samples per frame (PCM16 = 2 bytes/sample)
    arr = np.frombuffer(window[: n_frames * frame_bytes], dtype="<i2")
    frames = arr.reshape(n_frames, spf).astype(np.float32)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    quiet = rms < quiet_rms
    runs = []
    start = 0
    while start < n_frames:
        if not quiet[start]:
            start += 1
            continue
        end = start + 1
        while end < n_frames and quiet[end]:
            end += 1
        if end - start >= min_quiet_frames:
            runs.append((start, end))
        start = end
    if not runs:
        return None
    # Prefer the longest pause, then the lowest-energy one; this avoids cutting
    # at a fleeting low-RMS phoneme when there is a real inter-word pause nearby.
    start, end = min(runs, key=lambda r: (-(r[1] - r[0]), float(rms[r[0]:r[1]].mean())))
    return lo + (start + end) * frame_bytes // 2


class AsrServicer(asr_pb2_grpc.AsrServicer):
    def __init__(self, model):
        self.model = model

    async def Transcribe(self, request_iterator, context):
        buf = bytearray()
        st = {"sid": "", "turn_has_audio": False}
        # segments holds (enqueued_at, pcm_bytes, turn_complete) tuples, or None
        # as the session-over sentinel. A mid-turn cut pushes turn_complete=False
        # and RETAINS the remainder of buf (unlike END_TURN, which clears it) —
        # the next segment continues from where this one was cut, not empty.
        # The global GPU slot limit bounds executor work across streams. This
        # per-RPC queue stays independent so a slow client cannot deadlock its
        # reader while the response generator is suspended at a yielded result.
        # It is itself bounded (by count here, by age at dequeue below): an
        # unbounded backlog would let a stalled consumer accumulate unbounded —
        # or merely count-bounded but still minutes-long — stale audio.
        segments: asyncio.Queue = asyncio.Queue(maxsize=SEGMENT_QUEUE_MAX)
        with _SEGQ_LOCK:
            _ACTIVE_SEGMENT_QUEUES.add(segments)
        overflow: list[str] = []  # non-empty once the backlog is rejected

        async def enqueue(pcm: bytes, turn_complete: bool) -> bool:
            try:
                segments.put_nowait((time.monotonic(), pcm, turn_complete))
                return True
            except asyncio.QueueFull:
                overflow.append("depth")
                return False

        async def reader():
            try:
                async for ev in request_iterator:
                    if overflow:
                        return  # stop accepting new audio; already-queued work still drains
                    k = ev.WhichOneof("msg")
                    if k == "audio":
                        buf.extend(ev.audio.pcm16)
                        st["turn_has_audio"] = st["turn_has_audio"] or bool(ev.audio.pcm16)
                        st["sid"] = ev.audio.session_id or st["sid"]
                        if len(buf) >= SEG_THRESHOLD_BYTES:
                            # Scan only the trailing window that _find_quiet_cut
                            # actually looks at, not the whole (up to threshold+
                            # grace) buffer on every incoming frame. The offset
                            # it returns is relative to the slice we pass in, so
                            # rebase it back into buf's own coordinate space
                            # before using it to cut/delete from buf.
                            tail_start = max(0, len(buf) - QUIET_WINDOW_BYTES)
                            tail_cut = _find_quiet_cut(bytes(buf[tail_start:]), QUIET_WINDOW_BYTES,
                                                       QUIET_FRAME_BYTES, QUIET_RMS_THRESHOLD)
                            cut = tail_start + tail_cut if tail_cut is not None else None
                            forced = False
                            if cut is None and len(buf) >= SEG_THRESHOLD_BYTES + SEG_GRACE_BYTES:
                                cut, forced = len(buf), True  # no gap found in time — bound worst-case compute anyway
                            if cut is not None:
                                if not await enqueue(bytes(buf[:cut]), False):
                                    return
                                M_SEGMENT_CUTS.labels("forced" if forced else "quiet").inc()
                                del buf[:cut]
                    elif k == "control":
                        st["sid"] = ev.control.session_id or st["sid"]
                        t = ev.control.type
                        if t == asr_pb2.Control.END_TURN:
                            if buf:  # close out this turn; next turn starts fresh
                                if not await enqueue(bytes(buf), True):
                                    return
                                buf.clear()
                            elif st["turn_has_audio"]:
                                # A preceding mid-turn cut can leave no tail to
                                # transcribe. Still emit the final marker so the
                                # orchestrator commits the already-emitted parts.
                                if not await enqueue(b"", True):
                                    return
                            st["turn_has_audio"] = False
                        elif t == asr_pb2.Control.SESSION_CLOSE:
                            break
            finally:
                if not overflow:
                    if buf:  # flush the last in-progress turn as complete
                        await enqueue(bytes(buf), True)
                        buf.clear()
                    elif st["turn_has_audio"]:
                        await enqueue(b"", True)
                st["turn_has_audio"] = False
                # Sentinel: session over (or overflow -- the consumer decides
                # which once it drains what was already accepted). A blocking
                # put is correct even under overflow: the consumer keeps
                # draining accepted segments, so room always frees up.
                await segments.put(None)

        rt = asyncio.create_task(reader())
        loop = asyncio.get_running_loop()

        async def run(snap: bytes, turn_complete: bool):
            # proc_ms = dispatch->result wall time. It exceeds compute_ms only when the
            # single GPU worker is busy with another session — that gap IS queue wait.
            t0 = time.perf_counter()
            M_INFLIGHT.inc()
            acquired_slot = False
            released = False
            forward_release_pending = False

            def release_slot():
                nonlocal released
                if released:
                    return
                released = True
                if acquired_slot:
                    GPU_SLOTS.release()
                M_INFLIGHT.dec()

            try:
                await GPU_SLOTS.acquire()
                acquired_slot = True
                global _current_forward_started_at
                _current_forward_started_at = time.monotonic()
                work = loop.run_in_executor(GPU_POOL, _transcribe_pcm_tracked, self.model, snap)
                work.add_done_callback(_clear_forward_started_at)
                try:
                    text, compute_ms, win_s, pad_s = await asyncio.shield(work)
                except asyncio.CancelledError:
                    # A native forward can't be preempted: the executor thread
                    # keeps running it even though this await is torn down. Hold
                    # the slot until the forward actually exits (mirrors TTS),
                    # so a wedged/slow forward can never let a second forward
                    # run concurrently against the same non-thread-safe model.
                    forward_release_pending = True
                    work.add_done_callback(lambda _: release_slot())
                    raise
            except Exception:
                M_ERRORS.inc()
                raise
            finally:
                if not forward_release_pending:
                    release_slot()
            proc_ms = (time.perf_counter() - t0) * 1000.0
            wait_ms = max(proc_ms - compute_ms, 0.0)
            rtf = (compute_ms / 1000.0) / win_s if win_s else 0.0
            M_COMPUTE.observe(compute_ms / 1000.0)
            M_WAIT.observe(wait_ms / 1000.0)
            M_PROC.observe(proc_ms / 1000.0)
            if win_s:
                M_RTF.observe(rtf)
                M_WINDOW.observe(win_s)
                M_PAD.observe(pad_s)
            if LOG_INFERENCE:
                log.info("%s %-8s window %4.1fs (pad %.1fs) | compute %5.0fms  wait %4.0fms  "
                         "proc %5.0fms  rtf %.2f", st["sid"], "TURN-END" if turn_complete else "mid-cut",
                         win_s, pad_s, compute_ms, wait_ms, proc_ms, rtf)
            return text, proc_ms, _profile(self.model, compute_ms, wait_ms, win_s, pad_s, rtf)

        try:
            seg_idx = 0
            while True:
                item = await segments.get()  # queue-driven, no polling: nothing to wait on but the next segment
                if item is None:
                    break
                enqueued_at, snap, turn_complete = item
                if time.monotonic() - enqueued_at > SEGMENT_QUEUE_MAX_AGE_SEC:
                    # Already-accepted audio is honored (see above), but audio
                    # this stale is no longer useful for a live conversation —
                    # stop rather than feed an increasingly-outdated turn.
                    overflow.append("age")
                    break
                text, proc_ms, profile = await run(snap, turn_complete)
                if text or turn_complete:
                    M_TRANSCRIPTS.labels("turn_complete" if turn_complete else "mid_turn").inc()
                    _transcript_event(st["sid"], seg_idx, text, turn_complete, proc_ms)
                    yield asr_pb2.Transcript(session_id=st["sid"], text=text,
                                             turn_complete=turn_complete, segment_index=seg_idx,
                                             ts_ms=_now_ms(), proc_ms=int(proc_ms), profile=profile)
                seg_idx = 0 if turn_complete else seg_idx + 1
            log.info("session %s done", st["sid"])
        finally:
            rt.cancel()
            with _SEGQ_LOCK:
                _ACTIVE_SEGMENT_QUEUES.discard(segments)
        if overflow:
            reason = overflow[0]
            M_BACKLOG_OVERFLOW.labels(reason).inc()
            log.warning("session %s: audio backlog overflow (%s); aborting stream", st["sid"], reason)
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED,
                                f"audio backlog overflow ({reason}); reconnect the session")


async def serve():
    model = load_model()
    # Pay cold start and prime all shapes a real segment can land on, so no real
    # request pays a kernel-autotune stall. Operators can retain the old faster
    # startup trade-off with STT_WARM_ALL_BUCKETS=0.
    log.info("warming up (paying one-time cold start + shape autotune) ...")
    t0 = time.time()
    worst_sec = math.ceil((MAX_SEGMENT_SEC + SEGMENT_GRACE_SEC) / BUCKET_SEC) * BUCKET_SEC
    warm_secs = (range(1, int(worst_sec) + 1) if WARM_ALL_BUCKETS
                 else sorted({BUCKET_SEC, 2 * BUCKET_SEC, worst_sec}))
    for sec in warm_secs:
        model.warmup(np.zeros(int(sec * SR), dtype=np.float32))
    log.info("warmup done in %.1fs — ready", time.time() - t0)

    metrics_port = int(os.environ.get("STT_METRICS_PORT", "9103"))
    metrics_host = os.environ.get("STT_METRICS_HOST", "0.0.0.0")
    start_http_server(metrics_port, addr=metrics_host)  # own thread; never touches the event loop
    log.info("metrics on http://%s:%d/metrics", metrics_host, metrics_port)

    addr = os.environ.get("STT_ADDR", "[::]:50052")
    server = grpc.aio.server()
    asr_pb2_grpc.add_AsrServicer_to_server(AsrServicer(model), server)
    health_servicer = health.aio.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    await health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)
    server.add_insecure_port(addr)
    await server.start()
    asyncio.create_task(_loop_lag_monitor())
    asyncio.create_task(_wedge_watchdog(health_servicer))
    if PROFILE_RES:
        asyncio.create_task(resource_logger())
        log.info("resource profiling on (every %.0fs)", PROFILE_INTERVAL)
    _startup_banner([
        f"gRPC      {addr}",
        f"Model     {model.model_name} | {model.engine} ({model.device})",
        f"Segments  {MAX_SEGMENT_SEC:.1f}s + {SEGMENT_GRACE_SEC:.1f}s grace",
        f"GPU queue  {MAX_INFLIGHT} forwards max | warm all buckets: {WARM_ALL_BUCKETS}",
        f"Metrics   http://{metrics_host}:{metrics_port}/metrics",
        f"Transcript display: {TRANSCRIPT_LOG_MODE}  |  LOG_LEVEL=INFO",
    ])
    log.info("STT gRPC listening on %s (max segment %.1fs + %.1fs grace)",
             addr, MAX_SEGMENT_SEC, SEGMENT_GRACE_SEC)
    await server.wait_for_termination()


if __name__ == "__main__":
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass
