"""Static, single-query profiler for the LLM: load the model once, run one (or
--repeat N) queries through it directly — no gRPC, no orchestrator — and report
where time and resources actually go: model LOAD vs GENERATION, prefill (time to
first token) vs decode (per-token generation), GPU util/memory, CPU/RAM.

Reuses server.py's load_engine()/build_prompt() so this always profiles the exact
same engine construction + prompt formatting the real service uses (same .env,
same GGUF/gating/memory workarounds — see server.py's module docstring).

NOTE: deliberately NOT named profile.py — that shadows Python's own stdlib
`profile` module (used internally by cProfile/torch._dynamo) when run as
`python profile.py` from this directory, breaking vllm's import chain with a
confusing circular-import error. Don't rename it back to that.

Usage:
    cd llm && source .venv/bin/activate
    # .env (or exported env) must resolve the same way it does for server.py
    python query_profile.py "What are the symptoms of the flu?"
    python query_profile.py "..." --repeat 3   # run 3x; reveals any first-call warmup
"""
import argparse
import asyncio
import shutil
import subprocess
import threading
import time
import uuid

import psutil
import torch
from vllm import SamplingParams

import server  # reuses load_engine(), build_prompt(), and the resolved .env config


def _peak_mib() -> float:
    return torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0


def _reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


class _GpuSampler(threading.Thread):
    """Poll nvidia-smi in the background for GPU utilization % and memory (MiB).
    Same pattern as stt/sravaani_profile.py's sampler — kept local since llm/ and
    stt/ are separate venvs with no shared import path. Single-GPU box assumed
    (device 0); add a --gpu-index flag if this ever needs to target a specific
    card on a multi-GPU host."""
    def __init__(self, period: float = 0.05):
        super().__init__(daemon=True)
        self.period = period
        self.util: list = []
        self.mem: list = []
        self._stopflag = False

    def run(self):
        q = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits", "-i", "0"]
        while not self._stopflag:
            try:
                out = subprocess.check_output(q, text=True).strip().splitlines()[0]
                u, m = out.split(",")
                self.util.append(float(u))
                self.mem.append(float(m))
            except Exception:
                pass
            time.sleep(self.period)

    def stop(self):
        self._stopflag = True


class _CpuSampler(threading.Thread):
    """Poll this process's CPU% and RSS in the background."""
    def __init__(self, period: float = 0.1):
        super().__init__(daemon=True)
        self.period = period
        self.proc = psutil.Process()
        self.proc.cpu_percent()  # prime: first call always returns 0.0
        self.cpu: list = []
        self.rss_mib: list = []
        self._stopflag = False

    def run(self):
        while not self._stopflag:
            self.cpu.append(self.proc.cpu_percent())
            self.rss_mib.append(self.proc.memory_info().rss / 2**20)
            time.sleep(self.period)

    def stop(self):
        self._stopflag = True


def _run_phase():
    """Start a fresh GPU+CPU sampler pair for one phase (load, or generation).
    Returns (gpu_sampler_or_None, cpu_sampler)."""
    gpu = None
    if torch.cuda.is_available() and shutil.which("nvidia-smi"):
        gpu = _GpuSampler()
        gpu.start()
    cpu = _CpuSampler()
    cpu.start()
    return gpu, cpu


def _stop_phase(gpu, cpu):
    if gpu:
        gpu.stop()
        gpu.join(timeout=1.0)
    cpu.stop()
    cpu.join(timeout=1.0)


def _report_resources(label: str, gpu, cpu):
    if gpu and gpu.util:
        print(f"  GPU: util avg {sum(gpu.util)/len(gpu.util):5.1f}% / peak {max(gpu.util):5.1f}%   "
              f"|   mem avg {sum(gpu.mem)/len(gpu.mem):6.0f} MiB / peak {max(gpu.mem):6.0f} MiB")
    elif torch.cuda.is_available():
        print("  GPU: (nvidia-smi not found — no samples)")
    if cpu.cpu:
        print(f"  CPU: avg {sum(cpu.cpu)/len(cpu.cpu):5.1f}% / peak {max(cpu.cpu):5.1f}%   "
              f"|   RSS avg {sum(cpu.rss_mib)/len(cpu.rss_mib):6.0f} MiB / peak {max(cpu.rss_mib):6.0f} MiB")


async def run_once(engine, tokenizer, query: str) -> dict:
    """One query through the exact same code path server.py's Generate() RPC
    uses. Returns timing/token stats — the model's actual reply text included."""
    prompt = server.build_prompt(tokenizer, query)
    n_prompt_tokens = len(tokenizer(prompt).input_ids)
    sampling = SamplingParams(temperature=server.TEMPERATURE, max_tokens=server.MAX_TOKENS)
    request_id = str(uuid.uuid4())

    t0 = time.perf_counter()
    t_first = None
    final = None
    async for output in engine.generate(prompt, sampling, request_id):
        if t_first is None:
            t_first = time.perf_counter()
        final = output
    t_done = time.perf_counter()

    text = final.outputs[0].text if final and final.outputs else ""
    n_out = len(final.outputs[0].token_ids) if final and final.outputs else 0
    ttft_s = (t_first - t0) if t_first else 0.0
    total_s = t_done - t0
    decode_s = max(total_s - ttft_s, 0.0)
    return {
        "text": text, "n_prompt_tokens": n_prompt_tokens, "n_out": n_out,
        "ttft_s": ttft_s, "decode_s": decode_s, "total_s": total_s,
        "prefill_tok_s": n_prompt_tokens / ttft_s if ttft_s else 0.0,
        # first token's latency is already counted in TTFT, not decode — standard
        # convention for this metric in LLM serving benchmarks.
        "decode_tok_s": (n_out - 1) / decode_s if n_out > 1 and decode_s > 0 else 0.0,
    }


def report_runs(stats_list: list):
    for i, s in enumerate(stats_list):
        print(f"\n=== run {i + 1}/{len(stats_list)} ===")
        print(f"  reply: {s['text']!r}")
        print(f"  {s['n_prompt_tokens']} prompt tok -> {s['n_out']} reply tok")
        print(f"  prefill (TTFT):  {s['ttft_s']*1000:7.1f} ms   ({s['prefill_tok_s']:6.1f} tok/s)")
        print(f"  decode:          {s['decode_s']*1000:7.1f} ms   ({s['decode_tok_s']:6.1f} tok/s)")
        print(f"  TOTAL:           {s['total_s']*1000:7.1f} ms")
    if len(stats_list) > 1:
        totals = sorted(s["total_s"] for s in stats_list)
        med = totals[len(totals) // 2]
        print(f"\n=== {len(stats_list)} runs: total time min/median/max = "
              f"{totals[0]*1000:.0f}/{med*1000:.0f}/{totals[-1]*1000:.0f} ms ===")
        if stats_list[0]["total_s"] > 1.3 * med:
            print("  (run 1 was notably slower than the rest — some first-call warmup exists "
                  "despite CUDA graph capture happening at engine-load time)")


async def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", help="the user text to send through the model")
    ap.add_argument("--repeat", type=int, default=1,
                    help="run the same query N times (default 1) — reveals any first-call warmup")
    args = ap.parse_args()

    print(f"loading {server.MODEL_PATH} (tokenizer={server.TOKENIZER}) ...")
    print(f"config: max_model_len={server.MAX_MODEL_LEN or 'model default'}  "
          f"gpu_memory_utilization={server.GPU_MEMORY_UTILIZATION}  "
          f"max_num_batched_tokens={server.MAX_NUM_BATCHED_TOKENS or 'vLLM default'}  "
          f"enforce_eager={server.ENFORCE_EAGER}  enable_thinking={server.ENABLE_THINKING}")

    _reset_peak()
    gpu, cpu = _run_phase()
    t0 = time.perf_counter()
    engine, tokenizer = server.load_engine()
    load_s = time.perf_counter() - t0
    _stop_phase(gpu, cpu)

    print(f"\n--- LOAD phase: {load_s:.1f}s"
          + (f", peak VRAM (torch-visible) {_peak_mib():.0f} MiB" if torch.cuda.is_available() else "")
          + " ---")
    _report_resources("load", gpu, cpu)

    gpu, cpu = _run_phase()
    stats = [await run_once(engine, tokenizer, args.query) for _ in range(max(args.repeat, 1))]
    _stop_phase(gpu, cpu)

    print("\n--- GENERATION phase ---")
    report_runs(stats)
    _report_resources("generation", gpu, cpu)


if __name__ == "__main__":
    asyncio.run(main())
