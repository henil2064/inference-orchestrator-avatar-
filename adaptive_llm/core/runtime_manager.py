"""Worker process management for the adaptive LLM facade."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import asyncio
import os
import sys
from pathlib import Path

import grpc

from adaptive_llm.core.config import Config, ROOT
from adaptive_llm.planning.compatibility import ENGINE_LLAMACPP, ENGINE_SGLANG, ENGINE_VLLM
from adaptive_llm.planning.planner import ExecutionPlan
from adaptive_llm.planning.registry import ModelSpec

_TAIL_LINES = 40


class OutputTail:
    """Tees a worker subprocess's stdout/stderr to this process's own streams
    in real time -- so following worker logs live during normal operation is
    unaffected -- while keeping a bounded tail. On a startup failure that tail
    is the difference between "exit code 1" and the model's actual load error
    (unsupported architecture, CUDA OOM, a corrupt GGUF, ...)."""

    def __init__(self, maxlen: int = _TAIL_LINES):
        self.lines: deque[str] = deque(maxlen=maxlen)

    async def _pump(self, stream: asyncio.StreamReader | None, echo) -> None:
        if stream is None:
            return
        while True:
            raw = await stream.readline()
            if not raw:
                return
            text = raw.decode("utf-8", errors="replace").rstrip("\n")
            echo(text)
            self.lines.append(text)

    def tasks(self, process: asyncio.subprocess.Process) -> list[asyncio.Task]:
        return [
            asyncio.create_task(self._pump(process.stdout, lambda line: print(line, flush=True))),
            asyncio.create_task(self._pump(process.stderr, lambda line: print(line, file=sys.stderr, flush=True))),
        ]

    def text(self) -> str:
        return "\n".join(self.lines) if self.lines else "(worker produced no output)"


@dataclass
class WorkerHandle:
    address: str
    process: asyncio.subprocess.Process | None
    tail: OutputTail | None = None
    pump_tasks: list[asyncio.Task] = field(default_factory=list)


async def start_worker(plan: ExecutionPlan, model: ModelSpec, config: Config) -> WorkerHandle:
    if config.dry_run or not config.auto_start_worker:
        return WorkerHandle(address=config.worker_addr, process=None)

    module_name = _worker_module(plan)
    env = os.environ.copy()
    env["LLM_ADDR"] = config.worker_addr
    env["LLM_MODEL_PATH"] = _resolved_model_path(model)
    env["LLM_MAX_MODEL_LEN"] = str(plan.context_tokens)
    env["LLM_MAX_TOKENS"] = str(plan.max_output_tokens)
    for key, value in plan.engine_env.items():
        env.setdefault(key, value)
    if plan.gpu_index is not None:
        # The planner's memory estimate is for this physical GPU. Restricting
        # visibility also keeps vLLM from silently choosing a different card.
        env["CUDA_VISIBLE_DEVICES"] = str(plan.gpu_index)
    if model.tokenizer:
        env.setdefault("LLM_TOKENIZER", model.tokenizer)
    if plan.engine == ENGINE_LLAMACPP:
        if plan.gpu_index is not None:
            # CUDA_VISIBLE_DEVICES renumbers the selected physical GPU to 0.
            env["LLM_MAIN_GPU"] = "0"
        if plan.n_gpu_layers is not None:
            env["LLM_N_GPU_LAYERS"] = str(plan.n_gpu_layers)
        if plan.cpu_threads is not None:
            env["LLM_N_THREADS"] = str(plan.cpu_threads)
    if plan.engine == ENGINE_SGLANG:
        # SGLang runs its HTTP server behind the local gRPC worker. Keep it on
        # loopback: only the Adaptive LLM public API is externally exposed.
        env.setdefault("LLM_SGLANG_HOST", "127.0.0.1")
        env.setdefault("LLM_SGLANG_PORT", "30000")

    process = await asyncio.create_subprocess_exec(
        config.worker_python,
        "-m",
        module_name,
        cwd=str(ROOT),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    tail = OutputTail()
    return WorkerHandle(address=config.worker_addr, process=process, tail=tail, pump_tasks=tail.tasks(process))


async def stop_worker(handle: WorkerHandle) -> None:
    try:
        if handle.process and handle.process.returncode is None:
            handle.process.terminate()
            try:
                await asyncio.wait_for(handle.process.wait(), timeout=10)
            except asyncio.TimeoutError:
                handle.process.kill()
                await handle.process.wait()
    finally:
        for task in handle.pump_tasks:
            if not task.done():
                task.cancel()
        if handle.pump_tasks:
            await asyncio.gather(*handle.pump_tasks, return_exceptions=True)


async def wait_for_worker_ready(handle: WorkerHandle, config: Config) -> None:
    if config.dry_run:
        return
    channel = grpc.aio.insecure_channel(handle.address)
    tasks: set[asyncio.Task] = set()
    try:
        ready_task = asyncio.create_task(channel.channel_ready())
        tasks.add(ready_task)
        if handle.process is None:
            await asyncio.wait_for(ready_task, timeout=config.worker_ready_timeout_s)
            return

        exit_task = asyncio.create_task(handle.process.wait())
        tasks.add(exit_task)
        done, pending = await asyncio.wait(
            tasks,
            timeout=config.worker_ready_timeout_s,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if ready_task in done:
            return
        if exit_task in done:
            # The process has already exited, so its pipes are at EOF -- the
            # tee tasks just need a scheduling turn to drain what's already
            # buffered. Without this wait, a worker that fails almost
            # instantly can race the RuntimeError below and get built with an
            # empty tail.
            if handle.pump_tasks:
                await asyncio.wait(handle.pump_tasks, timeout=1.0)
            tail_text = handle.tail.text() if handle.tail else None
            raise RuntimeError(
                f"LLM worker exited during startup (exit code {exit_task.result()})"
                + (f"\n--- worker output (last {len(handle.tail.lines)} lines) ---\n{tail_text}" if tail_text else "")
            )
        tail_text = handle.tail.text() if handle.tail else None
        raise TimeoutError(
            f"timed out waiting {config.worker_ready_timeout_s}s for LLM worker at {handle.address} "
            "(still running -- this may be a slow model load, not a failure)"
            + (f"\n--- worker output so far (last {len(handle.tail.lines)} lines) ---\n{tail_text}" if tail_text else "")
        )
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await channel.close()


def _worker_module(plan: ExecutionPlan) -> str:
    if plan.engine == ENGINE_VLLM:
        return "adaptive_llm.engines.vllm_worker"
    if plan.engine == ENGINE_LLAMACPP:
        return "adaptive_llm.engines.llamacpp_worker"
    if plan.engine == ENGINE_SGLANG:
        return "adaptive_llm.engines.sglang_worker"
    raise ValueError(f"unsupported engine: {plan.engine}")


def _resolved_model_path(model: ModelSpec) -> str:
    path = Path(model.model_path)
    if not path.is_absolute():
        path = ROOT / path
    return str(path)
