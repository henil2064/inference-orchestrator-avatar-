"""A worker crashing during startup should surface its actual error, not just
an exit code -- otherwise "the model didn't load" is undiagnosable from the
facade's own log alone. See DECISION.md D-27 for the same fail-loud-with-a-
reason philosophy applied to GPU device selection."""

from __future__ import annotations

import asyncio
import sys

import pytest

from adaptive_llm.core import runtime_manager
from adaptive_llm.core.config import Config


def _config(**overrides) -> Config:
    values = dict(
        model_id="x", model_profiles_path=None, model_ref=None, model_file=None,
        model_dir=None, model_profile=None, auto_download=False,
        public_addr="[::]:0", worker_addr="127.0.0.1:1", metrics_port=0,
        engine_override=None, dry_run=False, auto_start_worker=True,
        worker_ready_timeout_s=5, vram_headroom_fraction=0.1, ram_headroom_fraction=0.1,
        min_headroom_mib=0, max_context_tokens=None, max_output_tokens=None,
        worker_python=sys.executable, max_inflight_requests=1, max_queued_requests=0,
        resource_metrics_interval_s=5, profile_bandwidth=False,
    )
    values.update(overrides)
    return Config(**values)


async def _spawn(code: str) -> runtime_manager.WorkerHandle:
    """Stand in for start_worker's subprocess wiring with a plain -c script,
    so this doesn't need a real engine module to exercise the crash path."""
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", code,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    tail = runtime_manager.OutputTail()
    return runtime_manager.WorkerHandle(
        address="127.0.0.1:1", process=process, tail=tail, pump_tasks=tail.tasks(process))


def test_worker_crash_message_includes_its_actual_error(capsys):
    async def scenario() -> str:
        handle = await _spawn(
            "import sys; print('loading model...'); "
            "print('RuntimeError: unsupported architecture geglu-v2', file=sys.stderr); "
            "sys.exit(1)"
        )
        try:
            with pytest.raises(RuntimeError) as exc_info:
                await runtime_manager.wait_for_worker_ready(handle, _config())
        finally:
            await runtime_manager.stop_worker(handle)
        return str(exc_info.value)

    message = asyncio.run(scenario())

    assert "exit code 1" in message
    assert "unsupported architecture geglu-v2" in message

    # The tee must not swallow live output while also retaining the tail.
    captured = capsys.readouterr()
    assert "loading model..." in captured.out
    assert "unsupported architecture geglu-v2" in captured.err


def test_worker_that_exits_cleanly_before_ready_reports_zero_exit(capsys):
    async def scenario() -> str:
        handle = await _spawn("pass")  # exits 0 without ever becoming ready
        try:
            with pytest.raises(RuntimeError) as exc_info:
                await runtime_manager.wait_for_worker_ready(handle, _config())
        finally:
            await runtime_manager.stop_worker(handle)
        return str(exc_info.value)

    message = asyncio.run(scenario())
    assert "exit code 0" in message


def test_stop_worker_cleans_up_tee_tasks_even_after_a_crash():
    async def scenario() -> runtime_manager.WorkerHandle:
        handle = await _spawn("import sys; sys.exit(1)")
        await handle.process.wait()
        await runtime_manager.stop_worker(handle)
        return handle

    handle = asyncio.run(scenario())
    assert all(task.done() for task in handle.pump_tasks)
