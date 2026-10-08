from __future__ import annotations

from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Apply coarse repo-wide tier markers while tests remain service-local."""
    for item in items:
        rel = Path(str(item.fspath)).resolve().relative_to(ROOT)
        rel_posix = rel.as_posix()
        markers = _markers_for(rel_posix, item.name)
        for marker in markers:
            item.add_marker(getattr(pytest.mark, marker))


def _markers_for(rel_posix: str, test_name: str) -> tuple[str, ...]:
    if rel_posix == "pipeline_test/test_stream_wav.py":
        return ("unit",)
    if rel_posix == "tts/tests/test_client.py":
        return ("integration",)
    if rel_posix == "scripts/orchestration/test_supervisor.py":
        return ("integration",)
    if rel_posix == "orchestrator/tests/test_replica_fanout.py" and test_name == "test_replica_channel_round_robins_and_excludes_unhealthy":
        return ("integration",)
    if rel_posix.startswith("adaptive_llm/tests/test_llamacpp_"):
        return ("model",)
    if rel_posix.startswith("scripts/orchestration/") or rel_posix.startswith("tests/"):
        return ("unit",)
    if rel_posix.endswith("_test.py") or "/tests/test_" in rel_posix or rel_posix.endswith("/test_segment.py"):
        return ("service",)
    return ()
