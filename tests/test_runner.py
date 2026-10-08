from __future__ import annotations

from pathlib import Path

import tests.run as runner


def _manifest():
    return {
        "suites": {
            "fast-a": {"tier": "unit", "default": True},
            "fast-b": {"tier": "service", "default": True},
            "manual": {"tier": "integration", "default": False},
        }
    }


def test_no_target_selects_default_suites_only():
    assert runner._targets(_manifest(), []) == ["fast-a", "fast-b"]


def test_targets_are_expanded_and_deduped_in_order():
    assert runner._targets(_manifest(), ["service", "fast-a", "all"]) == [
        "fast-b",
        "fast-a",
    ]


def test_unknown_target_reports_available_names():
    try:
        runner._targets(_manifest(), ["missing"])
    except SystemExit as exc:
        assert "unknown test target 'missing'" in str(exc)
        assert "fast-a" in str(exc)
    else:
        raise AssertionError("unknown targets must fail")


def test_suite_python_uses_configured_interpreter_relative_to_cwd(tmp_path: Path):
    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")

    assert runner._suite_python({"python": ".venv/bin/python"}, tmp_path) == str(python)


def test_suite_python_allows_test_python_override(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("TEST_PYTHON", "/tmp/custom-python")

    assert runner._suite_python({"python": ".venv/bin/python"}, tmp_path) == "/tmp/custom-python"


def test_suite_python_reports_missing_configured_interpreter(tmp_path: Path):
    try:
        runner._suite_python({"python": ".venv/bin/python"}, tmp_path)
    except SystemExit as exc:
        assert "configured Python interpreter does not exist" in str(exc)
        assert str(tmp_path / ".venv" / "bin" / "python") in str(exc)
    else:
        raise AssertionError("missing configured interpreter must fail")


def test_pytest_command_uses_service_python_with_repo_pytest_bootstrap():
    assert runner._resolved_command(
        ["python", "-m", "pytest", "tests/test_runner.py"],
        "/service/.venv/bin/python",
    ) == [
        "/service/.venv/bin/python",
        str(runner.PYTEST_BOOTSTRAP),
        "tests/test_runner.py",
    ]


def test_non_pytest_python_command_uses_service_python():
    assert runner._resolved_command(
        ["python", "stream_wav.py"],
        "/service/.venv/bin/python",
    ) == ["/service/.venv/bin/python", "stream_wav.py"]
