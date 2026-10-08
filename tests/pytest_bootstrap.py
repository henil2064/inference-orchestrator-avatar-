#!/usr/bin/env python3
"""Run pytest under a service interpreter using repo test tooling.

Service venvs own service dependencies. The repo test venv owns pytest.
"""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
TEST_VENV = ROOT / "tests" / ".venv"


def _add_test_site_packages() -> None:
    candidates = [
        *TEST_VENV.glob("lib/python*/site-packages"),
        TEST_VENV / "Lib" / "site-packages",
    ]
    for path in candidates:
        if path.is_dir():
            sys.path.append(str(path))


def main(argv: list[str]) -> int:
    _add_test_site_packages()
    try:
        import pytest
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"pytest is not available from {TEST_VENV}. "
            "Create tests/.venv or override with TEST_PYTHON=/path/to/python."
        ) from exc
    return pytest.main(argv)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
