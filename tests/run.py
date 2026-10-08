#!/usr/bin/env python3
"""Repo-level test dispatcher.

The test logic stays in service-local test directories. This runner only
selects the right command, working directory, and Python path for each suite.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tests" / "suites.yaml"
PYTEST_BOOTSTRAP = ROOT / "tests" / "pytest_bootstrap.py"


def _load_manifest() -> dict[str, Any]:
    with MANIFEST.open(encoding="utf-8") as f:
        return json.load(f)


def _merged_env(overrides: dict[str, str] | None, cwd: Path) -> dict[str, str]:
    env = os.environ.copy()
    if not overrides:
        return env
    for key, value in overrides.items():
        if key == "PYTHONPATH":
            entries = [str((cwd / item).resolve()) for item in value.split(os.pathsep) if item]
            existing = env.get("PYTHONPATH")
            if existing:
                entries.append(existing)
            env[key] = os.pathsep.join(entries)
        else:
            env[key] = value
    return env


def _suite_python(cfg: dict[str, Any], cwd: Path) -> str:
    override = os.environ.get("TEST_PYTHON")
    if override:
        return override

    configured = cfg.get("python")
    if not configured:
        return sys.executable

    path = Path(configured)
    if not path.is_absolute():
        path = cwd / path
    path = path.absolute()
    if not path.exists():
        raise SystemExit(
            f"configured Python interpreter does not exist: {path}\n"
            "Create the service venv first, or override with TEST_PYTHON=/path/to/python."
        )
    return str(path)


def _resolved_command(command: list[str], python: str) -> list[str]:
    if command[:3] == ["python", "-m", "pytest"]:
        return [python, str(PYTEST_BOOTSTRAP), *command[3:]]
    return [python if part == "python" else part for part in command]


def _targets(manifest: dict[str, Any], names: list[str]) -> list[str]:
    suites = manifest["suites"]
    if not names:
        return [name for name, cfg in suites.items() if cfg.get("default")]

    selected: list[str] = []
    for name in names:
        if name == "all":
            selected.extend(suite for suite, cfg in suites.items() if cfg.get("default"))
        elif name == "unit":
            selected.extend(suite for suite, cfg in suites.items() if cfg.get("tier") == "unit")
        elif name == "service":
            selected.extend(suite for suite, cfg in suites.items() if cfg.get("tier") == "service")
        elif name in {"integration", "e2e", "model", "gpu", "slow", "bench"}:
            selected.extend(suite for suite, cfg in suites.items() if cfg.get("tier") == name)
        elif name in suites:
            selected.append(name)
        else:
            valid = ", ".join(sorted([
                *suites, "all", "unit", "service", "integration", "e2e", "model", "gpu", "slow", "bench"
            ]))
            raise SystemExit(f"unknown test target {name!r}; valid targets: {valid}")

    deduped: list[str] = []
    for name in selected:
        if name not in deduped:
            deduped.append(name)
    return deduped


def _list(manifest: dict[str, Any]) -> None:
    for name, cfg in sorted(manifest["suites"].items()):
        default = "default" if cfg.get("default") else "opt-in"
        print(f"{name:14} {cfg.get('tier', '-'):<12} {default:8} {cfg.get('description', '')}")


def run_suite(name: str, cfg: dict[str, Any]) -> int:
    cwd = (ROOT / cfg.get("cwd", ".")).resolve()
    env = _merged_env(cfg.get("env"), cwd)
    python = _suite_python(cfg, cwd)
    print(f"\n==> {name} ({cwd.relative_to(ROOT) if cwd != ROOT else '.'})", flush=True)
    for command in cfg["commands"]:
        resolved_command = _resolved_command(command, python)
        print("+ " + " ".join(resolved_command), flush=True)
        result = subprocess.run(resolved_command, cwd=cwd, env=env)
        if result.returncode:
            return result.returncode
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "targets",
        nargs="*",
        help="suite name, all, or tier: unit, service, integration, e2e, model, gpu, slow, bench",
    )
    parser.add_argument("--list", action="store_true", help="list configured suites")
    args = parser.parse_args(argv)

    manifest = _load_manifest()
    if args.list:
        _list(manifest)
        return 0

    selected = _targets(manifest, args.targets)
    if not selected:
        print("No suites matched. This repository currently has no pure unit-only suite.", file=sys.stderr)
        return 2

    for name in selected:
        rc = run_suite(name, manifest["suites"][name])
        if rc:
            print(f"\n{name} failed with exit code {rc}", file=sys.stderr)
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
