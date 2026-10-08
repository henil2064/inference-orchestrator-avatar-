"""Writes Prometheus file_sd target files for STT/TTS replicas.

The normal Compose deployment runs Prometheus in Docker, so its targets use
``host.docker.internal``. A host-installed Prometheus cannot resolve that
Docker-only name; set ``PROMETHEUS_TARGET_HOST=127.0.0.1`` for that layout.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def write_file_sd(path: Path, ports: list[int], *, job: str) -> None:
    """Atomic write (temp file + rename) so Prometheus's file watcher never
    reads a half-written target list."""
    target_host = os.environ.get("PROMETHEUS_TARGET_HOST", "host.docker.internal")
    targets = [f"{target_host}:{port}" for port in ports]
    doc = [{"targets": targets, "labels": {"job": job}}]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2) + "\n")
    os.replace(tmp, path)
