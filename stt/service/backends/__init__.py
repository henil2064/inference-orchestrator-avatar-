"""STT runtime selection facade.

The gRPC service speaks only in PCM16 and transcript segments.  This module
chooses the implementation that turns one normalized waveform into text,
keeping the transport and segmenting policy independent of model format.
"""
from __future__ import annotations

import os

from .base import SttBackend

__all__ = ["SttBackend", "load_backend"]


def load_backend() -> SttBackend:
    """Load the configured ASR runtime.

    ``transformers`` is deliberately the default, preserving deployments that
    only set the long-standing ``SRAVAANI_MODEL`` variable.  ONNX is opt-in so
    its model files and runtime are never selected accidentally.
    """
    runtime = os.environ.get("STT_RUNTIME", "transformers").strip().lower()
    if runtime in {"transformers", "torchscript", "pytorch"}:
        from .transformers_backend import TransformersBackend
        return TransformersBackend()
    if runtime == "onnx":
        from .onnx_backend import OnnxBackend
        return OnnxBackend()
    raise RuntimeError("STT_RUNTIME must be one of: transformers, onnx")
