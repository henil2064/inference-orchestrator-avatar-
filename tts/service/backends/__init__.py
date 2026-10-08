"""Backend selection by model format.

Same gate the adaptive LLM runtime uses: a ``.gguf`` path routes to the
llama.cpp backend, anything else (a HF model directory) to bare PyTorch.
"""
from __future__ import annotations

import os
from pathlib import Path

from .base import TtsBackend, pcm16_bytes, split_sentences

__all__ = ["TtsBackend", "pcm16_bytes", "split_sentences", "load_backend"]


def load_backend() -> TtsBackend:
    default_dir = Path(__file__).resolve().parents[1] / "models/DhVaani-0.5-fp16"
    model_path = Path(os.environ.get("TTS_MODEL_DIR", default_dir)).resolve()
    fmt = os.environ.get("TTS_FORMAT", "").lower() or ("gguf" if model_path.suffix.lower() == ".gguf" else "pytorch")

    if fmt == "gguf":
        from .llamacpp_backend import LlamaCppBackend
        return LlamaCppBackend(model_path)
    from .pytorch_backend import PyTorchBackend
    return PyTorchBackend(model_path)
