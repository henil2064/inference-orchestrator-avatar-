"""TTS backend facade.

One loaded model plus one fixed reference voice, behind a uniform call. The
gRPC server depends only on this surface, so swapping a PyTorch model for a
``.gguf`` one is a backend change, not a service change -- the same split the
adaptive LLM runtime draws between vLLM and llama.cpp engines.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Iterator

import numpy as np


class TtsBackend(ABC):
    """A voice-cloning TTS model conditioned on a single reference voice.

    Concrete backends own model loading and the render path. They set the
    profile fields below in ``__init__`` (or ``load``) so the server can report
    inference telemetry without knowing which engine produced the audio.
    """

    # Populated by concrete backends before serving.
    engine: str = "unknown"          # profile.engine label, e.g. "zipvoice-flow-matching"
    model_name: str = "unknown"      # profile.model label
    device: str = "cpu"              # "cpu" | "cuda"
    compute_type: str = "float32"    # "float16" | "float32" | "gguf"
    sampling_rate: int = 24000       # Hz of the returned waveform
    profile_attributes: dict[str, str] = {}  # engine-specific knobs for telemetry
    # True only for a backend that can genuinely decode ahead of its own full
    # output (an autoregressive token generator whose codec can render a
    # partial token run). Flow-matching/diffusion samplers cannot -- they
    # need the whole schedule before any audio exists -- so this stays False
    # for them and the default synthesize_stream() below is the honest answer.
    supports_stream: bool = False

    @abstractmethod
    def synthesize(self, text: str) -> np.ndarray:
        """Render ``text`` to mono float32 samples in [-1, 1] at ``sampling_rate``."""

    def synthesize_stream(self, text: str) -> Iterator[np.ndarray]:
        """Render ``text`` as a sequence of audio chunks, earliest-available
        first. Default: no true streaming -- render the whole utterance, then
        yield it as a single chunk. A backend with ``supports_stream = True``
        overrides this to decode and yield before the full render is done."""
        yield self.synthesize(text)

    def warmup(self) -> None:
        """Exercise the full render path once so readiness is meaningful."""
        self.synthesize("Ready.")

    def split_sentences(self, text: str, max_chars: int = 360) -> list[str]:
        """Smallest safe streaming units: sentences (incl. Indic danda), then
        hard-wrapped at ``max_chars``. Flow-matching samplers are not
        token-streamable, so one sentence is the finest unit the caller can
        emit before the next is rendered."""
        return split_sentences(text, max_chars)


def split_sentences(text: str, max_chars: int = 360) -> list[str]:
    sentences = [p.strip() for p in re.split(r"(?<=[.!?।！？])\s+", text) if p.strip()]
    pieces: list[str] = []
    for sentence in sentences or [text]:
        for start in range(0, len(sentence), max_chars):
            pieces.append(sentence[start:start + max_chars])
    return pieces


def pcm16_bytes(audio: np.ndarray) -> bytes:
    """Exact wire format: clamp to avoid int16 wrap-around, little-endian PCM16."""
    return (np.clip(np.asarray(audio, dtype=np.float32), -1, 1) * 32767).astype("<i2", copy=False).tobytes()
