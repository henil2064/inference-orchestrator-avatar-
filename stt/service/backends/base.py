"""Model-runtime interface used by the STT service."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class SttBackend(ABC):
    """One loaded ASR model that accepts a 16 kHz mono float32 waveform."""

    engine: str = "unknown"
    model_name: str = "unknown"
    device: str = "cpu"
    compute_type: str = "float32"
    profile_attributes: dict[str, str] = {}
    # The TorchScript model benefits from server-side shape bucketing. Some
    # runtimes (notably the supplied ONNX export) normalize over every input
    # frame and must see the original duration exactly.
    preserve_input_length: bool = False

    @abstractmethod
    def transcribe_waveform(self, waveform: np.ndarray) -> str:
        """Return the greedy transcript for a normalized mono waveform."""

    def warmup(self, waveform: np.ndarray) -> None:
        self.transcribe_waveform(waveform)
