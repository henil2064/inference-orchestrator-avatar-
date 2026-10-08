"""Small audio helpers shared by the orchestrator runtime and tests."""
from __future__ import annotations

import math
import struct


def rms16(pcm: bytes) -> float:
    """Return RMS amplitude for little-endian signed PCM16 samples.

    PCM frames are allowed to arrive in chunks, so an incomplete trailing byte
    is ignored rather than raising an unpacking error.
    """
    usable_bytes = len(pcm) - (len(pcm) % 2)
    if usable_bytes == 0:
        return 0.0
    samples = struct.unpack(f"<{usable_bytes // 2}h", pcm[:usable_bytes])
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))
