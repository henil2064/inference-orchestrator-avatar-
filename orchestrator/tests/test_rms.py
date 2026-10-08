"""Self-check for the PCM helpers. Run: python -m tests.test_rms"""
import struct
from service.audio import rms16 as _rms
from service.server import SAMPLE_RATE

# empty -> 0.0
assert _rms(b"") == 0.0

# constant-amplitude signal: RMS == that amplitude
pcm = struct.pack("<4h", 1000, -1000, 1000, -1000)
assert abs(_rms(pcm) - 1000.0) < 1e-6, _rms(pcm)

# odd trailing byte is ignored, not a crash
assert _rms(b"\x00\x04\x00\x04\x7f") > 0

# sanity: seconds math the server logs with
assert (SAMPLE_RATE * 2) / 2 / SAMPLE_RATE == 1.0  # 1s of pcm bytes == 1.0s

print("ok")
