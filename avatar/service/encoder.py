"""H.264 encoding for rendered frames.

Raw 720p I420 is ~1.4 MB a frame, far too much to hand the edge per session,
and the edge would then need a video encoder of its own. Encoding here keeps
the payload to a few kB and lets NVENC do the work on the GPU that produced the
frame. libx264 is the fallback so the service still runs on a build of FFmpeg
without NVENC, and on a box with no GPU at all (the synthetic backend).
"""
from __future__ import annotations

import logging
from fractions import Fraction

import av
import numpy as np
from av.video.frame import PictureType

log = logging.getLogger("avatar.encoder")

# Per-codec private options. Everything here is about latency: one packet out
# per frame in, no B-frames, no lookahead. Baseline profile because that is the
# one H.264 flavour every WebRTC browser is required to decode.
_CODEC_OPTIONS: dict[str, dict[str, str]] = {
    "h264_nvenc": {
        "preset": "p4", "tune": "ull", "profile": "baseline", "rc": "cbr",
        "zerolatency": "1", "delay": "0",
        # Without this NVENC answers a forced I-frame with a non-IDR picture,
        # which does not let a browser that lost its reference recover.
        "forced-idr": "1",
    },
    "libx264": {
        "preset": "veryfast", "tune": "zerolatency", "profile": "baseline",
    },
}


class H264Encoder:
    """One encoder per session; frames in, Annex-B access units out."""

    def __init__(self, *, width: int, height: int, fps: int, bitrate: int,
                 gop_frames: int, codecs: list[str]) -> None:
        self.width, self.height = width, height
        self._pts = 0
        self._ctx = None
        self.name = ""
        errors = []
        for name in codecs:
            try:
                self._ctx = self._open(name, width, height, fps, bitrate, gop_frames)
                self.name = name
                break
            except Exception as exc:  # unavailable codec, no NVENC session, bad option
                errors.append(f"{name}: {exc}")
        if self._ctx is None:
            raise RuntimeError("no usable H.264 encoder (" + "; ".join(errors) + ")")
        if errors:
            log.warning("fell back to %s (%s)", self.name, "; ".join(errors))

    @staticmethod
    def _open(name: str, width: int, height: int, fps: int, bitrate: int, gop_frames: int):
        ctx = av.CodecContext.create(name, "w")
        ctx.width, ctx.height = width, height
        ctx.pix_fmt = "yuv420p"
        ctx.time_base = Fraction(1, fps)
        ctx.framerate = Fraction(fps, 1)
        ctx.bit_rate = bitrate
        ctx.gop_size = gop_frames
        ctx.max_b_frames = 0
        ctx.options = dict(_CODEC_OPTIONS.get(name, {}))
        # No global-header flag is set, so SPS/PPS are emitted in-band with
        # every keyframe -- which is what an RTP receiver joining late needs.
        ctx.open()
        return ctx

    def encode(self, i420: np.ndarray, *, force_key: bool = False) -> tuple[bytes, bool]:
        """Encode one packed I420 frame of shape (height * 3 // 2, width)."""
        frame = av.VideoFrame.from_ndarray(i420, format="yuv420p")
        frame.pts = self._pts
        self._pts += 1
        if force_key:
            frame.pict_type = PictureType.I
        packets = self._ctx.encode(frame)
        if not packets:
            return b"", False
        return b"".join(bytes(p) for p in packets), any(p.is_keyframe for p in packets)

    def close(self) -> None:
        self._ctx = None  # PyAV frees the codec context with the object
