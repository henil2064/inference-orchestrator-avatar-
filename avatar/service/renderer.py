"""Renderer backends: audio chunk in, I420 frames out.

``avtr1`` is the real thing and needs the upstream AVTR-1 environment, its
TensorRT engines, and an Ampere-or-newer NVIDIA GPU (see avatar/SETUP.md).
``synthetic`` draws a test card that reacts to the audio; it exists so the
gRPC contract, the edge's video track and the audio/video sync can all be
exercised on a machine with no GPU.
"""
from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_RATE = 16_000


class Avtr1Renderer:
    engine = "avtr-1"

    def __init__(self, *, width: int, height: int, avatar_ids: list[str] | None = None,
                 portraits_dir: str | None = None) -> None:
        # Imported here, not at module top: these pull in torch and TensorRT,
        # which only exist inside the upstream pixi environment.
        from avtr1_renderer.avtr1_artifact_manager import get_artifact_manager
        from avtr1_renderer.pipeline import TRANSPARENT_BG_ID, Pipeline
        from avtr1_renderer.types import Chunk, RenderOptions

        self._chunk_cls, self._options_cls = Chunk, RenderOptions
        portraits = Path(portraits_dir) if portraits_dir else Path(
            get_artifact_manager().get_artifact_path("reference_frames"))
        if not avatar_ids:
            avatar_ids = [p.stem for p in sorted(portraits.glob("*.png"))]
        if not avatar_ids:
            raise RuntimeError(f"no avatar portraits (*.png) found in {portraits}")
        self._pipeline, self._registry = Pipeline.from_artifacts(
            avatar_ids=avatar_ids, portraits_dir=portraits, out_size=(height, width))
        mg = self._pipeline._motion_generator
        self.frames_per_chunk = mg.chunk_size
        self.frame_samples = mg.frame_len
        self.step = mg.chunk_size * mg.frame_len
        self.window = (mg.chunk_size + mg.future_size) * mg.frame_len + mg.audio_shift
        self.width, self.height = width, height
        self.avatars = sorted(self._registry)
        self.backgrounds = sorted(b for b in self._pipeline._backgrounds if b != TRANSPARENT_BG_ID)

    def render_chunk(self, avatar_id: str, background_id: str, speech: np.ndarray,
                     listen: np.ndarray, state: Any) -> tuple[Any, Iterator[np.ndarray]]:
        """Pass ``state=None`` for a session's first chunk."""
        chunk = self._chunk_cls(audio_speech=np.ascontiguousarray(speech, dtype=np.float32),
                                audio_listen=np.ascontiguousarray(listen, dtype=np.float32))
        options = self._options_cls(pixel_format="yuv_i420", bg_id=background_id)
        state, frames = self._pipeline.process_chunk(self._registry[avatar_id], chunk, state, options)
        return state, (frame.data for frame in frames)


class SyntheticRenderer:
    """Same chunk geometry as AVTR-1, no model: a bar that follows the voice."""

    engine = "synthetic"
    frames_per_chunk = 5
    frame_samples = 640
    step = 5 * 640
    window = 10 * 640 + 80
    avatars = ["synthetic"]
    backgrounds = ["plain"]

    def __init__(self, *, width: int, height: int, **_: Any) -> None:
        self.width, self.height = width, height

    def render_chunk(self, avatar_id: str, background_id: str, speech: np.ndarray,
                     listen: np.ndarray, state: Any) -> tuple[Any, Iterator[np.ndarray]]:
        frame_index = state or 0

        def frames() -> Iterator[np.ndarray]:
            h, w = self.height, self.width
            for i in range(self.frames_per_chunk):
                lo = i * self.frame_samples
                level = _rms(speech[lo:lo + self.frame_samples])
                heard = _rms(listen[lo:lo + self.frame_samples])
                i420 = np.full((h * 3 // 2, w), 128, dtype=np.uint8)
                i420[:h] = 40
                # Agent voice: a centred bar whose height is the loudness.
                bar = int(min(1.0, level * 6) * h * 0.8)
                i420[(h - bar) // 2:(h + bar) // 2, w // 3:2 * w // 3] = 220
                # User voice: a strip along the bottom.
                i420[h - h // 20:h, :int(min(1.0, heard * 6) * w)] = 150
                # A moving tick so a frozen stream is obvious at a glance.
                x = (frame_index + i) * 8 % max(1, w - 8)
                i420[:h // 20, x:x + 8] = 255
                yield i420

        return frame_index + self.frames_per_chunk, frames()


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples)))) if len(samples) else 0.0


def load_renderer(*, width: int, height: int):
    backend = os.environ.get("AVATAR_BACKEND", "avtr1").strip().lower()
    if backend == "synthetic":
        return SyntheticRenderer(width=width, height=height)
    if backend != "avtr1":
        raise ValueError(f"AVATAR_BACKEND must be avtr1 or synthetic, got {backend!r}")
    # Each portrait costs face detection and feature extraction at startup, so
    # only the default is loaded unless more are asked for.
    wanted = os.environ.get("AVATAR_IDS", "").strip()
    if wanted.lower() == "all":
        ids = None
    else:
        ids = [s.strip() for s in wanted.split(",") if s.strip()] or [os.environ.get("AVATAR_DEFAULT_ID", "maria")]
    return Avtr1Renderer(width=width, height=height, avatar_ids=ids,
                         portraits_dir=os.environ.get("AVATAR_PORTRAITS_DIR") or None)
