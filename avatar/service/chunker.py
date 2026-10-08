"""Cuts the edge's continuous audio timeline into model-sized chunks.

The motion model renders ``step`` samples of video at a time but has to see
``window`` samples to do it (the chunk itself plus lookahead), so consecutive
chunks overlap. The edge sends small frames whenever its playout clock ticks;
this turns them into exactly the windows the model wants and keeps track of
where on the session timeline each one starts.

Pure numpy and no I/O, so the timeline rules are testable without a GPU.
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16_000
_INT16_SCALE = 32768.0


class ChunkAssembler:
    def __init__(self, *, window: int, step: int, max_backlog_steps: int = 2,
                 max_gap: int = SAMPLE_RATE) -> None:
        if step <= 0 or window < step:
            raise ValueError(f"need 0 < step <= window, got step={step} window={window}")
        self.window = window
        self.step = step
        self.max_backlog_steps = max(0, max_backlog_steps)
        self.max_gap = max(0, max_gap)
        self._speech = np.zeros(0, dtype=np.int16)
        self._listen = np.zeros(0, dtype=np.int16)
        self._base: int | None = None  # timeline position of _speech[0]
        self.dropped_chunks = 0
        self.resets = 0

    def push(self, speech_pcm16: bytes, listen_pcm16: bytes, pts: int) -> None:
        if len(speech_pcm16) != len(listen_pcm16) or len(speech_pcm16) % 2:
            raise ValueError(
                f"speech and listen must be equal-length PCM16 "
                f"(got {len(speech_pcm16)} and {len(listen_pcm16)} bytes)")
        speech = np.frombuffer(speech_pcm16, dtype="<i2")
        listen = np.frombuffer(listen_pcm16, dtype="<i2")
        if self._base is None:
            self._base = pts
        gap = pts - (self._base + len(self._speech))
        if gap < 0 or gap > self.max_gap:
            # The edge restarted its timeline (or lost more audio than is worth
            # papering over). Nothing buffered still lines up with it.
            self._speech = np.zeros(0, dtype=np.int16)
            self._listen = np.zeros(0, dtype=np.int16)
            self._base = pts
            self.resets += 1
            gap = 0
        if gap:
            # The edge drops frames rather than block its playout clock when the
            # stream backs up. Hold the timeline with silence so every later
            # frame still lands on the sample it was stamped with.
            pad = np.zeros(gap, dtype=np.int16)
            speech = np.concatenate([pad, speech])
            listen = np.concatenate([pad, listen])
        self._speech = np.concatenate([self._speech, speech])
        self._listen = np.concatenate([self._listen, listen])

    def flush_speech(self) -> None:
        """Barge-in: the buffered agent audio will never be heard."""
        self._speech = np.zeros_like(self._speech)

    def pop(self) -> tuple[int, np.ndarray, np.ndarray] | None:
        """Next (pts, speech, listen) window as float32 in [-1, 1], or None."""
        if self._base is None or len(self._speech) < self.window:
            return None
        backlog = (len(self._speech) - self.window) // self.step
        if backlog > self.max_backlog_steps:
            # Rendering fell behind real time. Video that is already late is
            # worthless to a live call, so skip to the newest chunk instead of
            # drifting further and further behind the audio.
            self._advance(backlog)
            self.dropped_chunks += backlog
        pts = self._base
        speech = self._speech[:self.window].astype(np.float32) / _INT16_SCALE
        listen = self._listen[:self.window].astype(np.float32) / _INT16_SCALE
        self._advance(1)
        return pts, speech, listen

    def _advance(self, steps: int) -> None:
        n = steps * self.step
        self._speech = self._speech[n:]
        self._listen = self._listen[n:]
        self._base += n
