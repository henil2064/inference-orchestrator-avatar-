"""Timeline rules of the chunk assembler: pure numpy, no GPU or codec."""
import numpy as np
import pytest

from service.chunker import ChunkAssembler

WINDOW, STEP = 6480, 3200  # AVTR-1's geometry: (5 + 5) * 640 + 80, and 5 * 640
FRAME = 320                # the edge's 20 ms tick at 16 kHz


def pcm(value: int, n: int = FRAME) -> bytes:
    return np.full(n, value, dtype="<i2").tobytes()


def feed(assembler: ChunkAssembler, frames: int, *, start: int = 0, value: int = 1000) -> None:
    for i in range(frames):
        assembler.push(pcm(value), pcm(-value), start + i * FRAME)


def test_no_chunk_until_a_full_window_is_buffered():
    a = ChunkAssembler(window=WINDOW, step=STEP)
    feed(a, WINDOW // FRAME)  # 20 frames = 6400 samples, 80 short
    assert a.pop() is None
    a.push(pcm(1000), pcm(-1000), (WINDOW // FRAME) * FRAME)
    pts, speech, listen = a.pop()
    assert pts == 0
    assert speech.dtype == np.float32 and speech.shape == (WINDOW,)
    assert speech[0] == pytest.approx(1000 / 32768) and listen[0] == pytest.approx(-1000 / 32768)


def test_consecutive_chunks_advance_by_one_step_and_overlap():
    a = ChunkAssembler(window=WINDOW, step=STEP, max_backlog_steps=10)
    for i in range(60):
        a.push(pcm(i), pcm(0), i * FRAME)  # each frame carries its own index
    first, second = a.pop(), a.pop()
    assert (first[0], second[0]) == (0, STEP)
    # The second window starts exactly where the first one's lookahead began.
    assert np.array_equal(second[1][:WINDOW - STEP], first[1][STEP:])


def test_timeline_starts_at_the_first_pts_not_zero():
    a = ChunkAssembler(window=WINDOW, step=STEP)
    feed(a, 21, start=48_000)
    assert a.pop()[0] == 48_000


def test_falling_behind_skips_to_the_newest_chunk():
    a = ChunkAssembler(window=WINDOW, step=STEP, max_backlog_steps=1)
    feed(a, 21 + 40)  # one window plus four further steps
    pts, _, _ = a.pop()
    assert pts == 4 * STEP
    assert a.dropped_chunks == 4
    assert a.pop() is None


def test_flush_silences_buffered_speech_but_keeps_listen_and_position():
    a = ChunkAssembler(window=WINDOW, step=STEP)
    feed(a, 21)
    a.flush_speech()
    pts, speech, listen = a.pop()
    assert pts == 0
    assert not speech.any()
    assert listen.all()


def test_small_gap_is_filled_with_silence_so_later_audio_keeps_its_position():
    a = ChunkAssembler(window=WINDOW, step=STEP)
    a.push(pcm(1000), pcm(1000), 0)
    feed(a, 20, start=2 * FRAME)  # the frame at pts=320 never arrived
    pts, speech, _ = a.pop()
    assert pts == 0
    assert speech[:FRAME].all() and not speech[FRAME:2 * FRAME].any() and speech[2 * FRAME:].all()
    assert a.resets == 0


@pytest.mark.parametrize("restart_pts", [0, 10_000_000])
def test_backwards_or_huge_jump_restarts_the_timeline(restart_pts):
    a = ChunkAssembler(window=WINDOW, step=STEP)
    feed(a, 10, start=100_000)
    feed(a, 21, start=restart_pts, value=2000)
    pts, speech, _ = a.pop()
    assert pts == restart_pts
    assert speech[0] == pytest.approx(2000 / 32768)
    assert a.resets == 1


def test_mismatched_tracks_are_rejected():
    a = ChunkAssembler(window=WINDOW, step=STEP)
    with pytest.raises(ValueError):
        a.push(pcm(1), pcm(1, FRAME - 1), 0)
