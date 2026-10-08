"""Unit checks for local, dependency-free harness helpers."""

import tempfile
import unittest
from pathlib import Path

from stream_wav import signal_url, wav_files


class StreamWavHelpersTest(unittest.TestCase):
    def test_edge_url_becomes_signal_url(self):
        self.assertEqual(signal_url("https://127.0.0.1:8443"), "wss://127.0.0.1:8443/ws")
        self.assertEqual(signal_url("http://edge.local:8080/base"), "ws://edge.local:8080/ws")

    def test_directory_inputs_are_sorted_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            b = root / "b.wav"
            a = root / "a.wav"
            a.touch()
            b.touch()
            self.assertEqual(wav_files([str(a)], str(root)), [a.resolve(), b.resolve()])

    def test_missing_input_is_rejected(self):
        with self.assertRaises(FileNotFoundError):
            wav_files(["does-not-exist.wav"], None)


class TurnCompletionTest(unittest.IsolatedAsyncioTestCase):
    """The harness must take the turn's end from the edge's own signal, not
    infer it from a gap in inbound audio. Under load an inter-clause pause
    longer than the idle window reads as 'drained', ending the turn early and
    understating playout exactly when the stress numbers matter most."""

    async def test_completion_reports_the_turn_reason(self):
        from stream_wav import TurnCompletion
        tc = TurnCompletion()
        tc.record({"turn_id": 1, "reason": "interrupted"})
        got = await tc.wait_past(0, timeout_s=1)
        self.assertEqual(got, {"turn_id": 1, "reason": "interrupted"})

    async def test_missing_reason_defaults_to_complete(self):
        from stream_wav import TurnCompletion
        tc = TurnCompletion()
        tc.record({"turn_id": 2})
        self.assertEqual((await tc.wait_past(0, timeout_s=1))["reason"], "complete")

    async def test_each_turn_consumes_its_own_completion(self):
        """Turn N must not be satisfied by turn N-1's signal."""
        from stream_wav import TurnCompletion
        tc = TurnCompletion()
        tc.record({"turn_id": 1, "reason": "complete"})
        tc.record({"turn_id": 2, "reason": "error"})
        self.assertEqual((await tc.wait_past(0, timeout_s=1))["turn_id"], 1)
        self.assertEqual((await tc.wait_past(1, timeout_s=1))["turn_id"], 2)

    async def test_timeout_returns_none_so_the_caller_can_fall_back(self):
        from stream_wav import TurnCompletion
        tc = TurnCompletion()
        self.assertIsNone(await tc.wait_past(0, timeout_s=0.01))

    async def test_a_late_signal_still_lands(self):
        import asyncio
        from stream_wav import TurnCompletion
        tc = TurnCompletion()

        async def emit_late():
            await asyncio.sleep(0.02)
            tc.record({"turn_id": 7, "reason": "complete"})

        asyncio.create_task(emit_late())
        got = await tc.wait_past(0, timeout_s=2)
        self.assertEqual(got["turn_id"], 7)


class AgentAudioSnapshotTest(unittest.TestCase):
    """A turn whose end the edge reported needs no silence inference."""

    def test_no_audio_turn_reports_zero(self):
        from stream_wav import AgentAudio
        audio = AgentAudio()
        audio.frames = 5
        self.assertEqual(audio.snapshot(5, None),
                         {"audio_frames_received": 0, "playout_wait_ms": 0})

    def test_frames_are_counted_from_the_turn_baseline(self):
        import time
        from stream_wav import AgentAudio
        audio = AgentAudio()
        audio.frames = 12
        audio.last_at = time.monotonic()
        got = audio.snapshot(4, audio.last_at - 0.5)
        self.assertEqual(got["audio_frames_received"], 8)
        self.assertGreaterEqual(got["playout_wait_ms"], 500)


class LateAndDelayedDrainTest(unittest.IsolatedAsyncioTestCase):
    """The two orderings a stress run actually hits at the end of a session.

    Both are about the LAST turn, which is where the harness is most likely to
    truncate its own measurement: the turn starts late, or its audio is still
    draining when the edge reports the session finished. Cutting either short
    silently understates playout in exactly the runs that matter.
    """

    async def test_a_late_starting_final_turn_is_still_measured(self):
        """utterance_end arriving well after the turn was queued must still be
        picked up, not timed out into the silence-heuristic fallback."""
        import asyncio
        from stream_wav import TurnCompletion
        tc = TurnCompletion()

        async def late_turn():
            await asyncio.sleep(0.15)  # LLM/TTS cold start on the final turn
            tc.record({"turn_id": 9, "reason": "complete"})

        asyncio.create_task(late_turn())
        got = await tc.wait_past(0, timeout_s=5)
        self.assertIsNotNone(got, "a late final turn fell back to the heuristic")
        self.assertEqual(got["turn_id"], 9)

    async def test_drain_completing_after_session_complete_still_counts(self):
        """session_complete is a session-level signal and carries no promise
        that the last turn's audio has drained. A drain landing after it must
        still be attributed to that turn."""
        import asyncio
        from stream_wav import AgentAudio, TurnCompletion
        tc, audio = TurnCompletion(), AgentAudio()
        session_complete = asyncio.Event()
        audio.frames = 10  # baseline at the start of the final turn

        async def edge_timeline():
            session_complete.set()          # edge says the session is done...
            await asyncio.sleep(0.05)
            audio.frames = 34               # ...but the tail is still arriving
            audio.last_at = __import__("time").monotonic()
            tc.record({"turn_id": 4, "reason": "complete"})

        asyncio.create_task(edge_timeline())
        completion = await tc.wait_past(0, timeout_s=5)
        self.assertTrue(session_complete.is_set(),
                        "precondition: session_complete should already have fired")
        self.assertIsNotNone(completion)
        self.assertEqual(audio.snapshot(10, None)["audio_frames_received"], 24,
                         "frames that arrived after session_complete were dropped")

    async def test_a_turn_that_never_drains_falls_back_rather_than_hanging(self):
        """No utterance_end at all (an edge too old to send it) must return
        None promptly so the caller can use the heuristic and label it."""
        import time
        from stream_wav import TurnCompletion
        tc = TurnCompletion()
        started = time.monotonic()
        self.assertIsNone(await tc.wait_past(0, timeout_s=0.05))
        self.assertLess(time.monotonic() - started, 2.0, "fallback did not trigger promptly")
