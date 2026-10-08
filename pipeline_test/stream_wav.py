#!/usr/bin/env python3
"""Replay WAVs as sequential turns of one full-duplex WebRTC call.

The media edge returns TTS on an inbound WebRTC Opus track.  This client keeps
one peer connection open, sends one recording at a time, and waits until the
returned audio has drained before it sends the next recording.
"""
from __future__ import annotations

import argparse
import asyncio
from fractions import Fraction
import json
from pathlib import Path
import ssl
import sys
import time
from typing import Any
from urllib.parse import urlsplit, urlunsplit

SAMPLE_RATE = 48_000
SAMPLES_PER_FRAME = 960  # 20 ms Opus frames


def signal_url(edge_url: str) -> str:
    """Turn an edge HTTP(S) base URL into its WebSocket signaling endpoint."""
    parsed = urlsplit(edge_url)
    if parsed.scheme not in {"http", "https", "ws", "wss"} or not parsed.netloc:
        raise ValueError("edge URL must look like https://host:8443 or http://host:8443")
    scheme = {"http": "ws", "https": "wss"}.get(parsed.scheme, parsed.scheme)
    return urlunsplit((scheme, parsed.netloc, "/ws", "", ""))


def wav_files(inputs: list[str], directory: str | None) -> list[Path]:
    paths = [Path(value).expanduser().resolve() for value in inputs]
    if directory:
        paths.extend(sorted(Path(directory).expanduser().resolve().glob("*.wav")))
    unique = list(dict.fromkeys(paths))
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError("WAV file(s) not found: " + ", ".join(missing))
    if not unique:
        raise ValueError("supply one or more --wav files or --wav-dir")
    return unique


def decode_wav(path: Path) -> bytes:
    """Decode an input WAV to packed 48 kHz mono PCM16 using PyAV."""
    try:
        import av
    except ImportError as exc:
        raise SystemExit("install dependencies first: python -m pip install -r pipeline_test/requirements.txt") from exc
    chunks: list[bytes] = []
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise ValueError(f"{path} has no decodable audio stream")
            resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
            for frame in container.decode(container.streams.audio[0]):
                for converted in resampler.resample(frame):
                    chunks.append(bytes(converted.planes[0])[:converted.samples * 2])
            for converted in resampler.resample(None):
                chunks.append(bytes(converted.planes[0])[:converted.samples * 2])
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"could not decode {path}: {exc}") from exc
    pcm = b"".join(chunks)
    if not pcm:
        raise ValueError(f"{path} decoded to no audio")
    return pcm


def make_turn_track(MediaStreamTrack, AudioFrame):  # imported lazily for dependency-free tests
    class TurnTrack(MediaStreamTrack):
        kind = "audio"

        def __init__(self) -> None:
            super().__init__()
            self.active = b""
            self.done: asyncio.Future[None] | None = None
            self.timestamp = 0
            self.started_at: float | None = None

        def queue_turn(self, pcm: bytes) -> asyncio.Future[None]:
            if self.active:
                raise RuntimeError("cannot queue a recording while another is active")
            self.active = pcm
            self.done = asyncio.get_running_loop().create_future()
            return self.done

        async def recv(self):
            if self.readyState != "live":
                raise RuntimeError("input track is closed")
            self.started_at = self.started_at or time.monotonic()
            delay = self.started_at + self.timestamp / SAMPLE_RATE - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            size = SAMPLES_PER_FRAME * 2
            if self.active:
                data, self.active = self.active[:size], self.active[size:]
                data = data.ljust(size, b"\0")
                if not self.active and self.done and not self.done.done():
                    self.done.set_result(None)
            else:
                # Continuous silence keeps this single full-duplex call alive
                # between turns; edge VAD ignores it as a user utterance.
                data = b"\0" * size
            frame = AudioFrame(format="s16", layout="mono", samples=SAMPLES_PER_FRAME)
            frame.planes[0].update(data)
            frame.sample_rate, frame.pts, frame.time_base = SAMPLE_RATE, self.timestamp, Fraction(1, SAMPLE_RATE)
            self.timestamp += SAMPLES_PER_FRAME
            return frame

    return TurnTrack()


class TurnCompletion:
    """The edge's authoritative per-turn playout signal.

    The edge emits utterance_end (aliased as playback_complete for older
    clients) once a turn's audio tail has actually gone out on the wire, with
    metadata["reason"] = complete | interrupted | error. That is a real
    statement about the turn; inferring the same thing from a gap in inbound
    audio is not. Under load any inter-clause pause longer than the idle window
    reads as "drained", which ends the turn early and understates playout
    exactly when the pipeline is most stressed -- the opposite of what a stress
    test needs.

    playback_complete is deliberately ignored: the edge sends both names for
    the same event, so counting both would double every turn.
    """

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []
        self.updated = asyncio.Event()

    def record(self, message: dict[str, Any]) -> None:
        self.events.append({"turn_id": message.get("turn_id"),
                            "reason": message.get("reason") or "complete"})
        self.updated.set()

    async def wait_past(self, baseline: int, timeout_s: float) -> dict[str, Any] | None:
        """The first completion recorded after `baseline`, or None on timeout."""
        deadline = time.monotonic() + timeout_s
        while len(self.events) <= baseline:
            self.updated.clear()
            if len(self.events) > baseline:
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            try:
                await asyncio.wait_for(self.updated.wait(), remaining)
            except asyncio.TimeoutError:
                return None
        return self.events[baseline]


class AgentAudio:
    """Consumes inbound WebRTC audio and exposes its last-received timestamp."""

    def __init__(self) -> None:
        self.frames = 0
        self.last_at: float | None = None
        self.updated = asyncio.Event()
        self.tasks: list[asyncio.Task] = []

    def consume(self, track) -> None:
        async def run() -> None:
            try:
                while True:
                    await track.recv()
                    self.frames += 1
                    self.last_at = time.monotonic()
                    self.updated.set()
            except Exception:  # remote track ends when the peer is torn down
                return
        self.tasks.append(asyncio.create_task(run()))

    def snapshot(self, baseline: int, started_at: float | None) -> dict[str, int]:
        """Frames received and elapsed playout for a turn whose end the edge
        told us about, so nothing has to be inferred from a silence gap."""
        if self.frames == baseline:
            return {"audio_frames_received": 0, "playout_wait_ms": 0}
        origin = started_at or self.last_at or time.monotonic()
        return {"audio_frames_received": self.frames - baseline,
                "playout_wait_ms": round(((self.last_at or time.monotonic()) - origin) * 1000)}

    async def wait_until_drained(self, baseline: int, idle_s: float, start_timeout_s: float) -> dict[str, int]:
        """Fallback for an edge that never sends utterance_end. Infers the end
        of playout from a quiet period, which is a guess: see TurnCompletion."""
        deadline = time.monotonic() + start_timeout_s
        while self.frames == baseline and time.monotonic() < deadline:
            self.updated.clear()
            if self.frames != baseline:
                break
            try:
                await asyncio.wait_for(self.updated.wait(), deadline - time.monotonic())
            except asyncio.TimeoutError:
                break
        if self.frames == baseline:
            return {"audio_frames_received": 0, "playout_wait_ms": 0}
        started_at = self.last_at or time.monotonic()
        while True:
            remaining = idle_s - (time.monotonic() - (self.last_at or time.monotonic()))
            if remaining <= 0:
                return {"audio_frames_received": self.frames - baseline,
                        "playout_wait_ms": round((time.monotonic() - started_at) * 1000)}
            self.updated.clear()
            try:
                await asyncio.wait_for(self.updated.wait(), remaining)
            except asyncio.TimeoutError:
                pass


async def stream_all(paths: list[Path], ws_url: str, timeout_s: float, verify_tls: bool,
                     idle_s: float, audio_start_timeout_s: float) -> list[dict[str, Any]]:
    try:
        import websockets
        from aiortc import RTCPeerConnection, RTCSessionDescription
        from aiortc.mediastreams import MediaStreamTrack
        from aiortc.sdp import candidate_from_sdp, candidate_to_sdp
        from av import AudioFrame
    except ImportError as exc:
        raise SystemExit("install dependencies first: python -m pip install -r pipeline_test/requirements.txt") from exc
    tls = None
    if ws_url.startswith("wss://") and not verify_tls:
        tls = ssl.create_default_context(); tls.check_hostname = False; tls.verify_mode = ssl.CERT_NONE
    pc = RTCPeerConnection()
    input_track = make_turn_track(MediaStreamTrack, AudioFrame)
    pc.addTrack(input_track)
    channel = pc.createDataChannel("agent-events", ordered=True)
    agent_audio = AgentAudio()
    turn_completion = TurnCompletion()
    replies: list[dict[str, Any]] = []
    reply_updated, connected, channel_open = asyncio.Event(), asyncio.Event(), asyncio.Event()
    session_complete, saved_acknowledged = asyncio.Event(), asyncio.Event()
    errors: list[str] = []
    saved: dict[str, Any] | None = None

    @pc.on("connectionstatechange")
    async def on_connection_state_change() -> None:
        if pc.connectionState == "connected":
            connected.set()
        elif pc.connectionState in {"failed", "closed"}:
            errors.append(f"WebRTC connection {pc.connectionState}")
            connected.set(); channel_open.set(); reply_updated.set(); saved_acknowledged.set()

    @pc.on("track")
    def on_track(track) -> None:
        if track.kind == "audio":
            agent_audio.consume(track)

    @channel.on("open")
    def on_channel_open() -> None:
        channel_open.set()

    @channel.on("message")
    def on_agent_event(raw: str | bytes) -> None:
        try:
            message = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        except (TypeError, json.JSONDecodeError):
            return
        if message.get("type") == "agent_reply":
            replies.append(message); reply_updated.set()
        elif message.get("type") == "utterance_end":
            # Authoritative end of this turn's playout. Also unblocks the reply
            # wait below, so a turn that produced no reply at all (empty
            # transcript, rejected admission, barge-in before any token) ends
            # here instead of stalling until the whole-session timeout.
            turn_completion.record(message); reply_updated.set()
        elif message.get("type") == "session_complete":
            session_complete.set()
        elif message.get("type") == "error":
            errors.append(str(message.get("error", "unknown agent event error")))
            reply_updated.set(); saved_acknowledged.set()

    try:
        async with websockets.connect(ws_url, ssl=tls, max_size=1_000_000) as ws:
            async def send_candidate(candidate) -> None:
                if candidate is not None:
                    await ws.send(json.dumps({"type": "ice", "candidate": {
                        "candidate": candidate_to_sdp(candidate), "sdpMid": candidate.sdpMid,
                        "sdpMLineIndex": candidate.sdpMLineIndex}}))
            @pc.on("icecandidate")
            def on_icecandidate(candidate) -> None:
                asyncio.create_task(send_candidate(candidate))
            async def receive_signals() -> None:
                nonlocal saved
                async for raw in ws:
                    message = json.loads(raw)
                    kind = message.get("type")
                    if kind == "answer":
                        answer = message["answer"]
                        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
                    elif kind == "ice" and message.get("candidate"):
                        item = message["candidate"]; candidate = candidate_from_sdp(item["candidate"])
                        candidate.sdpMid, candidate.sdpMLineIndex = item.get("sdpMid"), item.get("sdpMLineIndex")
                        await pc.addIceCandidate(candidate)
                    elif kind == "saved":
                        saved = message; saved_acknowledged.set()
                    elif kind == "error":
                        errors.append(str(message.get("error", "media edge error")))
                        reply_updated.set(); saved_acknowledged.set()
            receiver = asyncio.create_task(receive_signals())
            try:
                offer = await pc.createOffer(); await pc.setLocalDescription(offer)
                await ws.send(json.dumps({"type": "offer", "offer": {"type": pc.localDescription.type, "sdp": pc.localDescription.sdp}}))
                await asyncio.wait_for(asyncio.gather(connected.wait(), channel_open.wait()), timeout=timeout_s)
                if errors: raise RuntimeError(errors[-1])
                results = []
                for path in paths:
                    pcm, reply_count, audio_frames = decode_wav(path), len(replies), agent_audio.frames
                    turn_count = len(turn_completion.events)
                    await asyncio.wait_for(input_track.queue_turn(pcm), timeout=timeout_s)
                    # Either a reply or the turn ending is progress. Waiting on
                    # the reply alone stalls a turn that legitimately produces
                    # none.
                    while len(replies) == reply_count and len(turn_completion.events) == turn_count:
                        reply_updated.clear()
                        if len(replies) != reply_count or len(turn_completion.events) != turn_count:
                            break
                        if errors: raise RuntimeError(errors[-1])
                        await asyncio.wait_for(reply_updated.wait(), timeout=timeout_s)
                    audio_started_at = agent_audio.last_at
                    completion = await turn_completion.wait_past(turn_count, timeout_s)
                    if completion is None:
                        # No utterance_end within the budget: fall back to the
                        # silence heuristic and say so, so a run that inferred
                        # its numbers is never mistaken for a measured one.
                        playout = await asyncio.wait_for(
                            agent_audio.wait_until_drained(audio_frames, idle_s, audio_start_timeout_s),
                            timeout=timeout_s)
                        playout["playout_source"] = "idle_heuristic"
                        outcome = "unknown"
                    else:
                        playout = agent_audio.snapshot(audio_frames, audio_started_at)
                        playout["playout_source"] = "utterance_end"
                        outcome = completion["reason"]
                    reply = replies[-1] if len(replies) > reply_count else {}
                    results.append({"file": str(path), "turn_outcome": outcome,
                                    "reply": {"text": reply.get("text", ""), "proc_ms": reply.get("proc_ms")},
                                    **playout})
                # Only now can the client half-close the one input stream.
                await ws.send(json.dumps({"type": "stop"}))
                await asyncio.wait_for(asyncio.gather(session_complete.wait(), saved_acknowledged.wait()), timeout=timeout_s)
                if errors: raise RuntimeError(errors[-1])
                if not saved: raise RuntimeError("media edge did not confirm session completion")
                for result in results:
                    result.update(session_id=saved.get("file"), pcm_bytes_forwarded=saved.get("bytes"), edge_error=saved.get("error"))
                return results
            finally:
                receiver.cancel(); await asyncio.gather(receiver, return_exceptions=True)
    finally:
        input_track.stop()
        for task in agent_audio.tasks: task.cancel()
        await asyncio.gather(*agent_audio.tasks, return_exceptions=True)
        await pc.close()


async def main_async(args: argparse.Namespace) -> int:
    files, ws_url = wav_files(args.wav, args.wav_dir), args.signal_url or signal_url(args.edge_url)
    try:
        print(f"streaming {len(files)} turn(s) through one full-duplex call at {ws_url}", flush=True)
        results = await stream_all(files, ws_url, args.timeout_s, args.verify_tls, args.playout_idle_s, args.audio_start_timeout_s)
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr, flush=True); return 1
    for result in results: print(json.dumps(result, ensure_ascii=False), flush=True)
    if args.output: Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    return 0 if all(not result.get("edge_error") for result in results) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", action="append", default=[], help="WAV file to stream; repeatable")
    parser.add_argument("--wav-dir", default=str(Path(__file__).with_name("recordings")), help="stream every *.wav file in this directory, alphabetically")
    parser.add_argument("--edge-url", default="https://127.0.0.1:8443", help="media-edge base URL (default: %(default)s)")
    parser.add_argument("--signal-url", help="explicit WebSocket signaling URL; overrides --edge-url")
    parser.add_argument("--timeout-s", type=float, default=90, help="maximum wait per turn or final close (default: %(default)s)")
    parser.add_argument("--playout-idle-s", type=float, default=.35, help="quiet inbound-audio period that marks playout complete (default: %(default)s)")
    parser.add_argument("--audio-start-timeout-s", type=float, default=2, help="wait for RTP after a text reply before treating it as text-only (default: %(default)s)")
    parser.add_argument("--verify-tls", action="store_true", help="verify the local edge TLS certificate")
    parser.add_argument("--output", help="write printed results to this JSON file")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(parse_args())))
