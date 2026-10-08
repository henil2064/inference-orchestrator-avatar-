#!/usr/bin/env python3
"""Repeatable simulated-LLM-stream -> TTS profiler.

It has no LLM dependency. It simulates token arrival from supplied reply text,
then follows the current CPU-safe orchestrator boundary: one completed reply
becomes one SynthesizeStream call. TTS device selection belongs to the server.
"""
import argparse
import asyncio
import json
import statistics
import time
import wave
import re

import grpc
from proto import tts_pb2, tts_pb2_grpc

def percentile(values, p):
    values = sorted(values)
    return values[round((len(values) - 1) * p)] if values else 0.0


async def simulate_llm_stream(text, token_chars, token_delay_ms):
    """Emit text-like token chunks without involving an LLM service."""
    started = time.perf_counter()
    first_phrase_ms = None
    first_phrase_end = re.search(r"[.!?।！？](?:\s|$)", text)
    first_phrase_chars = first_phrase_end.end() if first_phrase_end else min(token_chars, len(text))
    for offset in range(0, len(text), token_chars):
        _ = text[offset:offset + token_chars]
        if token_delay_ms:
            await asyncio.sleep(token_delay_ms / 1000)
        if first_phrase_ms is None and offset + token_chars >= first_phrase_chars:
            first_phrase_ms = (time.perf_counter() - started) * 1000
    total_ms = (time.perf_counter() - started) * 1000
    return total_ms, (first_phrase_ms if first_phrase_ms is not None else total_ms)


async def tts_reply(stub, text, sid, timeout):
    started, first_audio_ms, pcm, profile, rate = time.perf_counter(), None, bytearray(), None, 0
    request = tts_pb2.SynthesizeRequest(session_id=sid, text=text, ts_ms=int(time.time() * 1000))
    async for chunk in stub.SynthesizeStream(request, timeout=timeout):
        if chunk.is_last:
            continue
        if chunk.session_id != sid or chunk.sample_rate_hz <= 0 or len(chunk.pcm16) % 2:
            raise RuntimeError("TTS returned invalid or foreign PCM")
        if rate and rate != chunk.sample_rate_hz:
            raise RuntimeError("TTS changed sample rate in one stream")
        rate = chunk.sample_rate_hz; pcm.extend(chunk.pcm16)
        if first_audio_ms is None and chunk.pcm16:
            first_audio_ms = (time.perf_counter() - started) * 1000
        if chunk.profile.model:
            profile = chunk.profile
    if not pcm or not rate or profile is None:
        raise RuntimeError("TTS stream had no profiled PCM")
    audio_s = len(pcm) / 2 / rate
    return bytes(pcm), rate, (time.perf_counter() - started) * 1000, first_audio_ms, audio_s, profile


async def run(args):
    if not args.text:
        raise SystemExit("provide reply text to simulate LLM streaming")
    async with grpc.aio.insecure_channel(args.target) as tts_channel:
        tts = tts_pb2_grpc.TtsStub(tts_channel)
        results, last = [], None
        for i in range(args.warmup + args.runs):
            sid = f"tts-profile-{int(time.time() * 1000)}-{i}"
            llm_ms, first_phrase_ms = await simulate_llm_stream(args.text, args.token_chars, args.token_delay_ms)
            pcm, rate, wall_ms, first_audio_ms, audio_s, profile = await tts_reply(tts, args.text, sid, args.timeout)
            if args.expected_device and profile.device.lower() != args.expected_device:
                raise RuntimeError(f"expected TTS device={args.expected_device}, got {profile.device or 'unknown'}")
            last = (pcm, rate)
            if i < args.warmup:
                continue
            row = {"run": i - args.warmup + 1, "chars": len(args.text), "simulated_llm_stream_ms": round(llm_ms, 2),
                   "tts_wall_ms": round(wall_ms, 2), "tts_compute_ms": round(profile.compute_ms, 2),
                   "tts_first_audio_ms": round(first_audio_ms, 2),
                   # The current client boundary sends the completed reply to TTS.
                   # Therefore this includes the time from the first phrase becoming
                   # available until the rest of the simulated LLM stream completes.
                   "first_phrase_to_first_audio_ms": round(llm_ms - first_phrase_ms + first_audio_ms, 2),
                   "audio_seconds": round(audio_s, 3), "rtf": round(profile.compute_ms / 1000 / audio_s, 4),
                   "device": profile.device, "tts_engine": profile.engine}
            results.append(row); print(json.dumps(row), flush=True)
        if args.out:
            with wave.open(args.out, "wb") as out:
                out.setnchannels(1); out.setsampwidth(2); out.setframerate(last[1]); out.writeframes(last[0])
        summary = {}
        for key in ("simulated_llm_stream_ms", "tts_wall_ms", "tts_first_audio_ms",
                    "first_phrase_to_first_audio_ms", "tts_compute_ms", "audio_seconds", "rtf"):
            values = [float(row[key]) for row in results]
            summary[key] = {"mean": round(statistics.fmean(values), 3), "p50": round(percentile(values, .5), 3), "p95": round(percentile(values, .95), 3)}
        print(json.dumps({"runs": args.runs, "summary": summary}, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Profile simulated LLM-stream to TTS behavior")
    parser.add_argument("text", help="completed reply text whose token arrival is simulated")
    parser.add_argument("--target", default="localhost:50054", help="TTS gRPC target")
    parser.add_argument("--runs", type=int, default=5); parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--token-chars", type=int, default=4, help="simulated characters per token chunk")
    parser.add_argument("--token-delay-ms", type=float, default=20, help="delay between simulated chunks")
    parser.add_argument("--expected-device", choices=("cpu", "cuda"))
    parser.add_argument("--out", help="save final PCM response as WAV")
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0 or args.token_chars < 1 or args.token_delay_ms < 0:
        parser.error("invalid run, warmup, token size, or delay")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
