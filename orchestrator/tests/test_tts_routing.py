"""Self-check that a completed LLM reply is rendered and framed for media-edge."""
import asyncio

from proto import asr_pb2, edge_pb2, llm_pb2, tts_pb2
from service.server import MediaBridge


class Stt:
    async def Transcribe(self, requests):
        seen = asyncio.Event()
        async def drain():
            async for request in requests:
                if (request.WhichOneof("msg") == "control"
                        and request.control.type == asr_pb2.Control.END_TURN):
                    seen.set()
        task = asyncio.create_task(drain())
        await seen.wait()
        try:
            yield asr_pb2.Transcript(session_id="tts-test", text="hello", turn_complete=True)
        finally:
            task.cancel()


class Llm:
    async def GenerateStream(self, request, timeout=None):
        yield llm_pb2.GenerateChunk(session_id=request.session_id, text_delta="spoken reply")
        yield llm_pb2.GenerateChunk(session_id=request.session_id, is_final=True)


class Tts:
    def __init__(self): self.requests = []
    async def SynthesizeStream(self, request, timeout=None):
        self.requests.append(request)
        # 1,200 bytes at 24 kHz is 25 ms, so orchestrator must frame it 960+240.
        yield tts_pb2.SynthesizeChunk(session_id=request.session_id, pcm16=b"\x01\x00" * 600,
                                      sample_rate_hz=24000,
                                      profile=tts_pb2.InferenceProfile(model="test", compute_ms=8,
                                                                         audio_seconds=.025, realtime_factor=.32))
        yield tts_pb2.SynthesizeChunk(session_id=request.session_id, sample_rate_hz=24000, is_last=True)


class Telemetry:
    def __init__(self): self.events = []
    def event(self, session_id, event_type, **fields): self.events.append((session_id, event_type, fields))
    def summary(self, *args, **kwargs): pass
    def forget(self, session_id): pass


async def events():
    yield edge_pb2.ClientEvent(control=edge_pb2.Control(session_id="tts-test", type=edge_pb2.Control.SESSION_OPEN))
    yield edge_pb2.ClientEvent(control=edge_pb2.Control(session_id="tts-test", type=edge_pb2.Control.USER_SPEECH_STARTED))
    yield edge_pb2.ClientEvent(control=edge_pb2.Control(session_id="tts-test", type=edge_pb2.Control.END_TURN,
                                                         ts_ms=1))
    yield edge_pb2.ClientEvent(control=edge_pb2.Control(session_id="tts-test", type=edge_pb2.Control.SESSION_CLOSE))
