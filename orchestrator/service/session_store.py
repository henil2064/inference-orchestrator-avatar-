"""The sole persistent store for the orchestrator's logical session state.

Live media and backend objects deliberately stay out of Redis. Redis holds the
data useful after a reconnect/restart: session metadata, completed turns, and a
bounded event feed for a future client WebSocket.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Awaitable, TypeVar

import redis.asyncio as redis

_T = TypeVar("_T")


class SessionStore:
    def __init__(self, client: redis.Redis, active_ttl_s: int, closed_ttl_s: int,
                 event_maxlen: int, *, op_timeout_s: float = 3.0,
                 turn_retention: int = 50):
        self.client = client
        self.active_ttl_s = active_ttl_s
        self.closed_ttl_s = closed_ttl_s
        self.event_maxlen = event_maxlen
        # Every Redis round trip is wrapped in this bound so a slow/wedged
        # Redis (not just connection-refused) can never hang a live call --
        # the caller's own except-Exception handling (server.py's _store())
        # already treats any failure, including this timeout, as best-effort.
        self.op_timeout_s = op_timeout_s
        # How many completed turns record_turn() retains per session, trimmed
        # atomically in the same pipeline as the append. Deliberately decoupled
        # from LLM_HISTORY_MAX_TURNS (the orchestrator's much smaller prompt
        # window) -- this bounds Redis storage, not prompt context.
        self.turn_retention = turn_retention

    async def _bounded(self, awaitable: Awaitable[_T]) -> _T:
        return await asyncio.wait_for(awaitable, timeout=self.op_timeout_s)

    @classmethod
    async def connect(cls, url: str, active_ttl_s: int, closed_ttl_s: int,
                      event_maxlen: int, *, pool_size: int = 64,
                      op_timeout_s: float = 3.0, connect_timeout_s: float = 3.0,
                      turn_retention: int = 50) -> "SessionStore":
        # A sized pool bounds worst-case connection fan-out at high session
        # counts; socket timeouts bound a single slow op instead of a default
        # of "wait forever" (redis-py's default is no socket timeout at all).
        client = redis.from_url(url, decode_responses=True, max_connections=pool_size,
                                socket_timeout=op_timeout_s, socket_connect_timeout=connect_timeout_s,
                                health_check_interval=30)
        await asyncio.wait_for(client.ping(), timeout=connect_timeout_s)
        return cls(client, active_ttl_s, closed_ttl_s, event_maxlen,
                   op_timeout_s=op_timeout_s, turn_retention=turn_retention)

    @staticmethod
    def _keys(session_id: str) -> tuple[str, str, str, str]:
        base = f"session:{session_id}"
        return f"{base}:meta", f"{base}:turns", f"{base}:events", f"{base}:control"

    async def open(self, session_id: str, started_ts_ms: int) -> None:
        meta, _, events, control = self._keys(session_id)
        now_ms = int(time.time() * 1000)
        async with self.client.pipeline(transaction=True) as pipe:
            pipe.hset(meta, mapping={
                "session_id": session_id,
                "status": "active",
                "started_ts_ms": started_ts_ms,
                "opened_at_ms": now_ms,
            })
            pipe.hsetnx(control, "active_turn", 0)
            pipe.hsetnx(control, "cancel_generation", 0)
            pipe.xadd(events, {"type": "session_open", "ts_ms": started_ts_ms},
                      maxlen=self.event_maxlen, approximate=True)
            for key in self._keys(session_id):
                pipe.expire(key, self.active_ttl_s)
            await self._bounded(pipe.execute())

    async def event(self, session_id: str, event_type: str, **fields: Any) -> None:
        _, _, events, _ = self._keys(session_id)
        values = {"type": event_type, "at_ms": int(time.time() * 1000)}
        values.update({key: str(value) for key, value in fields.items() if value is not None})
        async with self.client.pipeline(transaction=True) as pipe:
            pipe.xadd(events, values, maxlen=self.event_maxlen, approximate=True)
            for key in self._keys(session_id):
                pipe.expire(key, self.active_ttl_s)
            await self._bounded(pipe.execute())

    async def record_turn(self, session_id: str, user_text: str, assistant_text: str | None,
                          stt_proc_ms: int, llm_proc_ms: int | None,
                          interrupted: bool = False, delivery: str = "unknown") -> None:
        _, turns, events, control = self._keys(session_id)
        record = json.dumps({
            "turn": await self._bounded(self.client.hincrby(control, "active_turn", 1)),
            "completed_at_ms": int(time.time() * 1000),
            "user_text": user_text,
            # On barge-in this is the prefix actually spoken before the cut, so
            # multi-turn history reflects what the user really heard.
            "assistant_text": assistant_text,
            "interrupted": interrupted,
            # What we actually know happened to assistant_text: "queued" (audio
            # frames/text were handed to the edge relay -- never "sent"/"heard",
            # since there is no playback ack), "failed" (attempted, delivered
            # nothing), or "none" (nothing was attempted, e.g. LLM failure).
            "delivery": delivery,
            "stt_proc_ms": stt_proc_ms,
            "llm_proc_ms": llm_proc_ms,
        }, ensure_ascii=False)
        async with self.client.pipeline(transaction=True) as pipe:
            pipe.rpush(turns, record)
            # Atomic with the append: bounds Redis storage per session
            # regardless of how long the call runs (previously unbounded).
            pipe.ltrim(turns, -self.turn_retention, -1)
            pipe.xadd(events, {"type": "turn_complete", "user_text": user_text,
                               "assistant_text": assistant_text or "",
                               "interrupted": "1" if interrupted else "0"},
                      maxlen=self.event_maxlen, approximate=True)
            for key in self._keys(session_id):
                pipe.expire(key, self.active_ttl_s)
            await self._bounded(pipe.execute())

    async def recent_turns(self, session_id: str, limit: int) -> list[dict[str, Any]]:
        """Return up to ``limit`` newest completed turns in chronological order.

        The Redis list is already chronological, so a negative LRANGE bounds the
        read without loading the full session. Malformed legacy rows are ignored;
        every valid row remains the canonical user/assistant pair written by
        record_turn().
        """
        if limit <= 0:
            return []
        _, turns, _, _ = self._keys(session_id)
        rows = await self._bounded(self.client.lrange(turns, -limit, -1))
        result = []
        for row in rows:
            try:
                turn = json.loads(row)
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(turn, dict) and isinstance(turn.get("user_text"), str):
                result.append(turn)
        return result

    async def close(self, session_id: str) -> None:
        meta, _, events, _ = self._keys(session_id)
        now_ms = int(time.time() * 1000)
        async with self.client.pipeline(transaction=True) as pipe:
            pipe.hset(meta, mapping={"status": "closed", "closed_at_ms": now_ms})
            pipe.xadd(events, {"type": "session_close", "ts_ms": now_ms},
                      maxlen=self.event_maxlen, approximate=True)
            for key in self._keys(session_id):
                pipe.expire(key, self.closed_ttl_s)
            await self._bounded(pipe.execute())

    async def aclose(self) -> None:
        try:
            await asyncio.wait_for(self.client.aclose(), timeout=self.op_timeout_s)
        except asyncio.TimeoutError:
            pass
