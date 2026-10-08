"""Self-check for Redis session retention. Run: python -m tests.test_session_store"""

import asyncio
import json

from service.session_store import SessionStore


class FakePipeline:
    def __init__(self, client):
        self.client = client
        self.commands = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def _add(self, name, *args, **kwargs):
        self.commands.append((name, args, kwargs))
        return self

    def hset(self, *args, **kwargs):
        return self._add("hset", *args, **kwargs)

    def hsetnx(self, *args, **kwargs):
        return self._add("hsetnx", *args, **kwargs)

    def xadd(self, *args, **kwargs):
        return self._add("xadd", *args, **kwargs)

    def expire(self, *args, **kwargs):
        return self._add("expire", *args, **kwargs)

    def rpush(self, *args, **kwargs):
        return self._add("rpush", *args, **kwargs)

    def ltrim(self, *args, **kwargs):
        return self._add("ltrim", *args, **kwargs)

    async def execute(self):
        self.client.transactions.append(self.commands)
        return []


class FakeRedis:
    def __init__(self):
        self.transactions = []
        self.turn = 0
        self.rows = []

    def pipeline(self, transaction=True):
        assert transaction is True
        return FakePipeline(self)

    async def hincrby(self, key, field, amount):
        self.turn += amount
        return self.turn

    async def lrange(self, key, start, end):
        assert start < 0 and end == -1
        return self.rows[start:]


class SlowFakePipeline(FakePipeline):
    def __init__(self, client, delay_s):
        super().__init__(client)
        self.delay_s = delay_s

    async def execute(self):
        await asyncio.sleep(self.delay_s)
        return await super().execute()


class SlowFakeRedis(FakeRedis):
    """A Redis stand-in whose pipeline never returns within op_timeout_s --
    exercises the Phase 3 bounded-wait path (a slow backend, not just
    connection-refused, must not hang the caller)."""

    def __init__(self, delay_s):
        super().__init__()
        self.delay_s = delay_s

    def pipeline(self, transaction=True):
        return SlowFakePipeline(self, self.delay_s)


def expirations(transaction):
    return [(args[0], args[1]) for name, args, _ in transaction if name == "expire"]


async def test_record_turn_trims_atomically():
    """The ltrim must ride in the same pipeline/transaction as the rpush
    (Phase 3: bounded Redis storage per session, decoupled from
    LLM_HISTORY_MAX_TURNS), not as a separate follow-up call."""
    client = FakeRedis()
    store = SessionStore(client, active_ttl_s=60, closed_ttl_s=300, event_maxlen=10,
                         turn_retention=5)
    await store.record_turn("s1", "hi", "hello", 1, 2)
    transaction = client.transactions[-1]
    names = [name for name, _, _ in transaction]
    assert "rpush" in names and "ltrim" in names
    ltrim_args = next(args for name, args, _ in transaction if name == "ltrim")
    assert ltrim_args[1:] == (-5, -1)
    print("ok test_record_turn_trims_atomically")


async def test_slow_redis_is_bounded():
    """A Redis op that never returns must raise, not hang, within
    op_timeout_s -- the caller (server.py's _store()) already treats any
    exception as best-effort and continues the live call."""
    client = SlowFakeRedis(delay_s=1.0)
    store = SessionStore(client, active_ttl_s=60, closed_ttl_s=300, event_maxlen=10,
                         op_timeout_s=0.05)
    start = asyncio.get_event_loop().time()
    try:
        await store.open("s1", 1)
        raised = False
    except asyncio.TimeoutError:
        raised = True
    elapsed = asyncio.get_event_loop().time() - start
    assert raised, "a wedged Redis pipeline must raise TimeoutError, not hang"
    assert elapsed < 0.5, f"bounded wait took {elapsed}s, expected ~op_timeout_s"
    print("ok test_slow_redis_is_bounded")


async def test_session_lifecycle_ttls_and_history_filtering():
    client = FakeRedis()
    store = SessionStore(client, active_ttl_s=60, closed_ttl_s=300, event_maxlen=10)
    sid = "ttl-test"
    keys = set(store._keys(sid))

    await store.open(sid, 1)
    await store.event(sid, "transcript_segment", text="hello")
    await store.record_turn(sid, "hello", "hi", 10, 20)
    await store.close(sid)

    assert len(client.transactions) == 4
    for transaction in client.transactions[:3]:
        assert set(expirations(transaction)) == {(key, 60) for key in keys}
    assert set(expirations(client.transactions[3])) == {(key, 300) for key in keys}

    xadds = [(args, kwargs) for transaction in client.transactions
             for name, args, kwargs in transaction if name == "xadd"]
    assert xadds and all(kwargs["maxlen"] == 10 for _, kwargs in xadds)

    client.rows = [
        json.dumps({"user_text": "old", "assistant_text": "old reply"}),
        "not-json",
        json.dumps({"user_text": "new", "assistant_text": "new reply"}),
    ]
    assert await store.recent_turns(sid, 2) == [
        {"user_text": "new", "assistant_text": "new reply"},
    ]
    assert await store.recent_turns(sid, 0) == []
