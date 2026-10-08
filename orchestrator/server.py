"""Backward-compatible orchestrator entrypoint.

The implementation lives in :mod:`service.server` so runtime code and tests
can be organized without breaking ``cd orchestrator && python server.py``.
"""
import asyncio

from service.server import serve


if __name__ == "__main__":
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass
