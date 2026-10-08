"""Backward-compatible STT service entrypoint.

The implementation lives in :mod:`service.server`; this file keeps existing
commands such as ``cd stt && python server.py`` working during the refactor.
"""
import asyncio

from service.server import serve


if __name__ == "__main__":
    asyncio.run(serve())
