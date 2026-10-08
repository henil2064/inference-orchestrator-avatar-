"""Avatar service entrypoint."""
import asyncio

from service.server import serve


if __name__ == "__main__":
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass
