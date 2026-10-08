"""Small, consistent terminal logger for the adaptive LLM facade and workers."""

from __future__ import annotations

import logging
import os


def configure_logging() -> None:
    """Configure root logging from LOG_LEVEL, defaulting to errors only."""
    level_name = os.environ.get("LOG_LEVEL", "ERROR").upper()
    level = getattr(logging, level_name, logging.ERROR)
    if not isinstance(level, int):
        level = logging.ERROR
    logging.basicConfig(level=level, format="%(levelname)s %(name)s | %(message)s", force=True)


def startup_banner(service: str, lines: list[str]) -> None:
    """Print one concise readiness panel, independently of runtime log level."""
    if os.environ.get("STARTUP_BANNER", "1") == "0":
        return
    # Memory accounting is more useful untruncated. Keep a sane upper bound so
    # a path or malformed environment value cannot produce a giant terminal box.
    width = min(118, max(60, max(len(service), *(len(line) for line in lines)) + 2))
    content_width = width - 2
    print("\n+" + "=" * width + "+")
    print(f"|  {service:<{content_width}.{content_width}}|")
    print("+" + "-" * width + "+")
    for line in lines:
        print(f"|  {line:<{content_width}.{content_width}}|")
    print("+" + "=" * width + "+\n", flush=True)


def _enabled(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _compact(value: str) -> str:
    """Keep opt-in transcript output single-line and bounded for real terminals."""
    try:
        limit = max(64, int(os.environ.get("ADAPTIVE_LLM_LOG_MAX_CHARS", "800")))
    except ValueError:
        limit = 800
    value = " ".join(value.split())
    return value if len(value) <= limit else value[:limit] + "… [truncated]"


def exchange_event(session_id: str, prompt: str, reply: str) -> None:
    """Print an explicitly enabled, privacy-sensitive user/assistant exchange."""
    if not _enabled("ADAPTIVE_LLM_LOG_EXCHANGES"):
        return
    print(
        f'EXCHANGE adaptive-llm | session={session_id} | user="{_compact(prompt)}" '
        f'| assistant="{_compact(reply)}"',
        flush=True,
    )


def profile_event(component: str, session_id: str, details: str) -> None:
    """Print a compact opt-in inference profile without requiring LOG_LEVEL=INFO."""
    if _enabled("ADAPTIVE_LLM_PROFILE_INFERENCE"):
        print(f"PROFILE {component} | session={session_id} | {details}", flush=True)


def context_event(engine: str, session_id: str, payload, prompt_tokens: int | None = None) -> None:
    """Print the exact payload handed to the model: system prompt, trimmed
    history, and the current user query, uncompacted and untruncated (unlike
    exchange_event/profile_event, this is meant for full inspection).

    ``payload`` is either the chat-message list llama.cpp consumes or the
    already-rendered prompt string vLLM/SGLang consume after chat-template
    rendering -- either way, it is the literal thing sent to the model.
    """
    if not _enabled("ADAPTIVE_LLM_LOG_CONTEXT"):
        return
    header = f"CONTEXT {engine} | session={session_id}"
    if prompt_tokens is not None:
        header += f" | ~{prompt_tokens} tokens"
    rule = "=" * 100
    print(f"\n{rule}\n{header}\n{rule}", flush=True)
    if isinstance(payload, str):
        print(payload, flush=True)
    else:
        for index, message in enumerate(payload):
            print(f"--- [{index}] {message['role'].upper()} ---", flush=True)
            print(message["content"], flush=True)
    print(f"{rule}\n", flush=True)
