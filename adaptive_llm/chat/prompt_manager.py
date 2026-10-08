"""Versioned system-prompt catalog loading for adaptive LLM workers."""

from __future__ import annotations

import json
import os
from pathlib import Path


DEFAULT_PROMPT_ID = "voice-assistant-v1"
DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise voice assistant. Keep replies short and natural to speak aloud. "
    "The earlier conversation is completed history: assistant messages were already delivered. "
    "Never repeat, rewrite, or re-answer a historical assistant message. Respond only to the newest "
    "user message; if it is merely an acknowledgement, respond briefly and naturally."
)
CATALOG_PATH = Path(__file__).resolve().parents[1] / "prompts.json"


def load_system_prompt(environ: dict[str, str] | None = None) -> tuple[str, str]:
    """Return ``(prompt_id, text)`` selected by the environment.

    ``LLM_PROMPT_ID`` selects an entry from ``LLM_PROMPTS_FILE`` (or the bundled
    catalog). ``LLM_SYSTEM_PROMPT`` remains an explicit compatibility override
    while existing deployments migrate to the catalog.
    """
    env = os.environ if environ is None else environ
    override = env.get("LLM_SYSTEM_PROMPT")
    if override:
        return "env-override", override

    prompt_id = env.get("LLM_PROMPT_ID", DEFAULT_PROMPT_ID)
    catalog_path = Path(env.get("LLM_PROMPTS_FILE", CATALOG_PATH))
    if not catalog_path.is_absolute():
        catalog_path = Path.cwd() / catalog_path

    try:
        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"prompt catalog not found: {catalog_path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON in prompt catalog {catalog_path}: {exc.msg}") from exc

    prompts = raw.get("prompts") if isinstance(raw, dict) else None
    if not isinstance(prompts, list):
        raise RuntimeError(f"prompt catalog {catalog_path} must contain a 'prompts' list")

    matches = [entry for entry in prompts if isinstance(entry, dict) and entry.get("id") == prompt_id]
    if len(matches) != 1:
        available = ", ".join(
            str(entry.get("id")) for entry in prompts if isinstance(entry, dict) and entry.get("id")
        )
        raise RuntimeError(
            f"prompt id {prompt_id!r} was not found exactly once in {catalog_path}; "
            f"available IDs: {available or '(none)'}"
        )

    text = matches[0].get("text")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError(f"prompt {prompt_id!r} in {catalog_path} needs a non-empty string 'text'")
    return prompt_id, text
