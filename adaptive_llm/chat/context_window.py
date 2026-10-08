"""Model-agnostic chat-history validation and token-budget trimming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


USER = 2       # llm.v1.ChatMessage.USER
ASSISTANT = 3  # llm.v1.ChatMessage.ASSISTANT


@dataclass
class PreparedContext:
    payload: Any
    prompt_tokens: int
    prompt_chars: int
    history_turns_used: int
    history_turns_trimmed: int
    current_truncated: bool


def request_dialogue(request) -> list[dict[str, str]]:
    """Convert a protobuf request into validated USER/ASSISTANT chat messages.

    ``text`` remains the compatibility fallback. System messages are owned by
    the worker configuration and cannot be overridden by request content.
    """
    if not request.messages:
        return [{"role": "user", "content": request.text}]

    dialogue = []
    for message in request.messages:
        if message.role == USER:
            role = "user"
        elif message.role == ASSISTANT:
            role = "assistant"
        else:
            raise ValueError("LLM request messages may only use USER or ASSISTANT roles")
        dialogue.append({"role": role, "content": message.content})

    if not dialogue or dialogue[-1]["role"] != "user":
        raise ValueError("LLM request must end with the current USER message")
    history = dialogue[:-1]
    if len(history) % 2:
        raise ValueError("LLM history must contain complete USER/ASSISTANT pairs")
    for index, message in enumerate(history):
        expected = "user" if index % 2 == 0 else "assistant"
        if message["role"] != expected:
            raise ValueError("LLM history roles must alternate USER then ASSISTANT")
    return dialogue


def trim_to_budget(
    dialogue: list[dict[str, str]],
    input_budget: int,
    render_and_count: Callable[[list[dict[str, str]]], tuple[Any, int]],
    truncate_current: Callable[[str, int], str],
) -> PreparedContext:
    """Drop oldest whole turns, then token-truncate only the current user input.

    ``render_and_count`` owns the backend's authoritative exact or conservative
    token count. The final returned count is always <= ``input_budget`` or this
    function raises.
    """
    if input_budget <= 0:
        raise ValueError("LLM input token budget must be positive")
    history = [message.copy() for message in dialogue[:-1]]
    current = dialogue[-1].copy()
    original_turns = len(history) // 2
    current_truncated = False

    def render():
        return render_and_count(history + [current])

    payload, prompt_tokens = render()
    while prompt_tokens > input_budget and history:
        del history[:2]  # preserve complete USER/ASSISTANT pairs
        payload, prompt_tokens = render()

    while prompt_tokens > input_budget and current["content"]:
        excess = prompt_tokens - input_budget
        shortened = truncate_current(current["content"], max(1, excess))
        if shortened == current["content"]:
            shortened = ""
        current["content"] = shortened
        current_truncated = True
        payload, prompt_tokens = render()

    if prompt_tokens > input_budget:
        raise ValueError(
            f"system/chat-template overhead is {prompt_tokens} tokens, above "
            f"the {input_budget}-token input budget"
        )

    used = len(history) // 2
    return PreparedContext(
        payload=payload,
        prompt_tokens=prompt_tokens,
        prompt_chars=sum(len(message["content"]) for message in history + [current]),
        history_turns_used=used,
        history_turns_trimmed=original_turns - used,
        current_truncated=current_truncated,
    )
