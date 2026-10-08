"""Self-check for structured history validation and token-budget trimming."""

from types import SimpleNamespace

from adaptive_llm.chat.context_window import request_dialogue, trim_to_budget
from adaptive_llm.proto import llm_pb2


def request(*messages):
    return SimpleNamespace(messages=[
        SimpleNamespace(role=role, content=content) for role, content in messages
    ], text="fallback")


def render_and_count(dialogue):
    # One token per whitespace-delimited word plus fixed system/template cost.
    tokens = 5 + sum(len(message["content"].split()) for message in dialogue)
    return dialogue, tokens


def truncate_current(text, minimum_drop):
    return " ".join(text.split()[minimum_drop:])


def test_oldest_complete_turns_are_trimmed_first():
    dialogue = request_dialogue(request(
        (llm_pb2.ChatMessage.USER, "old user words"),
        (llm_pb2.ChatMessage.ASSISTANT, "old assistant words"),
        (llm_pb2.ChatMessage.USER, "new user"),
        (llm_pb2.ChatMessage.ASSISTANT, "new assistant"),
        (llm_pb2.ChatMessage.USER, "current request"),
    ))
    prepared = trim_to_budget(dialogue, 11, render_and_count, truncate_current)
    assert prepared.prompt_tokens <= 11
    assert prepared.history_turns_used == 1
    assert prepared.history_turns_trimmed == 1
    assert [message["content"] for message in prepared.payload] == [
        "new user", "new assistant", "current request",
    ]
    assert prepared.current_truncated is False


def test_current_input_is_last_resort_and_final_count_is_safe():
    dialogue = request_dialogue(request(
        (llm_pb2.ChatMessage.USER, "past"),
        (llm_pb2.ChatMessage.ASSISTANT, "reply"),
        (llm_pb2.ChatMessage.USER, "one two three four five six"),
    ))
    prepared = trim_to_budget(dialogue, 8, render_and_count, truncate_current)
    assert prepared.history_turns_used == 0
    assert prepared.current_truncated is True
    assert prepared.prompt_tokens <= 8
    assert prepared.payload[-1]["content"] == "four five six"


def test_legacy_text_fallback_and_role_validation():
    legacy = SimpleNamespace(messages=[], text="hello")
    assert request_dialogue(legacy) == [{"role": "user", "content": "hello"}]

    malformed = request((llm_pb2.ChatMessage.ASSISTANT, "not a current user"))
    try:
        request_dialogue(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError("assistant-final request must be rejected")
