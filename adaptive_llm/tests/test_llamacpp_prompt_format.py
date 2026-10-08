from types import SimpleNamespace

from adaptive_llm.engines import llamacpp_worker


def test_gemma_embeds_system_prompt_in_first_user_turn(monkeypatch):
    monkeypatch.setattr(llamacpp_worker, "SYSTEM_PROMPT", "System instruction")
    model = SimpleNamespace(metadata={"general.architecture": "gemma3"})

    messages = llamacpp_worker._messages_for_model(model, [{"role": "user", "content": "Hello"}])

    assert messages == [{"role": "user", "content": "System instruction\n\nHello"}]


def test_non_gemma_keeps_a_system_role(monkeypatch):
    monkeypatch.setattr(llamacpp_worker, "SYSTEM_PROMPT", "System instruction")
    model = SimpleNamespace(metadata={"general.architecture": "llama"})

    messages = llamacpp_worker._messages_for_model(model, [{"role": "user", "content": "Hello"}])

    assert messages == [
        {"role": "system", "content": "System instruction"},
        {"role": "user", "content": "Hello"},
    ]
