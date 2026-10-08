import json

import pytest

from adaptive_llm.chat.prompt_manager import load_system_prompt


def test_loads_selected_catalog_prompt(tmp_path):
    catalog = tmp_path / "prompts.json"
    catalog.write_text(json.dumps({"prompts": [{"id": "test-v1", "text": "Test prompt"}]}))

    assert load_system_prompt({"LLM_PROMPTS_FILE": str(catalog), "LLM_PROMPT_ID": "test-v1"}) == (
        "test-v1",
        "Test prompt",
    )


def test_env_override_wins_over_catalog():
    assert load_system_prompt({"LLM_SYSTEM_PROMPT": "Temporary prompt"}) == (
        "env-override",
        "Temporary prompt",
    )


def test_unknown_prompt_id_fails_with_available_ids(tmp_path):
    catalog = tmp_path / "prompts.json"
    catalog.write_text(json.dumps({"prompts": [{"id": "test-v1", "text": "Test prompt"}]}))

    with pytest.raises(RuntimeError, match="available IDs: test-v1"):
        load_system_prompt({"LLM_PROMPTS_FILE": str(catalog), "LLM_PROMPT_ID": "missing"})
