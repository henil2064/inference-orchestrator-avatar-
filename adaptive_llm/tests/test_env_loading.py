import os

from adaptive_llm.engines.common import load_dotenv


def test_load_dotenv_preserves_hash_in_token_and_strips_inline_comment(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\nHF_TOKEN=hf_prefix#suffix\nOTHER=value # comment\nQUOTED='has # a hash'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("LLM_ENV_FILE", str(env_file))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("OTHER", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)

    load_dotenv()

    assert os.environ["HF_TOKEN"] == "hf_prefix#suffix"
    assert os.environ["OTHER"] == "value"
    assert os.environ["QUOTED"] == "has # a hash"
