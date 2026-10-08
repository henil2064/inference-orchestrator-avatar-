from adaptive_llm.core.config import ROOT, load_config


def test_default_model_profiles_path_tracks_repo_layout(monkeypatch):
    monkeypatch.delenv("ADAPTIVE_LLM_MODEL_PROFILES", raising=False)
    monkeypatch.delenv("ADAPTIVE_LLM_REGISTRY", raising=False)

    config = load_config()

    assert ROOT.name == "inference-orchestrator"
    assert config.model_profiles_path == ROOT / "adaptive_llm" / "model-profiles.json"
    assert config.model_profiles_path.is_file()
