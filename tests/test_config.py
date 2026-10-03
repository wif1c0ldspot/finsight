"""Tests for provider configuration.

The agent was Ollama-only, with the base URL, model and collection name duplicated
across modules. These tests pin the resolution rules that replace that.
"""

import pytest

from finsight.config import DEFAULT_BASE_URLS, Settings
from finsight.llm import build_chat_model


def test_defaults_are_offline_ollama():
    settings = Settings()
    assert settings.llm_provider == "ollama"
    assert settings.embed_provider == "ollama"
    assert settings.resolved_llm_base_url() == "http://localhost:11434"
    assert settings.resolved_embed_base_url() == "http://localhost:11434"


def test_explicit_base_url_wins_over_the_provider_default():
    settings = Settings(llm_provider="openai", llm_base_url="https://proxy.internal/v1")
    assert settings.resolved_llm_base_url() == "https://proxy.internal/v1"


def test_hosted_providers_use_their_documented_defaults():
    assert Settings(llm_provider="openai").resolved_llm_base_url() == DEFAULT_BASE_URLS["openai"]
    assert Settings(llm_provider="anthropic").resolved_llm_base_url() == (
        DEFAULT_BASE_URLS["anthropic"]
    )


def test_legacy_ollama_base_url_is_still_honoured():
    """Existing FINSIGHT_OLLAMA_BASE_URL configuration must keep working."""
    settings = Settings(ollama_base_url="http://box.local:11434")
    assert settings.resolved_llm_base_url() == "http://box.local:11434"
    assert settings.resolved_embed_base_url() == "http://box.local:11434"


def test_embeddings_and_chat_can_use_different_providers():
    settings = Settings(
        llm_provider="anthropic",
        llm_model="claude-sonnet-4-5",
        embed_provider="openai",
        embed_model="text-embedding-3-small",
    )
    assert settings.resolved_llm_base_url() == DEFAULT_BASE_URLS["anthropic"]
    assert settings.resolved_embed_base_url() == DEFAULT_BASE_URLS["openai"]


def test_compatible_provider_requires_an_explicit_base_url():
    with pytest.raises(ValueError, match="llm_base_url is required"):
        Settings(llm_provider="openai_compatible")


def test_compatible_embed_provider_requires_an_explicit_base_url():
    with pytest.raises(ValueError, match="embed_base_url is required"):
        Settings(embed_provider="openai_compatible")


def test_compatible_provider_accepts_an_explicit_base_url():
    settings = Settings(
        llm_provider="openai_compatible",
        llm_base_url="http://localhost:8000/v1",
        llm_model="Qwen/Qwen3-8B",
    )
    assert settings.resolved_llm_base_url() == "http://localhost:8000/v1"


# --- judge -----------------------------------------------------------------


def test_judge_defaults_to_the_answering_model_and_says_so():
    settings = Settings(llm_model="qwen3:8b")
    assert settings.judge_is_same_model is True
    provider, model, _base_url, _key = settings.resolved_judge()
    assert provider == "ollama"
    assert model == "qwen3:8b"


def test_a_separate_judge_clears_the_self_grading_flag():
    settings = Settings(llm_model="qwen3:8b", judge_model="llama3.3:70b")
    assert settings.judge_is_same_model is False
    _provider, model, _base_url, _key = settings.resolved_judge()
    assert model == "llama3.3:70b"


def test_judge_can_use_a_different_provider_entirely():
    settings = Settings(
        llm_provider="ollama",
        judge_provider="openai",
        judge_model="gpt-4o-mini",
        judge_api_key="sk-test",
    )
    provider, model, base_url, api_key = settings.resolved_judge()
    assert provider == "openai"
    assert model == "gpt-4o-mini"
    assert base_url == DEFAULT_BASE_URLS["openai"]
    assert api_key == "sk-test"


# --- guard config ----------------------------------------------------------


def test_collection_name_is_configurable():
    assert Settings().collection_name == "finsight"
    assert Settings(collection_name="other").collection_name == "other"


def test_grounding_and_mcp_defaults_are_enabled():
    settings = Settings()
    assert settings.enforce_grounding is True
    assert settings.use_mcp_tools is True


def test_context_budget_is_configurable():
    assert Settings().context_max_chars == 12_000
    assert Settings(context_max_chars=500).context_max_chars == 500


def test_timeouts_and_retries_have_defaults():
    """The original llm factory set neither, so failures were unbounded."""
    settings = Settings()
    assert settings.llm_timeout_s > 0
    assert settings.llm_max_retries >= 0


# --- factory ---------------------------------------------------------------


@pytest.mark.parametrize(
    "provider,type_name,module",
    [("ollama", "ChatOllama", "langchain_ollama"),
     ("openai", "ChatOpenAI", "langchain_openai"),
     ("anthropic", "ChatAnthropic", "langchain_anthropic")],
)
def test_build_chat_model_returns_a_model_for_each_provider(provider, type_name, module):
    """Each provider must construct its own client type.

    A dummy key is supplied because the hosted SDKs validate credentials at
    construction. In practice the key comes from the environment
    (``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY``) when not set in Settings.
    """
    pytest.importorskip(module)
    model = build_chat_model(
        provider,  # type: ignore[arg-type]
        "some-model",
        DEFAULT_BASE_URLS[provider],
        "sk-test-key",
        temperature=0.0,
        timeout_s=5.0,
        max_retries=1,
    )
    assert type(model).__name__ == type_name


def test_openai_compatible_provider_builds_an_openai_client():
    pytest.importorskip("langchain_openai")
    model = build_chat_model(
        "openai_compatible",
        "Qwen/Qwen3-8B",
        "http://localhost:8000/v1",
        "sk-test-key",
        temperature=0.0,
        timeout_s=5.0,
        max_retries=1,
    )
    assert type(model).__name__ == "ChatOpenAI"


def test_build_chat_model_rejects_an_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        build_chat_model(
            "mystery",  # type: ignore[arg-type]
            "m",
            None,
            None,
            temperature=0.0,
            timeout_s=5.0,
            max_retries=1,
        )


@pytest.mark.parametrize("provider", ["ollama", "openai", "openai_compatible", "anthropic"])
def test_default_judge_inherits_custom_answering_endpoint(provider):
    settings = Settings(
        llm_provider=provider,
        llm_base_url="http://localhost:8000/v1",
        llm_api_key="local-key",
    )
    assert settings.resolved_judge()[2:] == ("http://localhost:8000/v1", "local-key")


def test_explicit_judge_endpoint_wins():
    settings = Settings(llm_base_url="http://answer:11434", judge_base_url="http://judge:11434")
    assert settings.resolved_judge()[2] == "http://judge:11434"


def test_separate_compatible_judge_requires_endpoint():
    with pytest.raises(ValueError, match="judge_base_url is required"):
        Settings(judge_provider="openai_compatible")


def test_separate_judge_does_not_inherit_answering_credentials():
    settings = Settings(
        llm_provider="openai", llm_api_key="answer-key", judge_provider="anthropic"
    )
    assert settings.resolved_judge()[2:] == (DEFAULT_BASE_URLS["anthropic"], None)


@pytest.mark.parametrize("value", [0, -1, float("inf"), float("nan")])
def test_embedding_timeout_requires_a_finite_positive_value(value):
    with pytest.raises(ValueError):
        Settings(embed_timeout_s=value)


def test_embedding_retry_budget_cannot_be_negative():
    with pytest.raises(ValueError):
        Settings(embed_max_retries=-1)


@pytest.mark.parametrize("field,value", [
    ("llm_timeout_s", "0"), ("llm_timeout_s", "-1"),
    ("llm_timeout_s", "inf"), ("llm_timeout_s", "nan"),
    ("embed_timeout_s", "inf"), ("llm_temperature", "nan"),
    ("llm_max_retries", "-1"), ("embed_max_retries", "-1"),
    ("max_retrieval_attempts", "-1"),
    ("chunk_size", "0"), ("chunk_overlap", "-1"),
    ("retrieval_top_k", "0"), ("retrieval_candidates", "-1"),
    ("context_max_chars", "0"), ("retrieval_top_k", "1.5"),
])
def test_invalid_runtime_environment_fails_at_configuration(monkeypatch, field, value):
    monkeypatch.setenv(f"FINSIGHT_{field.upper()}", value)
    with pytest.raises(ValueError):
        Settings(_env_file=None)


@pytest.mark.parametrize("overlap", [600, 601])
def test_overlap_must_leave_room_for_progress(overlap):
    with pytest.raises(ValueError, match="chunk_overlap must be smaller"):
        Settings(chunk_size=600, chunk_overlap=overlap)


def test_zero_retry_budgets_and_overlap_are_supported():
    settings = Settings(llm_max_retries=0, embed_max_retries=0,
                        max_retrieval_attempts=0, chunk_overlap=0)
    assert settings.max_retrieval_attempts == 0


@pytest.mark.parametrize("kwargs", [
    {"judge_provider": "openai"},
    {"judge_base_url": "http://other:11434"},
    {"judge_model": "other-model"},
])
def test_self_grading_caveat_requires_same_resolved_identity(kwargs):
    assert not Settings(**kwargs).judge_is_same_model


def test_self_grading_uses_resolved_endpoint_and_ignores_trailing_slash():
    settings = Settings(llm_base_url="http://answer:11434/",
                        judge_base_url="http://answer:11434")
    assert settings.judge_is_same_model


@pytest.mark.parametrize("field", ["retrieval_top_k", "max_retrieval_attempts", "llm_max_retries"])
def test_boolean_counts_are_not_integers(field):
    with pytest.raises(ValueError, match="not booleans"):
        Settings(**{field: True})
