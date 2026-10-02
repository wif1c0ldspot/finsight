"""Offline regression coverage for provider construction and stdio contracts."""

import builtins

import pytest

from finsight.config import Settings
from finsight.llm import ProviderNotInstalledError, build_chat_model, build_embeddings
from finsight.mcp.client import MCPToolError, _server_params, call_tool


def test_ollama_embeddings_have_bounded_sync_and_async_timeouts():
    embeddings = build_embeddings(Settings(embed_timeout_s=2.5))
    assert embeddings._client._client.timeout.read == 2.5
    assert embeddings._async_client._client.timeout.read == 2.5


def test_hosted_embeddings_have_timeout_and_retry_budget():
    pytest.importorskip("langchain_openai")
    embeddings = build_embeddings(Settings(
        embed_provider="openai", embed_api_key="test-key",
        embed_timeout_s=2.5, embed_max_retries=2,
    ))
    assert embeddings.request_timeout == 2.5
    assert embeddings.max_retries == 2


@pytest.mark.parametrize("provider,module", [
    ("openai", "langchain_openai"), ("anthropic", "langchain_anthropic"),
])
def test_missing_hosted_extra_has_actionable_error(monkeypatch, provider, module):
    original_import = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == module:
            raise ImportError("simulated base installation")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(ProviderNotInstalledError, match="Install it with: uv sync --extra"):
        build_chat_model(provider, "model", None, "test-key",
                         temperature=0, timeout_s=1, max_retries=0)


def test_mcp_subprocess_forwards_only_application_and_provider_settings(monkeypatch):
    monkeypatch.setenv("FINSIGHT_LLM_BASE_URL", "http://local:8000")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic")
    monkeypatch.setenv("UNRELATED_SECRET", "do-not-forward")
    env = _server_params().env
    assert env["FINSIGHT_LLM_BASE_URL"] == "http://local:8000"
    assert env["OPENAI_API_KEY"] == "test-openai"
    assert env["ANTHROPIC_API_KEY"] == "test-anthropic"
    assert "UNRELATED_SECRET" not in env


def test_stdio_uses_configured_corpus_and_reports_tool_errors(monkeypatch, tmp_path):
    (tmp_path / "unique.md").write_text("# Unique corpus\n\nOnly this document.")
    monkeypatch.setenv("FINSIGHT_CORPUS_DIR", str(tmp_path))
    documents = call_tool("list_documents")
    assert [document["doc_id"] for document in documents] == ["unique"]
    with pytest.raises(MCPToolError, match="Unknown document") as error:
        call_tool("get_document", doc_id="missing")
    assert error.value.tool == "get_document"
