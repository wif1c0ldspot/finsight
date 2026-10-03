"""Doctor probes configured services without making model requests."""

from io import StringIO

import httpx
import pytest
from rich.console import Console

from finsight import cli
from finsight.config import Settings


def run_doctor(monkeypatch, tmp_path, **settings):
    resolved = Settings(
        _env_file=None, corpus_dir=tmp_path / "corpus", index_dir=tmp_path / "index",
        **settings,
    )
    output = StringIO()
    monkeypatch.setattr(cli, "get_settings", lambda: resolved)
    monkeypatch.setattr(cli, "console", Console(file=output, width=200, color_system=None))
    cli.doctor()
    return output.getvalue()


def test_doctor_checks_separate_chat_and_embedding_services(monkeypatch, tmp_path):
    requests = []

    def get(url, *, timeout):
        requests.append((url, timeout))
        return httpx.Response(200 if "chat.local" in url else 404)

    monkeypatch.setattr(httpx, "get", get)
    output = run_doctor(
        monkeypatch, tmp_path, llm_base_url="http://chat.local:11434/",
        embed_base_url="http://embeddings.local:11434",
    )
    assert requests == [
        ("http://chat.local:11434/api/tags", 3.0),
        ("http://embeddings.local:11434/api/tags", 3.0),
    ]
    assert "chat endpoint" in output and "reachable (HTTP 200)" in output
    assert "embedding endpoint" in output and "failed (HTTP 404)" in output
    assert "reachable (HTTP 404)" not in output


@pytest.mark.parametrize("provider", ["openai", "anthropic", "openai_compatible"])
def test_doctor_probes_local_embeddings_with_hosted_chat(monkeypatch, tmp_path, provider):
    requests = []

    def get(url, *, timeout):
        requests.append(url)
        return httpx.Response(200)

    monkeypatch.setattr(httpx, "get", get)
    output = run_doctor(
        monkeypatch, tmp_path, llm_provider=provider,
        llm_base_url="https://hosted.example/v1", embed_base_url="http://embeddings.local",
    )
    assert requests == ["http://embeddings.local/api/tags"]
    assert f"not checked ({provider}; authenticated probe required)" in output
    assert "embedding endpoint" in output and "reachable (HTTP 200)" in output


def test_doctor_reports_hosted_embeddings_as_not_checked(monkeypatch, tmp_path):
    requests = []

    def get(url, *, timeout):
        requests.append(url)
        raise httpx.ConnectError("service offline")

    monkeypatch.setattr(httpx, "get", get)
    output = run_doctor(monkeypatch, tmp_path, embed_provider="openai")
    assert requests == ["http://localhost:11434/api/tags"]
    assert "unreachable: service offline" in output
    assert "not checked (openai; authenticated probe required)" in output


@pytest.mark.parametrize("status", [301, 401, 404, 500])
def test_doctor_marks_non_success_statuses_as_failed(monkeypatch, tmp_path, status):
    monkeypatch.setattr(httpx, "get", lambda *args, **kwargs: httpx.Response(status))
    output = run_doctor(monkeypatch, tmp_path)
    assert output.count(f"failed (HTTP {status})") == 2
    assert "reachable (HTTP" not in output


def test_doctor_reports_invalid_port_without_traceback(monkeypatch, tmp_path):
    # HTTPX rejects the port before making any network request. Both configured
    # roles must still be reported instead of aborting after the first failure.
    output = run_doctor(
        monkeypatch, tmp_path,
        llm_base_url="http://localhost:invalid",
        embed_base_url="http://localhost:invalid",
    )
    assert output.count("unreachable: Invalid port: 'invalid'") == 2
