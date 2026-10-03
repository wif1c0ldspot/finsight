"""LLM, embedding and judge factories.

Every model construction in the codebase routes through this module, so the
backend is swappable from configuration alone and exactly one place knows about
provider-specific constructor arguments.

Hosted providers are optional dependencies, so the base install stays offline and
Ollama-only. Install extra backends with::

    uv sync --extra openai      # or --extra anthropic / --extra all
"""

from typing import Any, cast

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel

from finsight.config import LLMProvider, Settings


class ProviderNotInstalledError(RuntimeError):
    """Raised when a configured provider's optional dependency is missing."""


def _missing_extra(provider: str, package: str, extra: str) -> ProviderNotInstalledError:
    return ProviderNotInstalledError(
        f"Provider {provider!r} requires {package}, which is not installed. "
        f"Install it with: uv sync --extra {extra}"
    )


def build_chat_model(
    provider: LLMProvider,
    model: str,
    base_url: str | None,
    api_key: str | None,
    *,
    temperature: float,
    timeout_s: float,
    max_retries: int,
    max_output_tokens: int | None = None,
) -> BaseChatModel:
    """Construct a chat model for any supported provider.

    A timeout and retry budget are set on every backend. The original
    implementation set neither, so a stalled or unstarted model service surfaced
    as an unhandled traceback instead of a bounded, legible failure.
    """
    if provider not in ("ollama", "openai", "anthropic", "openai_compatible"):
        raise ValueError(
            f"Unknown provider {provider!r}; expected one of "
            "'ollama', 'openai', 'anthropic', 'openai_compatible'."
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            # ChatOllama takes client options via client_kwargs, not timeout=.
            client_kwargs={"timeout": timeout_s},
            num_predict=max_output_tokens,
        )

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise _missing_extra("anthropic", "langchain-anthropic", "anthropic") from exc

        anthropic_kwargs: dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "timeout": timeout_s,
            "max_retries": max_retries,
        }
        if max_output_tokens is not None:
            anthropic_kwargs["max_tokens"] = max_output_tokens
        if base_url:
            anthropic_kwargs["base_url"] = base_url
        if api_key:
            anthropic_kwargs["api_key"] = api_key
        return cast(BaseChatModel, ChatAnthropic(**anthropic_kwargs))

    # ``openai`` and ``openai_compatible`` share one client; the compatible case
    # only differs by requiring an explicit base_url (enforced in Settings).
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise _missing_extra(provider, "langchain-openai", "openai") from exc

    openai_kwargs: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "timeout": timeout_s,
        "max_retries": max_retries,
    }
    if max_output_tokens is not None:
        openai_kwargs["max_completion_tokens"] = max_output_tokens
    if base_url:
        openai_kwargs["base_url"] = base_url
    if api_key:
        openai_kwargs["api_key"] = api_key
    return cast(BaseChatModel, ChatOpenAI(**openai_kwargs))


def build_llm(settings: Settings) -> BaseChatModel:
    """Chat model for the verify / reformulate / answer nodes."""
    return build_chat_model(
        settings.llm_provider,
        settings.llm_model,
        settings.resolved_llm_base_url(),
        settings.llm_api_key,
        temperature=settings.llm_temperature,
        timeout_s=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        max_output_tokens=settings.llm_max_output_tokens,
    )


def build_judge(settings: Settings) -> BaseChatModel:
    """Chat model for LLM-as-judge scoring.

    Point ``judge_model`` at a different family from ``llm_model``. Judging with
    the answering model is self-grading and inflates faithfulness.
    """
    provider, model, base_url, api_key = settings.resolved_judge()
    return build_chat_model(
        provider,
        model,
        base_url,
        api_key,
        temperature=0.0,
        timeout_s=settings.llm_timeout_s,
        max_retries=settings.llm_max_retries,
        max_output_tokens=settings.llm_max_output_tokens,
    )


def build_embeddings(settings: Settings) -> Embeddings:
    """Embedding model used to vectorize chunks and queries.

    Ingestion and retrieval both call this factory, so they cannot disagree about
    which embedding model produced the stored vectors.
    """
    base_url = settings.resolved_embed_base_url()

    if settings.embed_provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=settings.embed_model,
            base_url=base_url,
            # Ollama has no automatic retries; a failed request returns after
            # one bounded attempt. Hosted clients below retry within a budget.
            client_kwargs={"timeout": settings.embed_timeout_s},
        )

    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise _missing_extra(settings.embed_provider, "langchain-openai", "openai") from exc

    embed_kwargs: dict[str, Any] = {
        "model": settings.embed_model,
        "request_timeout": settings.embed_timeout_s,
        "max_retries": settings.embed_max_retries,
    }
    if base_url:
        embed_kwargs["base_url"] = base_url
    if settings.embed_api_key:
        embed_kwargs["api_key"] = settings.embed_api_key
    return cast(Embeddings, OpenAIEmbeddings(**embed_kwargs))
