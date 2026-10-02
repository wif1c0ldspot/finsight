"""Application configuration via pydantic-settings.

Every setting is overridable via a ``FINSIGHT_<FIELD>`` environment variable or a
local ``.env`` file, so the same code runs against a local Ollama stack, a hosted
OpenAI/Anthropic endpoint, or any OpenAI-compatible gateway without edits.

Provider selection
------------------
``llm_provider`` accepts ``ollama`` (default, fully offline), ``openai``,
``anthropic``, or ``openai_compatible`` for any gateway that speaks the OpenAI
chat-completions API (vLLM, LM Studio, OpenRouter, Azure-style proxies, ...).

``embed_provider`` accepts ``ollama``, ``openai`` or ``openai_compatible``.
Anthropic does not publish an embeddings endpoint, so it is deliberately not a
valid embedding provider.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["ollama", "openai", "anthropic", "openai_compatible"]
EmbedProvider = Literal["ollama", "openai", "openai_compatible"]

#: Provider default base URLs, used when no explicit base URL is configured.
DEFAULT_BASE_URLS: dict[str, str | None] = {
    "ollama": "http://localhost:11434",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "openai_compatible": None,  # must be supplied by the operator
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FINSIGHT_", env_file=".env", extra="ignore"
    )

    # --- Chat model ---
    llm_provider: LLMProvider = "ollama"
    llm_model: str = "qwen3:8b"
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_temperature: float = 0.0
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 3
    # Kept for backwards compatibility with the original Ollama-only config.
    ollama_base_url: str = "http://localhost:11434"

    # --- Embeddings ---
    embed_provider: EmbedProvider = "ollama"
    embed_model: str = "nomic-embed-text"
    embed_base_url: str | None = None
    embed_api_key: str | None = None
    embed_timeout_s: float = Field(default=60.0, gt=0, allow_inf_nan=False)
    embed_max_retries: int = Field(default=3, ge=0)

    # --- Judge (evaluation) ---
    # When unset the judge reuses the chat model, which introduces self-grading
    # bias. Set a different model (ideally a different family) for honest scores.
    judge_provider: LLMProvider | None = None
    judge_model: str | None = None
    judge_base_url: str | None = None
    judge_api_key: str | None = None

    # --- RAG knobs ---
    chunk_size: int = 600
    chunk_overlap: int = 100
    retrieval_top_k: int = 4
    retrieval_candidates: int = 8  # candidates per method before fusion
    collection_name: str = "finsight"
    # Upper bound on the rendered context block, so a large top_k cannot blow
    # past the model's window silently.
    context_max_chars: int = 12_000

    # --- Paths (relative to the project root; run commands from there) ---
    data_dir: Path = Path("data")
    corpus_dir: Path = Path("data/corpus")
    index_dir: Path = Path("data/index")
    chroma_dir: Path = Path("data/chroma")
    golden_file: Path = Path("data/golden/golden.json")

    # --- Agent ---
    max_retrieval_attempts: int = 2
    # Re-query when an answer cites nothing, instead of shipping it silently.
    enforce_grounding: bool = True

    # --- MCP ---
    # Route the agent's retrieval through the MCP tool layer. The tools are the
    # same functions the standalone server exposes, so the agent and any external
    # MCP host share one retrieval contract.
    use_mcp_tools: bool = True

    @model_validator(mode="after")
    def _check_provider_config(self) -> "Settings":
        if self.llm_provider == "openai_compatible" and not self.llm_base_url:
            raise ValueError(
                "llm_base_url is required when llm_provider='openai_compatible'"
            )
        if self.embed_provider == "openai_compatible" and not self.embed_base_url:
            raise ValueError(
                "embed_base_url is required when embed_provider='openai_compatible'"
            )
        provider, _, base_url, _ = self.resolved_judge()
        if provider == "openai_compatible" and not base_url:
            raise ValueError(
                "judge_base_url is required when judge_provider='openai_compatible' "
                "and the answering endpoint cannot be inherited"
            )
        return self

    # --- Resolvers -------------------------------------------------------

    def resolved_llm_base_url(self) -> str | None:
        if self.llm_base_url:
            return self.llm_base_url
        if self.llm_provider == "ollama":
            return self.ollama_base_url
        return DEFAULT_BASE_URLS[self.llm_provider]

    def resolved_embed_base_url(self) -> str | None:
        if self.embed_base_url:
            return self.embed_base_url
        if self.embed_provider == "ollama":
            return self.ollama_base_url
        return DEFAULT_BASE_URLS[self.embed_provider]

    def resolved_judge(self) -> tuple[LLMProvider, str, str | None, str | None]:
        """Return (provider, model, base_url, api_key) for the evaluation judge."""
        provider = self.judge_provider or self.llm_provider
        model = self.judge_model or self.llm_model
        base_url: str | None
        if self.judge_base_url:
            base_url = self.judge_base_url
        elif provider == self.llm_provider:
            base_url = self.resolved_llm_base_url()
        elif provider == "ollama":
            base_url = self.ollama_base_url
        else:
            base_url = DEFAULT_BASE_URLS[provider]
        api_key = self.judge_api_key or (
            self.llm_api_key if provider == self.llm_provider else None
        )
        return provider, model, base_url, api_key

    @property
    def judge_is_same_model(self) -> bool:
        """True when the judge is the answering model (self-grading bias)."""
        return self.judge_model is None or self.judge_model == self.llm_model


def get_settings() -> Settings:
    return Settings()
