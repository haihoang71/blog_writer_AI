"""
config/settings.py
──────────────────
Centralised application settings powered by Pydantic BaseSettings.
All values are read from environment variables (or a .env file).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── OpenAI ──────────────────────────────────────────────────────────
    openai_api_key: SecretStr = Field(
        default="sk-placeholder",
        description="OpenAI API key",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="Primary OpenAI model identifier",
    )
    openai_model_strong: str = Field(
        default="gpt-4o",
        description="Stronger model for Critic / complex reasoning",
    )
    openai_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    openai_max_tokens: int = Field(default=4096, ge=256, le=16384)
    openai_api_base: str = Field(
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
        description=(
            "Base URL for the OpenAI-compatible chat completions API. "
            "Defaults to Google's Gemini OpenAI-compatibility endpoint since "
            "this project is configured to run against Gemini models via "
            "OPENAI_API_KEY/OPENAI_MODEL. Set to an empty string to use the "
            "real OpenAI endpoint instead."
        ),
    )

    # ── Tavily ──────────────────────────────────────────────────────────
    tavily_api_key: SecretStr = Field(
        default="tvly-placeholder",
        description="Tavily Search API key",
    )
    tavily_max_results: int = Field(default=5, ge=1, le=20)

    # ── ArXiv ───────────────────────────────────────────────────────────
    arxiv_api_key: SecretStr = Field(
        default="",
        description=(
            "Optional arXiv API key. Note: arXiv's public search API is free "
            "and keyless — this field exists so the Academic Researcher agent "
            "can authenticate against arXiv-adjacent services (e.g. higher "
            "rate-limit tiers or citation-enrichment APIs) if configured. "
            "Leave empty to use the public, keyless arXiv endpoint."
        ),
    )
    arxiv_max_results: int = Field(default=5, ge=1, le=20)

    # ── Langfuse ────────────────────────────────────────────────────────
    langfuse_public_key: str = Field(
        default="",
        description="Langfuse public key (leave empty to disable tracing)",
    )
    langfuse_secret_key: SecretStr = Field(
        default="",
        description="Langfuse secret key",
    )
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com",
        description="Langfuse server URL (use http://localhost:3000 for self-hosted)",
    )

    # ── Redis / Celery ───────────────────────────────────────────────────
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis broker URL for Celery",
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/1",
    )

    # ── Application ──────────────────────────────────────────────────────
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
    )
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1024, le=65535)

    # ── Feature Flags ────────────────────────────────────────────────────
    enable_tracing: bool = Field(
        default=True,
        description="Toggle Langfuse tracing on/off",
    )
    enable_code_sandbox: bool = Field(
        default=True,
        description="Toggle the local code sandbox used by the Critic agent",
    )
    enable_hitl: bool = Field(
        default=True,
        description="Toggle Human-in-the-Loop interrupt node",
    )

    # ── Computed helpers ─────────────────────────────────────────────────
    #
    # NOTE: these all used to only check for the field's own Python default
    # (e.g. key.startswith("sk-placeholder")). But the shipped .env.example
    # placeholders are different strings ("sk-your-openai-api-key-here",
    # "tvly-your-tavily-api-key-here", "e2b-your-e2b-api-key-here") — so
    # copying .env.example to .env without filling in a real key was
    # incorrectly treated as "configured", and the code went on to make a
    # real (failing) API call instead of falling back to mock mode / the
    # local sandbox. `_looks_like_placeholder` catches both forms.
    @staticmethod
    def _looks_like_placeholder(key: str, *extra_markers: str) -> bool:
        if not key:
            return True
        markers = ("placeholder", "your-", *extra_markers)
        return any(marker in key for marker in markers)

    @property
    def is_tavily_configured(self) -> bool:
        key = self.tavily_api_key.get_secret_value()
        return bool(key) and not self._looks_like_placeholder(key)

    @property
    def is_langfuse_configured(self) -> bool:
        return bool(self.langfuse_public_key) and bool(
            self.langfuse_secret_key.get_secret_value()
        )

    @property
    def is_openai_configured(self) -> bool:
        key = self.openai_api_key.get_secret_value()
        return bool(key) and not self._looks_like_placeholder(key)

    @property
    def is_arxiv_configured(self) -> bool:
        """
        Whether a real arXiv API key was supplied.

        Not required for normal operation — ``tools.search_tools.arxiv_search``
        works against the free public arXiv API even when this is False.
        """
        key = self.arxiv_api_key.get_secret_value()
        return bool(key) and not self._looks_like_placeholder(key)

    @field_validator("openai_temperature")
    @classmethod
    def _validate_temperature(cls, v: float) -> float:
        if not (0.0 <= v <= 2.0):
            raise ValueError("temperature must be between 0.0 and 2.0")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
