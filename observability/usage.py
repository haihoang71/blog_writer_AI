"""Token/cost usage contract compatible with AgentLens Langfuse adapter.

AgentLens ``backend/ingestion/langfuse_client.py::_map_observation`` reads:

    usageDetails.input
    usageDetails.output
    totalCost

and maps them to ``usage.input``, ``usage.output``, ``usage.total_cost``,
which the normalizer then stores as ``prompt_tokens`` / ``completion_tokens``
/ ``cost_usd``.

This module is the single place synthetic and live usage is shaped so those
fields stay aligned. It never invents cost when the source is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

UsageSource = Literal[
    "provider_reported",
    "langfuse_estimated",
    "synthetic",
    "unavailable",
]


@dataclass(frozen=True)
class UsageRecord:
    """One span's token/cost numbers plus how they were obtained."""

    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    cost_usd: float | None
    source: UsageSource
    model: str | None = None

    def to_langfuse_usage_details(self) -> dict[str, int]:
        """Wire shape Langfuse v2 stores as ``usageDetails``."""
        return {
            "input": int(self.input_tokens or 0),
            "output": int(self.output_tokens or 0),
        }

    def to_agentlens_usage(self) -> dict[str, Any]:
        """Wire shape AgentLens ``_map_observation`` produces."""
        return {
            "input": int(self.input_tokens or 0),
            "output": int(self.output_tokens or 0),
            "total_cost": float(self.cost_usd or 0.0),
        }

    def metadata_fields(self) -> dict[str, Any]:
        """Safe metadata flags (never secrets)."""
        return {
            "usage_source": self.source,
            "usage_cost_is_estimate": self.source
            in {"langfuse_estimated", "synthetic"},
            "usage_is_synthetic": self.source == "synthetic",
        }


def normalize_usage(
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    cost_usd: float | None = None,
    source: UsageSource,
    model: str | None = None,
) -> UsageRecord:
    """Accept either Langfuse (input/output) or OpenAI (prompt/completion) names."""
    inn = input_tokens if input_tokens is not None else prompt_tokens
    out = output_tokens if output_tokens is not None else completion_tokens
    total = total_tokens
    if total is None and inn is not None and out is not None:
        total = inn + out
    if source == "unavailable":
        return UsageRecord(
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            cost_usd=None,
            source="unavailable",
            model=model,
        )
    return UsageRecord(
        input_tokens=inn,
        output_tokens=out,
        total_tokens=total,
        cost_usd=cost_usd,
        source=source,
        model=model,
    )


def usage_from_langchain_response(response: Any, *, model: str | None = None) -> UsageRecord:
    """Read provider-reported usage off a LangChain AIMessage when present."""
    meta = getattr(response, "response_metadata", None) or {}
    usage = meta.get("token_usage") or meta.get("usage") or {}
    # Newer LangChain integrations expose normalized counts on the message
    # itself, while OpenAI-compatible providers often keep token_usage in
    # response_metadata. Support both so the trace reflects provider data.
    if not usage:
        usage = getattr(response, "usage_metadata", None) or {}
    inn = usage.get("prompt_tokens", usage.get("input_tokens"))
    out = usage.get("completion_tokens", usage.get("output_tokens"))
    total = usage.get("total_tokens")
    if inn is None and out is None:
        return normalize_usage(source="unavailable", model=model)
    return normalize_usage(
        prompt_tokens=inn,
        completion_tokens=out,
        total_tokens=total,
        source="provider_reported",
        model=model,
    )


def synthetic_usage(
    *,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float | None = None,
    model: str | None = None,
) -> UsageRecord:
    """Fault-injector usage. Must not be presented as a real bill."""
    return normalize_usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd,
        source="synthetic",
        model=model,
    )
