"""Pure mirror of AgentLens ``langfuse_client._map_observation`` usage mapping.

Kept here so blog-writer tests can prove a synthetic Langfuse-shaped payload
survives the same rename AgentLens applies at ingestion. This is not a copy
of production AgentLens code paths — only the field contract.
"""

from __future__ import annotations

from typing import Any


def map_langfuse_observation(row: dict[str, Any]) -> dict[str, Any]:
    """Rename one Langfuse v2 observation into AgentLens normalizer input.

    AgentLens reads ``usageDetails.input``, ``usageDetails.output``,
    ``totalCost`` — not ``prompt_tokens`` / ``completion_tokens``.
    """
    usage_details = row.get("usageDetails") or {}
    metadata = row.get("metadata")
    return {
        "id": row.get("id"),
        "trace_id": row.get("traceId"),
        "parent_observation_id": row.get("parentObservationId"),
        "type": row.get("type"),
        "name": row.get("name"),
        "start_time": row.get("startTime"),
        "end_time": row.get("endTime"),
        "level": row.get("level"),
        "status_message": row.get("statusMessage"),
        "model": row.get("providedModelName"),
        "input": row.get("input"),
        "output": row.get("output"),
        "metadata": metadata if isinstance(metadata, dict) else {},
        "usage": {
            "input": usage_details.get("input") or 0,
            "output": usage_details.get("output") or 0,
            "total_cost": row.get("totalCost") or 0.0,
        },
        "is_root_observation": bool(row.get("isRootObservation")),
        "trace_name": row.get("traceName"),
        "environment": row.get("environment"),
        "latency": row.get("latency"),
    }


def tokens_from_mapped(mapped: dict[str, Any]) -> tuple[int, int, int, float]:
    """prompt, completion, total, cost — AgentLens span fields."""
    usage = mapped.get("usage") or {}
    prompt = int(usage.get("input") or 0)
    completion = int(usage.get("output") or 0)
    cost = float(usage.get("total_cost") or 0.0)
    return prompt, completion, prompt + completion, cost
