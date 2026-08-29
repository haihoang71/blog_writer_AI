"""Pure mirror of AgentLens ``langfuse_client._map_observation`` usage mapping.

Kept here so blog-writer tests can prove a synthetic Langfuse-shaped payload
survives the same rename AgentLens applies at ingestion. This is not a copy
of production AgentLens code paths — only the field contract.
"""

from __future__ import annotations

import json
from typing import Any


def _decode_io(value: Any) -> Any:
    """Decode JSON-encoded I/O returned by the Langfuse HTTP API."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value


def map_langfuse_observation(row: dict[str, Any]) -> dict[str, Any]:
    """Rename one Langfuse HTTP observation into AgentLens normalizer input.

    AgentLens reads ``usageDetails.input``, ``usageDetails.output``,
    ``totalCost`` — not ``prompt_tokens`` / ``completion_tokens``.
    """
    usage_details = row.get("usageDetails") or row.get("usage_details") or {}
    metadata = row.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    usage_input = usage_details.get("input") or usage_details.get("prompt_tokens") or 0
    usage_output = usage_details.get("output") or usage_details.get("completion_tokens") or 0
    total_cost = row.get("totalCost")
    if total_cost is None:
        total_cost = row.get("total_cost")
    if total_cost is None:
        total_cost = usage_details.get("total_cost")
    if total_cost is None:
        total_cost = usage_details.get("cost")
    if metadata.get("usage_source") is None:
        if usage_input or usage_output:
            metadata = {**metadata, "usage_source": "provider_reported"}
        elif total_cost:
            metadata = {**metadata, "usage_source": "langfuse_estimated"}
        else:
            metadata = {**metadata, "usage_source": "unavailable"}
    return {
        "id": row.get("id"),
        "trace_id": row.get("traceId") or row.get("trace_id"),
        "parent_observation_id": row.get("parentObservationId") or row.get("parentSpanId") or row.get("parent_observation_id"),
        "type": row.get("type") or row.get("span_type"),
        "name": row.get("name") or row.get("span_name"),
        "start_time": row.get("startTime") or row.get("start_time"),
        "end_time": row.get("endTime") or row.get("end_time"),
        "level": row.get("level"),
        "status_message": row.get("statusMessage") or row.get("status_message"),
        "model": row.get("providedModelName") or row.get("model"),
        "input": _decode_io(row.get("input")),
        "output": _decode_io(row.get("output")),
        "metadata": metadata,
        "usage": {
            "input": usage_input,
            "output": usage_output,
            "total_cost": total_cost,
        },
        "is_root_observation": bool(row.get("isRootObservation") or row.get("is_root_observation")),
        "trace_name": row.get("traceName"),
        "environment": row.get("environment"),
        "latency": row.get("latency"),
    }


def tokens_from_mapped(mapped: dict[str, Any]) -> tuple[int, int, int, float | None]:
    """prompt, completion, total, cost — AgentLens span fields."""
    usage = mapped.get("usage") or {}
    prompt = int(usage.get("input") or 0)
    completion = int(usage.get("output") or 0)
    raw_cost = usage.get("total_cost")
    cost = float(raw_cost) if raw_cost is not None else None
    return prompt, completion, prompt + completion, cost
