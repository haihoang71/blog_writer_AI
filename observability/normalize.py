"""Build a normalized span tree from local Langfuse-shaped observations."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any

from observability.agentlens_adapter import map_langfuse_observation, tokens_from_mapped


def _parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _duration_ms(start: datetime | None, end: datetime | None) -> int | None:
    if start is None or end is None:
        return None
    return max(int((end - start).total_seconds() * 1000), 0)


def _input_hash(value: Any) -> str | None:
    if value is None:
        return None
    canonical = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_observations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Return spans with parent/self-time/sequence matching AgentLens fields."""
    mapped = [map_langfuse_observation(row) for row in rows]
    by_external = {str(item["id"]): item for item in mapped if item.get("id")}
    children: dict[str | None, list[str]] = {}
    for item in mapped:
        parent = item.get("parent_observation_id")
        children.setdefault(parent, []).append(str(item["id"]))

    def _start_key(external_id: str) -> str:
        return str(by_external[external_id].get("start_time") or "")

    for parent, child_ids in list(children.items()):
        children[parent] = sorted(child_ids, key=_start_key)

    spans: list[dict[str, Any]] = []
    sequence = 0

    def walk(external_id: str, depth: int) -> None:
        nonlocal sequence
        item = by_external[external_id]
        start = _parse_dt(item.get("start_time"))
        end = _parse_dt(item.get("end_time"))
        duration = _duration_ms(start, end)
        child_ids = children.get(external_id, [])
        prompt, completion, total, cost = tokens_from_mapped(item)
        metadata = item.get("metadata") or {}
        agent_name = metadata.get("agent_name") or metadata.get("langgraph_node") or item.get("name")
        child_durations = [
            _duration_ms(
                _parse_dt(by_external[cid].get("start_time")),
                _parse_dt(by_external[cid].get("end_time")),
            )
            or 0
            for cid in child_ids
            if cid in by_external
        ]
        self_time = None
        if duration is not None:
            self_time = max(duration - sum(child_durations), 0)
        level = str(item.get("level") or "").upper()
        status_message = item.get("status_message")
        if level == "ERROR" and status_message and "timeout" in str(status_message).lower():
            status = "timeout"
        elif level == "ERROR":
            status = "error"
        elif level:
            status = "ok"
        else:
            status = "unknown"
        seq = sequence
        sequence += 1
        spans.append(
            {
                "id": external_id,
                "parent_span_id": item.get("parent_observation_id"),
                "sequence_index": seq,
                "agent_name": agent_name,
                "name": item.get("name"),
                "span_type": item.get("type"),
                "started_at": item.get("start_time"),
                "ended_at": item.get("end_time"),
                "duration_ms": duration,
                "self_time_ms": self_time,
                "status": status,
                "status_message": status_message,
                "error_class": metadata.get("error_class"),
                "model": item.get("model"),
                "input": item.get("input"),
                "output": item.get("output"),
                "input_tokens": prompt,
                "output_tokens": completion,
                "total_tokens": total,
                "cost_usd": cost,
                "metadata": metadata,
                "reads_state_keys": metadata.get("reads_state_keys"),
                "writes_state_keys": metadata.get("writes_state_keys"),
                "input_hash": _input_hash(item.get("input")),
                "usage_source": metadata.get("usage_source"),
                "depth": depth,
            }
        )
        for cid in child_ids:
            if cid in by_external:
                walk(cid, depth + 1)

    roots = children.get(None, [])
    if not roots and mapped:
        roots = [str(mapped[0]["id"])]
    for root_id in roots:
        if root_id in by_external:
            walk(root_id, 0)

    return {
        "span_count": len(spans),
        "agent_count": len({s["agent_name"] for s in spans if s.get("agent_name")}),
        "error_count": sum(1 for s in spans if s["status"] in {"error", "timeout"}),
        "spans": spans,
    }


def usage_rollup(spans: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate tokens/cost with source labels — never invent unavailable cost."""
    sources = {s.get("usage_source") for s in spans if s.get("usage_source")}
    input_tokens = sum(int(s.get("input_tokens") or 0) for s in spans)
    output_tokens = sum(int(s.get("output_tokens") or 0) for s in spans)
    costs = [s.get("cost_usd") for s in spans if s.get("cost_usd") is not None]
    synthetic = "synthetic" in sources
    unavailable = sources == {"unavailable"} or not sources
    cost: float | None
    cost_label: str
    if unavailable and not costs:
        cost = None
        cost_label = "Unavailable"
    elif synthetic and sources <= {"synthetic", "unavailable"}:
        cost = float(sum(c or 0 for c in costs)) if costs else None
        cost_label = "Synthetic"
    elif "provider_reported" in sources:
        cost = float(sum(c or 0 for c in costs)) if costs else None
        cost_label = "Provider reported"
    elif "langfuse_estimated" in sources:
        cost = float(sum(c or 0 for c in costs)) if costs else None
        cost_label = "Langfuse estimated"
    else:
        cost = float(sum(c or 0 for c in costs)) if costs else None
        cost_label = "Langfuse estimated" if cost else "Unavailable"
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": cost,
        "cost_label": cost_label,
        "sources": sorted(s for s in sources if s),
        "disclaimer": (
            "Synthetic token/cost numbers are injected for detector tests; "
            "they are not a provider bill. Langfuse estimated cost uses the "
            "pricing catalog and is not an invoice. A free API key does not "
            "mean estimated cost is 0."
        ),
    }
