"""Run / trace / SSE endpoints used by the Trace Explorer."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from api.security import require_admin
from config.settings import get_settings
from faults.ground_truth import get as get_ground_truth
from faults.scenarios import ALL_SCENARIOS
from observability.langfuse_io import fetch_raw_trace, redact_raw_for_public
from observability.normalize import normalize_observations, usage_rollup
from observability.redact import strip_secrets
from storage.run_store import get_run, list_events, list_observations, list_runs

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["runs"])


def _public_run(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": row.get("run_id"),
        "task_id": row.get("task_id"),
        "langfuse_trace_id": row.get("langfuse_trace_id"),
        "topic": row.get("topic"),
        "fault_scenario": row.get("fault_scenario"),
        "execution_mode": row.get("execution_mode"),
        "provider": row.get("provider"),
        "model": row.get("model"),
        "status": row.get("status"),
        "enable_hitl": row.get("enable_hitl"),
        "error": row.get("error"),
        "started_at": row.get("started_at"),
        "ended_at": row.get("ended_at"),
        "metadata": row.get("metadata") or {},
    }


def _load_run(run_id: str) -> dict[str, Any]:
    row = get_run(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return row


@router.get("/faults")
async def list_faults() -> dict[str, Any]:
    """Named synthetic scenarios. Labels never go onto Langfuse traces."""
    return {
        "scenarios": list(ALL_SCENARIOS),
        "note": (
            "Faults are a bounded branch at runtime_probe. Recoverable "
            "scenarios still produce a blog post. Ground truth is stored "
            "off-trace and is never written as expected_detector on spans."
        ),
    }


@router.get("/runs")
async def api_list_runs(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    rows = list_runs(limit=limit, offset=offset)
    return {"runs": [_public_run(row) for row in rows]}


@router.get("/runs/{run_id}")
async def api_get_run(run_id: str) -> dict[str, Any]:
    return _public_run(_load_run(run_id))


@router.get("/runs/{run_id}/trace")
async def api_get_trace(run_id: str) -> dict[str, Any]:
    row = _load_run(run_id)
    observations = [strip_secrets(item) for item in list_observations(row["run_id"])]
    normalized = normalize_observations(observations)
    usage = usage_rollup(normalized["spans"])
    pending = bool(row.get("langfuse_trace_id")) and get_settings().is_langfuse_configured
    return {
        "run": _public_run(row),
        "normalized": normalized,
        "usage": usage,
        "observation_count": len(observations),
        "langfuse_pending": pending and not observations,
        "source": "local_sqlite",
        "graph_hint": [
            "input_guard",
            "runtime_probe",
            "planner",
            "researcher",
            "academic_researcher?",
            "writer",
            "critic",
            "human_review",
            "output_guard",
        ],
    }


@router.get("/runs/{run_id}/trace/raw")
async def api_get_raw_trace(
    run_id: str,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    row = _load_run(run_id)
    settings = get_settings()
    local = {
        "trace": {
            "id": row.get("langfuse_trace_id") or row["run_id"],
            "name": row.get("topic"),
            "timestamp": row.get("started_at"),
            "metadata": {
                "execution_mode": row.get("execution_mode"),
                "fault_scenario": row.get("fault_scenario"),
            },
        },
        "observations": list_observations(row["run_id"]),
        "source": "local_sqlite",
    }
    remote = None
    if row.get("langfuse_trace_id"):
        remote = fetch_raw_trace(str(row["langfuse_trace_id"]))
        if remote:
            local["langfuse"] = remote
            local["source"] = "local+langfuse"
    admin = False
    if authorization:
        try:
            require_admin(authorization)
            admin = True
        except HTTPException:
            admin = False
    payload = strip_secrets(local)
    if admin or settings.enable_raw_trace_public:
        return redact_raw_for_public(payload) if not admin else payload
    return redact_raw_for_public(payload)


@router.get("/runs/{run_id}/usage")
async def api_get_usage(run_id: str) -> dict[str, Any]:
    row = _load_run(run_id)
    observations = list_observations(row["run_id"])
    normalized = normalize_observations(observations)
    rollup = usage_rollup(normalized["spans"])
    rollup["run_id"] = row["run_id"]
    rollup["execution_mode"] = row.get("execution_mode")
    return rollup


@router.get("/runs/{run_id}/events")
async def api_get_events(
    run_id: str,
    after_id: int = Query(default=0, ge=0),
) -> dict[str, Any]:
    row = _load_run(run_id)
    return {"run_id": row["run_id"], "events": list_events(row["run_id"], after_id=after_id)}


@router.get("/runs/{run_id}/ground-truth")
async def api_ground_truth(
    run_id: str,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    require_admin(authorization)
    row = _load_run(run_id)
    truth = get_ground_truth(row["run_id"])
    if truth is None:
        raise HTTPException(status_code=404, detail="No ground truth for this run")
    return truth


@router.get("/runs/{run_id}/stream")
async def api_stream_run(run_id: str, request: Request) -> StreamingResponse:
    row = _load_run(run_id)
    target = row["run_id"]

    async def event_gen():
        last = 0
        idle = 0
        while True:
            if await request.is_disconnected():
                break
            events = list_events(target, after_id=last)
            for event in events:
                last = int(event["id"])
                idle = 0
                yield f"event: {event['kind']}\ndata: {json.dumps(event, default=str)}\n\n"
            current = get_run(target)
            status = (current or {}).get("status")
            if status in {"completed", "failed", "timeout"}:
                yield f"event: done\ndata: {json.dumps({'status': status, 'run_id': target})}\n\n"
                break
            idle += 1
            if idle > 1500:
                yield "event: timeout\ndata: {}\n\n"
                break
            await asyncio.sleep(0.4)

    return StreamingResponse(event_gen(), media_type="text/event-stream")
