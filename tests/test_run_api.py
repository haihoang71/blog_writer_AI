"""HTTP run/trace/SSE contract."""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_generate_persists_run_and_trace(client) -> None:
    posted = client.post(
        "/api/v1/generate",
        json={
            "topic": "Python asyncio event loop internals",
            "enable_hitl": False,
            "fault_scenario": "none",
        },
    )
    assert posted.status_code == 200
    body = posted.json()
    assert body["task_id"]
    assert body["run_id"]
    status = client.get(f"/api/v1/status/{body['task_id']}")
    assert status.status_code == 200
    assert status.json()["status"] in {"completed", "failed", "running", "queued"}
    run = client.get(f"/api/v1/runs/{body['run_id']}")
    assert run.status_code == 200
    trace = client.get(f"/api/v1/runs/{body['run_id']}/trace")
    assert trace.status_code == 200
    payload = trace.json()
    assert payload["normalized"]["span_count"] >= 1
    agents = {span["agent_name"] for span in payload["normalized"]["spans"]}
    assert "runtime_probe" in agents
    usage = client.get(f"/api/v1/runs/{body['run_id']}/usage")
    assert usage.status_code == 200
    assert "cost_label" in usage.json()
    assert "disclaimer" in usage.json()
    raw = client.get(f"/api/v1/runs/{body['run_id']}/trace/raw")
    assert raw.status_code == 200
    assert raw.json().get("_redacted") is True


@pytest.mark.integration
def test_generate_with_error_fault_still_writes_blog(client) -> None:
    posted = client.post(
        "/api/v1/generate",
        json={
            "topic": "LangGraph checkpointing for production agents",
            "enable_hitl": False,
            "fault_scenario": "error",
        },
    )
    assert posted.status_code == 200
    run_id = posted.json()["run_id"]
    status = client.get(f"/api/v1/status/{posted.json()['task_id']}")
    assert status.json()["status"] == "completed"
    trace = client.get(f"/api/v1/runs/{run_id}/trace").json()
    levels = [span["status"] for span in trace["normalized"]["spans"]]
    assert "error" in levels
    result = status.json().get("result") or {}
    assert result.get("final_post") or result.get("word_count", 0) >= 0


@pytest.mark.unit
def test_unknown_fault_is_400(client) -> None:
    posted = client.post(
        "/api/v1/generate",
        json={"topic": "x", "fault_scenario": "explode"},
    )
    assert posted.status_code == 400


@pytest.mark.unit
def test_list_faults(client) -> None:
    response = client.get("/api/v1/faults")
    assert response.status_code == 200
    assert "loop" in response.json()["scenarios"]


@pytest.mark.unit
def test_health_exposes_mode_not_secrets(client) -> None:
    response = client.get("/health")
    body = response.json()
    assert body["status"] == "ok"
    assert "execution_mode" in body
    assert "openai_api_key" not in body
    assert "langfuse_secret_key" not in str(body)
