"""Usage contract: Langfuse usageDetails.input/output + totalCost."""

from __future__ import annotations

import pytest

from observability.agentlens_adapter import map_langfuse_observation, tokens_from_mapped
from observability.normalize import _input_hash, normalize_observations
from observability.usage import normalize_usage, synthetic_usage, usage_from_langchain_response
from faults.injector import baseline_probe, observation


@pytest.mark.unit
def test_usage_details_survive_agentlens_rename() -> None:
    record = synthetic_usage(input_tokens=40, output_tokens=12, cost_usd=0.0001)
    row = {
        "id": "obs-1",
        "traceId": "tr-1",
        "name": "token_probe",
        "usageDetails": record.to_langfuse_usage_details(),
        "totalCost": record.cost_usd,
        "metadata": record.metadata_fields(),
        "startTime": "2026-01-01T00:00:00Z",
        "endTime": "2026-01-01T00:00:01Z",
    }
    mapped = map_langfuse_observation(row)
    prompt, completion, total, cost = tokens_from_mapped(mapped)
    assert mapped["usage"]["input"] == 40
    assert mapped["usage"]["output"] == 12
    assert mapped["usage"]["total_cost"] == pytest.approx(0.0001)
    assert prompt == 40
    assert completion == 12
    assert total == 52
    assert cost == pytest.approx(0.0001)


@pytest.mark.unit
def test_unavailable_usage_does_not_invent_cost() -> None:
    record = normalize_usage(source="unavailable")
    assert record.input_tokens is None
    assert record.cost_usd is None
    assert record.to_agentlens_usage()["total_cost"] == 0.0
    # Adapter still yields 0 for missing numbers — UI must label the source.
    assert record.source == "unavailable"


@pytest.mark.unit
def test_langchain_response_without_usage_is_unavailable() -> None:
    class Msg:
        response_metadata: dict = {}

    record = usage_from_langchain_response(Msg(), model="gpt-4o-mini")
    assert record.source == "unavailable"
    assert record.cost_usd is None


@pytest.mark.unit
def test_baseline_probe_unique_input_hash_same_agent() -> None:
    a = baseline_probe(run_id="run-a")
    b = baseline_probe(run_id="run-b")
    assert a["metadata"]["agent_name"] == b["metadata"]["agent_name"] == "runtime_probe"
    assert a["name"] == b["name"] == "token_probe"
    assert _input_hash(a["input"]) != _input_hash(b["input"])
    assert "expected_detector" not in a["metadata"]


@pytest.mark.unit
def test_normalized_spans_expose_prompt_tokens() -> None:
    row = observation(
        name="planner",
        agent_name="planner",
        start=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        end=__import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        input_value={"x": 1},
        usage_input=9,
        usage_output=3,
        cost=0.01,
    )
    spans = normalize_observations([row])["spans"]
    assert spans[0]["input_tokens"] == 9
    assert spans[0]["output_tokens"] == 3
    assert spans[0]["cost_usd"] == pytest.approx(0.01)
    assert spans[0]["usage_source"] == "synthetic"
