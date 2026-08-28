"""Fault injectors produce detector-shaped observations and keep labels off-trace."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from faults.ground_truth import get as get_truth
from faults.injector import build_fault_observations, inject, observation
from faults.scenarios import ALL_SCENARIOS, parse_scenario
from observability.normalize import _input_hash, normalize_observations
from storage.run_store import list_observations


def _parent() -> str:
    now = datetime.now(timezone.utc)
    return observation(
        name="token_probe",
        agent_name="runtime_probe",
        start=now,
        end=now,
    )["id"]


@pytest.mark.unit
def test_unknown_scenario_rejected() -> None:
    with pytest.raises(ValueError):
        parse_scenario("not-a-fault")


@pytest.mark.unit
@pytest.mark.parametrize("scenario", ALL_SCENARIOS)
def test_no_expected_detector_on_observations(scenario: str) -> None:
    rows, truth = build_fault_observations(
        scenario,  # type: ignore[arg-type]
        run_id="r1",
        parent_id=_parent(),
        real_sleep=False,
    )
    for row in rows:
        assert "expected_detector" not in (row.get("metadata") or {})
    if scenario != "none":
        assert truth.get("expected_detector")


@pytest.mark.unit
def test_loop_is_bounded_ab_cycle() -> None:
    rows, truth = build_fault_observations(
        "loop", run_id="r", parent_id=_parent(), loop_cycles=3, real_sleep=False
    )
    names = [row["metadata"]["agent_name"] for row in rows]
    assert names == ["loop_agent_a", "loop_agent_b"] * 3
    assert truth["cycles"] == 3
    assert len(rows) == 6


@pytest.mark.unit
def test_error_span_is_error_level() -> None:
    rows, truth = build_fault_observations(
        "error", run_id="r", parent_id=_parent(), real_sleep=False
    )
    assert rows[0]["level"] == "ERROR"
    assert "timeout" not in (rows[0]["statusMessage"] or "").lower()
    assert truth["expected_detector"] == "error"


@pytest.mark.unit
def test_timeout_status_message_contains_timeout() -> None:
    rows, _truth = build_fault_observations(
        "timeout", run_id="r", parent_id=_parent(), real_sleep=False
    )
    assert rows[0]["level"] == "ERROR"
    assert "timeout" in rows[0]["statusMessage"].lower()
    normalized = normalize_observations(rows)
    assert normalized["spans"][0]["status"] == "timeout"


@pytest.mark.unit
def test_redundant_identical_input_hash() -> None:
    rows, truth = build_fault_observations(
        "redundant", run_id="r", parent_id=_parent(), real_sleep=False
    )
    assert len(rows) == 2
    assert rows[0]["metadata"]["agent_name"] == rows[1]["metadata"]["agent_name"] == "writer"
    assert _input_hash(rows[0]["input"]) == _input_hash(rows[1]["input"])
    assert truth["expected_detector"] == "redundant"


@pytest.mark.unit
def test_threshold_spike_tokens() -> None:
    rows, truth = build_fault_observations(
        "threshold",
        run_id="r",
        parent_id=_parent(),
        threshold_input_tokens=50_000,
        real_sleep=False,
    )
    assert rows[0]["usageDetails"]["input"] == 50_000
    assert truth["expected_detector"] == "threshold"


@pytest.mark.unit
def test_bottleneck_long_self_time() -> None:
    rows, truth = build_fault_observations(
        "bottleneck",
        run_id="r",
        parent_id=_parent(),
        bottleneck_ms=800,
        real_sleep=False,
    )
    norm = normalize_observations(rows)["spans"][0]
    assert (norm["duration_ms"] or 0) >= 800
    assert truth["expected_detector"] == "bottleneck"


@pytest.mark.unit
def test_hallucination_has_state_keys_and_ungrounded_numbers() -> None:
    rows, truth = build_fault_observations(
        "hallucination", run_id="r", parent_id=_parent(), real_sleep=False
    )
    meta = rows[0]["metadata"]
    assert "research_data" in meta["reads_state_keys"]
    assert "draft" in meta["writes_state_keys"]
    blob = str(rows[0]["output"])
    assert "99.4%" in blob
    assert "max_iterations=999" in blob
    assert truth["expected_detector"] == "hallucination"


@pytest.mark.unit
def test_prompt_injection_in_tool_output_not_system_prompt() -> None:
    rows, truth = build_fault_observations(
        "prompt_injection", run_id="r", parent_id=_parent(), real_sleep=False
    )
    output = str(rows[0]["output"])
    assert "Ignore previous instructions" in output
    assert "system prompt" in output.lower()
    assert truth["jailbreak_in_tool_output"] is True
    assert truth["jailbreak_in_system_prompt"] is False


@pytest.mark.unit
def test_cascading_root_then_symptom() -> None:
    rows, truth = build_fault_observations(
        "cascading", run_id="r", parent_id=_parent(), real_sleep=False
    )
    assert rows[0]["level"] == "ERROR"
    assert rows[1]["metadata"]["caused_by"] == rows[0]["id"]
    assert truth["cascade"] is True
    assert truth["root_span"] == rows[0]["id"]


@pytest.mark.unit
def test_inject_persists_ground_truth_separately(isolate_data_dir) -> None:
    result = inject(
        "error",
        run_id="run-gt",
        task_id="task-gt",
        langfuse_trace_id=None,
        real_sleep=False,
    )
    obs = list_observations("run-gt")
    assert result["extra"] == 1
    for row in obs:
        assert "expected_detector" not in (row.get("metadata") or {})
    truth = get_truth("run-gt")
    assert truth is not None
    assert truth["payload"]["expected_detector"] == "error"


@pytest.mark.unit
def test_parent_child_links() -> None:
    parent_id = _parent()
    rows, _ = build_fault_observations(
        "error", run_id="r", parent_id=parent_id, real_sleep=False
    )
    assert rows[0]["parentObservationId"] == parent_id
    assert rows[0]["isRootObservation"] is False
