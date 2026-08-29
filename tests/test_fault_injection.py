"""Tests for deterministic AgentLens fault scenarios."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from faults import injector
from faults import ground_truth as ground_truth_store
from faults.scenarios import FaultScenario, coerce_fault_scenario
from state.blog_state import initial_state


class _FakeObservation:
    def __init__(self) -> None:
        self.updates: list[dict] = []

    def update(self, **kwargs):
        self.updates.append(kwargs)
        return self


@pytest.fixture
def observed(monkeypatch):
    calls: list[dict] = []

    @contextmanager
    def fake_observation(**kwargs):
        observation = _FakeObservation()
        calls.append({"start": kwargs, "observation": observation})
        yield observation

    monkeypatch.setattr(injector, "_observation", fake_observation)
    return calls


@pytest.fixture
def ground_truth(monkeypatch):
    records: list[dict] = []

    def fake_write(run_id: str, record: dict):
        records.append(record)
        return Path("fault_ground_truth") / f"{run_id}.json"

    monkeypatch.setattr(injector, "write_ground_truth", fake_write)
    return records


def _run(scenario: FaultScenario | str):
    state = initial_state("Testing AgentLens fault traces")
    return injector.fault_injection_node(
        state,
        fault_scenario=(
            scenario.value if isinstance(scenario, FaultScenario) else scenario
        ),
    )


def test_fault_scenario_validation():
    assert coerce_fault_scenario(None) is FaultScenario.NONE
    assert coerce_fault_scenario(" LOOP ") is FaultScenario.LOOP
    with pytest.raises(ValueError, match="Unknown fault scenario"):
        coerce_fault_scenario("deadlock")


def test_ground_truth_is_written_outside_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(ground_truth_store, "GROUND_TRUTH_ROOT", tmp_path)
    path = ground_truth_store.write_ground_truth(
        "run-123",
        {"scenario": "error", "root_cause_span_name": "source_registry_lookup"},
    )

    assert path == tmp_path / "run-123.json"
    assert json.loads(path.read_text(encoding="utf-8"))["scenario"] == "error"


def test_api_request_validates_fault_scenario():
    from main import GenerateRequest

    request = GenerateRequest(topic="Agent observability", fault_scenario="loop")
    assert request.fault_scenario is FaultScenario.LOOP

    with pytest.raises(Exception):
        GenerateRequest(topic="Agent observability", fault_scenario="deadlock")


def test_clean_run_emits_only_threshold_baseline(observed, ground_truth):
    result = _run(FaultScenario.NONE)

    assert result["metadata"]["synthetic_fault"] is False
    assert result["metadata"]["token_probe_usage"]["total_tokens"] == 240
    assert ground_truth == []
    assert [call["start"]["name"] for call in observed] == [
        injector.TOKEN_PROBE_NAME
    ]


def test_loop_scenario_emits_four_a_b_repetitions(observed, ground_truth):
    result = _run(FaultScenario.LOOP)
    truth = ground_truth[0]

    assert truth["expected_detector"] == "loop"
    assert truth["repeat_count"] == 4
    assert truth["cycle"] == ["research_coordinator", "source_retriever"]
    loop_calls = [
        call for call in observed if call["start"]["name"].startswith("research_handoff_")
    ]
    assert len(loop_calls) == 8
    assert result["metadata"]["fault_ground_truth_run_id"] == result["run_id"]
    assert loop_calls[0]["start"]["metadata"]["agent_name"] == "research_coordinator"
    assert loop_calls[1]["start"]["metadata"]["agent_name"] == "source_retriever"
    assert "fault_role" not in loop_calls[0]["start"]["metadata"]
    assert "fault_scenario" not in loop_calls[0]["start"]["metadata"]
    assert "expected_detector" not in loop_calls[0]["start"]["metadata"]


def test_bottleneck_scenario_creates_slow_tool(
    monkeypatch, observed, ground_truth
):
    sleeps: list[float] = []
    monkeypatch.setattr(injector.time, "sleep", sleeps.append)

    result = _run(FaultScenario.BOTTLENECK)
    truth = ground_truth[0]

    assert sleeps == [injector.BOTTLENECK_DELAY_SECONDS]
    assert truth["root_cause_span_name"] == "full_corpus_scan"
    assert truth["delay_seconds"] == injector.BOTTLENECK_DELAY_SECONDS


def test_threshold_scenario_sets_large_synthetic_usage(observed, ground_truth):
    result = _run(FaultScenario.THRESHOLD)
    truth = ground_truth[0]
    probe = observed[0]["start"]

    assert truth["metric"] == "total_tokens"
    assert truth["value"] == 30_000
    assert truth["minimum_baseline_samples"] == 20
    assert probe["usage_details"]["prompt_tokens"] == 25_000
    assert probe["usage_details"]["completion_tokens"] == 5_000


def test_error_scenario_marks_tool_error_but_keeps_run_recoverable(
    observed, ground_truth
):
    result = _run(FaultScenario.ERROR)
    truth = ground_truth[0]
    error_call = next(
        call for call in observed
        if call["start"]["name"] == "source_registry_lookup"
    )

    assert truth["error_class"] == "IntentionalFaultError"
    assert truth["recoverable"] is True
    assert result["error_logs"][-1].recoverable is True
    assert error_call["observation"].updates[-1]["level"] == "ERROR"
    assert "IntentionalFaultError" in (
        error_call["observation"].updates[-1]["status_message"]
    )


def test_redundant_scenario_repeats_identical_input_for_one_agent(
    observed, ground_truth
):
    _run(FaultScenario.REDUNDANT)
    truth = ground_truth[0]
    duplicate_calls = [
        call for call in observed
        if call["start"]["name"] == "duplicate_outline_generation"
    ]

    assert truth["expected_detector"] == "redundant"
    assert truth["duplicate_count"] == injector.REDUNDANT_REPEAT_COUNT
    assert len(duplicate_calls) == injector.REDUNDANT_REPEAT_COUNT
    assert len({json.dumps(call["start"]["input"], sort_keys=True)
                for call in duplicate_calls}) == 1
    assert {
        call["start"]["metadata"]["agent_name"] for call in duplicate_calls
    } == {"outline_planner"}
    assert sum(
        call["start"]["usage_details"]["total_tokens"]
        for call in duplicate_calls
    ) == 450


def test_hallucination_scenario_has_claim_missing_from_its_evidence(
    observed, ground_truth
):
    _run(FaultScenario.HALLUCINATION)
    truth = ground_truth[0]
    writer = next(
        call for call in observed
        if call["start"]["name"] == "unsupported_market_summary"
    )["start"]
    evidence = json.dumps(writer["input"], sort_keys=True)
    claim = writer["output"]["summary"]

    assert truth["expected_detector"] == "hallucination"
    assert '"Northstar Quantum"' in claim
    assert "4.7 billion" in claim
    assert "Northstar Quantum" not in evidence
    assert "4.7" not in evidence
    assert writer["metadata"]["reads_state_keys"] == ["verified_research"]


def test_prompt_injection_scenario_places_jailbreak_only_in_tool_output(
    observed, ground_truth
):
    result = _run(FaultScenario.PROMPT_INJECTION)
    truth = ground_truth[0]
    tool = next(
        call for call in observed
        if call["start"]["name"] == "untrusted_web_content"
    )["start"]

    assert truth["expected_detector"] == "prompt_injection"
    assert "Ignore all previous instructions" in tool["output"]["content"]
    assert tool["output"]["trusted"] is False
    assert "Ignore all previous instructions" not in json.dumps(result)
    assert "fault_scenario" not in tool["metadata"]
    assert "expected_detector" not in tool["metadata"]


@pytest.mark.parametrize(
    "scenario",
    [
        FaultScenario.LOOP,
        FaultScenario.BOTTLENECK,
        FaultScenario.THRESHOLD,
        FaultScenario.ERROR,
        FaultScenario.REDUNDANT,
        FaultScenario.HALLUCINATION,
        FaultScenario.PROMPT_INJECTION,
    ],
)
def test_full_graph_completes_with_each_fault(
    scenario, monkeypatch, observed, ground_truth
):
    from graph.workflow import build_graph

    monkeypatch.setattr(injector.time, "sleep", lambda _seconds: None)
    graph = build_graph(enable_hitl=False, fault_scenario=scenario.value)
    state = initial_state("Python async programming patterns")
    result = graph.invoke(
        state,
        config={"configurable": {"thread_id": f"fault-{scenario.value}"}},
    )

    assert result.get("final_post")
    assert result["metadata"]["synthetic_fault"] is True
    assert ground_truth[0]["scenario"] == scenario.value
