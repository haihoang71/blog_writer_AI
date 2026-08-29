"""Synthetic fault injection for AgentLens detector evaluation."""

from faults.ground_truth import (
    get as get_ground_truth,
    record as record_ground_truth,
    write_ground_truth,
)
from faults.injector import (
    build_fault_observations,
    inject,
    observation,
    register_scenario,
)
from faults.scenarios import (
    ALL_SCENARIOS,
    FaultScenario,
    coerce_fault_scenario,
    parse_scenario,
)

__all__ = [
    "ALL_SCENARIOS",
    "FaultScenario",
    "build_fault_observations",
    "coerce_fault_scenario",
    "get_ground_truth",
    "inject",
    "observation",
    "parse_scenario",
    "record_ground_truth",
    "register_scenario",
    "write_ground_truth",
]
