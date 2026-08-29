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
]
"""Controlled fault injection for AgentLens trace generation."""

from faults.injector import fault_injection_node
from faults.ground_truth import write_ground_truth
from faults.scenarios import FaultScenario, coerce_fault_scenario

__all__ = [
    "FaultScenario",
    "coerce_fault_scenario",
    "fault_injection_node",
    "write_ground_truth",
]
