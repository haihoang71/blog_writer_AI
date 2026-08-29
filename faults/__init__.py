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
