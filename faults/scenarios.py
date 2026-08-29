"""Fault scenario definitions shared by the CLI, API, UI, and graph."""

from __future__ import annotations

from enum import Enum


class FaultScenario(str, Enum):
    """Supported synthetic anomalies.

    The string values intentionally match AgentLens' detector identifiers.
    """

    NONE = "none"
    LOOP = "loop"
    ERROR = "error"
    REDUNDANT = "redundant"
    THRESHOLD = "threshold"
    BOTTLENECK = "bottleneck"
    HALLUCINATION = "hallucination"
    PROMPT_INJECTION = "prompt_injection"


def coerce_fault_scenario(value: FaultScenario | str | None) -> FaultScenario:
    """Return a validated scenario, treating blank values as ``none``."""
    if isinstance(value, FaultScenario):
        return value
    normalised = (value or FaultScenario.NONE.value).strip().lower()
    try:
        return FaultScenario(normalised)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in FaultScenario)
        raise ValueError(
            f"Unknown fault scenario {value!r}. Expected one of: {allowed}"
        ) from exc
