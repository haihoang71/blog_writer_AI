"""Named fault scenarios. Ground-truth labels never go into production traces."""

from __future__ import annotations

from enum import Enum


class FaultScenario(str, Enum):
    """Stable wire values used by the API, CLI and eval fixtures."""

    NONE = "none"
    LOOP = "loop"
    ERROR = "error"
    REDUNDANT = "redundant"
    THRESHOLD = "threshold"
    BOTTLENECK = "bottleneck"
    HALLUCINATION = "hallucination"
    PROMPT_INJECTION = "prompt_injection"
    TIMEOUT = "timeout"
    CASCADING = "cascading"


ALL_SCENARIOS: tuple[str, ...] = tuple(item.value for item in FaultScenario)

RECOVERABLE: dict[FaultScenario | str, bool] = {
    scenario: True for scenario in FaultScenario
}


def parse_scenario(value: str | FaultScenario | None) -> FaultScenario:
    """Normalize a JSON/CLI value while retaining Enum compatibility."""
    if isinstance(value, FaultScenario):
        return value
    name = str(value or FaultScenario.NONE.value).strip().lower()
    try:
        return FaultScenario(name)
    except ValueError as exc:
        raise ValueError(f"Unknown fault scenario: {value!r}") from exc


def coerce_fault_scenario(value: str | FaultScenario | None) -> FaultScenario:
    """Backward-compatible name used by the original CLI/tests."""
    return parse_scenario(value)
