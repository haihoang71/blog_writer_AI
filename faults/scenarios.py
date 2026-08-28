"""Named fault scenarios. Ground-truth labels never go into production traces."""

from __future__ import annotations

from typing import Literal

FaultScenario = Literal[
    "none",
    "loop",
    "error",
    "redundant",
    "threshold",
    "bottleneck",
    "hallucination",
    "prompt_injection",
    "timeout",
    "cascading",
]

ALL_SCENARIOS: tuple[FaultScenario, ...] = (
    "none",
    "loop",
    "error",
    "redundant",
    "threshold",
    "bottleneck",
    "hallucination",
    "prompt_injection",
    "timeout",
    "cascading",
)

RECOVERABLE: dict[FaultScenario, bool] = {
    "none": True,
    "loop": True,
    "error": True,
    "redundant": True,
    "threshold": True,
    "bottleneck": True,
    "hallucination": True,
    "prompt_injection": True,
    "timeout": True,
    "cascading": True,
}


def parse_scenario(value: str | None) -> FaultScenario:
    name = (value or "none").strip().lower()
    if name not in ALL_SCENARIOS:
        raise ValueError(f"unknown fault scenario: {value!r}")
    return name  # type: ignore[return-value]
