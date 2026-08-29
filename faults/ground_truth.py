"""Persist fault labels outside Langfuse traces.

AgentLens must diagnose the trace without seeing the answer. Ground truth is
therefore written to a local eval artifact keyed by the graph run id instead of
being attached to span or trace metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
GROUND_TRUTH_ROOT = _PROJECT_ROOT / "fault_ground_truth"


def write_ground_truth(run_id: str, record: dict[str, Any]) -> Path:
    """Atomically write one JSON ground-truth record and return its path."""
    GROUND_TRUTH_ROOT.mkdir(parents=True, exist_ok=True)
    target = GROUND_TRUTH_ROOT / f"{run_id}.json"
    temporary = GROUND_TRUTH_ROOT / f".{run_id}.tmp"
    temporary.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(target)
    return target
