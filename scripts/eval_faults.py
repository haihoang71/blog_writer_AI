"""Eval-only runner: inject every fault scenario and write an off-trace report.

Does not call AgentLens and does not put expected_detector on Langfuse spans.
Run:  python scripts/eval_faults.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.settings import get_settings
from config.tracing import flush_langfuse, get_langfuse_callback
from faults.ground_truth import get as get_truth
from faults.injector import inject
from faults.scenarios import ALL_SCENARIOS
from observability.normalize import normalize_observations
from storage.run_store import list_observations, upsert_run


def main() -> int:
    out_dir = ROOT / "data" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scenarios": [],
        "note": "Ground truth is in data/ground_truth.sqlite3 and fault_ground_truth/*.json; labels are not on traces.",
    }
    for scenario in ALL_SCENARIOS:
        run_id = str(uuid4())
        task_id = str(uuid4())
        upsert_run(
            {
                "run_id": run_id,
                "task_id": task_id,
                "status": "eval",
                "fault_scenario": scenario,
                "topic": f"eval:{scenario}",
            }
        )
        result = None
        if get_settings().is_langfuse_configured and get_langfuse_callback():
            # Standalone eval runs do not have LangGraph's callback root, so
            # create one explicitly. The scenario label stays in ground truth
            # and the local report, never in the remote trace name/metadata.
            from langfuse import get_client

            client = get_client()
            with client.start_as_current_observation(
                name="eval-fixture",
                as_type="chain",
                input={"run_id": run_id, "purpose": "detector-evaluation"},
                metadata={"source": "blog-writer-eval"},
            ) as root:
                result = inject(
                    scenario,
                    run_id=run_id,
                    task_id=task_id,
                    langfuse_trace_id=getattr(root, "trace_id", None),
                    real_sleep=False,
                )
            flush_langfuse()
        else:
            result = inject(
                scenario,
                run_id=run_id,
                task_id=task_id,
                langfuse_trace_id=None,
                real_sleep=False,
            )
        observations = list_observations(run_id)
        normalized = normalize_observations(observations)
        leaked = [
            obs.get("metadata", {}).get("expected_detector")
            for obs in observations
            if (obs.get("metadata") or {}).get("expected_detector")
        ]
        truth = get_truth(run_id)
        report["scenarios"].append(
            {
                "scenario": scenario,
                "run_id": run_id,
                "span_count": normalized["span_count"],
                "error_count": normalized["error_count"],
                "expected_detector": (truth or {}).get("payload", {}).get("expected_detector"),
                "leaked_expected_detector_on_spans": leaked,
                "langfuse_trace_id": (truth or {}).get("langfuse_trace_id") or (result or {}).get("langfuse_trace_id"),
            }
        )
    path = out_dir / "fault_report.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {path}")
    leaked_any = any(item["leaked_expected_detector_on_spans"] for item in report["scenarios"])
    return 1 if leaked_any else 0


if __name__ == "__main__":
    raise SystemExit(main())
