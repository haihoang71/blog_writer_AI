"""Baseline token probe + optional synthetic fault branch.

Clean runs emit one probe span with a unique input (so redundant-detector
does not fire across traces) but a stable ``agent_name`` / span name (so
threshold baselines accumulate). Faults are extra sibling observations.
"""

from __future__ import annotations

import logging
from typing import Any

from faults.injector import inject
from faults.scenarios import parse_scenario
from state.blog_state import BlogState
from storage.run_store import append_event, get_run, upsert_run

logger = logging.getLogger(__name__)


def runtime_probe_node(state: BlogState) -> BlogState:
    """LangGraph node sitting between input_guard and planner."""
    metadata: dict[str, Any] = dict(state.get("metadata", {}))
    run_id = str(state.get("run_id") or "")
    scenario = parse_scenario(metadata.get("fault_scenario") or "none")
    task_id = str(metadata.get("task_id") or run_id)
    stored = get_run(run_id) or {}
    langfuse_trace_id = stored.get("langfuse_trace_id")
    try:
        from config.settings import get_settings
        from langfuse import get_client

        if get_settings().is_langfuse_configured:
            langfuse_trace_id = langfuse_trace_id or get_client().get_current_trace_id()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Langfuse trace id unavailable: %s", type(exc).__name__)

    result = inject(
        scenario,
        run_id=run_id,
        task_id=task_id,
        langfuse_trace_id=langfuse_trace_id,
        real_sleep=True,
    )
    metadata["runtime_probe"] = {
        "scenario": scenario,
        "probe_id": result.get("probe_id"),
        "extra_observations": result.get("extra"),
    }
    metadata["fault_scenario"] = scenario
    append_event(run_id, "node_completed", {"node": "runtime_probe", "scenario": scenario})
    if langfuse_trace_id:
        upsert_run({"run_id": run_id, "task_id": task_id, "langfuse_trace_id": langfuse_trace_id, "status": "running"})
    logger.info("runtime_probe: scenario=%s extra=%s", scenario, result.get("extra"))
    return {**state, "metadata": metadata}
