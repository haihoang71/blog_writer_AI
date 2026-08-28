"""
graph/middlewares.py
─────────────────────
LangGraph middleware: structured logging and token-usage tracking
injected around every node invocation.

Usage
-----
The ``wrap_node`` function wraps a node callable with pre/post logging:

    from graph.middlewares import wrap_node

    wrapped_planner = wrap_node("planner", planner_node)
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable

from state.blog_state import BlogState, ErrorLog

logger = logging.getLogger(__name__)


def wrap_node(
    node_name: str,
    node_fn: Callable[[BlogState], BlogState],
) -> Callable[[BlogState], BlogState]:
    """
    Wrap a node function with structured logging middleware.

    Captures:
    - Node entry / exit timestamps
    - Wall-clock latency
    - Any uncaught exceptions (converted to error log entries)

    Parameters
    ----------
    node_name:
        Human-readable name for this node (used in log messages).
    node_fn:
        The original node callable.

    Returns
    -------
    Callable
        Wrapped node that passes through the original return value.
    """

    @wraps(node_fn)
    def wrapper(state: BlogState) -> BlogState:
        run_id = state.get("run_id", "unknown")
        revision = state.get("revision_count", 0)

        logger.info(
            "━━━ [%s] ENTER node=%s run=%s revision=%d ━━━",
            "NODE", node_name, run_id[:8], revision,
        )
        t0 = time.perf_counter()
        wall_start = datetime.now(timezone.utc)
        try:
            from storage.run_store import append_event

            append_event(str(run_id), "node_started", {"node": node_name})
        except Exception as persist_exc:  # noqa: BLE001
            logger.debug("Could not persist node_started for %s: %s", node_name, persist_exc)

        try:
            result: BlogState = node_fn(state)
        except Exception as exc:  # noqa: BLE001
            # HITL uses LangGraph interrupt() which raises GraphInterrupt.
            # Catching it here would skip the pause and keep the graph running.
            exc_name = type(exc).__name__
            if "Interrupt" in exc_name:
                raise
            elapsed_ms = (time.perf_counter() - t0) * 1000
            logger.error(
                "✖ [NODE] node=%s FAILED in %.0fms: %s",
                node_name, elapsed_ms, exc,
            )
            try:
                from storage.run_store import append_event as _ae

                _ae(str(run_id), "node_failed", {"node": node_name, "error": exc_name})
            except Exception as persist_exc:  # noqa: BLE001
                logger.debug("Could not persist node_failed for %s: %s", node_name, persist_exc)
            # Append to error log but don't raise — let the graph decide.
            # NOTE: this previously called sys.exit(), which killed the entire
            # Python process (CLI run or FastAPI server) on any single node
            # exception. Removed — we return the updated state instead so the
            # graph/router can decide how to proceed.
            error_logs = list(state.get("error_logs", []))
            error_logs.append(
                ErrorLog(
                    node=node_name,
                    error_type=exc_name,
                    message=str(exc),
                    recoverable=False,
                )
            )
            return {**state, "error_logs": error_logs}

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Log key state changes
        changes: list[str] = []
        for key in ["outline", "research_data", "draft", "critique", "final_post"]:
            old_val = state.get(key)
            new_val = result.get(key)
            if old_val is None and new_val is not None:
                changes.append(f"+{key}")
            elif new_val != old_val and new_val is not None:
                changes.append(f"~{key}")

        logger.info(
            "✔ [NODE] node=%s done in %.0fms | changes=%s",
            node_name, elapsed_ms, ", ".join(changes) or "none",
        )

        # Track latency in metadata
        metadata: dict[str, Any] = dict(result.get("metadata", {}))
        metadata[f"{node_name}_latency_ms"] = round(elapsed_ms, 1)
        wall_end = datetime.now(timezone.utc)
        try:
            from faults.injector import observation
            from storage.run_store import add_observation, append_event as _ae

            _ae(str(run_id), "node_completed", {"node": node_name, "latency_ms": round(elapsed_ms, 1)})
            if node_name != "runtime_probe":
                add_observation(
                    str(run_id),
                    observation(
                        name=node_name,
                        agent_name=node_name,
                        start=wall_start,
                        end=wall_end,
                        input_value={
                            "run_id": run_id,
                            "revision": revision,
                            "node": node_name,
                        },
                        output_value={"changes": changes},
                        extra_metadata={
                            "reads_state_keys": ["topic"] if node_name == "planner" else ["draft"],
                            "writes_state_keys": [node_name],
                        },
                    ),
                )
        except Exception as persist_exc:  # noqa: BLE001
            logger.debug("Could not persist node observation for %s: %s", node_name, persist_exc)
        return {**result, "metadata": metadata}

    return wrapper


def log_graph_start(state: BlogState) -> None:
    """Log graph execution start."""
    logger.info(
        "═══════════════════════════════════════════════════\n"
        "  BLOG GENERATOR GRAPH START\n"
        "  Run ID : %s\n"
        "  Topic  : %s\n"
        "═══════════════════════════════════════════════════",
        state.get("run_id", "?"),
        state.get("topic", "?"),
    )


def log_graph_end(state: BlogState) -> None:
    """Log graph execution end with summary."""
    metadata = state.get("metadata", {})
    logger.info(
        "═══════════════════════════════════════════════════\n"
        "  BLOG GENERATOR GRAPH END\n"
        "  Run ID    : %s\n"
        "  Revisions : %d\n"
        "  Approved  : %s\n"
        "  Words     : %s\n"
        "  Errors    : %d\n"
        "═══════════════════════════════════════════════════",
        state.get("run_id", "?"),
        state.get("revision_count", 0),
        state.get("is_approved", False),
        len(state.get("final_post", "").split()),
        len(state.get("error_logs", [])),
    )
