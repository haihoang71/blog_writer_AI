"""Run the LangGraph pipeline with timeout, persistence, and HITL resume."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from datetime import datetime, timezone
from typing import Any, Optional

from config.settings import get_settings
from config.tracing import build_run_config, flush_langfuse
from faults.injector import clear_scenario, register_scenario
from faults.scenarios import parse_scenario
from graph.workflow import get_graph
from state.blog_state import initial_state
from storage.post_manager import save_post
from storage.run_store import append_event, get_run, upsert_run

logger = logging.getLogger(__name__)

_IN_FLIGHT: dict[str, dict[str, Any]] = {}


def extract_interrupt_payload(result: dict) -> Optional[dict]:
    """Unwrap LangGraph ``__interrupt__`` into a plain dict."""
    raw = result.get("__interrupt__")
    if not raw:
        return None
    item = raw[0] if isinstance(raw, (list, tuple)) else raw
    return getattr(item, "value", item)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _langfuse_trace_id() -> str | None:
    try:
        from langfuse import get_client

        return get_client().get_current_trace_id()
    except Exception:  # noqa: BLE001
        return None


def resolve_execution_mode(result: dict[str, Any] | None) -> str:
    """Prefer planner_mode from state; otherwise infer from settings."""
    metadata = (result or {}).get("metadata") or {}
    mode = metadata.get("planner_mode")
    if mode in {"mock", "live", "mock_fallback"}:
        return str(mode)
    settings = get_settings()
    return "live" if settings.is_openai_configured else "mock"


def cache_task(task_id: str, payload: dict[str, Any]) -> None:
    """Keep HITL resume config in process memory (MemorySaver is in-memory)."""
    current = _IN_FLIGHT.get(task_id, {})
    current.update(payload)
    _IN_FLIGHT[task_id] = current


def get_cached_task(task_id: str) -> dict[str, Any] | None:
    """Return the in-process task overlay, if any."""
    return _IN_FLIGHT.get(task_id)


def _persist_final_post(result: dict[str, Any]) -> dict[str, Any]:
    final_post = result.get("final_post") or result.get("draft", "")
    record = None
    if final_post:
        try:
            record = save_post(result)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to save post: %s", exc)
    return {
        "final_post": final_post,
        "revision_count": result.get("revision_count", 0),
        "is_approved": result.get("is_approved", False),
        "word_count": len((final_post or "").split()),
        "metadata": result.get("metadata", {}),
        "error_count": len(result.get("error_logs", [])),
        "post_id": record["id"] if record else None,
    }


def _invoke_with_timeout(graph: Any, payload: Any, config: dict[str, Any], timeout: int) -> dict[str, Any]:
    # A context-manager executor waits for a running worker during __exit__,
    # defeating the request timeout. Shutdown without waiting so the API can
    # report timeout promptly. Python cannot safely kill an already-running
    # thread; the graph worker may finish in the background and its Langfuse
    # spans are still flushed by the worker when it returns.
    def invoke() -> dict[str, Any]:
        settings = get_settings()
        if settings.is_langfuse_configured and config.get("callbacks"):
            try:
                from langfuse import get_client

                client = get_client()
                context_manager = client.start_as_current_observation(
                    name=str(config.get("run_name") or "blog-generation"),
                    as_type="chain",
                    metadata=dict(config.get("metadata") or {}),
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Could not create Langfuse root context: %s", type(exc).__name__)
            else:
                # LangChain's v3 callback handler starts child observations,
                # but does not itself install a current OTEL span. Establish
                # one in the same worker thread so synthetic runtime-probe
                # spans can join the exact trace instead of becoming detached.
                with context_manager:
                    return graph.invoke(payload, config)
        return graph.invoke(payload, config)

    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(invoke)
    try:
        return future.result(timeout=timeout)
    except FuturesTimeout as exc:
        future.cancel()
        raise TimeoutError(f"graph invoke exceeded {timeout}s") from exc
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def start_generation(
    *,
    topic: str,
    enable_hitl: bool,
    tags: list[str],
    fault_scenario: str,
    task_id: str,
    run_id: str | None = None,
) -> str:
    """Create the run row and execute the graph (blocking — call from a worker)."""
    settings = get_settings()
    scenario = parse_scenario(fault_scenario)
    state = initial_state(topic)
    if run_id:
        state["run_id"] = run_id
    run_id = str(state["run_id"])
    metadata = dict(state.get("metadata") or {})
    metadata["task_id"] = task_id
    state["metadata"] = metadata

    mode = "live" if settings.is_openai_configured else "mock"
    upsert_run(
        {
            "run_id": run_id,
            "task_id": task_id,
            "topic": topic,
            "fault_scenario": scenario.value,
            "execution_mode": mode,
            "provider": "openai-compatible" if mode == "live" else "mock",
            "model": settings.openai_model,
            "status": "running",
            "enable_hitl": enable_hitl,
            "started_at": _now(),
            "metadata_json": {"tags": tags},
        }
    )
    append_event(run_id, "run_started", {"topic": topic, "fault_scenario": scenario.value})
    register_scenario(run_id, scenario)
    cache_task(
        task_id,
        {
            "run_id": run_id,
            "status": "running",
            "enable_hitl": enable_hitl,
            "result": None,
            "error": None,
        },
    )

    graph = get_graph(enable_hitl=enable_hitl)
    config = build_run_config(
        run_name=f"api-{topic[:30]}",
        tags=["api", "synthetic-ready"] + tags,
        metadata={"topic": topic, "task_id": task_id},
    )
    config["configurable"] = {"thread_id": task_id}
    cache_task(task_id, {"config": config, "enable_hitl": enable_hitl})

    try:
        result = _invoke_with_timeout(
            graph, state, config, settings.graph_timeout_seconds
        )
    except TimeoutError as exc:
        logger.exception("Generation timed out for run %s", run_id)
        upsert_run(
            {
                "run_id": run_id,
                "status": "timeout",
                "error": str(exc),
                "ended_at": _now(),
                "langfuse_trace_id": _langfuse_trace_id(),
            }
        )
        append_event(run_id, "run_timeout", {"error": str(exc)})
        cache_task(task_id, {"status": "timeout", "error": str(exc)})
        clear_scenario(run_id)
        flush_langfuse()
        return run_id
    except Exception as exc:  # noqa: BLE001
        logger.exception("Generation failed for run %s", run_id)
        upsert_run(
            {
                "run_id": run_id,
                "status": "failed",
                "error": str(exc),
                "ended_at": _now(),
                "langfuse_trace_id": _langfuse_trace_id(),
            }
        )
        append_event(run_id, "run_failed", {"error": type(exc).__name__})
        cache_task(task_id, {"status": "failed", "error": str(exc)})
        clear_scenario(run_id)
        flush_langfuse()
        return run_id

    return _finalise_result(
        task_id=task_id,
        run_id=run_id,
        result=result,
        enable_hitl=enable_hitl,
    )


def _finalise_result(
    *,
    task_id: str,
    run_id: str,
    result: dict[str, Any],
    enable_hitl: bool,
) -> str:
    trace_id = _langfuse_trace_id()
    mode = resolve_execution_mode(result)
    interrupt = extract_interrupt_payload(result)
    if enable_hitl and interrupt:
        upsert_run(
            {
                "run_id": run_id,
                "status": "paused",
                "execution_mode": mode,
                "langfuse_trace_id": trace_id,
            }
        )
        append_event(run_id, "hitl_paused", {"revision": interrupt.get("revision_count")})
        cache_task(
            task_id,
            {
                "status": "paused",
                "result": {"interrupt": interrupt},
                "run_id": run_id,
                "enable_hitl": True,
            },
        )
        flush_langfuse()
        return run_id

    summary = _persist_final_post(result)
    upsert_run(
        {
            "run_id": run_id,
            "status": "completed",
            "execution_mode": mode,
            "langfuse_trace_id": trace_id,
            "ended_at": _now(),
            "error": None,
        }
    )
    append_event(run_id, "run_completed", {"word_count": summary.get("word_count")})
    cache_task(
        task_id,
        {"status": "completed", "result": summary, "run_id": run_id, "error": None},
    )
    flush_langfuse()
    return run_id


def resume_review(task_id: str, feedback: str) -> dict[str, Any]:
    """Resume a paused HITL graph on the shared checkpointer."""
    from langgraph.types import Command

    cached = get_cached_task(task_id) or {}
    stored = get_run(task_id)
    run_id = cached.get("run_id") or (stored or {}).get("run_id")
    if not run_id:
        raise KeyError(task_id)
    enable_hitl = bool(cached.get("enable_hitl"))
    if stored is not None:
        enable_hitl = bool(stored.get("enable_hitl"))
    config = cached.get("config") or {"configurable": {"thread_id": task_id}}
    graph = get_graph(enable_hitl=enable_hitl)
    settings = get_settings()
    result = _invoke_with_timeout(
        graph, Command(resume=feedback), config, settings.graph_timeout_seconds
    )
    _finalise_result(
        task_id=task_id,
        run_id=str(run_id),
        result=result,
        enable_hitl=enable_hitl,
    )
    overlay = get_cached_task(task_id) or {}
    return {
        "task_id": task_id,
        "run_id": run_id,
        "status": overlay.get("status"),
        "result": overlay.get("result"),
    }


def status_payload(task_id: str) -> dict[str, Any] | None:
    """Merge in-memory overlay with the SQLite run row."""
    cached = get_cached_task(task_id)
    stored = get_run(task_id)
    if cached is None and stored is None:
        return None
    run_id = (cached or {}).get("run_id") or (stored or {}).get("run_id")
    return {
        "task_id": task_id,
        "run_id": run_id,
        "status": (cached or {}).get("status") or (stored or {}).get("status"),
        "result": (cached or {}).get("result"),
        "error": (cached or {}).get("error") or (stored or {}).get("error"),
        "execution_mode": (stored or {}).get("execution_mode"),
        "fault_scenario": (stored or {}).get("fault_scenario"),
        "langfuse_trace_id": (stored or {}).get("langfuse_trace_id"),
        "enable_hitl": (stored or {}).get("enable_hitl"),
    }
