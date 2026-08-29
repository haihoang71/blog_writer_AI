"""HITL shares a checkpointer; graph invoke honors the wall-clock timeout."""

from __future__ import annotations

import time
from uuid import uuid4

import pytest

from services.generation import _invoke_with_timeout, extract_interrupt_payload


@pytest.mark.integration
def test_hitl_resume_uses_same_checkpointer() -> None:
    from config.tracing import build_run_config
    from graph.workflow import get_graph
    from langgraph.types import Command
    from state.blog_state import initial_state

    graph = get_graph(enable_hitl=True)
    thread_id = f"hitl-{uuid4()}"
    state = initial_state("Python asyncio event loop internals")
    state["metadata"] = {"fault_scenario": "none", "task_id": thread_id}
    config = build_run_config(run_name="hitl-test", tags=["test"])
    config["configurable"] = {"thread_id": thread_id}

    paused = graph.invoke(state, config=config)
    interrupt = extract_interrupt_payload(paused)
    assert interrupt is not None

    resumed = graph.invoke(Command(resume="approve"), config=config)
    assert extract_interrupt_payload(resumed) is None
    assert resumed.get("final_post") or resumed.get("draft")


@pytest.mark.unit
def test_graph_timeout_raises() -> None:
    class Slow:
        def invoke(self, *_args, **_kwargs):
            time.sleep(1.5)
            return {}

    with pytest.raises(TimeoutError):
        _invoke_with_timeout(Slow(), {}, {}, timeout=0.05)
