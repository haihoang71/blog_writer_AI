"""
graph/workflow.py
──────────────────
LangGraph StateGraph definition for the Multi-Agent Blog Generator.

Graph Architecture
------------------

  [START]
     │
     ▼
  input_guard ──(blocked)──► [END]
     │ (allowed)
     ▼
  planner
     │
     ▼
  researcher
     │
     ├─(academic-heavy topic)──► academic_researcher ──┐
     │                                                  │
     ▼ (otherwise)                                      │
  writer ◄──────────────────────────────────────────────┐
     │                                                   │
     ▼                                                   │
  critic                                                 │
     │                                                   │
     ├─(not approved & revisions left)───────────────────┘
     │
     ▼ (approved OR max revisions)
  human_review  ◄──(HITL interrupt)──┐
     │                               │
     ├─(human rejects)───────────────┘
     │
     ▼ (human approves)
  output_guard
     │
     ▼
  [END]

Key features:
- ``researcher`` conditionally routes to ``academic_researcher`` first (for
  topics that are heavily academic/scientific) before reaching ``writer``;
  otherwise it goes straight to ``writer``. Either way, only one unified
  ``research_data`` reaches the Writer.
- The Human-in-the-Loop node uses a dynamic ``interrupt()`` call (not
  ``interrupt_before``) to pause execution for HITL.
- All nodes are wrapped with logging middleware.
- Guardrail nodes are standalone nodes, not just router functions.
- MemorySaver enables graph state checkpointing.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from agents.academic_researcher import academic_researcher_node
from agents.critic import critic_node
from agents.planner import planner_node
from agents.researcher import researcher_node
from agents.writer import writer_node
from config.constants import MAX_REVISIONS
from config.settings import get_settings
from graph.middlewares import log_graph_end, log_graph_start, wrap_node
from graph.router import (
    NODE_ACADEMIC_RESEARCHER,
    NODE_CRITIC,
    NODE_END,
    NODE_HUMAN_REVIEW,
    NODE_INPUT_GUARD,
    NODE_OUTPUT_GUARD,
    NODE_PLANNER,
    NODE_RESEARCHER,
    NODE_WRITER,
    route_after_critic,
    route_after_human_review,
    route_after_input_guard,
    route_after_researcher,
)
from guardrails.hallucination_guard import check_hallucination
from guardrails.input_guard import GuardResult, check_input
from guardrails.output_guard import sanitise_output
from state.blog_state import BlogState, ErrorLog

logger = logging.getLogger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Guardrail Nodes (wrapped as LangGraph nodes)
# ─────────────────────────────────────────────────────────────────────────────


def input_guard_node(state: BlogState) -> BlogState:
    """
    Input guardrail node.

    Validates and sanitises the topic. If blocked, adds an error log
    entry with ``recoverable=False`` and clears ``sanitised_topic``.
    """
    topic: str = state.get("topic", "")
    error_logs: list[ErrorLog] = list(state.get("error_logs", []))

    logger.info("Input guard: validating topic %r", topic[:80])
    result: GuardResult = check_input(topic)

    if result.decision == "block":
        logger.warning(
            "Input guard: BLOCKED — %s | issues: %s",
            result.reason, result.detected_issues,
        )
        error_logs.append(
            ErrorLog(
                node=NODE_INPUT_GUARD,
                error_type="InputGuardBlock",
                message=f"Blocked: {result.reason}",
                recoverable=False,
            )
        )
        return {
            **state,
            "sanitised_topic": "",
            "error_logs": error_logs,
        }

    if result.decision == "clarify":
        logger.info("Input guard: CLARIFY — treating as allow. %s", result.reason)

    sanitised = result.sanitised_topic or topic
    logger.info("Input guard: ALLOWED — sanitised topic: %r", sanitised)
    return {**state, "sanitised_topic": sanitised, "error_logs": error_logs}


def _make_human_review_node(enable_hitl: bool) -> Any:
    """
    Build the Human-in-the-Loop node, closing over *enable_hitl*.

    Uses LangGraph ``interrupt()`` to pause execution and surface the
    current draft to a human reviewer. Resumes when the graph is invoked
    again with ``Command(resume=...)``.

    NOTE: this used to read the module-level ``settings.enable_hitl`` (a
    global .env flag) instead of the ``enable_hitl`` argument that callers
    actually pass to ``build_graph()`` (the CLI's ``--hitl/--no-hitl`` flag,
    or the API's ``enable_hitl`` request field). That meant a caller asking
    for ``--no-hitl`` could still get silently stuck waiting on
    ``interrupt()`` whenever ``ENABLE_HITL=true`` in ``.env`` (the shipped
    default), because the node's decision to interrupt and the graph's
    caller-facing "did we pause?" checks disagreed. Building the node as a
    closure over the same flag used everywhere else keeps them consistent.
    """

    def human_review_node(state: BlogState) -> BlogState:
        draft: str = state.get("draft", "")
        revision_count: int = state.get("revision_count", 0)
        critique = state.get("critique")

        logger.info("Human review node: pausing for HITL (revision %d).", revision_count)

        if enable_hitl:
            # LangGraph interrupt — execution pauses here until resumed
            human_input = interrupt(
                {
                    "message": "Please review the blog post draft and provide feedback.",
                    "draft_preview": draft[:500] + ("..." if len(draft) > 500 else ""),
                    "revision_count": revision_count,
                    "critique_summary": (
                        critique.summary_feedback if critique else "No critique available."
                    ),
                    "instructions": (
                        "Resume with: {'human_feedback': '<your feedback or APPROVE>'}\n"
                        "Use 'reject', 'revise', or 'redo' to send back for revision.\n"
                        "Use 'approve' or leave empty to publish."
                    ),
                }
            )
            feedback = human_input if isinstance(human_input, str) else str(human_input)
        else:
            # HITL disabled — auto-approve
            feedback = "approve"
            logger.info("HITL disabled — auto-approving draft.")

        return {**state, "human_feedback": feedback}

    return human_review_node


def output_guard_node(state: BlogState) -> BlogState:
    """
    Output guardrail node.

    Runs PII redaction, Markdown validation, and hallucination check.
    Writes the cleaned text to ``state["final_post"]``.
    """
    draft: str = state.get("draft", "")
    research = state.get("research_data")
    error_logs: list[ErrorLog] = list(state.get("error_logs", []))
    metadata: dict[str, Any] = dict(state.get("metadata", {}))

    logger.info("Output guard: sanitising final draft.")

    # ── PII redaction + Markdown validation ───────────────────────────────
    guard_result = sanitise_output(draft)

    if not guard_result.passed:
        logger.warning(
            "Output guard: issues found — %s (proceeding with best-effort output)",
            guard_result.issues,
        )
        error_logs.append(
            ErrorLog(
                node=NODE_OUTPUT_GUARD,
                error_type="OutputGuardWarning",
                message=f"Issues: {guard_result.issues}",
                recoverable=True,
            )
        )

    metadata["output_pii_redacted"] = guard_result.pii_items_redacted
    metadata["output_word_count"] = guard_result.word_count
    metadata["output_guard_issues"] = guard_result.issues

    # ── Hallucination check ───────────────────────────────────────────────
    if research:
        hall_report = check_hallucination(
            draft=guard_result.clean_text,
            research_data_dict=research.model_dump(),
        )
        metadata["faithfulness_score"] = hall_report.overall_faithfulness_score
        metadata["flagged_claims"] = len(hall_report.flagged_sentences)

        if not hall_report.passed:
            logger.warning(
                "Output guard: hallucination check failed — faithfulness=%.2f",
                hall_report.overall_faithfulness_score,
            )

    logger.info(
        "Output guard: final post ready — %d words, %d PII redacted.",
        guard_result.word_count, guard_result.pii_items_redacted,
    )

    return {
        **state,
        "final_post": guard_result.clean_text,
        "error_logs": error_logs,
        "metadata": metadata,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Graph Builder
# ─────────────────────────────────────────────────────────────────────────────


def build_graph(enable_hitl: bool = True) -> Any:
    """
    Construct and compile the LangGraph StateGraph.

    Parameters
    ----------
    enable_hitl:
        When True, the ``human_review`` node will use ``interrupt()``
        to pause execution for human input. Set False for fully automated runs.

    Returns
    -------
    CompiledGraph
        The compiled, executable graph with checkpointing enabled.
    """
    graph = StateGraph(BlogState)

    # ── Register nodes (all wrapped with logging middleware) ───────────────
    graph.add_node(
        NODE_INPUT_GUARD,
        wrap_node(NODE_INPUT_GUARD, input_guard_node),
    )
    graph.add_node(
        NODE_PLANNER,
        wrap_node(NODE_PLANNER, planner_node),
    )
    graph.add_node(
        NODE_RESEARCHER,
        wrap_node(NODE_RESEARCHER, researcher_node),
    )
    graph.add_node(
        NODE_ACADEMIC_RESEARCHER,
        wrap_node(NODE_ACADEMIC_RESEARCHER, academic_researcher_node),
    )
    graph.add_node(
        NODE_WRITER,
        wrap_node(NODE_WRITER, writer_node),
    )
    graph.add_node(
        NODE_CRITIC,
        wrap_node(NODE_CRITIC, critic_node),
    )
    graph.add_node(
        NODE_HUMAN_REVIEW,
        wrap_node(NODE_HUMAN_REVIEW, _make_human_review_node(enable_hitl)),
    )
    graph.add_node(
        NODE_OUTPUT_GUARD,
        wrap_node(NODE_OUTPUT_GUARD, output_guard_node),
    )

    # ── Static edges ───────────────────────────────────────────────────────
    graph.add_edge(START, NODE_INPUT_GUARD)
    graph.add_edge(NODE_PLANNER, NODE_RESEARCHER)
    graph.add_edge(NODE_ACADEMIC_RESEARCHER, NODE_WRITER)
    graph.add_edge(NODE_WRITER, NODE_CRITIC)
    graph.add_edge(NODE_OUTPUT_GUARD, END)

    # ── Conditional edges ──────────────────────────────────────────────────
    graph.add_conditional_edges(
        NODE_INPUT_GUARD,
        route_after_input_guard,
        {
            NODE_PLANNER: NODE_PLANNER,
            NODE_END: END,
        },
    )

    graph.add_conditional_edges(
        NODE_RESEARCHER,
        route_after_researcher,
        {
            NODE_ACADEMIC_RESEARCHER: NODE_ACADEMIC_RESEARCHER,
            NODE_WRITER: NODE_WRITER,
        },
    )

    graph.add_conditional_edges(
        NODE_CRITIC,
        route_after_critic,
        {
            NODE_HUMAN_REVIEW: NODE_HUMAN_REVIEW,
            NODE_WRITER: NODE_WRITER,
        },
    )

    graph.add_conditional_edges(
        NODE_HUMAN_REVIEW,
        route_after_human_review,
        {
            NODE_OUTPUT_GUARD: NODE_OUTPUT_GUARD,
            NODE_WRITER: NODE_WRITER,
        },
    )

    # ── Compile with checkpointing ─────────────────────────────────────────
    # NOTE: no `interrupt_before` here. The human_review node itself calls
    # LangGraph's dynamic `interrupt()` (see `_make_human_review_node`) when
    # `enable_hitl` is true, which is the mechanism that actually pauses
    # execution and produces the `__interrupt__` payload callers poll for.
    # Statically gating the same node via `interrupt_before` as well used to
    # stop the graph *before* the node ran at all, so the dynamic interrupt()
    # call — and its rich payload — never fired on the first pass.
    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)

    logger.info(
        "Graph compiled: HITL=%s, MAX_REVISIONS=%d", enable_hitl, MAX_REVISIONS
    )
    return compiled


# ── Module-level singleton graph ───────────────────────────────────────────
_graph_instance: Any = None


def get_graph(enable_hitl: bool | None = None) -> Any:
    """Return cached graph instance or build a new one."""
    global _graph_instance
    hitl = settings.enable_hitl if enable_hitl is None else enable_hitl
    if _graph_instance is None:
        _graph_instance = build_graph(enable_hitl=hitl)
    return _graph_instance


def reset_graph() -> None:
    """Force rebuild of the graph singleton (e.g., after config change)."""
    global _graph_instance
    _graph_instance = None
