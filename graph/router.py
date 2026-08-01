"""
graph/router.py
────────────────
Routing logic for conditional edges in the LangGraph workflow.

Routers are pure functions: they read the current ``BlogState`` and
return the name of the next node to visit.
"""

from __future__ import annotations

import logging

from config.constants import MAX_REVISIONS
from state.blog_state import BlogState

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Node Name Constants
# ─────────────────────────────────────────────────────────────────────────────

NODE_INPUT_GUARD = "input_guard"
NODE_PLANNER = "planner"
NODE_RESEARCHER = "researcher"
NODE_ACADEMIC_RESEARCHER = "academic_researcher"
NODE_WRITER = "writer"
NODE_CRITIC = "critic"
NODE_HUMAN_REVIEW = "human_review"
NODE_OUTPUT_GUARD = "output_guard"
NODE_END = "__end__"
NODE_BLOCKED = "blocked"


# ─────────────────────────────────────────────────────────────────────────────
# Routers
# ─────────────────────────────────────────────────────────────────────────────


def route_after_input_guard(state: BlogState) -> str:
    """
    Route after the input guardrail node.

    - If topic was blocked → end (blocked path)
    - Otherwise → proceed to Planner
    """
    sanitised = state.get("sanitised_topic", "")
    error_logs = state.get("error_logs", [])

    # Check if last error log is an input block
    for log in reversed(error_logs):
        if log.node == "input_guard" and not log.recoverable:
            logger.info("Router: input blocked — routing to end.")
            return NODE_END

    if not sanitised:
        # No sanitised topic — blocked
        logger.warning("Router: no sanitised topic after input guard — routing to end.")
        return NODE_END

    logger.info("Router: input accepted — routing to planner.")
    return NODE_PLANNER


def route_after_researcher(state: BlogState) -> str:
    """
    Route after the general Researcher node.

    - If the topic/outline is heavily academic/scientific (per
      ``agents.academic_researcher.is_academic_topic``) → also run the
      Academic Researcher node (arXiv search + citations) before Writer.
    - Otherwise → go straight to Writer, unchanged from before this feature.

    Imported lazily to avoid a module-import cycle (agents.* modules don't
    import graph.router, but importing at module scope here would still
    force agents.academic_researcher to load before graph.workflow finishes
    wiring — deferring the import keeps router.py import-order-agnostic).
    """
    from agents.academic_researcher import is_academic_topic

    if is_academic_topic(state):
        logger.info("Router: topic looks academic → also routing to academic_researcher.")
        return NODE_ACADEMIC_RESEARCHER

    logger.info("Router: topic not academic-heavy → routing directly to writer.")
    return NODE_WRITER


def route_after_critic(state: BlogState) -> str:
    """
    Route after the Critic node.

    Decision logic:
    - If ``is_approved == True`` OR ``revision_count >= MAX_REVISIONS``
      → Human review (HITL) node
    - Else → Writer node for revision
    """
    is_approved: bool = state.get("is_approved", False)
    revision_count: int = state.get("revision_count", 0)

    if is_approved:
        logger.info(
            "Router: draft approved (revision %d) → HITL review.", revision_count
        )
        return NODE_HUMAN_REVIEW

    if revision_count >= MAX_REVISIONS:
        logger.info(
            "Router: max revisions (%d) reached → force HITL review.", MAX_REVISIONS
        )
        return NODE_HUMAN_REVIEW

    logger.info(
        "Router: draft not approved (revision %d/%d) → writer revision.",
        revision_count, MAX_REVISIONS,
    )
    return NODE_WRITER


def route_after_human_review(state: BlogState) -> str:
    """
    Route after the Human-in-the-Loop review node.

    - If human rejected → Writer for another pass
    - Otherwise → Output guardrail
    """
    human_feedback: str = state.get("human_feedback", "").strip().lower()

    # Explicit rejection keywords
    rejection_keywords = {"reject", "redo", "revise", "no", "rewrite"}
    if any(kw in human_feedback for kw in rejection_keywords):
        logger.info("Router: human rejected draft → routing to writer.")
        return NODE_WRITER

    logger.info("Router: human approved (or no feedback) → output guard.")
    return NODE_OUTPUT_GUARD
