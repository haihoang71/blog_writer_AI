"""
graph/router.py
Routing logic for conditional edges in the LangGraph workflow.
"""

from __future__ import annotations

import logging

from config.constants import MAX_REVISIONS
from state.blog_state import BlogState

logger = logging.getLogger(__name__)


NODE_INPUT_GUARD = "input_guard"

# Runtime probe vừa tạo baseline trace, vừa là điểm cắm fault injection.
NODE_RUNTIME_PROBE = "runtime_probe"

# Tên tương thích với workflow cũ.
NODE_FAULT_INJECTION = NODE_RUNTIME_PROBE

NODE_PLANNER = "planner"
NODE_RESEARCHER = "researcher"
NODE_ACADEMIC_RESEARCHER = "academic_researcher"
NODE_WRITER = "writer"
NODE_CRITIC = "critic"
NODE_HUMAN_REVIEW = "human_review"
NODE_OUTPUT_GUARD = "output_guard"
NODE_END = "__end__"
NODE_BLOCKED = "blocked"


def route_after_input_guard(state: BlogState) -> str:
    """
    Route sau input guardrail.

    Topic bị block  → kết thúc.
    Topic hợp lệ    → runtime_probe.
    """
    sanitised = state.get("sanitised_topic", "")
    error_logs = state.get("error_logs", [])

    for log in reversed(error_logs):
        if log.node == "input_guard" and not log.recoverable:
            logger.info("Router: input blocked — routing to end.")
            return NODE_END

    if not sanitised:
        logger.warning(
            "Router: no sanitised topic after input guard — routing to end."
        )
        return NODE_END

    logger.info("Router: input accepted — routing to runtime_probe.")
    return NODE_RUNTIME_PROBE


def route_after_researcher(state: BlogState) -> str:
    """
    Route sau general researcher.
    """
    from agents.academic_researcher import is_academic_topic

    if is_academic_topic(state):
        logger.info(
            "Router: topic looks academic → routing to academic_researcher."
        )
        return NODE_ACADEMIC_RESEARCHER

    logger.info(
        "Router: topic not academic-heavy → routing directly to writer."
    )
    return NODE_WRITER


def route_after_critic(state: BlogState) -> str:
    """
    Draft được approve hoặc đạt số revision tối đa → human review.
    Ngược lại → writer revision.
    """
    is_approved: bool = state.get("is_approved", False)
    revision_count: int = state.get("revision_count", 0)

    if is_approved:
        logger.info(
            "Router: draft approved (revision %d) → HITL review.",
            revision_count,
        )
        return NODE_HUMAN_REVIEW

    if revision_count >= MAX_REVISIONS:
        logger.info(
            "Router: max revisions (%d) reached → force HITL review.",
            MAX_REVISIONS,
        )
        return NODE_HUMAN_REVIEW

    logger.info(
        "Router: draft not approved (revision %d/%d) → writer revision.",
        revision_count,
        MAX_REVISIONS,
    )
    return NODE_WRITER


def route_after_human_review(state: BlogState) -> str:
    """
    Human reject → writer.
    Human approve hoặc không có feedback → output guard.
    """
    human_feedback: str = state.get("human_feedback", "").strip().lower()

    rejection_keywords = {"reject", "redo", "revise", "no", "rewrite"}

    if any(keyword in human_feedback for keyword in rejection_keywords):
        logger.info("Router: human rejected draft → routing to writer.")
        return NODE_WRITER

    logger.info("Router: human approved (or no feedback) → output guard.")
    return NODE_OUTPUT_GUARD