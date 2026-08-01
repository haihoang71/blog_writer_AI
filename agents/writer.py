"""
agents/writer.py
─────────────────
Writer Agent Node — drafts and revises the blog post in Markdown.

This node:
1. Loads the writer system prompt (includes critique if this is a revision).
2. Calls the LLM to generate a full Markdown blog post.
3. Increments ``revision_count`` in state.
4. Updates ``state["draft"]`` with the new text.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config.constants import (
    MAX_REVISIONS,
    WRITER_MAX_TOKENS,
    MAX_LLM_RETRIES,
)
from config.settings import get_settings
from prompts.loader import load_prompt
from state.blog_state import (
    BlogCritique,
    BlogOutline,
    BlogState,
    ErrorLog,
    ResearchData,
)

logger = logging.getLogger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# LLM Factory
# ─────────────────────────────────────────────────────────────────────────────


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        openai_api_base=settings.openai_api_base or None,
        temperature=0.7,           # slightly higher for creative writing
        max_tokens=WRITER_MAX_TOKENS,
        api_key=settings.openai_api_key.get_secret_value(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Critique Serialisation
# ─────────────────────────────────────────────────────────────────────────────


def _format_critique(critique: BlogCritique | None) -> str:
    """Convert a BlogCritique to a readable string for the prompt."""
    if critique is None:
        return "No critique yet — this is the first draft."

    lines: list[str] = [
        f"Overall Score: {critique.overall_score}/5",
        f"Summary: {critique.summary_feedback}",
        "",
        "Issues to Address:",
    ]
    for issue in critique.critical_issues:
        lines.append(
            f"  [{issue.severity.upper()}] Section '{issue.section}': "
            f"{issue.issue} → Suggestion: {issue.suggestion}"
        )
    if critique.positive_aspects:
        lines.append("\nPositive Aspects (keep these):")
        lines.extend(f"  + {p}" for p in critique.positive_aspects)

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Mock Draft (keyless fallback)
# ─────────────────────────────────────────────────────────────────────────────


def _mock_draft(
    topic: str,
    outline: BlogOutline,
    research: ResearchData,
    revision_count: int,
) -> str:
    """Return a structured mock Markdown draft."""
    logger.info("[MOCK] Generating mock draft — revision %d", revision_count)

    sections_md = ""
    for sec in outline.sections:
        sections_md += f"\n## {sec.title}\n\n"
        sections_md += f"{sec.purpose} This section explores the key aspects of {sec.title}.\n\n"

        # Add relevant findings
        for rsec in research.sections:
            if rsec.section_id == sec.id:
                for finding in rsec.findings[:2]:
                    sections_md += f"{finding.claim}\n\n"
                for snippet in rsec.code_snippets[:1]:
                    sections_md += f"Here is an example implementation:\n\n"
                    sections_md += f"```{snippet.language}\n{snippet.code}\n```\n\n"

    return f"""# {outline.title}

> **TL;DR** — This is a mock blog post generated without live API keys [1].
> In production, GPT-4o will produce a full technical article based on real research.
> Configure your `.env` file with valid API keys to enable live generation.

## Introduction

Welcome to this comprehensive guide on **{topic}** [1]. This article covers the fundamentals,
practical implementation, and best practices that every modern developer should know.
Whether you're just getting started or looking to deepen your expertise, this guide has
something valuable for you.

In today's rapidly evolving technology landscape, {topic} has become increasingly important.
Understanding it can significantly impact the quality and performance of the systems you build [2].
{sections_md}
## Key Takeaways

- {topic} is a critical concept in modern software development
- Understanding the core principles helps build more reliable systems
- Practical implementation requires attention to both theory and practice
- The community continues to evolve best practices rapidly
- Always measure and validate performance in your specific context [3]

## References

1. [MOCK Source — Configure OpenAI API Key for live generation](https://openai.com)
2. [Official Documentation](https://docs.example.com)
3. [Community Best Practices](https://github.com/example/best-practices)
"""


# ─────────────────────────────────────────────────────────────────────────────
# Agent Node
# ─────────────────────────────────────────────────────────────────────────────


def writer_node(state: BlogState) -> BlogState:
    """
    LangGraph node: Writer Agent.

    On first pass, writes a new draft from outline + research.
    On subsequent passes (revision_count > 0), incorporates critique feedback.

    Reads:  ``outline``, ``research_data``, ``critique``, ``revision_count``
    Writes: ``draft``, ``revision_count``
    """
    topic: str = state.get("sanitised_topic") or state.get("topic", "")
    outline: BlogOutline | None = state.get("outline")
    research: ResearchData | None = state.get("research_data")
    critique: BlogCritique | None = state.get("critique")
    revision_count: int = state.get("revision_count", 0)
    error_logs: list[ErrorLog] = list(state.get("error_logs", []))
    metadata: dict[str, Any] = dict(state.get("metadata", {}))

    logger.info(
        "Writer node — topic=%r revision=%d", topic, revision_count
    )

    if outline is None:
        logger.error("Writer: no outline in state — skipping.")
        return state

    if research is None:
        logger.warning("Writer: no research data — will write from outline only.")

    # ── Fallback if OpenAI not configured ────────────────────────────────
    if not settings.is_openai_configured:
        draft = _mock_draft(
            topic, outline, research or ResearchData(), revision_count
        )
        metadata["writer_mode"] = "mock"
        return {
            **state,
            "draft": draft,
            "revision_count": revision_count + 1,
            "metadata": metadata,
        }

    # ── Build prompt ──────────────────────────────────────────────────────
    outline_json = outline.model_dump_json(indent=2)
    research_json = research.model_dump_json(indent=2) if research else "{}"
    critique_str = _format_critique(critique)

    system_prompt = load_prompt(
        "writer",
        topic=topic,
        outline_json=outline_json,
        research_json=research_json,
        critique=critique_str,
        revision_count=revision_count,
        max_revisions=MAX_REVISIONS,
    )

    action_verb = "revise" if revision_count > 0 else "write"
    human_msg = (
        f"Please {action_verb} the full Markdown blog post for '{topic}'. "
        "Follow all formatting rules in the system prompt exactly. "
        "Return ONLY the Markdown — no explanations, no preamble."
    )

    # ── LLM call ──────────────────────────────────────────────────────────
    try:
        llm = _build_llm()

        @retry(
            stop=stop_after_attempt(MAX_LLM_RETRIES),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            reraise=True,
        )
        def _call() -> str:
            response = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=human_msg)]
            )
            return response.content

        draft = _call()
        logger.info(
            "Writer: produced %d-word draft (revision %d)",
            len(draft.split()), revision_count + 1,
        )
        metadata["writer_mode"] = "live"
        metadata[f"draft_word_count_rev{revision_count + 1}"] = len(draft.split())

    except Exception as exc:  # noqa: BLE001
        logger.error("Writer failed: %s — using mock draft.", exc)
        error_logs.append(
            ErrorLog(
                node="writer",
                error_type=type(exc).__name__,
                message=str(exc),
                recoverable=True,
            )
        )
        draft = _mock_draft(
            topic, outline, research or ResearchData(), revision_count
        )
        metadata["writer_mode"] = "mock_fallback"

    return {
        **state,
        "draft": draft,
        "revision_count": revision_count + 1,
        "error_logs": error_logs,
        "metadata": metadata,
    }
