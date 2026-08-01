"""
agents/planner.py
──────────────────
Planner Agent Node — generates a structured blog post outline.

This node:
1. Loads the planner system prompt from YAML.
2. Calls the LLM to produce a JSON outline.
3. Validates the JSON into a ``BlogOutline`` Pydantic model.
4. Updates the graph state with the validated outline.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config.constants import (
    ALLOWED_DOMAINS,
    MAX_OUTLINE_SECTIONS,
    PLANNER_MAX_TOKENS,
    MAX_LLM_RETRIES,
)
from config.settings import get_settings
from prompts.loader import load_prompt
from state.blog_state import (
    BlogOutline,
    BlogState,
    ErrorLog,
    OutlineSection,
    SubSection,
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
        temperature=settings.openai_temperature,
        max_tokens=PLANNER_MAX_TOKENS,
        api_key=settings.openai_api_key.get_secret_value(),
        model_kwargs={"response_format": {"type": "json_object"}},
    )


# ─────────────────────────────────────────────────────────────────────────────
# JSON Parsing Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _parse_json_response(raw: str) -> dict:
    """Extract and parse JSON from LLM response, handling markdown fences."""
    raw = raw.strip()
    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding bare JSON object
    match = re.search(r"\{[\s\S]+\}", raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM response: {raw[:300]}")


def _build_outline_from_dict(data: dict) -> BlogOutline:
    """Construct a BlogOutline from parsed JSON dict."""
    sections: list[OutlineSection] = []
    for raw_section in data.get("sections", []):
        subsections: list[SubSection] = []
        for raw_sub in raw_section.get("subsections", []):
            subsections.append(
                SubSection(
                    id=raw_sub.get("id", ""),
                    title=raw_sub.get("title", ""),
                    purpose=raw_sub.get("purpose", ""),
                    key_points=raw_sub.get("key_points", []),
                    needs_code=raw_sub.get("needs_code", False),
                )
            )
        sections.append(
            OutlineSection(
                id=raw_section.get("id", ""),
                title=raw_section.get("title", ""),
                purpose=raw_section.get("purpose", ""),
                key_points=raw_section.get("key_points", []),
                needs_code=raw_section.get("needs_code", False),
                depth=raw_section.get("depth", "intermediate"),
                subsections=subsections,
            )
        )

    return BlogOutline(
        title=data.get("title", ""),
        slug=data.get("slug", ""),
        meta_description=data.get("meta_description", ""),
        target_audience=data.get("target_audience", ""),
        estimated_read_time_minutes=data.get("estimated_read_time_minutes", 0),
        sections=sections,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mock Outline (keyless fallback)
# ─────────────────────────────────────────────────────────────────────────────


def _mock_outline(topic: str) -> BlogOutline:
    """Return a structured mock outline when OpenAI is not configured."""
    logger.info("[MOCK] Generating mock outline for topic: %s", topic)
    slug = re.sub(r"\s+", "-", topic.lower())
    slug = re.sub(r"[^a-z0-9\-]", "", slug)

    return BlogOutline(
        title=f"A Comprehensive Guide to {topic}",
        slug=slug[:60],
        meta_description=f"Learn everything about {topic} — concepts, code examples, and best practices.",
        target_audience="Senior software engineers and ML practitioners",
        estimated_read_time_minutes=12,
        sections=[
            OutlineSection(
                id="s1",
                title="Introduction",
                purpose=f"Set the context for {topic} and explain why it matters.",
                key_points=[f"What is {topic}?", "Why it's important in 2024", "What you'll learn"],
                needs_code=False,
                depth="introductory",
                subsections=[
                    SubSection(
                        id="s1.1",
                        title="Background & Motivation",
                        purpose="Historical context and current landscape.",
                        key_points=["Origins", "Industry adoption"],
                        needs_code=False,
                    )
                ],
            ),
            OutlineSection(
                id="s2",
                title="Core Concepts",
                purpose=f"Explain the fundamental building blocks of {topic}.",
                key_points=["Key terminology", "Architecture overview", "Design principles"],
                needs_code=True,
                depth="intermediate",
                subsections=[
                    SubSection(
                        id="s2.1",
                        title="Architecture Deep Dive",
                        purpose="Detailed look at the internals.",
                        key_points=["Component breakdown", "Data flow"],
                        needs_code=True,
                    )
                ],
            ),
            OutlineSection(
                id="s3",
                title="Implementation Guide",
                purpose=f"Walk through a practical implementation of {topic}.",
                key_points=["Setup & installation", "Step-by-step code walkthrough", "Common pitfalls"],
                needs_code=True,
                depth="advanced",
                subsections=[
                    SubSection(
                        id="s3.1",
                        title="Getting Started",
                        purpose="Environment setup and dependencies.",
                        key_points=["Prerequisites", "Installation steps"],
                        needs_code=True,
                    ),
                    SubSection(
                        id="s3.2",
                        title="Production Considerations",
                        purpose="Scaling, monitoring, and reliability.",
                        key_points=["Performance tuning", "Observability"],
                        needs_code=False,
                    ),
                ],
            ),
            OutlineSection(
                id="s4",
                title="Best Practices & Conclusion",
                purpose="Summarise learnings and provide actionable next steps.",
                key_points=["Do's and don'ts", "Resources for further learning", "Summary"],
                needs_code=False,
                depth="intermediate",
                subsections=[],
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent Node
# ─────────────────────────────────────────────────────────────────────────────


def planner_node(state: BlogState) -> BlogState:
    """
    LangGraph node: Planner Agent.

    Reads ``state["sanitised_topic"]`` (or ``topic`` as fallback),
    calls the LLM to generate a structured outline, validates it,
    and writes it back to ``state["outline"]``.

    Parameters
    ----------
    state:
        Current ``BlogState``.

    Returns
    -------
    BlogState
        Updated state with ``outline`` populated.
    """
    topic = state.get("sanitised_topic") or state.get("topic", "")
    error_logs: list[ErrorLog] = list(state.get("error_logs", []))
    metadata: dict[str, Any] = dict(state.get("metadata", {}))

    logger.info("Planner node — topic: %r", topic)

    # ── Fallback if OpenAI not configured ────────────────────────────────
    if not settings.is_openai_configured:
        outline = _mock_outline(topic)
        metadata["planner_mode"] = "mock"
        return {**state, "outline": outline, "metadata": metadata}

    # ── Build prompt ──────────────────────────────────────────────────────
    system_prompt = load_prompt(
        "planner",
        allowed_domains=", ".join(ALLOWED_DOMAINS),
        max_sections=MAX_OUTLINE_SECTIONS,
    )

    human_msg = (
        f"Generate a comprehensive blog post outline for the following topic:\n\n"
        f"**Topic:** {topic}\n\n"
        f"Return ONLY the JSON object as specified. No additional text."
    )

    # ── LLM call with retry ───────────────────────────────────────────────
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

        raw = _call()
        data = _parse_json_response(raw)
        outline = _build_outline_from_dict(data)

        logger.info(
            "Planner: generated outline '%s' with %d sections",
            outline.title, len(outline.sections),
        )
        metadata["planner_mode"] = "live"
        metadata["outline_title"] = outline.title

    except Exception as exc:  # noqa: BLE001
            logger.error("Planner failed: %s — using mock outline.", exc)
            error_logs.append(
                ErrorLog(
                    node="planner",
                    error_type=type(exc).__name__,
                    message=str(exc),
                    recoverable=True,
                )
            )
            outline = _mock_outline(topic)
            metadata["planner_mode"] = "mock_fallback" 

    return {**state, "outline": outline, "error_logs": error_logs, "metadata": metadata}
