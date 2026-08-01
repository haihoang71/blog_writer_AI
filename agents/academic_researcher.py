"""
agents/academic_researcher.py
──────────────────────────────
Academic Researcher Agent Node — specialised research for topics that are
heavily related to academic/scientific research (arXiv-backed).

This node only runs conditionally: ``graph.router.route_after_researcher``
inspects the topic/outline and sends the run here (in addition to the
general ``agents.researcher`` node) only when the topic looks academic. It
never replaces the general Researcher — it augments it.

This node:
1. Decides whether a topic is "academic enough" to warrant arXiv research
   (``is_academic_topic`` — reused by the router so both agree).
2. Loads the academic_researcher system prompt from YAML.
3. Builds a tool-calling LLM bound to ``arxiv_search`` only.
4. Invokes the agent to gather citable findings for relevant outline
   sections.
5. Merges the resulting findings (with citations) into the existing
   ``state["research_data"]`` produced by the general Researcher node,
   so the Writer agent sees a single unified research report.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config.constants import (
    ACADEMIC_TOPIC_KEYWORDS,
    MAX_ACADEMIC_PAPERS,
    MAX_LLM_RETRIES,
    RESEARCHER_MAX_TOKENS,
)
from config.settings import get_settings
from prompts.loader import load_prompt
from state.blog_state import (
    BlogOutline,
    BlogState,
    ErrorLog,
    ResearchData,
    ResearchFinding,
    ResearchSection,
)
from tools.search_tools import arxiv_search

logger = logging.getLogger(__name__)
settings = get_settings()

_ACADEMIC_TOOLS = [arxiv_search]


# ─────────────────────────────────────────────────────────────────────────────
# Topic Classification
# ─────────────────────────────────────────────────────────────────────────────


def is_academic_topic(state: BlogState) -> bool:
    """
    Decide whether a topic/outline is heavily related to academic/scientific
    research and therefore warrants also invoking the Academic Researcher.

    Lightweight keyword heuristic (no extra LLM call): checks the sanitised
    topic plus outline titles/purposes/key_points against
    ``config.constants.ACADEMIC_TOPIC_KEYWORDS``. Shared by
    ``graph.router.route_after_researcher`` so the router and this node's
    own gating logic can never disagree.
    """
    topic: str = (state.get("sanitised_topic") or state.get("topic") or "").lower()
    outline: BlogOutline | None = state.get("outline")

    haystack_parts: list[str] = [topic]
    if outline is not None:
        haystack_parts.append(outline.title.lower())
        for sec in outline.sections:
            haystack_parts.append(sec.title.lower())
            haystack_parts.append(sec.purpose.lower())
            haystack_parts.extend(kp.lower() for kp in sec.key_points)

    haystack = " ".join(haystack_parts)
    return any(keyword in haystack for keyword in ACADEMIC_TOPIC_KEYWORDS)


# ─────────────────────────────────────────────────────────────────────────────
# LLM Factory
# ─────────────────────────────────────────────────────────────────────────────


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        openai_api_base=settings.openai_api_base or None,
        temperature=0.1,
        max_tokens=RESEARCHER_MAX_TOKENS,
        api_key=settings.openai_api_key.get_secret_value(),
    ).bind_tools(_ACADEMIC_TOOLS)


# ─────────────────────────────────────────────────────────────────────────────
# Parse Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _parse_json_response(raw: str) -> dict:
    """Extract JSON from LLM response (same tolerant strategy as researcher.py)."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    for pat in [r"```(?:json)?\s*([\s\S]+?)\s*```", r"\{[\s\S]+\}"]:
        match = re.search(pat, raw)
        if match:
            try:
                return json.loads(match.group(1) if "```" in pat else match.group())
            except json.JSONDecodeError:
                continue

    raise ValueError(f"Cannot parse JSON from academic researcher response: {raw[:300]}")


def _parse_academic_sections(data: dict) -> list[ResearchSection]:
    """Build ``ResearchSection`` list from the parsed JSON dict."""
    sections: list[ResearchSection] = []
    for raw_sec in data.get("sections", []):
        findings: list[ResearchFinding] = [
            ResearchFinding(
                claim=f.get("claim", ""),
                source_url=f.get("source_url", "internal_knowledge"),
                confidence=f.get("confidence", "high"),
                excerpt=f.get("excerpt", ""),
                citation=f.get("citation", ""),
            )
            for f in raw_sec.get("findings", [])
            if f.get("claim")
        ]
        if findings:
            sections.append(
                ResearchSection(section_id=raw_sec.get("section_id", ""), findings=findings)
            )
    return sections


# ─────────────────────────────────────────────────────────────────────────────
# Tool Execution (manual tool loop, mirrors agents/researcher.py)
# ─────────────────────────────────────────────────────────────────────────────


def _execute_tool_calls(tool_calls: list[dict]) -> list[dict]:
    results = []
    tool_map = {t.name: t for t in _ACADEMIC_TOOLS}

    for tc in tool_calls:
        tool_name = tc.get("name", "")
        tool_args = tc.get("args", {})
        tool_id = tc.get("id", "")

        if tool_name not in tool_map:
            logger.warning("Academic researcher: unknown tool requested: %s", tool_name)
            continue

        try:
            output = tool_map[tool_name].invoke(tool_args)
            results.append({"tool_call_id": tool_id, "name": tool_name, "content": output})
            logger.info(
                "Academic researcher: executed tool %s → %d chars output",
                tool_name, len(output),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Academic researcher: tool %s failed: %s", tool_name, exc)
            results.append(
                {"tool_call_id": tool_id, "name": tool_name, "content": f"Error: {exc}"}
            )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Merge Helper
# ─────────────────────────────────────────────────────────────────────────────


def _merge_into_research_data(
    base: ResearchData | None, extra_sections: list[ResearchSection], extra_takeaways: list[str]
) -> ResearchData:
    """
    Merge academic findings into the existing ``ResearchData`` (produced by
    the general Researcher node), keyed by ``section_id``. Never overwrites
    or drops the general Researcher's findings/snippets — only appends.
    """
    if base is None:
        base = ResearchData()

    by_section_id = {sec.section_id: sec for sec in base.sections}

    for extra_sec in extra_sections:
        existing = by_section_id.get(extra_sec.section_id)
        if existing is not None:
            existing.findings.extend(extra_sec.findings)
        else:
            base.sections.append(extra_sec)
            by_section_id[extra_sec.section_id] = extra_sec

    for takeaway in extra_takeaways:
        if takeaway not in base.key_takeaways:
            base.key_takeaways.append(takeaway)

    return base


# ─────────────────────────────────────────────────────────────────────────────
# Agent Node
# ─────────────────────────────────────────────────────────────────────────────


def academic_researcher_node(state: BlogState) -> BlogState:
    """
    LangGraph node: Academic Researcher Agent.

    Runs only on the conditional edge from ``researcher`` when
    ``graph.router.route_after_researcher`` decides the topic is academic.
    Searches arXiv, produces citable findings, and merges them into the
    existing ``state["research_data"]`` before handing off to the Writer.
    """
    outline: BlogOutline | None = state.get("outline")
    topic: str = state.get("sanitised_topic") or state.get("topic", "")
    error_logs: list[ErrorLog] = list(state.get("error_logs", []))
    metadata: dict[str, Any] = dict(state.get("metadata", {}))

    if outline is None:
        logger.error("Academic researcher: no outline in state — skipping.")
        return state

    logger.info("Academic researcher node — topic=%r", topic)

    # ── Fallback if OpenAI not configured — still worth a keyless arXiv pass ──
    if not settings.is_openai_configured:
        merged = _merge_into_research_data(
            state.get("research_data"),
            _mock_academic_sections(outline),
            ["[MOCK] Academic backing generated without live API keys."],
        )
        metadata["academic_researcher_mode"] = "mock"
        return {**state, "research_data": merged, "metadata": metadata}

    outline_json = outline.model_dump_json(indent=2)
    system_prompt = load_prompt(
        "academic_researcher",
        topic=topic,
        outline_json=outline_json,
        max_papers=MAX_ACADEMIC_PAPERS,
    )

    human_msg = (
        "Find real arXiv papers that back up the academic/technical claims "
        "this blog post outline will need. Use the arxiv_search tool. "
        "Return your findings as the JSON object specified — only real, "
        "tool-backed papers and citations."
    )

    try:
        llm = _build_llm()
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_msg)]

        max_iterations = MAX_ACADEMIC_PAPERS + 2

        @retry(
            stop=stop_after_attempt(MAX_LLM_RETRIES),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            reraise=True,
        )
        def _call(msgs):
            return llm.invoke(msgs)

        for _ in range(max_iterations):
            response = _call(messages)
            messages.append(response)

            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_results = _execute_tool_calls(response.tool_calls)
                for tr in tool_results:
                    messages.append(
                        ToolMessage(content=tr["content"], tool_call_id=tr["tool_call_id"])
                    )
            else:
                raw = response.content
                break
        else:
            raw = response.content

        data = _parse_json_response(raw)
        extra_sections = _parse_academic_sections(data)
        extra_takeaways = data.get("key_takeaways", [])

        merged = _merge_into_research_data(
            state.get("research_data"), extra_sections, extra_takeaways
        )

        logger.info(
            "Academic researcher: added %d sections' worth of citable findings",
            len(extra_sections),
        )
        metadata["academic_researcher_mode"] = "live"
        metadata["academic_findings_added"] = sum(len(s.findings) for s in extra_sections)

    except Exception as exc:  # noqa: BLE001
        logger.error("Academic researcher failed: %s — using mock academic data.", exc)
        error_logs.append(
            ErrorLog(
                node="academic_researcher",
                error_type=type(exc).__name__,
                message=str(exc),
                recoverable=True,
            )
        )
        merged = _merge_into_research_data(
            state.get("research_data"),
            _mock_academic_sections(outline),
            ["[MOCK] Academic backing generated after a live-mode failure."],
        )
        metadata["academic_researcher_mode"] = "mock_fallback"

    return {**state, "research_data": merged, "error_logs": error_logs, "metadata": metadata}


# ─────────────────────────────────────────────────────────────────────────────
# Mock Academic Research (keyless / no-LLM fallback)
# ─────────────────────────────────────────────────────────────────────────────


def _mock_academic_sections(outline: BlogOutline) -> list[ResearchSection]:
    """Synthetic academic findings, still routed through arxiv_search's mock
    path so the ``citation`` field is populated realistically."""
    sections: list[ResearchSection] = []
    for sec in outline.sections[:2]:
        raw = arxiv_search.invoke({"query": sec.title, "max_results": 1})
        try:
            papers = json.loads(raw)
        except json.JSONDecodeError:
            papers = []

        findings = [
            ResearchFinding(
                claim=(
                    f"[MOCK] Recent work on '{sec.title}' reports measurable gains "
                    "over prior baselines."
                ),
                source_url=p.get("url", "internal_knowledge"),
                confidence="medium",
                excerpt=p.get("summary", "")[:200],
                citation=p.get("citation", ""),
            )
            for p in papers
        ]
        if findings:
            sections.append(ResearchSection(section_id=sec.id, findings=findings))

    return sections
