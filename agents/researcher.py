"""
agents/researcher.py
─────────────────────
Researcher Agent Node — gathers technical facts via search tools.

This node:
1. Loads the researcher system prompt from YAML.
2. Builds a tool-calling LLM bound to tavily_search and arxiv_search.
3. Invokes the agent to gather research for each outline section.
4. Parses and validates the research report into a ``ResearchData`` model.
5. Updates the graph state with research_data.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from config.constants import (
    MAX_RESEARCH_QUERIES,
    MAX_SEARCH_RESULTS,
    RESEARCHER_MAX_TOKENS,
    MAX_LLM_RETRIES,
)
from config.settings import get_settings
from prompts.loader import load_prompt
from state.blog_state import (
    BlogOutline,
    BlogState,
    CodeSnippet,
    ErrorLog,
    ResearchData,
    ResearchFinding,
    ResearchSection,
)
from tools.search_tools import tavily_search, arxiv_search, SEARCH_TOOLS

logger = logging.getLogger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# LLM Factory
# ─────────────────────────────────────────────────────────────────────────────


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model,
        openai_api_base=settings.openai_api_base or None,
        temperature=0.2,
        max_tokens=RESEARCHER_MAX_TOKENS,
        api_key=settings.openai_api_key.get_secret_value(),
    ).bind_tools(SEARCH_TOOLS)


# ─────────────────────────────────────────────────────────────────────────────
# Parse Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _parse_json_response(raw: str) -> dict:
    """Extract JSON from LLM response."""
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

    raise ValueError(f"Cannot parse JSON from researcher response: {raw[:300]}")


def _build_research_data(data: dict, outline: BlogOutline) -> ResearchData:
    """Construct a ``ResearchData`` from the parsed JSON dict."""
    sections: list[ResearchSection] = []
    for raw_sec in data.get("sections", []):
        findings: list[ResearchFinding] = [
            ResearchFinding(
                claim=f.get("claim", ""),
                source_url=f.get("source_url", "internal_knowledge"),
                confidence=f.get("confidence", "medium"),
                excerpt=f.get("excerpt", ""),
            )
            for f in raw_sec.get("findings", [])
        ]
        snippets: list[CodeSnippet] = [
            CodeSnippet(
                language=s.get("language", "python"),
                description=s.get("description", ""),
                code=s.get("code", ""),
                source_url=s.get("source_url", "internal_knowledge"),
            )
            for s in raw_sec.get("code_snippets", [])
        ]
        sections.append(
            ResearchSection(
                section_id=raw_sec.get("section_id", ""),
                findings=findings,
                code_snippets=snippets,
            )
        )

    return ResearchData(
        research_id=data.get("research_id", str(uuid.uuid4())),
        sections=sections,
        key_takeaways=data.get("key_takeaways", []),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mock Research (keyless fallback)
# ─────────────────────────────────────────────────────────────────────────────


def _mock_research(outline: BlogOutline) -> ResearchData:
    """Return synthetic research data without API calls."""
    logger.info("[MOCK] Generating mock research data.")
    sections: list[ResearchSection] = []

    for sec in outline.sections:
        findings = [
            ResearchFinding(
                claim=f"[MOCK] Key insight about '{sec.title}': This is a placeholder finding for testing.",
                source_url="internal_knowledge",
                confidence="medium",
                excerpt=f"Mock excerpt related to {sec.title}.",
            ),
            ResearchFinding(
                claim=f"[MOCK] Studies show that understanding '{sec.title}' improves developer productivity by 40%.",
                source_url="https://example.com/mock-source",
                confidence="low",
                excerpt="Mock research excerpt — replace with live data in production.",
            ),
        ]
        snippets: list[CodeSnippet] = []
        if sec.needs_code:
            snippets.append(
                CodeSnippet(
                    language="python",
                    description=f"Example implementation for {sec.title}",
                    code=(
                        f"# {sec.title} — Example Code\n"
                        "# This is a mock snippet. Run with live API keys for real examples.\n\n"
                        "def example_function(param: str) -> str:\n"
                        '    """Demonstrates the core concept."""\n'
                        "    result = f\"Processed: {param}\"\n"
                        "    return result\n\n"
                        "# Usage\n"
                        'output = example_function("hello")\n'
                        "print(output)"
                    ),
                    source_url="internal_knowledge",
                )
            )
        sections.append(ResearchSection(section_id=sec.id, findings=findings, code_snippets=snippets))

    return ResearchData(
        research_id=str(uuid.uuid4()),
        sections=sections,
        key_takeaways=[
            f"[MOCK] Key takeaway 1 about {outline.title}",
            f"[MOCK] Key takeaway 2 about {outline.title}",
            "[MOCK] Replace with live research for production blog posts.",
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tool Execution (manual tool loop)
# ─────────────────────────────────────────────────────────────────────────────


def _execute_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """Execute tool calls returned by the LLM."""
    results = []
    tool_map = {t.name: t for t in SEARCH_TOOLS}

    for tc in tool_calls:
        tool_name = tc.get("name", "")
        tool_args = tc.get("args", {})
        tool_id = tc.get("id", "")

        if tool_name not in tool_map:
            logger.warning("Unknown tool requested: %s", tool_name)
            continue

        try:
            output = tool_map[tool_name].invoke(tool_args)
            results.append({"tool_call_id": tool_id, "name": tool_name, "content": output})
            logger.info("Executed tool %s → %d chars output", tool_name, len(output))
        except Exception as exc:  # noqa: BLE001
            logger.error("Tool %s failed: %s", tool_name, exc)
            results.append(
                {"tool_call_id": tool_id, "name": tool_name, "content": f"Error: {exc}"}
            )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Agent Node
# ─────────────────────────────────────────────────────────────────────────────


def researcher_node(state: BlogState) -> BlogState:
    """
    LangGraph node: Researcher Agent.

    Reads ``state["outline"]``, issues search queries via Tavily/ArXiv
    tools, synthesises research into a ``ResearchData`` object, and
    writes it back to ``state["research_data"]``.
    """
    outline: BlogOutline | None = state.get("outline")
    error_logs: list[ErrorLog] = list(state.get("error_logs", []))
    metadata: dict[str, Any] = dict(state.get("metadata", {}))

    if outline is None:
        logger.error("Researcher: no outline in state — skipping.")
        return state

    logger.info("Researcher node — outline: %r", outline.title)

    # ── Fallback if OpenAI not configured ────────────────────────────────
    if not settings.is_openai_configured:
        research = _mock_research(outline)
        metadata["researcher_mode"] = "mock"
        return {**state, "research_data": research, "metadata": metadata}

    # ── Build prompt ──────────────────────────────────────────────────────
    outline_json = outline.model_dump_json(indent=2)
    system_prompt = load_prompt(
        "researcher",
        outline_json=outline_json,
        max_queries=MAX_RESEARCH_QUERIES,
        max_results=MAX_SEARCH_RESULTS,
    )

    human_msg = (
        "Research the outlined blog post thoroughly. "
        "Use the search tools to find accurate technical facts, "
        "code examples, and citations. "
        "Return your complete research as the JSON object specified."
    )

    # ── Agentic tool loop ─────────────────────────────────────────────────
    try:
        from langchain_core.messages import ToolMessage

        llm = _build_llm()
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=human_msg)]

        max_iterations = MAX_RESEARCH_QUERIES + 2

        @retry(
            stop=stop_after_attempt(MAX_LLM_RETRIES),
            wait=wait_exponential(multiplier=1, min=2, max=10),
            reraise=True,
        )
        def _call(msgs):
            return llm.invoke(msgs)

        for iteration in range(max_iterations):
            response = _call(messages)
            messages.append(response)

            # If the model made tool calls, execute them
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_results = _execute_tool_calls(response.tool_calls)
                for tr in tool_results:
                    messages.append(
                        ToolMessage(
                            content=tr["content"],
                            tool_call_id=tr["tool_call_id"],
                        )
                    )
            else:
                # No more tool calls — parse final answer
                raw = response.content
                break
        else:
            raw = response.content

        data = _parse_json_response(raw)
        research = _build_research_data(data, outline)

        logger.info(
            "Researcher: collected %d sections, %d total findings",
            len(research.sections),
            sum(len(s.findings) for s in research.sections),
        )
        metadata["researcher_mode"] = "live"

    except Exception as exc:  # noqa: BLE001
        logger.error("Researcher failed: %s — using mock research.", exc)
        error_logs.append(
            ErrorLog(
                node="researcher",
                error_type=type(exc).__name__,
                message=str(exc),
                recoverable=True,
            )
        )
        research = _mock_research(outline)
        metadata["researcher_mode"] = "mock_fallback"

    return {**state, "research_data": research, "error_logs": error_logs, "metadata": metadata}
