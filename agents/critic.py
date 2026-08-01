"""
agents/critic.py
─────────────────
Technical Critic Agent Node — reviews drafts and executes code snippets.

This node:
1. Extracts code blocks from the draft.
2. Runs each block through the code security guardrail.
3. Executes safe blocks in the sandbox.
4. Calls the LLM to generate a structured critique.
5. Validates the critique into a ``BlogCritique`` Pydantic model.
6. Sets ``is_approved`` based on the LLM's decision.
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
    CRITIC_MAX_TOKENS,
    MAX_REVISIONS,
    MAX_LLM_RETRIES,
)
from config.settings import get_settings
from guardrails.code_sandbox_guard import check_code_safety
from prompts.loader import load_prompt
from state.blog_state import (
    BlogCritique,
    BlogState,
    CodeExecutionResult,
    CritiqueIssue,
    CritiqueScores,
    ErrorLog,
    ResearchData,
)
from tools.code_interpreter import execute_code

logger = logging.getLogger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# LLM Factory
# ─────────────────────────────────────────────────────────────────────────────


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.openai_model_strong,  # Use stronger model for critique
        openai_api_base=settings.openai_api_base or None,
        temperature=0.1,
        max_tokens=CRITIC_MAX_TOKENS,
        api_key=settings.openai_api_key.get_secret_value(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Code Block Extraction & Execution
# ─────────────────────────────────────────────────────────────────────────────


def _extract_code_blocks(draft: str) -> list[tuple[str, str]]:
    """
    Extract fenced code blocks from Markdown.

    Returns
    -------
    list[tuple[str, str]]
        List of (language, code) pairs.
    """
    pattern = re.compile(r"```(\w*)\n([\s\S]*?)```", re.MULTILINE)
    blocks: list[tuple[str, str]] = []
    for match in pattern.finditer(draft):
        lang = match.group(1).lower() or "text"
        code = match.group(2).strip()
        if code:
            blocks.append((lang, code))
    return blocks


def _execute_code_blocks(
    blocks: list[tuple[str, str]],
    assets_dir: str | None = None,
) -> list[CodeExecutionResult]:
    """
    Run safe Python blocks through the sandbox.

    When *assets_dir* is given, any matplotlib figures the snippet produces
    (e.g. a chart illustrating a benchmark) are saved into it and recorded
    on the result so they can be embedded in the final blog post / surfaced
    in the UI.
    """
    results: list[CodeExecutionResult] = []
    for idx, (lang, code) in enumerate(blocks):
        # Security check first
        safety = check_code_safety(code, language=lang)
        if not safety.safe_to_execute:
            results.append(
                CodeExecutionResult(
                    snippet_index=idx,
                    language=lang,
                    status="blocked",
                    output=f"Blocked: {safety.recommendation}",
                )
            )
            continue

        exec_result = execute_code(code=code, language=lang, assets_dir=assets_dir)
        results.append(
            CodeExecutionResult(
                snippet_index=idx,
                language=lang,
                status=exec_result.status,
                output=(exec_result.output or exec_result.error)[:500],
                artifacts=exec_result.artifacts,
            )
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# JSON Parsing
# ─────────────────────────────────────────────────────────────────────────────


def _parse_critique_json(raw: str) -> dict:
    """Extract and parse JSON from LLM critic response."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    for pat in [r"```(?:json)?\s*([\s\S]+?)\s*```", r"\{[\s\S]+\}"]:
        m = re.search(pat, raw)
        if m:
            try:
                return json.loads(m.group(1) if "```" in pat else m.group())
            except json.JSONDecodeError:
                continue
    raise ValueError(f"Cannot parse critique JSON: {raw[:300]}")


def _to_int_score(value: Any, default: int = 3) -> int:
    """
    Coerce an LLM-provided "score" into the int CritiqueScores/BlogCritique
    expect.

    Gemini (and other models) sometimes return scores like ``2.3`` or ``4.5``
    despite the prompt asking for integers 1-5. Pydantic's int field rejects
    floats with a nonzero fractional part outright (`int_from_float` error),
    which was crashing the whole critique parse and silently falling back to
    the mock critique on every live run. Rounding here is more forgiving than
    the strict validation error.
    """
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _build_critique(data: dict, exec_results: list[CodeExecutionResult]) -> BlogCritique:
    """Construct a BlogCritique from parsed JSON + code execution results."""
    scores_data = data.get("scores", {})
    scores = CritiqueScores(
        technical_accuracy=_to_int_score(scores_data.get("technical_accuracy", 3)),
        code_quality=_to_int_score(scores_data.get("code_quality", 3)),
        completeness=_to_int_score(scores_data.get("completeness", 3)),
        clarity=_to_int_score(scores_data.get("clarity", 3)),
        structure=_to_int_score(scores_data.get("structure", 3)),
        seo_readability=_to_int_score(scores_data.get("seo_readability", 3)),
    )

    issues: list[CritiqueIssue] = []
    for raw_issue in data.get("critical_issues", []):
        issues.append(
            CritiqueIssue(
                section=raw_issue.get("section", "global"),
                issue=raw_issue.get("issue", ""),
                severity=raw_issue.get("severity", "major"),
                suggestion=raw_issue.get("suggestion", ""),
            )
        )

    return BlogCritique(
        approved=data.get("approved", False),
        overall_score=_to_int_score(data.get("overall_score", 3)),
        scores=scores,
        critical_issues=issues,
        code_execution_results=exec_results,
        positive_aspects=data.get("positive_aspects", []),
        summary_feedback=data.get("summary_feedback", ""),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Mock Critique (keyless fallback)
# ─────────────────────────────────────────────────────────────────────────────


def _mock_critique(revision_count: int) -> BlogCritique:
    """Return a structured mock critique."""
    logger.info("[MOCK] Generating mock critique — revision %d", revision_count)

    # Approve after 2 mock revisions
    approved = revision_count >= 2

    issues: list[CritiqueIssue] = []
    if not approved:
        issues = [
            CritiqueIssue(
                section="Introduction",
                issue="[MOCK] The introduction could be more engaging. Add a compelling hook.",
                severity="minor",
                suggestion="Start with a real-world problem statement or surprising statistic.",
            ),
            CritiqueIssue(
                section="Core Concepts",
                issue="[MOCK] Code snippet in section 2 needs more inline comments.",
                severity="major",
                suggestion="Add docstrings and inline comments explaining each step.",
            ),
        ]

    return BlogCritique(
        approved=approved,
        overall_score=4 if approved else 3,
        scores=CritiqueScores(
            technical_accuracy=4,
            code_quality=3 if not approved else 4,
            completeness=4,
            clarity=3 if not approved else 4,
            structure=4,
            seo_readability=4,
        ),
        critical_issues=issues,
        code_execution_results=[],
        positive_aspects=[
            "Good structure and heading hierarchy",
            "Appropriate technical depth",
            "Clear TL;DR section",
        ],
        summary_feedback=(
            "Draft approved — meets all quality criteria."
            if approved
            else (
                "[MOCK] The draft has a solid foundation but needs refinement "
                "in code documentation and introduction engagement."
            )
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Agent Node
# ─────────────────────────────────────────────────────────────────────────────


def critic_node(state: BlogState) -> BlogState:
    """
    LangGraph node: Technical Critic Agent.

    Steps:
    1. Extract code blocks from draft.
    2. Execute safe Python blocks in sandbox.
    3. Call LLM to generate structured critique.
    4. Set ``is_approved`` based on critique decision.

    Reads:  ``draft``, ``research_data``, ``revision_count``
    Writes: ``critique``, ``is_approved``
    """
    draft: str = state.get("draft", "")
    research: ResearchData | None = state.get("research_data")
    revision_count: int = state.get("revision_count", 0)
    error_logs: list[ErrorLog] = list(state.get("error_logs", []))
    metadata: dict[str, Any] = dict(state.get("metadata", {}))

    logger.info("Critic node — revision %d", revision_count)

    if not draft.strip():
        logger.error("Critic: empty draft — skipping.")
        return state

    # ── Code extraction + execution ───────────────────────────────────────
    code_blocks = _extract_code_blocks(draft)
    logger.info("Critic: found %d code blocks.", len(code_blocks))

    assets_dir: str | None = None
    run_id = state.get("run_id")
    if run_id:
        try:
            from storage.post_manager import get_generation_assets_dir

            assets_dir = str(get_generation_assets_dir(run_id))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not resolve assets dir for run %s: %s", run_id, exc)

    exec_results = _execute_code_blocks(code_blocks, assets_dir=assets_dir)
    total_artifacts = sum(len(r.artifacts) for r in exec_results)
    if total_artifacts:
        logger.info(
            "Critic: code execution produced %d chart/image artifact(s) in %s",
            total_artifacts, assets_dir,
        )

    failed_snippets = [r for r in exec_results if r.status == "failed"]
    if failed_snippets:
        logger.warning("Critic: %d code snippet(s) failed execution.", len(failed_snippets))

    metadata["chart_artifact_count"] = total_artifacts
    if assets_dir:
        metadata["assets_dir"] = assets_dir

    # ── Fallback if OpenAI not configured ────────────────────────────────
    if not settings.is_openai_configured:
        critique = _mock_critique(revision_count)
        critique.code_execution_results = exec_results
        metadata["critic_mode"] = "mock"
        return {
            **state,
            "critique": critique,
            "is_approved": critique.approved,
            "error_logs": error_logs,
            "metadata": metadata,
        }

    # ── Build prompt ──────────────────────────────────────────────────────
    research_json = research.model_dump_json(indent=2) if research else "{}"
    exec_summary = json.dumps(
        [
            {
                "snippet_index": r.snippet_index,
                "language": r.language,
                "status": r.status,
                "output": r.output[:200],
            }
            for r in exec_results
        ],
        indent=2,
    )

    # Inject code execution results into draft context
    draft_with_exec = (
        f"{draft}\n\n---\n**Code Execution Results:**\n```json\n{exec_summary}\n```"
    )

    system_prompt = load_prompt(
        "critic",
        draft=draft_with_exec[:8000],   # truncate to avoid context overflow
        research_json=research_json[:4000],
        revision_count=revision_count,
        max_revisions=MAX_REVISIONS,
    )

    human_msg = (
        "Review the blog post draft thoroughly. "
        "Apply all evaluation dimensions. "
        "Return your critique as the JSON object specified — nothing else."
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

        raw = _call()
        data = _parse_critique_json(raw)

        # Force approval if max revisions reached
        if revision_count >= MAX_REVISIONS:
            data["approved"] = True
            data["summary_feedback"] = (
                data.get("summary_feedback", "") +
                f" [Auto-approved: max revisions ({MAX_REVISIONS}) reached]"
            )

        critique = _build_critique(data, exec_results)
        logger.info(
            "Critic: approved=%s score=%d issues=%d",
            critique.approved, critique.overall_score, len(critique.critical_issues),
        )
        metadata["critic_mode"] = "live"
        metadata[f"critique_score_rev{revision_count}"] = critique.overall_score

    except Exception as exc:  # noqa: BLE001
        logger.error("Critic failed: %s — using mock critique.", exc)
        error_logs.append(
            ErrorLog(
                node="critic",
                error_type=type(exc).__name__,
                message=str(exc),
                recoverable=True,
            )
        )
        critique = _mock_critique(revision_count)
        critique.code_execution_results = exec_results
        metadata["critic_mode"] = "mock_fallback"

    return {
        **state,
        "critique": critique,
        "is_approved": critique.approved,
        "error_logs": error_logs,
        "metadata": metadata,
    }
