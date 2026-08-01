"""
state/blog_state.py
────────────────────
Shared graph state for the Multi-Agent Blog Generator.

Design principles
-----------------
- ``BlogState`` is a LangGraph ``TypedDict`` so it can be passed through
  graph nodes with full type safety.
- Nested ``Pydantic`` models validate rich sub-objects (outline, research,
  critique) at construction time.
- Every field has a sensible default so nodes can be written as pure
  transformers that only update the fields they own.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict


# ─────────────────────────────────────────────────────────────────────────────
#  Pydantic Sub-Models (rich validation for complex fields)
# ─────────────────────────────────────────────────────────────────────────────


class SubSection(BaseModel):
    """H3-level section within an outline H2."""

    id: str
    title: str
    purpose: str
    key_points: list[str] = Field(default_factory=list)
    needs_code: bool = False


class OutlineSection(BaseModel):
    """H2-level section in the blog post outline."""

    id: str
    title: str
    purpose: str
    key_points: list[str] = Field(default_factory=list)
    needs_code: bool = False
    depth: str = "intermediate"
    subsections: list[SubSection] = Field(default_factory=list)

    @field_validator("depth")
    @classmethod
    def _validate_depth(cls, v: str) -> str:
        allowed = {"introductory", "intermediate", "advanced"}
        if v not in allowed:
            raise ValueError(f"depth must be one of {allowed}, got '{v}'")
        return v


class BlogOutline(BaseModel):
    """Structured blog post outline produced by the Planner agent."""

    title: str
    slug: str = ""
    meta_description: str = ""
    target_audience: str = ""
    estimated_read_time_minutes: int = 0
    sections: list[OutlineSection] = Field(default_factory=list)

    @field_validator("sections")
    @classmethod
    def _validate_sections(cls, v: list[OutlineSection]) -> list[OutlineSection]:
        if len(v) < 1:
            raise ValueError("Outline must contain at least one section.")
        return v


class ResearchFinding(BaseModel):
    """A single research fact/citation."""

    claim: str
    source_url: str = "internal_knowledge"
    confidence: str = "medium"
    excerpt: str = ""
    citation: str = ""
    """Formatted academic citation string (e.g. author/year/title/arXiv id),
    populated by the Academic Researcher agent. Empty for findings sourced
    from the general Researcher agent (web search / internal knowledge)."""

    @field_validator("confidence")
    @classmethod
    def _validate_confidence(cls, v: str) -> str:
        allowed = {"high", "medium", "low"}
        if v not in allowed:
            return "medium"
        return v


class CodeSnippet(BaseModel):
    """A code snippet gathered during research."""

    language: str = "python"
    description: str = ""
    code: str
    source_url: str = "internal_knowledge"


class ResearchSection(BaseModel):
    """Research data for a single outline section."""

    section_id: str
    findings: list[ResearchFinding] = Field(default_factory=list)
    code_snippets: list[CodeSnippet] = Field(default_factory=list)


class ResearchData(BaseModel):
    """Complete research report produced by the Researcher agent."""

    research_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sections: list[ResearchSection] = Field(default_factory=list)
    key_takeaways: list[str] = Field(default_factory=list)


class CritiqueIssue(BaseModel):
    """A single issue identified by the Critic agent."""

    section: str = "global"
    issue: str
    severity: str = "major"
    suggestion: str = ""

    @field_validator("severity")
    @classmethod
    def _validate_severity(cls, v: str) -> str:
        allowed = {"critical", "major", "minor"}
        if v not in allowed:
            return "major"
        return v


class CodeExecutionResult(BaseModel):
    """Result of running a code snippet in the sandbox."""

    snippet_index: int = 0
    language: str = "python"
    status: str = "skipped"  # passed | failed | skipped
    output: str = ""
    artifacts: list[str] = Field(default_factory=list)
    """Filenames of any chart/plot images (e.g. matplotlib figures) this
    snippet produced, saved under the run's assets directory."""


class CritiqueScores(BaseModel):
    """Dimension-level scores from the Critic (1–5)."""

    technical_accuracy: int = 0
    code_quality: int = 0
    completeness: int = 0
    clarity: int = 0
    structure: int = 0
    seo_readability: int = 0


class BlogCritique(BaseModel):
    """Complete critique report produced by the Critic agent."""

    approved: bool = False
    overall_score: int = 0
    scores: CritiqueScores = Field(default_factory=CritiqueScores)
    critical_issues: list[CritiqueIssue] = Field(default_factory=list)
    code_execution_results: list[CodeExecutionResult] = Field(default_factory=list)
    positive_aspects: list[str] = Field(default_factory=list)
    summary_feedback: str = ""


class ErrorLog(BaseModel):
    """A structured error entry in the error log."""

    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    node: str
    error_type: str
    message: str
    recoverable: bool = True


# ─────────────────────────────────────────────────────────────────────────────
#  Main Graph State (LangGraph TypedDict)
# ─────────────────────────────────────────────────────────────────────────────


class BlogState(TypedDict, total=False):
    """
    Shared mutable state threaded through every node of the LangGraph workflow.

    Fields
    ------
    topic:
        The raw user-supplied topic string.
    sanitised_topic:
        Topic after input guardrail processing.
    outline:
        Validated ``BlogOutline`` produced by the Planner node.
    research_data:
        Validated ``ResearchData`` produced by the Researcher node.
    draft:
        Current Markdown draft string (updated by Writer on each pass).
    critique:
        Latest ``BlogCritique`` produced by the Critic node.
    revision_count:
        Number of completed Writer → Critic cycles.
    is_approved:
        True when the Critic approves the draft or max revisions is hit.
    human_feedback:
        Optional free-text from the HITL interrupt node.
    final_post:
        Output-guardrail-scrubbed final Markdown payload.
    run_id:
        Unique identifier for this graph run.
    error_logs:
        Accumulated structured error entries.
    metadata:
        Arbitrary key-value metadata (tags, timestamps, token counts).
    """

    # ── Inputs ───────────────────────────────────────────────────────────
    topic: str
    sanitised_topic: str

    # ── Agent Outputs ─────────────────────────────────────────────────────
    outline: Optional[BlogOutline]
    research_data: Optional[ResearchData]
    draft: str
    critique: Optional[BlogCritique]

    # ── Control Flow ──────────────────────────────────────────────────────
    revision_count: int
    is_approved: bool

    # ── HITL ──────────────────────────────────────────────────────────────
    human_feedback: str

    # ── Final Output ──────────────────────────────────────────────────────
    final_post: str

    # ── Observability ─────────────────────────────────────────────────────
    run_id: str
    error_logs: list[ErrorLog]
    metadata: dict[str, Any]


def initial_state(topic: str) -> BlogState:
    """
    Factory function that returns a fully-initialised ``BlogState`` with
    safe defaults for a new graph run.

    Parameters
    ----------
    topic:
        The user's requested blog post topic.
    """
    return BlogState(
        topic=topic,
        sanitised_topic="",
        outline=None,
        research_data=None,
        draft="",
        critique=None,
        revision_count=0,
        is_approved=False,
        human_feedback="",
        final_post="",
        run_id=str(uuid.uuid4()),
        error_logs=[],
        metadata={
            "created_at": datetime.now(timezone.utc).isoformat(),
            "topic": topic,
        },
    )
