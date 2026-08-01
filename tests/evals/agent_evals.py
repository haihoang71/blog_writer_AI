"""
tests/evals/agent_evals.py
───────────────────────────
Automated agent evaluations using RAGAS-style metrics.

Metrics evaluated:
- Faithfulness      : Are draft claims supported by research?
- Answer Relevance  : Does the draft address the topic and outline?
- Context Recall    : Is all outline information covered?
- Code Validity     : Do code snippets pass safety and syntax checks?

Run with:
    pytest tests/evals/agent_evals.py -v -m eval
"""

from __future__ import annotations

import re
import pytest
from typing import Any

from state.blog_state import BlogOutline, ResearchData, initial_state
from guardrails.hallucination_guard import check_hallucination
from guardrails.code_sandbox_guard import check_code_safety
from guardrails.output_guard import sanitise_output


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation Metrics
# ─────────────────────────────────────────────────────────────────────────────


def eval_faithfulness(draft: str, research: ResearchData) -> dict[str, Any]:
    """
    Faithfulness Metric: how many draft claims are supported by research?

    Returns
    -------
    dict with score (0.0-1.0) and details.
    """
    report = check_hallucination(draft, research.model_dump())
    return {
        "metric": "faithfulness",
        "score": report.overall_faithfulness_score,
        "total_claims": report.total_claims,
        "verified": report.verified,
        "flagged": report.flagged,
        "passed": report.passed,
    }


def eval_answer_relevance(draft: str, topic: str, outline: BlogOutline) -> dict[str, Any]:
    """
    Answer Relevance: does the draft address the stated topic and outline?

    Checks:
    - Topic keywords appear in the draft
    - All H2 section titles from the outline appear in the draft
    """
    topic_words = set(topic.lower().split())
    draft_lower = draft.lower()

    # Topic coverage
    covered_words = sum(1 for w in topic_words if w in draft_lower)
    topic_coverage = covered_words / len(topic_words) if topic_words else 0.0

    # Section coverage
    sections_covered = 0
    missing_sections: list[str] = []
    for section in outline.sections:
        title_words = set(section.title.lower().split())
        title_in_draft = all(w in draft_lower for w in title_words if len(w) > 3)
        if title_in_draft:
            sections_covered += 1
        else:
            missing_sections.append(section.title)

    section_coverage = (
        sections_covered / len(outline.sections) if outline.sections else 0.0
    )
    overall = (topic_coverage + section_coverage) / 2.0

    return {
        "metric": "answer_relevance",
        "score": round(overall, 4),
        "topic_coverage": round(topic_coverage, 4),
        "section_coverage": round(section_coverage, 4),
        "missing_sections": missing_sections,
        "passed": overall >= 0.5,
    }


def eval_code_validity(draft: str) -> dict[str, Any]:
    """
    Code Validity: check all Python code blocks for syntax and safety.

    Returns
    -------
    dict with per-snippet results and overall pass rate.
    """
    blocks = re.findall(r"```python\n([\s\S]*?)```", draft)
    if not blocks:
        return {
            "metric": "code_validity",
            "score": 1.0,
            "snippets_checked": 0,
            "snippets_safe": 0,
            "passed": True,
        }

    safe_count = 0
    snippet_results: list[dict] = []

    for i, code in enumerate(blocks):
        result = check_code_safety(code.strip(), language="python")
        is_safe = result.risk_level in ("safe", "warn")
        if is_safe:
            safe_count += 1
        snippet_results.append(
            {
                "snippet_index": i,
                "risk_level": result.risk_level,
                "safe": is_safe,
                "recommendation": result.recommendation,
            }
        )

    score = safe_count / len(blocks)
    return {
        "metric": "code_validity",
        "score": round(score, 4),
        "snippets_checked": len(blocks),
        "snippets_safe": safe_count,
        "snippet_results": snippet_results,
        "passed": score >= 0.8,
    }


def eval_output_format(draft: str) -> dict[str, Any]:
    """
    Output Format: verify the draft meets the required Markdown schema.
    """
    result = sanitise_output(draft)
    schema_score = sum([
        result.has_title,
        result.has_tldr,
        result.has_key_takeaways,
        result.has_references,
    ]) / 4.0

    return {
        "metric": "output_format",
        "score": round(schema_score, 4),
        "has_title": result.has_title,
        "has_tldr": result.has_tldr,
        "has_key_takeaways": result.has_key_takeaways,
        "has_references": result.has_references,
        "word_count": result.word_count,
        "issues": result.issues,
        "passed": schema_score >= 0.75,
    }


def run_full_evaluation(
    draft: str,
    topic: str,
    outline: BlogOutline,
    research: ResearchData,
) -> dict[str, Any]:
    """
    Run all evaluation metrics and return a consolidated report.

    Parameters
    ----------
    draft:    The generated Markdown blog post.
    topic:    The original user topic.
    outline:  The planned outline.
    research: The gathered research data.

    Returns
    -------
    dict with all metric results and an overall composite score.
    """
    faithfulness = eval_faithfulness(draft, research)
    relevance = eval_answer_relevance(draft, topic, outline)
    code_val = eval_code_validity(draft)
    format_check = eval_output_format(draft)

    composite = (
        faithfulness["score"] * 0.30 +
        relevance["score"] * 0.30 +
        code_val["score"] * 0.20 +
        format_check["score"] * 0.20
    )

    all_passed = all([
        faithfulness["passed"],
        relevance["passed"],
        code_val["passed"],
        format_check["passed"],
    ])

    return {
        "composite_score": round(composite, 4),
        "all_passed": all_passed,
        "metrics": {
            "faithfulness": faithfulness,
            "answer_relevance": relevance,
            "code_validity": code_val,
            "output_format": format_check,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pytest-based Eval Tests
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.eval
class TestAgentEvals:
    """Evaluation tests — marked with 'eval' for selective execution."""

    @pytest.fixture
    def sample_draft(self):
        return """# Understanding LangGraph for Production Systems

> **TL;DR** — LangGraph is a framework for building stateful multi-agent applications.
> It models workflows as directed graphs and supports human-in-the-loop patterns [1].
> This guide covers architecture, implementation, and production best practices.

## Introduction to LangGraph

LangGraph is a stateful multi-actor framework built on LangChain [1].
It reduces agent loop complexity by 40% in production systems [2].
The framework supports both synchronous and asynchronous execution.

## Building Your First Graph

Creating a graph requires defining state and adding nodes:

```python
from typing import TypedDict
from langgraph.graph import StateGraph

class MyState(TypedDict):
    message: str

graph = StateGraph(MyState)
print('Graph created!')
```

This pattern enables clean separation of concerns between agent nodes [1].

## Key Takeaways

- LangGraph models workflows as directed graphs
- TypedDict provides type-safe state management
- Human-in-the-loop support is built-in

## References

1. [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
2. [LangChain Blog Post](https://blog.langchain.dev)
"""

    @pytest.fixture
    def sample_outline(self):
        from state.blog_state import OutlineSection
        return BlogOutline(
            title="Understanding LangGraph for Production Systems",
            slug="langgraph-production",
            sections=[
                OutlineSection(
                    id="s1",
                    title="Introduction to LangGraph",
                    purpose="Introduce LangGraph",
                    key_points=["What is LangGraph"],
                    needs_code=False,
                    depth="introductory",
                ),
                OutlineSection(
                    id="s2",
                    title="Building Your First Graph",
                    purpose="Practical implementation guide",
                    key_points=["StateGraph", "TypedDict"],
                    needs_code=True,
                    depth="intermediate",
                ),
            ],
        )

    @pytest.fixture
    def sample_research(self):
        from state.blog_state import ResearchSection, ResearchFinding
        return ResearchData(
            sections=[
                ResearchSection(
                    section_id="s1",
                    findings=[
                        ResearchFinding(
                            claim="LangGraph is a stateful multi-actor framework built on LangChain.",
                            source_url="https://langchain-ai.github.io/langgraph/",
                            confidence="high",
                        )
                    ],
                ),
                ResearchSection(
                    section_id="s2",
                    findings=[
                        ResearchFinding(
                            claim="StateGraph requires a TypedDict state class.",
                            source_url="https://langchain-ai.github.io/langgraph/",
                            confidence="high",
                        )
                    ],
                ),
            ],
            key_takeaways=["LangGraph models workflows as directed graphs."],
        )

    def test_faithfulness_eval(self, sample_draft, sample_research):
        result = eval_faithfulness(sample_draft, sample_research)
        assert result["metric"] == "faithfulness"
        assert 0.0 <= result["score"] <= 1.0
        print(f"\n[EVAL] Faithfulness: {result['score']:.2f}")

    def test_answer_relevance_eval(self, sample_draft, sample_outline):
        result = eval_answer_relevance(
            sample_draft, "LangGraph production systems", sample_outline
        )
        assert result["metric"] == "answer_relevance"
        assert 0.0 <= result["score"] <= 1.0
        print(f"\n[EVAL] Answer Relevance: {result['score']:.2f}")

    def test_code_validity_eval(self, sample_draft):
        result = eval_code_validity(sample_draft)
        assert result["metric"] == "code_validity"
        assert result["snippets_checked"] >= 1
        assert result["passed"] is True
        print(f"\n[EVAL] Code Validity: {result['score']:.2f}")

    def test_output_format_eval(self, sample_draft):
        result = eval_output_format(sample_draft)
        assert result["metric"] == "output_format"
        assert result["has_title"] is True
        assert result["has_tldr"] is True
        print(f"\n[EVAL] Output Format: {result['score']:.2f}")

    def test_full_evaluation(self, sample_draft, sample_outline, sample_research):
        report = run_full_evaluation(
            draft=sample_draft,
            topic="LangGraph production systems",
            outline=sample_outline,
            research=sample_research,
        )
        assert 0.0 <= report["composite_score"] <= 1.0
        print(f"\n[EVAL] Composite Score: {report['composite_score']:.2f}")
        print(f"[EVAL] All Passed: {report['all_passed']}")
        for name, metric in report["metrics"].items():
            print(f"  - {name}: {metric['score']:.2f} (passed={metric['passed']})")
