"""
guardrails/input_guard.py
──────────────────────────
Input guardrail: validates user topic before entering the graph.

Checks
------
1. Prompt-injection pattern detection (regex + heuristic)
2. Off-topic / out-of-domain classification via LLM
3. Blocked topic detection (regex against BLOCKED_TOPICS list)
4. Length / format sanity check

Returns a ``GuardResult`` dataclass indicating allow/block/clarify.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from config.constants import ALLOWED_DOMAINS, BLOCKED_TOPICS
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Prompt injection heuristics ────────────────────────────────────────────

_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?", re.I),
    re.compile(r"act\s+as\s+(a\s+)?(?!researcher|writer|planner)", re.I),
    re.compile(r"(jailbreak|DAN\s+mode|developer\s+mode)", re.I),
    re.compile(r"you\s+are\s+now\s+(?!a\s+(technical|software|ML|AI))", re.I),
    re.compile(r"(forget|disregard|override)\s+(your\s+)?(system|instructions?)", re.I),
    re.compile(r"<\s*(script|iframe|img)[^>]*>", re.I),  # HTML injection
    re.compile(r"\{\{.*?\}\}", re.I),                    # Template injection
    re.compile(r"\$\{.*?\}"),                             # JS template injection
]

# ── Blocked topic quick check ──────────────────────────────────────────────

_BLOCKED_RE = re.compile(
    "|".join(re.escape(t) for t in BLOCKED_TOPICS), re.I
)


# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GuardResult:
    """Result of the input guardrail check."""

    decision: str               # "allow" | "block" | "clarify"
    sanitised_topic: str = ""
    reason: str = ""
    detected_issues: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Internal Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _check_injection(topic: str) -> list[str]:
    """Return list of injection pattern descriptions detected."""
    issues: list[str] = []
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(topic):
            issues.append(f"Injection pattern: '{pattern.pattern}'")
    return issues


def _check_blocked_topics(topic: str) -> list[str]:
    """Return list of blocked topics found in the topic string."""
    matches = _BLOCKED_RE.findall(topic)
    return [f"Blocked topic: '{m}'" for m in set(matches)]


def _sanitise(topic: str) -> str:
    """Strip dangerous characters while preserving meaning."""
    # Remove HTML tags
    topic = re.sub(r"<[^>]+>", "", topic)
    # Remove null bytes and control chars
    topic = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", topic)
    # Collapse whitespace
    topic = " ".join(topic.split())
    return topic.strip()


def _llm_classify(topic: str) -> GuardResult:
    """
    Use the LLM to classify the topic when heuristics are ambiguous.
    Falls back to a permissive 'allow' if the LLM is unavailable.
    """
    if not settings.is_openai_configured:
        logger.info(
            "OpenAI not configured — skipping LLM classification, defaulting to allow."
        )
        return GuardResult(
            decision="allow",
            sanitised_topic=_sanitise(topic),
            reason="LLM classification skipped (no OpenAI key)",
        )

    try:
        import json
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        from prompts.loader import load_prompt

        llm = ChatOpenAI(
            model=settings.openai_model,
            openai_api_base=settings.openai_api_base or None,
            temperature=0.0,
            api_key=settings.openai_api_key.get_secret_value(),
        )

        system_prompt = load_prompt(
            "input_moderation",
            topic=topic,
            allowed_domains=", ".join(ALLOWED_DOMAINS),
            blocked_topics=", ".join(BLOCKED_TOPICS),
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Classify this topic: {topic}"),
        ]

        response = llm.invoke(messages)
        raw = response.content.strip()

        # Try to parse JSON
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
            else:
                raise ValueError(f"Could not parse LLM response as JSON: {raw[:200]}")

        decision = data.get("decision", "clarify")
        if decision not in {"accept", "reject", "clarify"}:
            decision = "clarify"

        # Map to internal decision values
        decision_map = {"accept": "allow", "reject": "block", "clarify": "clarify"}
        mapped_decision = decision_map.get(decision, "clarify")

        return GuardResult(
            decision=mapped_decision,
            sanitised_topic=data.get("sanitised_topic", _sanitise(topic)),
            reason=data.get("reason", ""),
            detected_issues=data.get("detected_issues", []),
        )

    except Exception as exc:  # noqa: BLE001
        logger.error("LLM input classification failed: %s — defaulting to allow.", exc)
        return GuardResult(
            decision="allow",
            sanitised_topic=_sanitise(topic),
            reason=f"LLM classification error (allowing by default): {exc}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def check_input(topic: str) -> GuardResult:
    """
    Run the full input guardrail pipeline on a topic string.

    Pipeline
    --------
    1. Length check
    2. Prompt injection detection
    3. Blocked topic pattern match
    4. LLM semantic classification (if heuristics pass)

    Parameters
    ----------
    topic:
        Raw user-provided topic string.

    Returns
    -------
    GuardResult
        ``decision`` is "allow", "block", or "clarify".
    """
    topic = topic.strip()
    issues: list[str] = []

    # ── 1. Length check ──────────────────────────────────────────────────
    if len(topic) < 5:
        return GuardResult(
            decision="block",
            reason="Topic is too short to be meaningful (< 5 characters).",
            detected_issues=["Topic too short"],
        )
    if len(topic) > 500:
        return GuardResult(
            decision="block",
            reason="Topic exceeds maximum allowed length (500 characters).",
            detected_issues=["Topic too long"],
        )

    # ── 2. Injection detection ───────────────────────────────────────────
    injection_issues = _check_injection(topic)
    if injection_issues:
        logger.warning(
            "Prompt injection detected in topic: %s — issues: %s",
            topic[:80], injection_issues,
        )
        return GuardResult(
            decision="block",
            reason="Potential prompt injection detected.",
            detected_issues=injection_issues,
        )

    # ── 3. Blocked topics ────────────────────────────────────────────────
    blocked_issues = _check_blocked_topics(topic)
    if blocked_issues:
        logger.warning(
            "Blocked topic detected: %s — %s", topic[:80], blocked_issues
        )
        return GuardResult(
            decision="block",
            reason="Topic contains content that violates our content policy.",
            detected_issues=blocked_issues,
        )

    # ── 4. LLM semantic classification ───────────────────────────────────
    result = _llm_classify(topic)
    logger.info(
        "Input guard decision: %s — %s", result.decision, result.reason
    )
    return result
