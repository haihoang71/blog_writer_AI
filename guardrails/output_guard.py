"""
guardrails/output_guard.py
───────────────────────────
Output guardrail: PII redaction and Markdown format validation.

Steps
-----
1. PII Redaction — uses presidio-analyzer if available; regex fallback.
2. Markdown Schema Validation — checks required sections are present.
3. Word Count Check — enforces minimum length.
4. Final cleanup — normalise whitespace, ensure trailing newline.

Usage
-----
    from guardrails.output_guard import sanitise_output, OutputGuardResult

    result = sanitise_output(draft)
    print(result.clean_text)
    print(result.pii_items_redacted)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from config.constants import MIN_DRAFT_WORD_COUNT

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class OutputGuardResult:
    """Result of the output guardrail pipeline."""

    clean_text: str = ""
    passed: bool = True
    pii_items_redacted: int = 0
    word_count: int = 0
    has_title: bool = False
    has_tldr: bool = False
    has_key_takeaways: bool = False
    has_references: bool = False
    issues: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# PII Redaction
# ─────────────────────────────────────────────────────────────────────────────

# Regex fallback patterns (when presidio is not available)
_PII_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Email addresses
    (re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"), "[EMAIL]"),
    # US/international phone numbers
    (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE]"),
    # US Social Security Numbers
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    # Credit card numbers (basic)
    (re.compile(r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"), "[CARD_NUMBER]"),
    # IPv4 addresses (private/internal)
    (re.compile(r"\b(?:192\.168|10\.\d+|172\.(?:1[6-9]|2\d|3[01]))\.\d+\.\d+\b"), "[INTERNAL_IP]"),
    # API keys / tokens (common patterns)
    (re.compile(r"\b(sk|pk|api|token|key)[-_][A-Za-z0-9]{20,}\b", re.I), "[API_KEY]"),
    # AWS access keys
    (re.compile(r"\b(AKIA|ASIA)[A-Z0-9]{16}\b"), "[AWS_KEY]"),
]


def _redact_pii_regex(text: str) -> tuple[str, int]:
    """Apply regex-based PII redaction. Returns (cleaned_text, count_redacted)."""
    total_redacted = 0
    for pattern, replacement in _PII_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            text = pattern.sub(replacement, text)
            total_redacted += len(matches)
    return text, total_redacted


def _redact_pii_presidio(text: str) -> tuple[str, int]:
    """Use Microsoft Presidio for NLP-powered PII detection."""
    try:
        import spacy  # type: ignore

        # Presidio's default AnalyzerEngine() loads a spaCy model
        # ("en_core_web_lg") and — critically — will silently call
        # `spacy.cli.download(...)` to fetch it (several hundred MB) the
        # first time it's missing, blocking the whole generation pipeline
        # for many minutes. The README documents this model as optional
        # ("python -m spacy download en_core_web_lg" run ahead of time), so
        # we check it's actually installed first and skip straight to the
        # regex fallback instead of triggering a surprise download.
        if not spacy.util.is_package("en_core_web_lg"):
            logger.info(
                "presidio: spaCy model 'en_core_web_lg' not installed — "
                "using regex PII redaction instead of triggering an "
                "automatic ~400MB download. Run `python -m spacy download "
                "en_core_web_lg` to enable presidio's NLP-based redaction."
            )
            raise ImportError("en_core_web_lg not installed")

        from presidio_analyzer import AnalyzerEngine  # type: ignore
        from presidio_anonymizer import AnonymizerEngine  # type: ignore
        from presidio_anonymizer.entities import OperatorConfig  # type: ignore

        analyzer = AnalyzerEngine()
        anonymizer = AnonymizerEngine()

        results = analyzer.analyze(
            text=text,
            entities=[
                "EMAIL_ADDRESS", "PHONE_NUMBER", "PERSON",
                "CREDIT_CARD", "US_SSN", "IP_ADDRESS",
            ],
            language="en",
        )

        if not results:
            return text, 0

        anonymised = anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators={
                "DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"}),
                "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
                "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE]"}),
                "PERSON": OperatorConfig("replace", {"new_value": "[NAME]"}),
            },
        )
        return anonymised.text, len(results)

    except ImportError:
        logger.debug("presidio not installed — using regex PII redaction.")
        raise


def _apply_pii_redaction(text: str) -> tuple[str, int]:
    """Try presidio first; fall back to regex."""
    try:
        return _redact_pii_presidio(text)
    except (ImportError, Exception):  # noqa: BLE001
        return _redact_pii_regex(text)


# ─────────────────────────────────────────────────────────────────────────────
# Markdown Schema Validation
# ─────────────────────────────────────────────────────────────────────────────


def _validate_markdown_schema(text: str) -> tuple[dict[str, bool], list[str]]:
    """Check required Markdown sections are present."""
    structure = {
        "has_title": bool(re.search(r"^#\s+\S+", text, re.MULTILINE)),
        "has_tldr": bool(re.search(r"(?i)(tl;dr|tldr)", text)),
        "has_key_takeaways": bool(
            re.search(r"(?i)##\s+key\s+takeaways?", text, re.MULTILINE)
        ),
        "has_references": bool(
            re.search(r"(?i)##\s+references?", text, re.MULTILINE)
        ),
    }
    issues: list[str] = []
    if not structure["has_title"]:
        issues.append("Missing H1 title (# Title)")
    if not structure["has_tldr"]:
        issues.append("Missing TL;DR section")
    if not structure["has_key_takeaways"]:
        issues.append("Missing '## Key Takeaways' section")
    if not structure["has_references"]:
        issues.append("Missing '## References' section")
    return structure, issues


# ─────────────────────────────────────────────────────────────────────────────
# Text Cleanup
# ─────────────────────────────────────────────────────────────────────────────


def _cleanup(text: str) -> str:
    """Normalise whitespace and ensure single trailing newline."""
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing spaces from each line
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines)
    # Ensure single trailing newline
    return text.rstrip() + "\n"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def sanitise_output(text: str) -> OutputGuardResult:
    """
    Run the full output guardrail pipeline on a Markdown draft.

    Steps
    -----
    1. PII redaction (presidio → regex fallback)
    2. Markdown schema validation
    3. Word count check
    4. Whitespace cleanup

    Parameters
    ----------
    text:
        Raw Markdown blog post text from the Writer agent.

    Returns
    -------
    OutputGuardResult
        Contains the cleaned text and a list of any schema issues found.
    """
    if not text.strip():
        return OutputGuardResult(
            clean_text="",
            passed=False,
            issues=["Draft is empty"],
        )

    issues: list[str] = []

    # ── 1. PII redaction ──────────────────────────────────────────────────
    clean_text, pii_count = _apply_pii_redaction(text)
    if pii_count > 0:
        logger.info("Output guard: redacted %d PII items.", pii_count)

    # ── 2. Markdown schema validation ─────────────────────────────────────
    structure, schema_issues = _validate_markdown_schema(clean_text)
    issues.extend(schema_issues)
    if schema_issues:
        logger.warning("Output guard: Markdown schema issues: %s", schema_issues)

    # ── 3. Word count check ───────────────────────────────────────────────
    word_count = len(clean_text.split())
    if word_count < MIN_DRAFT_WORD_COUNT:
        issues.append(
            f"Draft too short: {word_count} words (min {MIN_DRAFT_WORD_COUNT})"
        )

    # ── 4. Cleanup ────────────────────────────────────────────────────────
    clean_text = _cleanup(clean_text)

    passed = len(issues) == 0

    return OutputGuardResult(
        clean_text=clean_text,
        passed=passed,
        pii_items_redacted=pii_count,
        word_count=word_count,
        has_title=structure.get("has_title", False),
        has_tldr=structure.get("has_tldr", False),
        has_key_takeaways=structure.get("has_key_takeaways", False),
        has_references=structure.get("has_references", False),
        issues=issues,
    )
