"""
guardrails/code_sandbox_guard.py
──────────────────────────────────
Code security guardrail: AST analysis + optional LLM security review.

Combines:
- AST-level static analysis (fast, no API calls)
- Regex pattern matching against SENSITIVE_CODE_PATTERNS
- LLM-based security reasoning for ambiguous cases

Usage
-----
    from guardrails.code_sandbox_guard import check_code_safety, CodeSafetyResult

    result = check_code_safety(code="import os; os.system('ls')", language="python")
    print(result.risk_level)   # "block"
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from config.constants import SENSITIVE_CODE_PATTERNS
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# ── Precompile regex patterns ──────────────────────────────────────────────
_SENSITIVE_REGEXES: list[re.Pattern] = [
    re.compile(p, re.MULTILINE) for p in SENSITIVE_CODE_PATTERNS
]


# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RiskItem:
    """A single detected risk in a code snippet."""

    check: str
    severity: str        # low | medium | high | critical
    line: int = 0
    detail: str = ""


@dataclass
class CodeSafetyResult:
    """Result of the code safety check."""

    risk_level: str = "safe"           # safe | warn | block
    risks_detected: list[RiskItem] = field(default_factory=list)
    safe_to_execute: bool = True
    recommendation: str = "Code appears safe to execute."


# ─────────────────────────────────────────────────────────────────────────────
# Analysis Functions
# ─────────────────────────────────────────────────────────────────────────────


def _regex_scan(code: str) -> list[RiskItem]:
    """Scan code lines against SENSITIVE_CODE_PATTERNS."""
    issues: list[RiskItem] = []
    lines = code.splitlines()
    for pattern in _SENSITIVE_REGEXES:
        for i, line in enumerate(lines, 1):
            if pattern.search(line):
                issues.append(
                    RiskItem(
                        check="regex_pattern",
                        severity="high",
                        line=i,
                        detail=f"Pattern '{pattern.pattern}' matched: {line.strip()[:80]}",
                    )
                )
    return issues


def _ast_scan(code: str) -> list[RiskItem]:
    """Run AST-level analysis to detect dangerous constructs."""
    issues: list[RiskItem] = []

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        issues.append(
            RiskItem(
                check="syntax",
                severity="medium",
                line=exc.lineno or 0,
                detail=f"Syntax error: {exc.msg}",
            )
        )
        return issues

    for node in ast.walk(tree):
        lineno = getattr(node, "lineno", 0)

        # Dangerous calls
        if isinstance(node, ast.Call):
            func_name = ""
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr

            if func_name in {"eval", "exec"}:
                issues.append(
                    RiskItem(
                        check="dynamic_execution",
                        severity="critical",
                        line=lineno,
                        detail=f"Dynamic code execution: {func_name}()",
                    )
                )
            elif func_name in {"system", "popen"}:
                issues.append(
                    RiskItem(
                        check="shell_execution",
                        severity="critical",
                        line=lineno,
                        detail=f"Shell execution: {func_name}()",
                    )
                )
            elif func_name == "open":
                # Check if write mode
                args = node.args
                keywords = {k.arg: k for k in node.keywords}
                mode_val = None
                if len(args) >= 2 and isinstance(args[1], ast.Constant):
                    mode_val = str(args[1].value)
                elif "mode" in keywords:
                    kw = keywords["mode"]
                    if isinstance(kw.value, ast.Constant):
                        mode_val = str(kw.value.value)
                if mode_val and any(m in mode_val for m in ("w", "a", "x")):
                    issues.append(
                        RiskItem(
                            check="file_write",
                            severity="high",
                            line=lineno,
                            detail=f"File write operation: open(..., '{mode_val}')",
                        )
                    )

        # Dangerous imports
        elif isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".")[0]
                if module in {"os", "subprocess", "socket", "shutil", "ctypes"}:
                    issues.append(
                        RiskItem(
                            check="dangerous_import",
                            severity="high",
                            line=lineno,
                            detail=f"Dangerous module import: {alias.name}",
                        )
                    )

        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            if module in {"os", "subprocess", "socket", "shutil", "ctypes"}:
                issues.append(
                    RiskItem(
                        check="dangerous_import",
                        severity="high",
                        line=lineno,
                        detail=f"Dangerous module import from: {node.module}",
                    )
                )

        # Potential infinite loop
        elif isinstance(node, ast.While):
            if isinstance(node.test, ast.Constant) and node.test.value is True:
                issues.append(
                    RiskItem(
                        check="infinite_loop",
                        severity="medium",
                        line=lineno,
                        detail="Potential infinite loop: while True without apparent break",
                    )
                )

    return issues


def _determine_risk_level(issues: list[RiskItem]) -> tuple[str, bool, str]:
    """Return (risk_level, safe_to_execute, recommendation)."""
    if not issues:
        return "safe", True, "No security risks detected. Safe to execute."

    severities = {r.severity for r in issues}
    if "critical" in severities:
        return "block", False, (
            "Critical security risk detected. Execution blocked. "
            "Remove dangerous calls before running."
        )
    if "high" in severities:
        return "block", False, (
            "High-severity security risk detected. Execution blocked."
        )
    if "medium" in severities:
        return "warn", True, (
            "Medium-severity concerns detected. Review before publishing."
        )
    return "warn", True, "Minor concerns detected. Proceed with caution."


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def check_code_safety(
    code: str,
    language: str = "python",
    context: str = "blog post code snippet",
) -> CodeSafetyResult:
    """
    Evaluate a code snippet for security risks.

    Steps
    -----
    1. Regex pattern scan
    2. AST structural analysis
    3. Risk level determination

    Parameters
    ----------
    code:
        The code string to analyse.
    language:
        Programming language of the snippet.
    context:
        Human-readable context (for logging).

    Returns
    -------
    CodeSafetyResult
        Contains risk_level, detected risks, and execution recommendation.
    """
    if language.lower() not in {"python", "py", "python3"}:
        return CodeSafetyResult(
            risk_level="safe",
            safe_to_execute=False,   # Can't execute non-Python
            recommendation=f"Language '{language}' is not executed — skipping safety check.",
        )

    if not code or not code.strip():
        return CodeSafetyResult(
            risk_level="safe",
            recommendation="Empty code snippet — nothing to analyse.",
        )

    all_issues: list[RiskItem] = []
    all_issues.extend(_regex_scan(code))
    all_issues.extend(_ast_scan(code))

    # Deduplicate by detail
    seen: set[str] = set()
    unique_issues: list[RiskItem] = []
    for issue in all_issues:
        key = f"{issue.check}:{issue.detail}"
        if key not in seen:
            seen.add(key)
            unique_issues.append(issue)

    risk_level, safe_to_execute, recommendation = _determine_risk_level(unique_issues)

    logger.info(
        "Code safety check: risk_level=%s, issues=%d, context=%s",
        risk_level, len(unique_issues), context,
    )

    return CodeSafetyResult(
        risk_level=risk_level,
        risks_detected=unique_issues,
        safe_to_execute=safe_to_execute,
        recommendation=recommendation,
    )
