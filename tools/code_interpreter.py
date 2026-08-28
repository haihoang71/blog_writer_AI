"""
tools/code_interpreter.py
──────────────────────────
Secure code execution sandbox — local subprocess only.

Execution chain
---------------
1. AST security scan via ``guardrails.code_sandbox_guard``
2. Run in a subprocess with best-effort resource limits (no cloud dependency)
3. Capture stdout/stderr and return structured result

NOTE: this used to also support E2B (a paid cloud sandbox) as the primary
path, falling back to this local subprocess only if E2B wasn't configured.
That's been removed — running everything locally avoids requiring a paid
E2B API key, at the cost of weaker isolation (this is documented as a
"best-effort" sandbox, not true containment; don't run untrusted code here).

Usage
-----
    from tools.code_interpreter import execute_code

    result = execute_code(code="print('hello')", language="python")
    print(result.status)   # "passed"
    print(result.output)   # "hello"
"""

from __future__ import annotations

import ast
import json
import logging
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from typing import Optional

from config.constants import SANDBOX_TIMEOUT_SECONDS
from config.settings import get_settings
from langchain_core.tools import tool

logger = logging.getLogger(__name__)
settings = get_settings()

_ARTIFACT_MARKER = "##CODE_INTERPRETER_ARTIFACTS##"


# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ExecutionResult:
    """Result of a sandbox code execution."""

    status: str = "skipped"       # passed | failed | skipped | blocked
    output: str = ""
    error: str = ""
    execution_time_ms: float = 0.0
    language: str = "python"
    sandbox_type: str = "none"    # local | none
    artifacts: list[str] = field(default_factory=list)
    """Filenames of any images (e.g. matplotlib figures) saved during
    execution. Present only when ``assets_dir`` was passed to
    ``execute_code`` and the snippet produced at least one figure. Callers
    resolve these against the same ``assets_dir`` they supplied."""


# ─────────────────────────────────────────────────────────────────────────────
# AST-Based Safety Pre-Check
# ─────────────────────────────────────────────────────────────────────────────

_DANGEROUS_CALLS = frozenset(
    {
        "eval", "exec", "compile", "__import__",
        "input", "breakpoint",
    }
)
"""Calls that are always unsafe. NOTE: ``open`` is intentionally NOT in this
blanket list — read-only file access (e.g. loading a small CSV to plot) is a
normal part of writing blog demo/visualisation code. Only write/append/
exclusive-mode ``open()`` calls are flagged, via ``_check_open_call`` below,
mirroring the behaviour of ``guardrails.code_sandbox_guard``."""


def _check_open_call(node: ast.Call) -> Optional[str]:
    """Return a warning string if this ``open(...)`` call uses a write mode."""
    args = node.args
    keywords = {k.arg: k for k in node.keywords}
    mode_val: Optional[str] = None
    if len(args) >= 2 and isinstance(args[1], ast.Constant):
        mode_val = str(args[1].value)
    elif "mode" in keywords and isinstance(keywords["mode"].value, ast.Constant):
        mode_val = str(keywords["mode"].value.value)

    if mode_val and any(m in mode_val for m in ("w", "a", "x")):
        return f"File write operation: open(..., '{mode_val}')"
    return None

_DANGEROUS_IMPORTS = frozenset(
    {
        "os", "sys", "subprocess", "socket", "shutil",
        "pathlib", "ctypes", "importlib", "pickle",
        "shelve", "multiprocessing", "threading",
        "signal", "resource", "mmap",
    }
)


def _ast_safety_check(code: str) -> tuple[bool, list[str]]:
    """
    Parse code AST and detect dangerous patterns.

    Returns
    -------
    tuple[bool, list[str]]
        (is_safe, list_of_warnings)
    """
    warnings: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return False, [f"Syntax error: {exc}"]

    for node in ast.walk(tree):
        # Detect dangerous built-in calls
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id == "open":
                    open_warning = _check_open_call(node)
                    if open_warning:
                        warnings.append(open_warning)
                elif node.func.id in _DANGEROUS_CALLS:
                    warnings.append(f"Dangerous call detected: {node.func.id}()")
            elif isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                if attr in {"system", "popen", "run", "call", "Popen", "connect",
                            "rmtree", "unlink", "remove", "listdir"}:
                    warnings.append(f"Dangerous method detected: .{attr}()")

        # Detect dangerous imports
        elif isinstance(node, ast.Import):
            for alias in node.names:
                module = alias.name.split(".")[0]
                if module in _DANGEROUS_IMPORTS:
                    warnings.append(f"Dangerous import detected: import {alias.name}")

        elif isinstance(node, ast.ImportFrom):
            module = (node.module or "").split(".")[0]
            if module in _DANGEROUS_IMPORTS:
                warnings.append(f"Dangerous import detected: from {node.module} import ...")

    is_safe = len(warnings) == 0
    return is_safe, warnings


# ─────────────────────────────────────────────────────────────────────────────
# Local Subprocess Sandbox
# ─────────────────────────────────────────────────────────────────────────────


def _run_local(
    code: str,
    timeout: int = SANDBOX_TIMEOUT_SECONDS,
    assets_dir: Optional[str] = None,
) -> ExecutionResult:
    """
    Execute Python code in a subprocess with a hard timeout.
    This is a best-effort sandbox (resource limits + AST pre-check), not a
    true isolation boundary — don't run untrusted code here.

    If *assets_dir* is provided, the subprocess is configured with a
    non-interactive matplotlib backend ("Agg") and, after the user code
    finishes, any open matplotlib figures are saved as PNGs into that
    directory. This is what lets blog-writing agents (and the sandbox UI)
    actually produce chart images rather than just text output.
    """
    import time

    assets_dir_repr = repr(assets_dir) if assets_dir else "None"

    # Wrap code with resource limits
    wrapped = textwrap.dedent(
        f"""
import sys
import signal

# Limit CPU time (Unix only — the resource module does not exist on Windows)
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, ({min(timeout, 10)}, {min(timeout, 10)}))
except Exception:
    pass

# Limit max output
class _LimitedStdout:
    MAX_BYTES = 10_000
    def __init__(self, inner): self._inner = inner; self._written = 0
    def write(self, s):
        if self._written < self.MAX_BYTES:
            self._inner.write(s); self._written += len(s)
    def flush(self): self._inner.flush()

_real_stdout = sys.stdout
sys.stdout = _LimitedStdout(_real_stdout)

_ASSETS_DIR = {assets_dir_repr}
if _ASSETS_DIR:
    import os
    os.makedirs(_ASSETS_DIR, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless — no display available in sandbox
    except Exception:
        pass

# === USER CODE ===
{code}
# === END USER CODE ===

if _ASSETS_DIR:
    try:
        import sys as _sys
        _plt_mod = _sys.modules.get("matplotlib.pyplot")
        if _plt_mod is not None and _plt_mod.get_fignums():
            import json as _json
            import uuid as _uuid
            _saved = []
            for _num in _plt_mod.get_fignums():
                _fig = _plt_mod.figure(_num)
                _fname = f"figure-{{_uuid.uuid4().hex[:8]}}.png"
                _fig.savefig(_sys.modules["os"].path.join(_ASSETS_DIR, _fname),
                             bbox_inches="tight", dpi=120)
                _saved.append(_fname)
            _plt_mod.close("all")
            if _saved:
                _real_stdout.write("\\n{_ARTIFACT_MARKER}" + _json.dumps(_saved) + "\\n")
                _real_stdout.flush()
    except Exception as _artifact_exc:
        _real_stdout.write(f"\\n[artifact-capture-warning] {{_artifact_exc}}\\n")
"""
    )

    start = time.perf_counter()
    try:
        proc = subprocess.run(
            [sys.executable, "-c", wrapped],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        status = "passed" if proc.returncode == 0 else "failed"

        raw_stdout = proc.stdout
        artifacts: list[str] = []
        marker_idx = raw_stdout.find(_ARTIFACT_MARKER)
        if marker_idx != -1:
            before = raw_stdout[:marker_idx]
            after_line = raw_stdout[marker_idx + len(_ARTIFACT_MARKER):].splitlines()
            json_line = after_line[0] if after_line else "[]"
            try:
                artifacts = json.loads(json_line)
            except json.JSONDecodeError:
                artifacts = []
            remaining = "\n".join(after_line[1:])
            raw_stdout = (before + remaining).strip()

        return ExecutionResult(
            status=status,
            output=raw_stdout[:5000],
            error=proc.stderr[:2000],
            execution_time_ms=round(elapsed_ms, 2),
            sandbox_type="local",
            language="python",
            artifacts=artifacts,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            status="failed",
            error=f"Execution timed out after {timeout}s",
            sandbox_type="local",
            language="python",
        )
    except Exception as exc:  # noqa: BLE001
        return ExecutionResult(
            status="failed",
            error=str(exc),
            sandbox_type="local",
            language="python",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Public Interface
# ─────────────────────────────────────────────────────────────────────────────


def execute_code(
    code: str,
    language: str = "python",
    skip_safety_check: bool = False,
    timeout: int = SANDBOX_TIMEOUT_SECONDS,
    assets_dir: Optional[str] = None,
) -> ExecutionResult:
    """
    Execute a code snippet securely.

    Steps
    -----
    1. Skip if language is not Python (only Python execution is supported).
    2. Run AST safety check unless ``skip_safety_check=True``.
    3. Run in the local subprocess sandbox.
    4. Return ``ExecutionResult``.

    Parameters
    ----------
    assets_dir:
        Optional directory to save any chart/plot images (e.g. matplotlib
        figures) produced by the snippet. When provided, ``result.artifacts``
        will list the saved filenames (relative to this directory). Used by
        the Critic agent and the sandbox UI to turn ad-hoc visualisation
        code into blog-ready images.
    """
    if language.lower() not in {"python", "py", "python3"}:
        logger.debug("Skipping execution for non-Python language: %s", language)
        return ExecutionResult(
            status="skipped",
            output=f"Execution skipped: language '{language}' not supported.",
            language=language,
        )

    if not settings.enable_code_sandbox:
        return ExecutionResult(
            status="skipped",
            output="Code sandbox disabled via ENABLE_CODE_SANDBOX=false.",
            language=language,
        )

    # ── AST safety check ──────────────────────────────────────────────────
    if not skip_safety_check:
        is_safe, warnings = _ast_safety_check(code)
        if not is_safe:
            logger.warning("Code blocked by AST safety check: %s", warnings)
            return ExecutionResult(
                status="blocked",
                error=f"Safety check failed: {'; '.join(warnings)}",
                language=language,
            )
        if warnings:
            logger.info("Code safety warnings (proceeding): %s", warnings)

    # ── Local sandbox ─────────────────────────────────────────────────────
    return _run_local(code, timeout=timeout, assets_dir=assets_dir)


def run_sandbox_snippet(
    code: str,
    assets_dir: str,
    timeout: int = SANDBOX_TIMEOUT_SECONDS,
) -> dict:
    """
    Convenience wrapper for ad-hoc "try this code" / visualisation requests
    (e.g. the UI's Sandbox tab). Runs a safety check + execution and returns
    a JSON-serialisable dict rather than a dataclass.
    """
    from guardrails.code_sandbox_guard import check_code_safety

    safety = check_code_safety(code, language="python")
    if not safety.safe_to_execute:
        return {
            "status": "blocked",
            "output": "",
            "error": safety.recommendation,
            "artifacts": [],
        }

    result = execute_code(code=code, language="python", timeout=timeout, assets_dir=assets_dir)
    return {
        "status": result.status,
        "output": result.output,
        "error": result.error,
        "execution_time_ms": result.execution_time_ms,
        "artifacts": result.artifacts,
    }


# ── LangChain Tool Wrapper ─────────────────────────────────────────────────


@tool
def python_repl(code: str) -> str:
    """
    Execute a Python code snippet in a secure sandbox and return the output.

    Use this tool to validate code examples, run demonstrations, or verify
    computational results before including them in the blog post.

    Parameters
    ----------
    code:
        Valid Python code to execute. Avoid dangerous system calls.

    Returns
    -------
    str
        JSON-encoded ExecutionResult with status, output, and error fields.
    """
    import json
    from dataclasses import asdict

    result = execute_code(code=code, language="python")
    return json.dumps(asdict(result), indent=2)


CODE_TOOLS = [python_repl]
"""List of code execution tools for binding to the Critic agent."""
