"""
prompts/loader.py
─────────────────
Dynamic YAML prompt loader with template variable interpolation.

Features
--------
- Loads prompts from ``prompts/templates/`` or ``prompts/guardrails/``
- Validates that all required template variables are supplied
- LRU-caches loaded YAML files to avoid repeated disk I/O
- Supports prompt versioning (logged at load time)

Usage
-----
    from prompts.loader import load_prompt

    system_prompt = load_prompt(
        "planner",
        topic="LangGraph for production",
        allowed_domains="ML, AI, Coding",
        max_sections=8,
    )
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────
_PROMPTS_ROOT = Path(__file__).parent
_TEMPLATES_DIR = _PROMPTS_ROOT / "templates"
_GUARDRAILS_DIR = _PROMPTS_ROOT / "guardrails"

_SEARCH_DIRS: list[Path] = [_TEMPLATES_DIR, _GUARDRAILS_DIR]


class PromptNotFoundError(FileNotFoundError):
    """Raised when no YAML template matches the requested prompt name."""


class MissingTemplateVariableError(KeyError):
    """Raised when a required template variable is absent from kwargs."""


# ── Internal helpers ──────────────────────────────────────────────────────


@lru_cache(maxsize=64)
def _load_yaml(path: Path) -> dict[str, Any]:
    """Load and parse a YAML file; cached per unique path."""
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    logger.debug(
        "Loaded prompt '%s' v%s from %s",
        data.get("name", path.stem),
        data.get("version", "?"),
        path,
    )
    return data


def _find_yaml(name: str) -> Path:
    """Locate a YAML file by stem name across known search directories."""
    candidates = [name, f"{name}.yaml", f"{name}.yml"]
    for directory in _SEARCH_DIRS:
        for candidate in candidates:
            path = directory / candidate
            if path.exists():
                return path
    searched = ", ".join(str(d) for d in _SEARCH_DIRS)
    raise PromptNotFoundError(
        f"No prompt template named '{name}' found in: {searched}"
    )


_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")
"""
Matches simple single-word ``{variable_name}`` placeholders only.

Deliberately does NOT match multi-line/complex braces (e.g. the literal JSON
schema blocks embedded in the prompt YAML files, such as
``{ "title": "...", ... }``) because those contain whitespace/quotes right
after the opening brace, which this pattern requires to be a bare word
character. This lets prompt authors write both ``{topic}``-style template
variables and literal JSON examples in the same prompt without escaping.
"""


def _extract_variables(template: str) -> set[str]:
    """Return the set of ``{variable}`` placeholders in a prompt template."""
    return set(_PLACEHOLDER_RE.findall(template))


def _render(template: str, variables: dict[str, Any]) -> str:
    """
    Substitute ``{key}`` placeholders with values from *variables*.

    Only exact ``{word}`` matches corresponding to known variable names are
    replaced; any other braces in the template (e.g. literal JSON examples)
    are left untouched.

    Raises
    ------
    MissingTemplateVariableError
        If any placeholder in *template* has no matching key in *variables*.
    """
    required = _extract_variables(template)
    missing = required - set(variables.keys())
    if missing:
        raise MissingTemplateVariableError(
            f"Template requires variables that were not supplied: {missing}"
        )

    def _substitute(match: re.Match[str]) -> str:
        key = match.group(1)
        return str(variables[key]) if key in variables else match.group(0)

    return _PLACEHOLDER_RE.sub(_substitute, template)


# ── Public API ────────────────────────────────────────────────────────────


def load_prompt(name: str, **kwargs: Any) -> str:
    """
    Load the ``system`` prompt for *name* and render it with *kwargs*.

    Parameters
    ----------
    name:
        Stem of the YAML file (e.g., ``"planner"``, ``"input_moderation"``).
    **kwargs:
        Template variables referenced in the prompt's ``system`` field.

    Returns
    -------
    str
        The fully rendered system prompt string.

    Raises
    ------
    PromptNotFoundError
        If no matching YAML file is found.
    MissingTemplateVariableError
        If required template variables are absent.
    """
    path = _find_yaml(name)
    data = _load_yaml(path)

    raw_system: str = data.get("system", "")
    if not raw_system:
        logger.warning("Prompt '%s' has an empty 'system' field.", name)
        return ""

    rendered = _render(raw_system, kwargs) if kwargs else raw_system
    return rendered.strip()


def load_prompt_metadata(name: str) -> dict[str, Any]:
    """
    Return the full YAML metadata dict for a prompt (excluding 'system').

    Useful for inspecting version, description, etc.
    """
    path = _find_yaml(name)
    data = _load_yaml(path)
    return {k: v for k, v in data.items() if k != "system"}


def list_available_prompts() -> list[str]:
    """Return sorted list of all available prompt names."""
    found: list[str] = []
    for directory in _SEARCH_DIRS:
        if directory.exists():
            found.extend(
                p.stem for p in directory.glob("*.yaml") if p.is_file()
            )
    return sorted(set(found))


def invalidate_cache() -> None:
    """Clear the YAML loader cache (useful for hot-reload in development)."""
    _load_yaml.cache_clear()
    logger.debug("Prompt cache cleared.")
