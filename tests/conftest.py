"""Shared fixtures. Isolate SQLite under tmp_path so tests never touch data/."""

from __future__ import annotations

import pytest


_SETTING_MODULES = (
    "main",
    "graph.workflow",
    "agents.planner",
    "agents.writer",
    "agents.researcher",
    "agents.critic",
    "agents.academic_researcher",
    "guardrails.input_guard",
    "guardrails.hallucination_guard",
    "guardrails.code_sandbox_guard",
    "tools.search_tools",
    "tools.code_interpreter",
)


@pytest.fixture(autouse=True)
def isolate_data_dir(tmp_path, monkeypatch):
    """Point DATA_DIR at a temp folder, force mock LLM keys, rebind settings."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-placeholder")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-placeholder")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "")
    monkeypatch.setenv("ADMIN_TOKEN", "")
    monkeypatch.setenv("ENVIRONMENT", "development")
    from config.settings import get_settings

    get_settings.cache_clear()
    fresh = get_settings()
    for name in _SETTING_MODULES:
        mod = __import__(name, fromlist=["settings"])
        if hasattr(mod, "settings"):
            mod.settings = fresh
    posts_root = tmp_path / "blog_posts"
    posts_root.mkdir()
    monkeypatch.setattr("storage.post_manager.BLOG_POSTS_ROOT", posts_root)
    yield tmp_path
    get_settings.cache_clear()


@pytest.fixture
def client(isolate_data_dir):
    """Fresh TestClient bound to the isolated data dir."""
    from fastapi.testclient import TestClient

    from main import create_fastapi_app

    with TestClient(create_fastapi_app()) as test_client:
        yield test_client
