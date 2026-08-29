"""CORS, rate limit, secret redaction, ground-truth auth."""

from __future__ import annotations

import pytest

from api.security import RateLimiter, cors_origin_list
from observability.redact import contains_secret_value, strip_secrets


@pytest.mark.unit
def test_rate_limiter_blocks_after_budget() -> None:
    limiter = RateLimiter(per_min=3)
    now = 1_000_000.0
    assert limiter.allow("1.1.1.1", now=now)
    assert limiter.allow("1.1.1.1", now=now + 1)
    assert limiter.allow("1.1.1.1", now=now + 2)
    assert limiter.allow("1.1.1.1", now=now + 3) is False
    assert limiter.allow("2.2.2.2", now=now + 3) is True
    assert limiter.allow("1.1.1.1", now=now + 61) is True


@pytest.mark.unit
def test_strip_secrets_and_expected_detector() -> None:
    payload = {
        "langfuse_secret_key": "sk-lf-real",
        "openai_api_key": "sk-openai",
        "metadata": {"expected_detector": "loop", "agent_name": "writer"},
        "nested": {"password": "hunter2", "ok": 1},
    }
    redacted = strip_secrets(payload)
    assert redacted["langfuse_secret_key"] == "[redacted]"
    assert redacted["openai_api_key"] == "[redacted]"
    assert "expected_detector" not in redacted["metadata"]
    assert redacted["nested"]["password"] == "[redacted]"
    assert redacted["nested"]["ok"] == 1


@pytest.mark.unit
def test_contains_secret_value() -> None:
    found = contains_secret_value({"x": "abcSECRET99zz"}, ["SECRET99"])
    assert found == ["SECRET99"]


@pytest.mark.unit
def test_cors_never_wildcard_in_production(monkeypatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("CORS_ORIGINS", "http://localhost:8000,*")
    from config.settings import get_settings

    get_settings.cache_clear()
    origins = cors_origin_list()
    assert "*" not in origins
    get_settings.cache_clear()


@pytest.mark.unit
def test_ground_truth_requires_admin(client, monkeypatch) -> None:
    monkeypatch.setenv("ADMIN_TOKEN", "test-admin-token")
    from config.settings import get_settings

    get_settings.cache_clear()
    from faults.injector import inject
    from storage.run_store import upsert_run

    upsert_run({"run_id": "r-sec", "task_id": "t-sec", "status": "completed", "topic": "x"})
    inject("none", run_id="r-sec", task_id="t-sec", langfuse_trace_id=None, real_sleep=False)
    denied = client.get("/api/v1/runs/r-sec/ground-truth")
    assert denied.status_code in {401, 403}
    ok = client.get(
        "/api/v1/runs/r-sec/ground-truth",
        headers={"Authorization": "Bearer test-admin-token"},
    )
    assert ok.status_code == 200
    get_settings.cache_clear()


@pytest.mark.unit
def test_trace_payload_does_not_echo_langfuse_secret(client) -> None:
    from config.settings import get_settings
    from faults.injector import inject
    from storage.run_store import upsert_run

    upsert_run({"run_id": "r-secret", "task_id": "t-secret", "status": "completed", "topic": "x"})
    inject("none", run_id="r-secret", task_id="t-secret", langfuse_trace_id=None, real_sleep=False)
    response = client.get("/api/v1/runs/r-secret/trace")
    assert response.status_code == 200
    secret = get_settings().langfuse_secret_key.get_secret_value()
    blob = response.text
    if secret and len(secret) >= 8 and "placeholder" not in secret.lower():
        assert secret not in blob
    assert "expected_detector" not in blob
