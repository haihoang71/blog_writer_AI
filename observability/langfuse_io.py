"""Fetch raw Langfuse traces with the server-side secret. Never send the key to the UI."""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_MAX_RAW_BYTES = 1_000_000


def fetch_raw_trace(trace_id: str) -> dict[str, Any] | None:
    """GET Langfuse public API for one trace + observations. Returns None if pending."""
    from config.settings import get_settings

    settings = get_settings()
    if not settings.is_langfuse_configured:
        return None
    if not trace_id:
        return None
    host = settings.langfuse_host.rstrip("/")
    auth = (settings.langfuse_public_key, settings.langfuse_secret_key.get_secret_value())
    try:
        with httpx.Client(timeout=8.0) as client:
            trace_resp = client.get(f"{host}/api/public/traces/{trace_id}", auth=auth)
            if trace_resp.status_code == 404:
                return None
            trace_resp.raise_for_status()
            obs_resp = client.get(
                f"{host}/api/public/v2/observations",
                params={"traceId": trace_id, "limit": 1000},
                auth=auth,
            )
            observations: list[Any] = []
            if obs_resp.status_code == 200:
                body = obs_resp.json()
                observations = body.get("data") or body.get("observations") or []
            payload = {
                "trace": trace_resp.json(),
                "observations": observations,
            }
            return _cap_payload(payload)
    except (httpx.HTTPError, ValueError) as exc:
        logger.info("Langfuse raw fetch not ready: %s", type(exc).__name__)
        return None


def _cap_payload(payload: dict[str, Any]) -> dict[str, Any]:
    raw = str(payload)
    if len(raw.encode("utf-8")) <= _MAX_RAW_BYTES:
        return payload
    return {
        **payload,
        "observations": (payload.get("observations") or [])[:50],
        "_truncated": True,
    }


def redact_raw_for_public(payload: dict[str, Any]) -> dict[str, Any]:
    """Drop secrets and trim large I/O for the public raw viewer."""
    import copy

    redacted = copy.deepcopy(payload)
    trace = redacted.get("trace")
    if isinstance(trace, dict):
        for key in list(trace.keys()):
            if "secret" in key.lower() or "password" in key.lower() or key.endswith("Key"):
                trace[key] = "[redacted]"
    for obs in redacted.get("observations") or []:
        if not isinstance(obs, dict):
            continue
        meta = obs.get("metadata")
        if isinstance(meta, dict):
            meta.pop("expected_detector", None)
        for io_key in ("input", "output"):
            value = obs.get(io_key)
            text = value if isinstance(value, str) else str(value)
            if len(text) > 4000:
                obs[io_key] = text[:4000] + "…[truncated]"
                obs["_io_truncated"] = True
    redacted["_redacted"] = True
    return redacted
