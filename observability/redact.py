"""Strip secrets and eval labels from payloads that leave the process."""

from __future__ import annotations

from typing import Any

_KEY_MARKERS = (
    "secret",
    "password",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "authorization",
    "private_key",
)


def is_secret_key(key: str) -> bool:
    """Return True when a mapping key looks like a credential field."""
    lowered = key.lower().replace("-", "_")
    if lowered.endswith("key") and "public" not in lowered:
        return True
    return any(marker in lowered for marker in _KEY_MARKERS)


def strip_secrets(value: Any) -> Any:
    """Recursively replace credential-looking fields with ``[redacted]``."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, inner in value.items():
            if is_secret_key(str(key)):
                out[key] = "[redacted]"
            elif str(key) == "expected_detector":
                continue
            else:
                out[key] = strip_secrets(inner)
        return out
    if isinstance(value, list):
        return [strip_secrets(item) for item in value]
    return value


def contains_secret_value(payload: Any, secrets: list[str]) -> list[str]:
    """Return which of ``secrets`` appear as substrings in ``payload`` JSON text."""
    blob = str(payload)
    found: list[str] = []
    for secret in secrets:
        if secret and len(secret) >= 8 and secret in blob:
            found.append(secret)
    return found
