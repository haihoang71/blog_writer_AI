"""
config/tracing.py
─────────────────
Langfuse observability setup and callback-handler factory.

Usage
-----
    from config.tracing import get_langfuse_callback, flush_langfuse

    callbacks = get_langfuse_callback()          # None when tracing disabled
    config = {"callbacks": callbacks} if callbacks else {}

    result = graph.invoke(state, config=config)
    flush_langfuse()                             # ensure spans are sent
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def get_langfuse_callback() -> list[Any] | None:
    """
    Build and return a list containing a Langfuse CallbackHandler, or
    return ``None`` if Langfuse is not configured / tracing is disabled.
    """
    from config.settings import get_settings

    settings = get_settings()

    if not settings.enable_tracing:
        logger.debug("Tracing disabled via ENABLE_TRACING=false")
        return None

    if not settings.is_langfuse_configured:
        logger.info(
            "Langfuse keys not configured — tracing disabled. "
            "Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY in your .env to enable."
        )
        return None

    try:
        # NOTE: The Langfuse Python SDK v3+ (installed here) replaced the old
        # `langfuse.callback.CallbackHandler` (v2 API) with a LangChain
        # integration at `langfuse.langchain.CallbackHandler`. The handler no
        # longer takes public_key/secret_key/host/debug directly — instead it
        # reads from the underlying `Langfuse` client singleton, so we
        # explicitly initialise that client first with our settings.
        from langfuse import Langfuse  # type: ignore[import]
        from langfuse.langchain import CallbackHandler  # type: ignore[import]

        Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key.get_secret_value(),
            host=settings.langfuse_host,
            debug=(settings.environment == "development"),
        )
        handler = CallbackHandler()
        logger.info(
            "Langfuse tracing enabled → host=%s", settings.langfuse_host
        )
        return [handler]

    except ImportError:
        logger.warning(
            "langfuse package not installed. Run: pip install langfuse"
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to initialise Langfuse: %s", exc)
        return None


def flush_langfuse() -> None:
    """Flush pending Langfuse spans. Call at end of request / CLI run."""
    try:
        from langfuse import get_client  # type: ignore[import]

        get_client().flush()
    except Exception:  # noqa: BLE001
        pass  # silently ignore if not configured


def build_run_config(
    run_name: str = "blog-generation",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a LangGraph ``config`` dict that includes Langfuse callbacks
    and useful run-level metadata.

    Parameters
    ----------
    run_name:
        Human-readable label shown in the Langfuse dashboard.
    tags:
        Optional list of string tags attached to the trace.
    metadata:
        Optional key/value metadata forwarded to Langfuse.

    Returns
    -------
    dict
        Ready to pass as ``graph.invoke(state, config=<this>)``.
    """
    config: dict[str, Any] = {
        "run_name": run_name,
        "tags": tags or ["blog-generator"],
        "metadata": metadata or {},
    }

    callbacks = get_langfuse_callback()
    if callbacks:
        config["callbacks"] = callbacks

    return config
