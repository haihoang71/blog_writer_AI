"""Make the test suite deterministic and network-free.

The application intentionally reads ``.env`` at import time. Without these
process-level overrides, a developer who has real credentials configured would
silently run integration tests against paid APIs and Langfuse. Conftest is
loaded before test-module collection, so these values take effect before any
module-level Settings singleton is created.
"""

from __future__ import annotations

import os


os.environ["OPENAI_API_KEY"] = "sk-placeholder"
os.environ["TAVILY_API_KEY"] = "tvly-placeholder"
os.environ["LANGFUSE_PUBLIC_KEY"] = ""
os.environ["LANGFUSE_SECRET_KEY"] = ""
os.environ["ENABLE_TRACING"] = "false"
