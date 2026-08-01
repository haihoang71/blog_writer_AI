"""
config/constants.py
───────────────────
Application-wide constants and limit definitions.
Centralise these here so every module imports from one source of truth.
"""

from __future__ import annotations

# ── Workflow Limits ──────────────────────────────────────────────────────────
MAX_REVISIONS: int = 3
"""Maximum Writer → Critic loops before forcing a HITL checkpoint."""

MAX_SEARCH_RESULTS: int = 5
"""Maximum web-search results returned per query."""

MAX_RESEARCH_QUERIES: int = 4
"""Maximum parallel search queries the Researcher may issue."""

MAX_ACADEMIC_PAPERS: int = 5
"""Maximum arXiv papers the Academic Researcher agent will pull in per run."""

MAX_OUTLINE_SECTIONS: int = 8
"""Maximum H2 sections the Planner may produce."""

MAX_CODE_SNIPPETS_PER_POST: int = 5
"""Maximum code blocks embedded in a single blog post."""

# ── Timeout Limits (seconds) ─────────────────────────────────────────────────
AGENT_TIMEOUT_SECONDS: int = 120
"""Hard timeout per individual agent node invocation."""

SANDBOX_TIMEOUT_SECONDS: int = 30
"""Maximum allowed execution time for sandboxed code."""

GRAPH_TIMEOUT_SECONDS: int = 600
"""Full-graph execution wall-clock timeout."""

SEARCH_TIMEOUT_SECONDS: int = 15
"""HTTP timeout for external search API calls."""

# ── Token Budgets ────────────────────────────────────────────────────────────
PLANNER_MAX_TOKENS: int = 2048
RESEARCHER_MAX_TOKENS: int = 2048
WRITER_MAX_TOKENS: int = 4096
CRITIC_MAX_TOKENS: int = 2048

# ── Retry Policy ─────────────────────────────────────────────────────────────
MAX_LLM_RETRIES: int = 3
RETRY_WAIT_SECONDS: float = 2.0
RETRY_BACKOFF_MULTIPLIER: float = 2.0

# ── Content Quality Thresholds ───────────────────────────────────────────────
MIN_DRAFT_WORD_COUNT: int = 600
"""Minimum word count for a draft to pass the output guardrail."""

MIN_OUTLINE_SECTIONS: int = 3
"""Minimum number of H2 sections required in an outline."""

HALLUCINATION_SIMILARITY_THRESHOLD: float = 0.35
"""Minimum cosine-similarity between claim and research context."""

# ── Guardrail Configuration ──────────────────────────────────────────────────
BLOCKED_TOPICS: list[str] = [
    "politics",
    "religion",
    "adult content",
    "gambling",
    "weapons",
    "drugs",
]
"""Topics that trigger the input guardrail to reject the request."""

ALLOWED_DOMAINS: list[str] = [
    "coding",
    "programming",
    "software engineering",
    "machine learning",
    "artificial intelligence",
    "data science",
    "devops",
    "cloud computing",
    "cybersecurity",
    "databases",
    "web development",
    "mobile development",
    "algorithms",
    "mathematics",
    "statistics",
    "robotics",
]
"""High-level domains the system is scoped to serve."""

ACADEMIC_TOPIC_KEYWORDS: list[str] = [
    # English
    "research paper", "research papers", "arxiv", "paper", "papers",
    "state of the art", "sota", "benchmark", "dataset", "neural network",
    "deep learning", "transformer", "reinforcement learning",
    "large language model", "llm", "diffusion model", "generative model",
    "theorem", "proof", "algorithm complexity", "peer-reviewed",
    "academic", "publication", "citation", "literature review",
    "survey paper", "novel approach", "empirical study", "ablation study",
    "neurips", "icml", "iclr", "acl", "cvpr", "conference paper",
    # Vietnamese
    "nghiên cứu khoa học", "bài báo khoa học", "công trình nghiên cứu",
    "học máy", "học sâu", "mạng nơ-ron", "mô hình ngôn ngữ lớn",
    "thuật toán", "định lý", "chứng minh", "khảo sát tài liệu",
]
"""
Keyword markers used by ``agents.academic_researcher`` to decide whether a
topic is heavily related to academic/scientific research and therefore
warrants also invoking the Academic Researcher agent (arXiv search +
citations) in addition to the general Researcher agent.
"""

SENSITIVE_CODE_PATTERNS: list[str] = [
    r"import\s+os\s*;?\s*os\.system",
    r"subprocess\.(run|call|Popen)",
    r"__import__\(",
    r"exec\s*\(",
    r"eval\s*\(",
    r"open\s*\(.*(w|a|x)\b",   # file-write modes
    r"socket\.connect",
    r"requests\.(get|post|put|delete)",
    r"urllib\.request",
    r"shutil\.(rmtree|move)",
    r"pathlib\.Path.*unlink",
]
"""Regex patterns that flag code snippets as potentially unsafe."""

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_FORMAT: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT: str = "%Y-%m-%dT%H:%M:%S"

# ── API Routes ────────────────────────────────────────────────────────────────
API_V1_PREFIX: str = "/api/v1"
GENERATE_ENDPOINT: str = f"{API_V1_PREFIX}/generate"
STATUS_ENDPOINT: str = f"{API_V1_PREFIX}/status"
REVIEW_ENDPOINT: str = f"{API_V1_PREFIX}/review"
