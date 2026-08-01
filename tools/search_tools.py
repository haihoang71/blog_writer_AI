"""
tools/search_tools.py
─────────────────────
Web and academic search tools with graceful mock fallback.

Tools exposed
-------------
- ``tavily_search``            : Tavily real-time web search (or mock)
- ``arxiv_search``              : ArXiv academic paper search (or mock)
- ``semantic_scholar_search``   : Semantic Scholar search — citation counts,
                                   abstracts, venue/year; complements arXiv
                                   for papers outside arXiv and for citation
                                   metrics. No API key required.
- ``crossref_search``           : CrossRef search — DOI-backed citation
                                   metadata across virtually all publishers
                                   (journals, conferences, books), useful for
                                   citing sources that aren't on arXiv. No
                                   API key required.

All tools are LangChain ``@tool``-decorated callables that can be bound
directly to an agent via ``.bind_tools([...])``.
"""

from __future__ import annotations

import json
import logging
import time
from functools import wraps
from typing import Any

from langchain_core.tools import tool

from config.constants import MAX_SEARCH_RESULTS, SEARCH_TIMEOUT_SECONDS
from config.settings import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


# ── Retry decorator ────────────────────────────────────────────────────────

def _with_retry(max_attempts: int = 3, delay: float = 1.5):
    """Simple exponential-backoff retry decorator."""
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    wait = delay * (2 ** (attempt - 1))
                    logger.warning(
                        "%s attempt %d/%d failed: %s — retrying in %.1fs",
                        fn.__name__, attempt, max_attempts, exc, wait,
                    )
                    time.sleep(wait)
            raise RuntimeError(
                f"{fn.__name__} failed after {max_attempts} attempts"
            ) from last_exc
        return wrapper
    return decorator


# ── Mock helpers ────────────────────────────────────────────────────────────

def _mock_tavily_results(query: str, max_results: int) -> list[dict[str, Any]]:
    """Return synthetic search results for testing without an API key."""
    logger.info("[MOCK] tavily_search — query=%r", query)
    return [
        {
            "title": f"[MOCK] Result {i + 1} for: {query}",
            "url": f"https://example.com/mock/{i + 1}",
            "content": (
                f"This is a mock search result ({i + 1}) for the query '{query}'. "
                "In production, real content would appear here from Tavily API. "
                "Configure TAVILY_API_KEY in your .env to enable live search."
            ),
            "score": round(0.95 - i * 0.05, 2),
        }
        for i in range(min(max_results, 3))
    ]


def _mock_arxiv_results(query: str, max_results: int) -> list[dict[str, Any]]:
    """Return synthetic arXiv results for testing."""
    logger.info("[MOCK] arxiv_search — query=%r", query)
    results = [
        {
            "title": f"[MOCK] arXiv Paper {i + 1}: {query}",
            "authors": ["A. Researcher", "B. Scientist"],
            "summary": (
                f"This is a mock arXiv abstract ({i + 1}) for '{query}'. "
                "Configure the system with live internet access to retrieve real papers."
            ),
            "url": f"https://arxiv.org/abs/2024.{10000 + i}",
            "published": "2024-01-01",
            "categories": ["cs.AI", "cs.LG"],
        }
        for i in range(min(max_results, 2))
    ]
    for r in results:
        r["citation"] = format_arxiv_citation(r)
    return results


def format_arxiv_citation(paper: dict[str, Any]) -> str:
    """
    Build a compact, human-readable citation string for an arXiv paper dict
    (as produced by ``arxiv_search`` / ``_mock_arxiv_results``).

    Format: ``Author1, Author2 et al. (Year). Title. arXiv:<id>. <url>``
    """
    authors: list[str] = paper.get("authors") or []
    if len(authors) == 1:
        author_str = authors[0]
    elif len(authors) == 2:
        author_str = f"{authors[0]} & {authors[1]}"
    elif authors:
        author_str = f"{authors[0]} et al."
    else:
        author_str = "Unknown Author"

    published: str = paper.get("published", "")
    year = published.split("-")[0] if published else "n.d."

    title = paper.get("title", "Untitled")
    url = paper.get("url", "")
    arxiv_id = url.rstrip("/").rsplit("/", 1)[-1] if url else "unknown"

    return f"{author_str} ({year}). {title}. arXiv:{arxiv_id}. {url}"


def _mock_semantic_scholar_results(query: str, max_results: int) -> list[dict[str, Any]]:
    """Return synthetic Semantic Scholar results for testing."""
    logger.info("[MOCK] semantic_scholar_search — query=%r", query)
    return [
        {
            "title": f"[MOCK] Semantic Scholar Paper {i + 1}: {query}",
            "authors": ["C. Researcher"],
            "year": 2024,
            "venue": "Mock Conference on AI",
            "citation_count": 100 - i * 10,
            "abstract": (
                f"This is a mock abstract ({i + 1}) for '{query}'. "
                "Live results require internet access to the Semantic Scholar API."
            ),
            "url": f"https://www.semanticscholar.org/paper/mock-{i + 1}",
            "doi": None,
        }
        for i in range(min(max_results, 3))
    ]


def _mock_crossref_results(query: str, max_results: int) -> list[dict[str, Any]]:
    """Return synthetic CrossRef results for testing."""
    logger.info("[MOCK] crossref_search — query=%r", query)
    return [
        {
            "title": f"[MOCK] CrossRef Work {i + 1}: {query}",
            "authors": ["D. Author"],
            "container_title": "Mock Journal of Software Engineering",
            "published": "2024-01-01",
            "doi": f"10.0000/mock.{i + 1}",
            "url": f"https://doi.org/10.0000/mock.{i + 1}",
        }
        for i in range(min(max_results, 3))
    ]


# ── Tavily Search ──────────────────────────────────────────────────────────


@tool
def tavily_search(query: str, max_results: int = MAX_SEARCH_RESULTS) -> str:
    """
    Search the web for up-to-date technical information using the Tavily API.

    Parameters
    ----------
    query:
        A focused search query string.
    max_results:
        Number of results to return (default 5, max 20).

    Returns
    -------
    str
        JSON-encoded list of search results with title, url, and content.
    """
    max_results = min(max_results, MAX_SEARCH_RESULTS)

    if not settings.is_tavily_configured:
        results = _mock_tavily_results(query, max_results)
        return json.dumps(results, ensure_ascii=False, indent=2)

    @_with_retry(max_attempts=3, delay=1.0)
    def _live_search() -> list[dict[str, Any]]:
        from tavily import TavilyClient  # type: ignore[import]

        client = TavilyClient(
            api_key=settings.tavily_api_key.get_secret_value()
        )
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="advanced",
            include_answer=False,
            timeout=SEARCH_TIMEOUT_SECONDS,
        )
        return response.get("results", [])

    try:
        results = _live_search()
        logger.info(
            "tavily_search: returned %d results for %r", len(results), query
        )
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.error("tavily_search failed, falling back to mock: %s", exc)
        results = _mock_tavily_results(query, max_results)
        return json.dumps(results, ensure_ascii=False, indent=2)


# ── ArXiv Search ───────────────────────────────────────────────────────────


@tool
def arxiv_search(query: str, max_results: int = 3) -> str:
    """
    Search arXiv for academic papers on ML/AI topics.

    Each result includes a ready-to-use ``citation`` string so downstream
    agents (e.g. the Academic Researcher) don't need to reformat citations
    themselves.

    Parameters
    ----------
    query:
        Search query optimised for academic paper titles/abstracts.
    max_results:
        Number of papers to retrieve (default 3).

    Returns
    -------
    str
        JSON-encoded list of papers with title, authors, summary, url, and
        a formatted ``citation`` string.

    Notes
    -----
    arXiv's public search API is free and keyless. ``settings.arxiv_api_key``
    is intentionally not required here — it's reserved for optional
    arXiv-adjacent services (rate-limit tiers, citation enrichment) that a
    future integration might use.
    """
    max_results = min(max_results, settings.arxiv_max_results)

    try:
        import arxiv  # type: ignore[import]

        client = arxiv.Client(num_retries=3, delay_seconds=1.0)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers = []
        for result in client.results(search):
            paper = {
                "title": result.title,
                "authors": [str(a) for a in result.authors[:3]],
                "summary": result.summary[:400],
                "url": result.entry_id,
                "published": str(result.published.date()),
                "categories": result.categories,
            }
            paper["citation"] = format_arxiv_citation(paper)
            papers.append(paper)
        logger.info(
            "arxiv_search: returned %d papers for %r", len(papers), query
        )
        return json.dumps(papers, ensure_ascii=False, indent=2)

    except ImportError:
        logger.warning("arxiv package not installed, using mock results.")
        results = _mock_arxiv_results(query, max_results)
        return json.dumps(results, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        logger.error("arxiv_search failed, falling back to mock: %s", exc)
        results = _mock_arxiv_results(query, max_results)
        return json.dumps(results, ensure_ascii=False, indent=2)


# ── Tool Registry ──────────────────────────────────────────────────────────

SEARCH_TOOLS = [tavily_search, arxiv_search]
"""List of all available search tools for binding to the Researcher agent."""
