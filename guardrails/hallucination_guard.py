"""
guardrails/hallucination_guard.py
──────────────────────────────────
Hallucination guardrail: verifies draft claims against research data.

Approach
--------
1. Extract all factual sentences from the draft (heuristic).
2. For each claim, compute similarity against research findings.
3. Flag claims with similarity below HALLUCINATION_SIMILARITY_THRESHOLD.
4. Return a structured report with flagged sentences.

For similarity, we use:
- TF-IDF cosine similarity (always available, no API needed)
- Optional: OpenAI embeddings for higher accuracy
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from config.constants import HALLUCINATION_SIMILARITY_THRESHOLD
from config.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ClaimVerification:
    """Verification result for a single factual claim."""

    claim: str
    max_similarity: float
    matched_finding: str = ""
    status: str = "verified"    # verified | flagged | unverifiable


@dataclass
class HallucinationReport:
    """Full hallucination check report for a blog draft."""

    total_claims: int = 0
    verified: int = 0
    flagged: int = 0
    unverifiable: int = 0
    overall_faithfulness_score: float = 1.0
    claim_verifications: list[ClaimVerification] = field(default_factory=list)
    flagged_sentences: list[str] = field(default_factory=list)
    passed: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Text Utilities
# ─────────────────────────────────────────────────────────────────────────────


def _extract_claims(text: str) -> list[str]:
    """
    Extract sentences that likely contain factual claims from Markdown text.
    Heuristic: sentences with numbers, technical terms, or specific assertions.
    """
    # Strip Markdown formatting
    clean = re.sub(r"```[\s\S]*?```", "", text)    # remove code blocks
    clean = re.sub(r"`[^`]+`", "", clean)           # remove inline code
    clean = re.sub(r"#+\s+", "", clean)             # remove headings
    clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", clean)  # bold → plain
    clean = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", clean)  # links → text

    # Split into sentences
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    claims: list[str] = []
    for s in sentences:
        s = s.strip()
        if len(s) < 20:
            continue
        # Heuristics: contains a number, a proper noun (capital), or claim words
        has_number = bool(re.search(r"\d", s))
        has_technical = bool(re.search(
            r"\b(is|are|was|were|uses|enables|achieves|reduces|increases|"
            r"outperforms|implements|supports|requires|allows|provides)\b", s, re.I
        ))
        if has_number or has_technical:
            claims.append(s)

    return claims[:50]  # Cap at 50 claims per draft


def _tfidf_similarity(query: str, corpus: list[str]) -> float:
    """Compute max TF-IDF cosine similarity between query and corpus."""
    if not corpus:
        return 0.0
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        all_docs = [query] + corpus
        vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
        tfidf_matrix = vectorizer.fit_transform(all_docs)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        return float(np.max(similarities))
    except ImportError:
        # Fallback: simple word overlap (Jaccard-like)
        query_words = set(query.lower().split())
        max_sim = 0.0
        for doc in corpus:
            doc_words = set(doc.lower().split())
            if not query_words and not doc_words:
                continue
            intersection = query_words & doc_words
            union = query_words | doc_words
            sim = len(intersection) / len(union) if union else 0.0
            max_sim = max(max_sim, sim)
        return max_sim


def _build_research_corpus(research_data_dict: dict) -> list[str]:
    """Flatten all research findings and snippets into a text corpus."""
    corpus: list[str] = []
    for section in research_data_dict.get("sections", []):
        for finding in section.get("findings", []):
            corpus.append(finding.get("claim", ""))
            corpus.append(finding.get("excerpt", ""))
        for snippet in section.get("code_snippets", []):
            corpus.append(snippet.get("description", ""))
    corpus.extend(research_data_dict.get("key_takeaways", []))
    return [c for c in corpus if c.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def check_hallucination(
    draft: str,
    research_data_dict: dict,
    threshold: float = HALLUCINATION_SIMILARITY_THRESHOLD,
) -> HallucinationReport:
    """
    Check draft claims against research data for potential hallucinations.

    Parameters
    ----------
    draft:
        Markdown blog post draft text.
    research_data_dict:
        Serialised ``ResearchData`` as a plain dict.
    threshold:
        Minimum cosine similarity for a claim to be considered verified.

    Returns
    -------
    HallucinationReport
        Summary with flagged sentences and overall faithfulness score.
    """
    if not draft.strip():
        return HallucinationReport(passed=True)

    corpus = _build_research_corpus(research_data_dict)
    if not corpus:
        logger.warning(
            "No research corpus available — hallucination check skipped."
        )
        return HallucinationReport(
            passed=True,
            overall_faithfulness_score=1.0,
        )

    claims = _extract_claims(draft)
    if not claims:
        return HallucinationReport(passed=True, total_claims=0)

    verifications: list[ClaimVerification] = []
    flagged_sentences: list[str] = []

    for claim in claims:
        sim = _tfidf_similarity(claim, corpus)
        best_match = corpus[0] if corpus else ""

        # Find best matching corpus item
        if corpus:
            scores = [(c, _tfidf_similarity(claim, [c])) for c in corpus]
            best_match, _ = max(scores, key=lambda x: x[1])

        if sim >= threshold:
            status = "verified"
        elif sim >= threshold * 0.5:
            status = "unverifiable"
        else:
            status = "flagged"
            flagged_sentences.append(claim)

        verifications.append(
            ClaimVerification(
                claim=claim,
                max_similarity=round(sim, 4),
                matched_finding=best_match[:100],
                status=status,
            )
        )

    total = len(verifications)
    verified_count = sum(1 for v in verifications if v.status == "verified")
    flagged_count = sum(1 for v in verifications if v.status == "flagged")
    unverifiable_count = sum(1 for v in verifications if v.status == "unverifiable")

    faithfulness = verified_count / total if total > 0 else 1.0
    passed = faithfulness >= 0.5  # At least 50% claims verifiable

    logger.info(
        "Hallucination check: total=%d verified=%d flagged=%d faithfulness=%.2f",
        total, verified_count, flagged_count, faithfulness,
    )

    return HallucinationReport(
        total_claims=total,
        verified=verified_count,
        flagged=flagged_count,
        unverifiable=unverifiable_count,
        overall_faithfulness_score=round(faithfulness, 4),
        claim_verifications=verifications,
        flagged_sentences=flagged_sentences,
        passed=passed,
    )
