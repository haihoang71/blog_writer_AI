"""
storage/post_manager.py
────────────────────────
Manages the on-disk library of generated blog posts.

Directory layout
-----------------
    blog_posts/
      _generation/<run_id>/assets/*.png   ← scratch space while a graph run
                                             (or an ad-hoc sandbox snippet) is
                                             still executing. Code interpreter
                                             output (e.g. matplotlib figures)
                                             lands here first.
      <post_id>/
        post.md                           ← final Markdown blog post
        metadata.json                     ← title, topic, scores, timestamps...
        assets/*.png                      ← charts/plots referenced by post.md

Usage
-----
    from storage.post_manager import (
        get_generation_assets_dir,
        save_post,
        list_posts,
        get_post,
    )

    assets_dir = get_generation_assets_dir(run_id)
    ...
    record = save_post(final_state)
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Project root = parent directory of this "storage" package.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
BLOG_POSTS_ROOT = _PROJECT_ROOT / "blog_posts"
_GENERATION_ROOT = BLOG_POSTS_ROOT / "_generation"


def _slugify(text: str) -> str:
    slug = re.sub(r"\s+", "-", text.strip().lower())
    slug = re.sub(r"[^a-z0-9\-]", "", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug[:60] or "untitled"


# ─────────────────────────────────────────────────────────────────────────────
# Scratch / in-progress generation assets
# ─────────────────────────────────────────────────────────────────────────────


def get_generation_assets_dir(run_id: str) -> Path:
    """
    Return (creating if necessary) the scratch assets directory for an
    in-progress graph run or ad-hoc sandbox execution. Code snippets executed
    by the Critic node (e.g. matplotlib plots) save their output PNGs here;
    once the run finishes, ``save_post`` moves them into the post's
    permanent folder.
    """
    path = _GENERATION_ROOT / run_id / "assets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def discard_generation_dir(run_id: str) -> None:
    """Remove a run's scratch directory (e.g. when a topic was blocked)."""
    path = _GENERATION_ROOT / run_id
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def cleanup_stale_generation_dirs(max_age_hours: float = 24.0) -> int:
    """Remove scratch generation directories older than *max_age_hours*."""
    if not _GENERATION_ROOT.exists():
        return 0
    removed = 0
    cutoff = time.time() - max_age_hours * 3600
    for child in _GENERATION_ROOT.iterdir():
        try:
            if child.is_dir() and child.stat().st_mtime < cutoff:
                shutil.rmtree(child, ignore_errors=True)
                removed += 1
        except OSError:
            continue
    return removed


def new_sandbox_run_id() -> str:
    """Generate a run id for an ad-hoc (not part of a blog generation) snippet."""
    return f"sandbox-{uuid.uuid4().hex[:12]}"


# ─────────────────────────────────────────────────────────────────────────────
# Post persistence
# ─────────────────────────────────────────────────────────────────────────────


def save_post(state: dict[str, Any]) -> dict[str, Any]:
    """
    Persist a completed (or best-effort) graph run as a post folder under
    ``blog_posts/``.

    Parameters
    ----------
    state:
        The final ``BlogState`` dict returned by ``graph.invoke()``.

    Returns
    -------
    dict
        Metadata record for the saved post (also written to metadata.json).
    """
    outline = state.get("outline")
    topic = state.get("sanitised_topic") or state.get("topic", "untitled")
    title = getattr(outline, "title", None) or topic
    slug = getattr(outline, "slug", None) or _slugify(topic)
    slug = slug or _slugify(title)

    final_post = state.get("final_post") or state.get("draft", "")
    run_id = state.get("run_id", str(uuid.uuid4()))
    metadata = dict(state.get("metadata", {}))
    critique = state.get("critique")

    created_at = datetime.now(timezone.utc)
    date_prefix = created_at.strftime("%Y%m%d")
    base_post_id = f"{date_prefix}-{_slugify(slug)}"

    post_dir = BLOG_POSTS_ROOT / base_post_id
    suffix = 2
    while post_dir.exists():
        post_dir = BLOG_POSTS_ROOT / f"{base_post_id}-{suffix}"
        suffix += 1
    post_id = post_dir.name

    post_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = post_dir / "assets"

    # ── Move any generated assets from the scratch dir ──────────────────
    gen_dir = _GENERATION_ROOT / run_id
    gen_assets_dir = gen_dir / "assets"
    asset_files: list[str] = []
    if gen_assets_dir.exists():
        assets_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted(gen_assets_dir.glob("*")):
            if f.is_file():
                dest = assets_dir / f.name
                shutil.move(str(f), str(dest))
                asset_files.append(f.name)
        shutil.rmtree(gen_dir, ignore_errors=True)

    # ── Write the Markdown post ──────────────────────────────────────────
    (post_dir / "post.md").write_text(final_post or "", encoding="utf-8")

    record: dict[str, Any] = {
        "id": post_id,
        "run_id": run_id,
        "title": title,
        "slug": slug,
        "topic": topic,
        "created_at": created_at.isoformat(),
        "revision_count": state.get("revision_count", 0),
        "is_approved": state.get("is_approved", False),
        "word_count": len((final_post or "").split()),
        "faithfulness_score": metadata.get("faithfulness_score"),
        "overall_score": getattr(critique, "overall_score", None),
        "assets": asset_files,
        "error_count": len(state.get("error_logs", []) or []),
    }

    (post_dir / "metadata.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    logger.info(
        "Saved post %r → %s (%d words, %d assets)",
        title, post_dir, record["word_count"], len(asset_files),
    )
    return record


def list_posts() -> list[dict[str, Any]]:
    """Return metadata for every saved post, newest first."""
    if not BLOG_POSTS_ROOT.exists():
        return []
    posts: list[dict[str, Any]] = []
    for meta_path in BLOG_POSTS_ROOT.glob("*/metadata.json"):
        try:
            posts.append(json.loads(meta_path.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to read %s: %s", meta_path, exc)
    posts.sort(key=lambda p: p.get("created_at", ""), reverse=True)
    return posts


def get_post(post_id: str) -> Optional[dict[str, Any]]:
    """Return a single post's metadata plus its Markdown content."""
    # Guard against path traversal via the id.
    safe_id = Path(post_id).name
    post_dir = BLOG_POSTS_ROOT / safe_id
    meta_path = post_dir / "metadata.json"
    md_path = post_dir / "post.md"
    if not meta_path.exists() or not md_path.exists():
        return None
    try:
        record = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        record = {"id": safe_id}
    record["content"] = md_path.read_text(encoding="utf-8")
    return record


def get_post_asset_path(post_id: str, filename: str) -> Optional[Path]:
    """Resolve a saved post's asset file path, guarding against path traversal."""
    safe_id = Path(post_id).name
    safe_name = Path(filename).name
    path = BLOG_POSTS_ROOT / safe_id / "assets" / safe_name
    if path.exists() and path.is_file():
        return path
    return None


def get_generation_asset_path(run_id: str, filename: str) -> Optional[Path]:
    """Resolve an in-progress run's scratch asset (e.g. live sandbox preview)."""
    safe_run_id = Path(run_id).name
    safe_name = Path(filename).name
    path = _GENERATION_ROOT / safe_run_id / "assets" / safe_name
    if path.exists() and path.is_file():
        return path
    return None
