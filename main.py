"""
main.py
────────
Entrypoint for the Multi-Agent Blog Generator.

Supports two modes:
1. CLI mode   — run ``python main.py generate --topic "..."``
2. API mode   — run ``python main.py serve`` to start FastAPI

Architecture
────────────
  CLI:
    typer CLI → graph.invoke() → rich console output

  API:
    FastAPI → Celery task → graph.invoke() → WebSocket or polling status

Usage
------
  # CLI (quick generation):
  python main.py generate --topic "LangGraph for production ML systems"

  # CLI (with HITL enabled):
  python main.py generate --topic "..." --hitl

  # API server:
  python main.py serve

  # Run tests:
  python main.py test
"""

from __future__ import annotations

import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from fastapi import BackgroundTasks, FastAPI, HTTPException

from pydantic import BaseModel, Field, field_validator

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

# ── Logging setup ──────────────────────────────────────────────────────────
from config.constants import LOG_FORMAT, LOG_DATE_FORMAT
from config.settings import get_settings
from faults.scenarios import FaultScenario

settings = get_settings()

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format=LOG_FORMAT,
    datefmt=LOG_DATE_FORMAT,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── App instances ──────────────────────────────────────────────────────────
app = typer.Typer(
    name="blog-generator",
    help="Enterprise Multi-Agent Blog Generator powered by LangGraph",
    add_completion=False,
)
console = Console()


class GenerateRequest(BaseModel):
    topic: str
    enable_hitl: bool = False
    tags: list[str] = Field(default_factory=list)
    fault_scenario: FaultScenario | str = FaultScenario.NONE

    @field_validator("fault_scenario", mode="before")
    @classmethod
    def _coerce_known_fault(cls, value):
        if isinstance(value, str):
            try:
                return FaultScenario(value.strip().lower())
            except ValueError:
                # Giữ string để endpoint trả HTTP 400 rõ ràng.
                return value
        return value


class ReviewRequest(BaseModel):
    task_id: str
    feedback: str = "approve"


class GenerateResponse(BaseModel):
    task_id: str
    run_id: str | None = None
    status: str
    message: str


class StatusResponse(BaseModel):
    task_id: str
    run_id: str | None = None
    status: str
    result: dict | None = None
    error: str | None = None
    execution_mode: str | None = None
    fault_scenario: str | None = None


class SandboxRequest(BaseModel):
    code: str


def extract_interrupt_payload(result: dict) -> Optional[dict]:
    """
    Normalise the ``__interrupt__`` value LangGraph puts in a graph result
    into a plain dict.

    LangGraph's dynamic ``interrupt()`` call populates ``result["__interrupt__"]``
    with a tuple/list of ``Interrupt`` objects (not a dict), each carrying the
    original payload on ``.value``. Code that naively did
    ``result["__interrupt__"].get(...)`` (treating it like a dict) would raise
    an ``AttributeError`` the moment a real interrupt actually fired — this
    unwraps it consistently for both the CLI and the API.
    """
    raw = result.get("__interrupt__")
    if not raw:
        return None
    item = raw[0] if isinstance(raw, (list, tuple)) else raw
    return getattr(item, "value", item)


# ─────────────────────────────────────────────────────────────────────────────
# CLI Commands
# ─────────────────────────────────────────────────────────────────────────────


@app.command("generate")
def cli_generate(
    topic: str = typer.Option(
        ...,
        "--topic", "-t",
        help="Blog post topic (e.g. 'LangGraph multi-agent systems')",
        prompt="Enter the blog topic",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save final post to this file path (e.g. output.md)",
    ),
    hitl: bool = typer.Option(
        False,
        "--hitl/--no-hitl",
        help="Enable Human-in-the-Loop review checkpoint",
    ),
    fault: FaultScenario = typer.Option(
        FaultScenario.NONE,
        "--fault",
        case_sensitive=False,
        help=(
            "Inject an AgentLens test fault: none, loop, error, redundant, "
            "threshold, bottleneck, hallucination, or prompt_injection"
        ),
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Show detailed agent steps",
    ),
    fault: str = typer.Option(
        "none",
        "--fault",
        help="Synthetic AgentLens fault scenario (none|loop|error|timeout|...)",
    ),
) -> None:
    """
    Generate a complete blog post for the given topic.

    The system will:
    1. Validate your topic through the input guardrail
    2. Generate a structured outline
    3. Research technical facts and code examples
    4. Write a full Markdown blog post
    5. Review and revise until quality criteria are met
    6. (Optional) Pause for human review
    7. Apply output guardrails (PII redaction, format verification)
    """
    console.print(
        Panel.fit(
            f"[bold cyan]Multi-Agent Blog Generator[/bold cyan]\n"
            f"Topic: [yellow]{topic}[/yellow]\n"
            f"HITL: [green]{hitl}[/green] | "
            f"Fault: [yellow]{fault.value}[/yellow] | "
            f"Verbose: [green]{verbose}[/green]",
            title="[bold white]🤖 Starting Generation[/bold white]",
        )
    )

    from config.tracing import build_run_config, flush_langfuse
    from faults.injector import clear_scenario, register_scenario
    from faults.scenarios import parse_scenario
    from graph.workflow import get_graph
    from state.blog_state import initial_state
    from storage.run_store import append_event, upsert_run

    thread_id = str(uuid.uuid4())
    state = initial_state(topic)
    scenario = parse_scenario(fault)
    metadata = dict(state.get("metadata") or {})
    metadata["task_id"] = thread_id
    state["metadata"] = metadata
    upsert_run(
        {
            "run_id": state["run_id"],
            "task_id": thread_id,
            "topic": topic,
            "fault_scenario": scenario.value,
            "execution_mode": "live" if settings.is_openai_configured else "mock",
            "status": "running",
            "enable_hitl": hitl,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    append_event(state["run_id"], "cli_started", {"topic": topic, "fault": scenario.value})
    register_scenario(state["run_id"], scenario)

    fault_tags = [] if fault is FaultScenario.NONE else ["synthetic-fault"]

    config = build_run_config(
        run_name=f"blog-{topic[:40]}",
        tags=["cli", "blog-generation", "synthetic-ready"],
        metadata={"topic": topic, "thread_id": thread_id},
    )
    config["configurable"] = {"thread_id": thread_id}

    graph = get_graph(enable_hitl=hitl)
    from services.generation import _invoke_with_timeout

    # ── Run graph ──────────────────────────────────────────────────────────
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Running agent pipeline...", total=None)

        try:
            result = _invoke_with_timeout(graph, state, config, settings.graph_timeout_seconds)
        except Exception as exc:
            console.print(f"[red]❌ Graph failed: {exc}[/red]")
            logger.exception("Graph execution failed")
            upsert_run(
                {
                    "run_id": state["run_id"],
                    "status": "failed",
                    "error": str(exc),
                    "ended_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            clear_scenario(state["run_id"])
            raise typer.Exit(1)

    # ── HITL loop ──────────────────────────────────────────────────────────
    # Loops (rather than a single if/resume) because a human "revise" verdict
    # sends the draft back through Writer → Critic → human_review again, so
    # more than one review round can happen before the post is finalised.
    if hitl:
        from langgraph.types import Command

        interrupt_data = extract_interrupt_payload(result)
        while interrupt_data:
            console.print(
                Panel(
                    f"[bold yellow]⏸️  Human Review Required[/bold yellow]\n\n"
                    f"Revision: {interrupt_data.get('revision_count', '?')}\n"
                    f"Critique: {interrupt_data.get('critique_summary', 'N/A')}\n\n"
                    f"[dim]Draft preview:[/dim]\n"
                    f"{interrupt_data.get('draft_preview', '')}",
                    title="[bold white]👤 HITL Checkpoint[/bold white]",
                )
            )
            feedback = typer.prompt(
                "Enter feedback (or press Enter to approve)",
                default="approve",
            )
            result = _invoke_with_timeout(
                graph, Command(resume=feedback), config, settings.graph_timeout_seconds
            )
            interrupt_data = extract_interrupt_payload(result)

    # ── Display results ────────────────────────────────────────────────────
    _display_results(result, verbose=verbose)

    # ── Save output ────────────────────────────────────────────────────────
    final_post = result.get("final_post") or result.get("draft", "")
    if output and final_post:
        output.write_text(final_post, encoding="utf-8")
        console.print(f"\n[green]✅ Saved to: {output}[/green]")

    if final_post:
        from storage.post_manager import save_post

        record = save_post(result)
        console.print(
            f"[green]📁 Added to post library:[/green] blog_posts/{record['id']}/post.md"
        )

    upsert_run(
        {
            "run_id": state["run_id"],
            "status": "completed",
            "ended_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    clear_scenario(state["run_id"])
    append_event(state["run_id"], "cli_completed", {})
    flush_langfuse()


def _display_results(result: dict, verbose: bool = False) -> None:
    """Display generation results in the console."""
    final_post = result.get("final_post") or result.get("draft", "")
    revision_count = result.get("revision_count", 0)
    is_approved = result.get("is_approved", False)
    error_logs = result.get("error_logs", [])
    metadata = result.get("metadata", {})

    # ── Summary table ──────────────────────────────────────────────────────
    table = Table(title="Generation Summary", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="cyan")

    table.add_row("Revisions", str(revision_count))
    table.add_row("Approved", "✅ Yes" if is_approved else "⚠️  Force-passed")
    table.add_row("Word Count", str(len(final_post.split())))
    table.add_row("Errors", str(len(error_logs)))

    if metadata.get("faithfulness_score") is not None:
        table.add_row(
            "Faithfulness", f"{metadata['faithfulness_score']:.2f}"
        )
    if metadata.get("output_word_count"):
        table.add_row("Output Words", str(metadata["output_word_count"]))

    console.print(table)

    if error_logs and verbose:
        console.print("\n[yellow]⚠️  Error Log:[/yellow]")
        for log in error_logs:
            console.print(f"  [{log.node}] {log.error_type}: {log.message}")

    if final_post:
        console.print(
            Panel(
                Markdown(final_post[:3000] + ("\n\n_[truncated — use --output to save full post]_" if len(final_post) > 3000 else "")),
                title="[bold white]📝 Final Blog Post[/bold white]",
            )
        )
    else:
        console.print("[red]No final post generated. Check error logs.[/red]")


@app.command("serve")
def cli_serve(
    host: str = typer.Option(settings.api_host, help="API host"),
    port: int = typer.Option(settings.api_port, help="API port"),
    reload: bool = typer.Option(False, help="Enable hot-reload (development only)"),
) -> None:
    """Start the FastAPI server."""
    import uvicorn

    # NOTE: `host` is the bind address passed to uvicorn — "0.0.0.0" (the
    # default) means "listen on all network interfaces", which is correct
    # for the server but is NOT a valid address to type into a browser.
    # Browsers reject it with ERR_ADDRESS_INVALID. Always show the user a
    # real browsable URL (localhost) regardless of what we bind to.
    browsable_host = "localhost" if host in ("0.0.0.0", "::") else host
    console.print(
        f"[bold green]🚀 Starting API server — open http://{browsable_host}:{port} "
        f"in your browser[/bold green]"
    )
    if host in ("0.0.0.0", "::"):
        console.print(
            f"[dim]   (bound to {host}:{port} for all interfaces — "
            f"use http://localhost:{port}, not http://{host}:{port})[/dim]"
        )
    # `api_app` is the FastAPI instance defined at module level in this file
    # (main.py), built once via create_fastapi_app(). There is no separate
    # `api.py` module — pointing uvicorn at "api:create_app" (as this used to)
    # raised ModuleNotFoundError on every `python main.py serve` call.
    #
    # `reload_excludes`: with --reload on, uvicorn's watchfiles watches the
    # ENTIRE working directory by default — including blog_posts/, which
    # the app itself writes to every time a post/asset is saved
    # (storage/post_manager.py) or a sandbox snippet runs. Without this
    # exclude, every generation triggers "N changes detected" and a full
    # server restart mid-run, wiping the in-memory `_tasks` dict and
    # potentially breaking a generation that's running or paused on HITL.
    # --reload is meant for editing source code, not for excluding the
    # app's own runtime output from being mistaken for a code change.
    uvicorn.run(
        "main:api_app",
        host=host,
        port=port,
        reload=reload,
        reload_excludes=["blog_posts/*", "blog_posts/**", "data/*", "web/node_modules/*"]
        if reload
        else None,
        log_level=settings.log_level.lower(),
    )


@app.command("test")
def cli_test(
    eval_only: bool = typer.Option(False, "--eval-only", help="Run only evaluation tests"),
) -> None:
    """Run the test suite."""
    import subprocess
    import sys

    args = [sys.executable, "-m", "pytest"]
    if eval_only:
        args.extend(["-m", "eval", "tests/evals/"])
    else:
        args.extend(["tests/", "-v", "--tb=short"])

    subprocess.run(args, cwd=Path(__file__).parent)


@app.command("list-prompts")
def cli_list_prompts() -> None:
    """List all available prompt templates."""
    from prompts.loader import list_available_prompts, load_prompt_metadata

    prompts = list_available_prompts()
    table = Table(title="Available Prompt Templates", show_header=True)
    table.add_column("Name", style="bold cyan")
    table.add_column("Version")
    table.add_column("Description")

    for name in prompts:
        try:
            meta = load_prompt_metadata(name)
            table.add_row(
                name,
                str(meta.get("version", "?")),
                str(meta.get("description", ""))[:80],
            )
        except Exception:
            table.add_row(name, "?", "Unable to load metadata")

    console.print(table)


@app.command("config")
def cli_config() -> None:
    """Show current configuration (safe values only)."""
    table = Table(title="Current Configuration", show_header=True)
    table.add_column("Setting", style="bold")
    table.add_column("Value")
    table.add_column("Status")

    rows = [
        ("OpenAI Model", settings.openai_model, "✅" if settings.is_openai_configured else "⚠️  Mock mode"),
        ("Strong Model", settings.openai_model_strong, ""),
        ("Tavily Search", "Configured" if settings.is_tavily_configured else "Not configured", "✅" if settings.is_tavily_configured else "⚠️  Mock mode"),
        ("Code Sandbox", "Local subprocess", "✅" if settings.enable_code_sandbox else "⚠️  Disabled"),
        ("Langfuse", "Configured" if settings.is_langfuse_configured else "Not configured", "✅" if settings.is_langfuse_configured else "ℹ️  Disabled"),
        ("HITL", str(settings.enable_hitl), ""),
        ("Environment", settings.environment, ""),
        ("Log Level", settings.log_level, ""),
    ]

    for row in rows:
        table.add_row(*row)

    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI Application
# ─────────────────────────────────────────────────────────────────────────────


def create_fastapi_app():
    """Build and return the FastAPI application."""
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    from api.runs import router as runs_router
    from api.security import RateLimitMiddleware, cors_origin_list
    from services.generation import resume_review, start_generation, status_payload
    from storage.run_store import init_db

    init_db()

    fast_app = FastAPI(
        title="Multi-Agent Blog Generator API",
        description="Enterprise-grade multi-agent system for generating technical blog posts.",
        version="1.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    origins = cors_origin_list()
    fast_app.add_middleware(RateLimitMiddleware)
    fast_app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    fast_app.include_router(runs_router)

    static_dir = Path(__file__).parent / "static"
    web_dist = Path(__file__).parent / "web" / "dist"
    if static_dir.exists():
        fast_app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    if web_dist.exists():
        assets = web_dist / "assets"
        if assets.exists():
            fast_app.mount("/assets", StaticFiles(directory=str(assets)), name="web-assets")

    def _index_response():
        spa = web_dist / "index.html"
        if spa.exists():
            return FileResponse(str(spa))
        index_path = static_dir / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return JSONResponse(
            {
                "message": "UI not built. Run `npm install && npm run build` in web/, or see /docs.",
            }
        )

    def _run_generation(
        task_id: str,
        topic: str,
        enable_hitl: bool,
        tags: list[str],
        fault_scenario: FaultScenario | str,
        run_id: str,
    ) -> None:
        start_generation(
            topic=topic,
            enable_hitl=enable_hitl,
            tags=tags,
            fault_scenario=fault_scenario,
            task_id=task_id,
            run_id=run_id,
        )

    @fast_app.get("/")
    async def index():
        return _index_response()

    @fast_app.get("/health")
    async def health_check():
        return {
            "status": "ok",
            "openai": settings.is_openai_configured,
            "tavily": settings.is_tavily_configured,
            "langfuse": settings.is_langfuse_configured,
            "execution_mode": "live" if settings.is_openai_configured else "mock",
            "cors_origins": origins,
        }

    @fast_app.post("/api/v1/generate", response_model=GenerateResponse)
    async def generate_blog(request: GenerateRequest, background_tasks: BackgroundTasks):
        from faults.scenarios import parse_scenario

        scenario_value = (
            request.fault_scenario.value
            if isinstance(request.fault_scenario, FaultScenario)
            else request.fault_scenario
        )
        try:
            parse_scenario(scenario_value)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        task_id = str(uuid.uuid4())
        run_id = str(uuid.uuid4())
        from storage.run_store import upsert_run

        upsert_run(
            {
                "run_id": run_id,
                "task_id": task_id,
                "topic": request.topic,
                "fault_scenario": scenario_value,
                "execution_mode": "live" if settings.is_openai_configured else "mock",
                "status": "queued",
                "enable_hitl": request.enable_hitl,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        background_tasks.add_task(
            _run_generation,
            task_id=task_id,
            topic=request.topic,
            enable_hitl=request.enable_hitl,
            tags=request.tags,
            fault_scenario=scenario_value,
            run_id=run_id,
        )
        return GenerateResponse(
            task_id=task_id,
            run_id=run_id,
            status="queued",
            message=(
                f"Blog generation queued. Poll /api/v1/status/{task_id} "
                f"or stream /api/v1/runs/{run_id}/stream."
            ),
        )

    @fast_app.get("/api/v1/status/{task_id}", response_model=StatusResponse)
    async def get_status(task_id: str):
        payload = status_payload(task_id)
        if payload is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return StatusResponse(
            task_id=task_id,
            run_id=payload.get("run_id"),
            status=payload["status"],
            result=payload.get("result"),
            error=payload.get("error"),
            execution_mode=payload.get("execution_mode"),
            fault_scenario=payload.get("fault_scenario"),
        )

    @fast_app.post("/api/v1/review")
    async def submit_review(request: ReviewRequest):
        """Submit human feedback for a paused HITL task."""
        try:
            return resume_review(request.task_id, request.feedback)
        except KeyError:
            raise HTTPException(status_code=404, detail="Task not found")
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # ── Post library ────────────────────────────────────────────────────────

    @fast_app.get("/api/v1/posts")
    async def api_list_posts():
        from storage.post_manager import list_posts

        return {"posts": list_posts()}

    @fast_app.get("/api/v1/posts/{post_id}")
    async def api_get_post(post_id: str):
        from storage.post_manager import get_post

        record = get_post(post_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Post not found")
        return record

    @fast_app.get("/api/v1/posts/{post_id}/assets/{filename}")
    async def api_get_post_asset(post_id: str, filename: str):
        from storage.post_manager import get_post_asset_path

        path = get_post_asset_path(post_id, filename)
        if path is None:
            raise HTTPException(status_code=404, detail="Asset not found")
        return FileResponse(str(path))

    # ── Live generation assets (preview before a post is saved, e.g. HITL) ──

    @fast_app.get("/api/v1/generation/{run_id}/assets/{filename}")
    async def api_get_generation_asset(run_id: str, filename: str):
        from storage.post_manager import get_generation_asset_path

        path = get_generation_asset_path(run_id, filename)
        if path is None:
            raise HTTPException(status_code=404, detail="Asset not found")
        return FileResponse(str(path))

    # ── Sandbox: run ad-hoc code / visualisations (matplotlib, etc.) ────────

    @fast_app.post("/api/v1/sandbox/execute")
    async def api_sandbox_execute(request: SandboxRequest):
        from storage.post_manager import get_generation_assets_dir, new_sandbox_run_id
        from tools.code_interpreter import run_sandbox_snippet

        run_id = new_sandbox_run_id()
        assets_dir = get_generation_assets_dir(run_id)
        outcome = run_sandbox_snippet(request.code, assets_dir=str(assets_dir))
        outcome["run_id"] = run_id
        outcome["artifact_urls"] = [
            f"/api/v1/generation/{run_id}/assets/{name}" for name in outcome.get("artifacts", [])
        ]
        return outcome

    @fast_app.get("/{path:path}")
    async def spa_fallback(path: str):
        blocked = ("api/", "docs", "redoc", "openapi.json", "health", "static/")
        if path.startswith(blocked) or path in {"docs", "redoc", "openapi.json", "health"}:
            raise HTTPException(status_code=404, detail="Not found")
        return _index_response()

    return fast_app


# ─────────────────────────────────────────────────────────────────────────────
# WSGI/ASGI App for uvicorn
# ─────────────────────────────────────────────────────────────────────────────

# This is used by `uvicorn main:api_app` directly
api_app = create_fastapi_app()


# ─────────────────────────────────────────────────────────────────────────────
# Entrypoint
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
