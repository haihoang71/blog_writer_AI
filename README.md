# 🤖 Enterprise Multi-Agent Blog Generator

> A production-ready, enterprise-grade **Multi-Agent System** built with **LangGraph**, **LangChain**, and **GPT-4o** that automatically researches, drafts, reviews, and edits high-quality technical blog posts about Coding, Machine Learning, and Artificial Intelligence.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Project Structure](#project-structure)
4. [Prerequisites](#prerequisites)
5. [Environment Setup](#environment-setup)
6. [Installing Dependencies](#installing-dependencies)
7. [Configuration](#configuration)
8. [Running Locally (CLI)](#running-locally-cli)
9. [Running the API Server](#running-the-api-server)
10. [Configuring Langfuse Tracing](#configuring-langfuse-tracing)
11. [Mock Mode (No API Keys Required)](#mock-mode-no-api-keys-required)
12. [Human-in-the-Loop (HITL)](#human-in-the-loop-hitl)
13. [Testing](#testing)
14. [Evaluation Metrics](#evaluation-metrics)
15. [Guardrail System](#guardrail-system)
16. [Prompt Management](#prompt-management)
17. [Extending the System](#extending-the-system)
18. [Troubleshooting](#troubleshooting)

---

## Quick Start

The fastest way to get the whole system (agents + Web UI) running:

```bash
# 1. Activate the existing virtual environment
#    Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
#    macOS / Linux:
source .venv/bin/activate

# 2. Install/refresh dependencies (matplotlib, langfuse v3+, etc.)
pip install -r requirements.txt

# 3. Make sure .env has your keys — a real OPENAI_API_KEY (or Gemini key,
#    see below) is what turns on live generation instead of mock mode.
#    (.env already exists in this project — just double-check it.)

# 4. Start the server (this runs BOTH the agent backend and the Web UI)
python main.py serve --reload
```

Then open **http://localhost:8000/** in your browser. That's the whole app:

- **Generate** tab — enter a topic, optionally tick "Human review (HITL)", watch it run, review/approve, download the Markdown.
- **Library** tab — every post ever generated (via UI, CLI, or API) shows up here automatically, pulled from `blog_posts/`.
- **Sandbox** tab — run ad-hoc Python (matplotlib charts included) in the same local sandbox the Critic agent uses.

There's no separate "start the UI" step — the UI *is* served by the same process, so `python main.py serve` is the only command you need day to day.

Prefer the terminal instead of the browser? The exact same pipeline is available as a CLI command, and it saves to the same `blog_posts/` library:

```bash
python main.py generate --topic "Building production RAG systems with LangChain" --no-hitl
```

Sanity-check your setup at any time with:

```bash
python main.py config     # shows which keys are live vs. mock/disabled
pytest tests/ -v          # runs the test suite
```

Notes on this project's current setup:
- This runs against **Gemini** models by default (`OPENAI_API_KEY` holds a Google AI Studio key, `OPENAI_MODEL`/`OPENAI_MODEL_STRONG` are `gemini-*` names, and `OPENAI_API_BASE` points at Gemini's OpenAI-compatible endpoint) — not OpenAI itself. See [Configuration](#configuration) if you want to switch to real OpenAI or another provider.
- The code sandbox (used by the Critic agent and the Sandbox tab) runs **locally only** — no E2B or other paid cloud sandbox is used or required.
- PII redaction falls back to regex by default; it will **not** auto-download the ~400MB spaCy model. Run `python -m spacy download en_core_web_lg` yourself first if you want Presidio's NLP-based redaction instead.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INPUT (Topic)                        │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │   Input Guardrail   │ ← Injection detection
              │   (input_guard.py)  │ ← Domain validation
              └──────────┬──────────┘ ← LLM classification
                         │ allow
                         ▼
              ┌─────────────────────┐
              │   Planner Agent     │ ← Structured outline (JSON)
              │   (planner.py)      │ ← GPT-4o-mini
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Researcher Agent   │ ← Tavily web search
              │  (researcher.py)    │ ← ArXiv academic search
              └──────────┬──────────┘ ← Agentic tool loop
                         │
              academic-heavy topic? ──Yes──┐
                         │ No               ▼
                         │        ┌─────────────────────┐
                         │        │ Academic Researcher  │ ← ArXiv search only
                         │        │ (academic_           │ ← Produces citations
                         │        │  researcher.py)      │ ← Merges into
                         │        └──────────┬───────────┘   research_data
                         │                   │
                         ▼◄──────────────────┘
                    ┌────▼────┐
                    │ WRITE   │ ◄──────────────────────────┐
                    │ LOOP    │                            │ (revision)
                    └────┬────┘                            │
                         │                                 │
                         ▼                                 │
              ┌─────────────────────┐                      │
              │   Writer Agent      │ ← Full Markdown draft │
              │   (writer.py)       │ ← Incorporates critique│
              └──────────┬──────────┘                      │
                         │                                 │
                         ▼                                 │
              ┌─────────────────────┐                      │
              │  Technical Critic   │ ← Code execution      │
              │  (critic.py)        │ ← 6-dimension scoring │
              └──────────┬──────────┘ ← Structured critique │
                         │                                 │
                 approved?  No ───────────────────────────┘
                         │ Yes (or max revisions)
                         ▼
              ┌─────────────────────┐
              │  Human Review (HITL)│ ← LangGraph interrupt()
              │  (workflow.py)      │ ← Human feedback
              └──────────┬──────────┘
                         │ approve
                         ▼
              ┌─────────────────────┐
              │  Output Guardrail   │ ← PII redaction
              │  (output_guard.py)  │ ← Markdown validation
              └──────────┬──────────┘ ← Hallucination check
                         │
                         ▼
              ┌─────────────────────┐
              │   FINAL BLOG POST   │
              └─────────────────────┘
```

### Key Design Principles

| Principle | Implementation |
|-----------|---------------|
| **No hardcoded prompts** | All prompts in `prompts/templates/*.yaml`, loaded dynamically |
| **Graceful degradation** | Every agent has a mock fallback when API keys are absent |
| **Multi-layer guardrails** | Input → Planner → Writer → Critic → Output guardrail chain |
| **Observability** | Langfuse traces every token, latency, and agent call |
| **Human-in-the-loop** | LangGraph `interrupt()` pauses graph for human review |
| **Type safety** | Pydantic v2 models validate every data boundary |

---

## Project Structure

```
multi-agent-blog/
├── README.md                   ← This file
├── main.py                     ← CLI + FastAPI entrypoint
├── requirements.txt            ← Project dependencies
├── pytest.ini                  ← Test configuration
├── .env.example                ← Environment variable template
├── .gitignore                  ├─────────────────────────────
│
├── config/
│   ├── settings.py             ← Pydantic BaseSettings (env vars)
│   ├── constants.py            ← System constants & limits
│   └── tracing.py              ← Langfuse callback setup
│
├── prompts/
│   ├── loader.py               ← Dynamic YAML prompt loader
│   ├── templates/
│   │   ├── planner.yaml        ← Planner system prompt
│   │   ├── researcher.yaml     ← Researcher system prompt
│   │   ├── academic_researcher.yaml ← Academic Researcher system prompt
│   │   ├── writer.yaml         ← Writer system prompt
│   │   └── critic.yaml         ← Critic system prompt
│   └── guardrails/
│       ├── input_moderation.yaml
│       └── code_security.yaml
│
├── guardrails/
│   ├── input_guard.py          ← Injection detection + LLM classifier
│   ├── code_sandbox_guard.py   ← AST safety analysis
│   ├── hallucination_guard.py  ← TF-IDF claim verification
│   └── output_guard.py         ← PII redaction + Markdown validation
│
├── state/
│   └── blog_state.py           ← TypedDict + Pydantic state models
│
├── tools/
│   ├── search_tools.py         ← Tavily + ArXiv LangChain tools
│   └── code_interpreter.py     ← Local subprocess sandbox
│
├── agents/
│   ├── planner.py              ← Planner node
│   ├── researcher.py           ← Researcher node (tool-calling)
│   ├── academic_researcher.py  ← Academic Researcher node (ArXiv + citations)
│   ├── writer.py               ← Writer node
│   └── critic.py               ← Critic node (code execution)
│
├── graph/
│   ├── workflow.py             ← StateGraph + HITL interrupt
│   ├── router.py               ← Conditional edge logic
│   └── middlewares.py          ← Node logging middleware
│
└── tests/
    ├── test_workflow.py        ← Unit + integration tests
    └── evals/
        └── agent_evals.py      ← Faithfulness, relevance, code evals
```

---

## Prerequisites

- **Python 3.10 or later** (3.11+ recommended)
- **pip** (Python package manager)
- **Git**
- **(Optional)** Redis for Celery-based async API
- **(Optional)** OpenAI API key for live generation
- **(Optional)** Tavily API key for web search
- **(Optional)** Langfuse account for observability

---

## Environment Setup

### 1. Clone or navigate to the project

```bash
cd multi-agent-blog
```

### 2. Create a Python virtual environment

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Copy and configure your environment file

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Then open `.env` in your editor and fill in your API keys (see [Configuration](#configuration)).

---

## Installing Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# (Optional) Install spacy language model for advanced PII detection
python -m spacy download en_core_web_lg
```

> **Note:** The system works without `spacy` — it falls back to regex-based PII detection.

---

## Configuration

All configuration is driven by `.env`. Here is a reference:

### Required for live generation

| Variable | Description | Where to get it |
|----------|-------------|-----------------|
| `OPENAI_API_KEY` | OpenAI API key | https://platform.openai.com/api-keys |
| `OPENAI_MODEL` | Primary model (default: `gpt-4o-mini`) | — |
| `OPENAI_MODEL_STRONG` | Critic model (default: `gpt-4o`) | — |
| `OPENAI_API_BASE` | OpenAI-compatible endpoint (default: Gemini's) | — |

### Optional — enhanced capabilities

| Variable | Description | Where to get it |
|----------|-------------|-----------------|
| `TAVILY_API_KEY` | Web search API | https://tavily.com |
| `ARXIV_API_KEY` | Optional — arXiv's public search API is free/keyless already; only fill this in if you have a key for an arXiv-adjacent service | — |
| `ARXIV_MAX_RESULTS` | Max papers per arXiv query (default: `5`) | — |
| `LANGFUSE_PUBLIC_KEY` | Observability public key | https://cloud.langfuse.com |
| `LANGFUSE_SECRET_KEY` | Observability secret key | https://cloud.langfuse.com |
| `LANGFUSE_HOST` | Langfuse server URL | Default: `https://cloud.langfuse.com` |

### Feature flags

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_TRACING` | `true` | Toggle Langfuse tracing |
| `ENABLE_CODE_SANDBOX` | `true` | Toggle code execution |
| `ENABLE_HITL` | `true` | Toggle human-in-the-loop |

---

## Running Locally (CLI)

The CLI is the quickest way to generate blog posts.

### View available commands

```bash
python main.py --help
```

### Show current configuration

```bash
python main.py config
```

Output shows which APIs are configured and which will use mock mode.

### Generate a blog post (fully automated)

```bash
python main.py generate --topic "Building production RAG systems with LangChain"
```

### Generate with human review checkpoint

```bash
python main.py generate \
  --topic "LangGraph multi-agent systems for enterprise" \
  --hitl
```

The graph will pause at the human review node, show a draft preview,
and prompt you to approve or reject it.

### Save output to a file

```bash
python main.py generate \
  --topic "Async Python for ML inference pipelines" \
  --output output/my-blog-post.md
```

### Enable verbose logging

```bash
python main.py generate \
  --topic "Transformer architecture internals" \
  --verbose
```

### List available prompt templates

```bash
python main.py list-prompts
```

---

## Running the API Server

### Start the FastAPI server

```bash
# Development (auto-reload)
python main.py serve --reload

# Or use uvicorn directly
uvicorn main:api_app --host 0.0.0.0 --port 8000 --reload
```

### Web UI

Once the server is running, open **http://localhost:8000/** in a browser for a
built-in UI (served from `static/index.html`) with three tabs:

- **Generate** — enter a topic, optionally enable HITL, watch the pipeline run,
  review/approve drafts, and download the final Markdown.
- **Sandbox** — run ad-hoc Python (including matplotlib) in the same sandbox
  the Critic agent uses, and see any generated charts inline. Useful for
  testing code snippets or producing a visualization before it goes in a post.
- **Library** — browse every post that has been generated (see
  [Post Library](#post-library) below), rendered with any charts embedded.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Web UI |
| `GET` | `/health` | Health check & config status |
| `POST` | `/api/v1/generate` | Queue blog generation task |
| `GET` | `/api/v1/status/{task_id}` | Poll task status |
| `POST` | `/api/v1/review` | Submit HITL feedback |
| `GET` | `/api/v1/posts` | List saved posts |
| `GET` | `/api/v1/posts/{post_id}` | Get a saved post's Markdown + metadata |
| `GET` | `/api/v1/posts/{post_id}/assets/{filename}` | Fetch a chart image for a post |
| `POST` | `/api/v1/sandbox/execute` | Run ad-hoc Python (matplotlib etc.) and get output + chart URLs |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/redoc` | ReDoc API docs |

### Post Library

Every completed generation (from the CLI, the API, or the UI) is saved to
`blog_posts/<date>-<slug>/`:

```
blog_posts/
└── 20260730-building-production-rag-systems/
    ├── post.md          ← final Markdown
    ├── metadata.json     ← title, topic, scores, timestamps, asset list
    └── assets/
        └── figure-a1b2c3d4.png   ← charts produced by the Critic's code sandbox
```

`storage/post_manager.py` provides `save_post()`, `list_posts()`, and
`get_post()` if you want to script against the library directly.

### Example API Usage

#### Queue a generation job

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "topic": "Building LangGraph agents for production",
    "enable_hitl": false,
    "tags": ["ml", "langchain"]
  }'
```

Response:
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "queued",
  "message": "Blog generation queued. Poll /api/v1/status/... for results."
}
```

#### Poll for status

```bash
curl http://localhost:8000/api/v1/status/550e8400-e29b-41d4-a716-446655440000
```

Response (when completed):
```json
{
  "task_id": "550e8400-...",
  "status": "completed",
  "result": {
    "final_post": "# Building LangGraph Agents...\n\n...",
    "revision_count": 2,
    "is_approved": true,
    "word_count": 1847,
    "metadata": { "faithfulness_score": 0.72, ... }
  }
}
```

#### Submit human review feedback

```bash
curl -X POST http://localhost:8000/api/v1/review \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "550e8400-...",
    "feedback": "approve"
  }'
```

---

## Configuring Langfuse Tracing

Langfuse provides end-to-end observability for every agent run.

### Option A: Cloud (recommended for getting started)

1. Sign up at [cloud.langfuse.com](https://cloud.langfuse.com)
2. Create a new project
3. Navigate to **Settings → API Keys** and create a key pair
4. Add to `.env`:
   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_HOST=https://cloud.langfuse.com
   ```

### Option B: Self-hosted Langfuse (Docker)

```bash
# Clone and start Langfuse
git clone https://github.com/langfuse/langfuse.git
cd langfuse
docker-compose up -d
```

Then set:
```
LANGFUSE_HOST=http://localhost:3000
```

### What gets traced?

- Every agent node call (latency, input, output)
- LLM token usage and cost per request
- Tool calls (Tavily, ArXiv, code sandbox)
- Graph run metadata (topic, revisions, approval)
- Error logs and fallback events

### Disabling tracing

```
ENABLE_TRACING=false
```

---

## Mock Mode (No API Keys Required)

The system is designed to run completely out-of-the-box without any API keys:

| Component | Mock Behaviour |
|-----------|---------------|
| Planner | Generates a 4-section outline based on topic name |
| Researcher | Produces placeholder research with dummy citations |
| Academic Researcher | Only runs for academic-heavy topics; produces mock arXiv findings with formatted citations |
| Writer | Creates a structured Markdown post using mock data |
| Critic | Approves after 2 mock revisions with minor feedback |
| Tavily | Returns 3 synthetic search results |
| ArXiv | Returns 2 synthetic paper entries (each with a formatted `citation` string) |

To run in mock mode, simply don't set `OPENAI_API_KEY` (or use the placeholder).

```bash
# This works without any API keys!
python main.py generate --topic "Python generators and coroutines" --no-hitl
```

---

## Human-in-the-Loop (HITL)

The HITL checkpoint uses LangGraph's `interrupt()` primitive to pause graph
execution before the final output guardrail.

### CLI workflow

```
1. Run: python main.py generate --topic "..." --hitl
2. Graph executes: input_guard → planner → researcher → writer → critic
3. [PAUSE] — Console shows draft preview and critique summary
4. You enter feedback (or press Enter to approve)
5. Graph resumes: output_guard → final post
```

### API workflow

```
1. POST /api/v1/generate   {"enable_hitl": true, "topic": "..."}
2. GET  /api/v1/status/{id} — poll until status is "paused" or "awaiting_review"
3. POST /api/v1/review      {"task_id": "...", "feedback": "approve"}
4. GET  /api/v1/status/{id} — poll until status is "completed"
```

### Feedback options

| Feedback | Effect |
|----------|--------|
| `approve`, `yes`, `ok`, `` (empty) | Accept draft → output guardrail |
| `reject`, `no`, `revise`, `redo` | Send back to Writer for revision |
| Any text | Appended as `human_feedback` to state |

### Disable HITL globally

```
ENABLE_HITL=false
```

---

## Testing

### Run the full test suite

```bash
pytest tests/ -v
```

### Run unit tests only (fast, no API keys)

```bash
pytest tests/test_workflow.py -v
```

### Run evaluation tests

```bash
pytest tests/evals/ -v -m eval
```

### Run a specific test class

```bash
pytest tests/test_workflow.py::TestInputGuard -v
pytest tests/test_workflow.py::TestFullGraphIntegration -v
```

### Test coverage

```bash
pytest tests/ --cov=. --cov-report=html
open htmlcov/index.html
```

---

## Evaluation Metrics

The `tests/evals/agent_evals.py` module implements four evaluation metrics:

| Metric | Weight | Description |
|--------|--------|-------------|
| **Faithfulness** | 30% | Claims in draft supported by research (TF-IDF cosine similarity) |
| **Answer Relevance** | 30% | Topic keywords and outline sections covered in draft |
| **Code Validity** | 20% | Code snippets pass AST safety check (0 = all fail, 1 = all safe) |
| **Output Format** | 20% | Markdown schema completeness (title, TL;DR, Key Takeaways, References) |

**Composite score formula:**
```
composite = 0.30*faithfulness + 0.30*relevance + 0.20*code_validity + 0.20*format
```

### Run evaluations

```python
from tests.evals.agent_evals import run_full_evaluation

report = run_full_evaluation(
    draft=my_draft,
    topic="LangGraph agents",
    outline=my_outline,
    research=my_research,
)
print(f"Composite Score: {report['composite_score']:.2f}")
```

---

## Guardrail System

Four independent guardrail modules form a defence-in-depth security layer:

### 1. Input Guardrail (`guardrails/input_guard.py`)

Protects the system from malicious or off-topic requests:

- **Regex injection detection**: 8 patterns catch common prompt injection attempts
- **Blocked topic filtering**: Keywords from `BLOCKED_TOPICS` in `constants.py`
- **Length validation**: 5–500 character topic limit
- **LLM semantic classifier**: Uses `input_moderation.yaml` prompt for nuanced decisions

### 2. Code Sandbox Guardrail (`guardrails/code_sandbox_guard.py`)

Protects against dangerous code snippets:

- **AST analysis**: Detects `eval()`, `exec()`, dangerous imports, file writes
- **Regex patterns**: Matches `SENSITIVE_CODE_PATTERNS` from `constants.py`
- **Risk levels**: `safe` → `warn` → `block`

### 3. Hallucination Guardrail (`guardrails/hallucination_guard.py`)

Verifies draft claims against research data:

- **Claim extraction**: Heuristic sentence filtering for factual assertions
- **TF-IDF similarity**: Cosine similarity between claims and research corpus
- **Jaccard fallback**: Used when `scikit-learn` is unavailable
- **Faithfulness score**: `verified_claims / total_claims`

### 4. Output Guardrail (`guardrails/output_guard.py`)

Final cleanup before delivery:

- **PII redaction**: Presidio-powered (regex fallback) — emails, phones, SSNs, API keys
- **Markdown validation**: Checks for required H1, TL;DR, Key Takeaways, References
- **Word count enforcement**: Minimum `MIN_DRAFT_WORD_COUNT` words (default: 600)
- **Whitespace cleanup**: Normalises blank lines and trailing spaces

---

## Prompt Management

All system prompts are version-controlled YAML files — **no hardcoded strings in Python**.

### Prompt file structure

```yaml
version: "1.2"
name: writer
description: >
  Short description of this prompt.

system: |
  You are a ...

  TOPIC: {topic}
  OUTLINE: {outline_json}
```

### Loading prompts

```python
from prompts.loader import load_prompt

system_prompt = load_prompt(
    "writer",
    topic="LangGraph",
    outline_json=outline.model_dump_json(),
    research_json=research.model_dump_json(),
    critique="No critique yet.",
    revision_count=0,
    max_revisions=3,
)
```

### Adding a new prompt

1. Create `prompts/templates/my_agent.yaml`
2. Define `version`, `name`, `description`, `system` fields
3. Use `{variable_name}` placeholders for dynamic values
4. Load it with `load_prompt("my_agent", variable_name="value")`

### Modifying prompts (no code changes needed)

Edit the YAML file and optionally bump `version`. The loader cache will
be invalidated on next import (or call `prompts.loader.invalidate_cache()`).

---

## Extending the System

### Add a new agent node

1. Create `agents/my_agent.py` with a function `def my_agent_node(state: BlogState) -> BlogState:`
2. Create `prompts/templates/my_agent.yaml`
3. Register in `graph/workflow.py`:
   ```python
   graph.add_node("my_agent", wrap_node("my_agent", my_agent_node))
   graph.add_edge("researcher", "my_agent")
   graph.add_edge("my_agent", "writer")
   ```

`agents/academic_researcher.py` is a worked example of a *conditionally*
invoked agent — it only runs when `graph.router.route_after_researcher`
decides the topic is academic-heavy (via `agents.academic_researcher.
is_academic_topic`, a keyword check against `config.constants.
ACADEMIC_TOPIC_KEYWORDS`), and it merges its output into the existing
`research_data` instead of owning a separate state field. Use this pattern
for any agent that should augment the pipeline for a subset of topics
rather than always run.

### Add a new search tool

1. Create a `@tool`-decorated function in `tools/search_tools.py`
2. Add it to `SEARCH_TOOLS` list
3. The Researcher agent automatically picks it up via `.bind_tools(SEARCH_TOOLS)`

### Add a new guardrail

1. Create `guardrails/my_guard.py` with a `check_*()` function
2. Add it to the appropriate node in `graph/workflow.py`
3. Update `guardrails/__init__.py`

### Configure maximum revisions

In `.env` or `config/constants.py`:
```python
MAX_REVISIONS = 5  # default: 3
```

---

## Troubleshooting

### `ModuleNotFoundError`

```bash
# Ensure virtual environment is activated
# Windows:
.\venv\Scripts\Activate.ps1

# Then reinstall:
pip install -r requirements.txt
```

### OpenAI authentication error

```
openai.AuthenticationError: Incorrect API key provided
```

Check your `.env` file:
- Key starts with `sk-` (not `sk-placeholder`)
- No quotes around the value
- `.env` is in the same directory as `main.py`

### Tavily returns mock results

This is expected behaviour when `TAVILY_API_KEY` is not set. The system
logs `[MOCK] tavily_search` at INFO level. Set a real key to enable live search.

### LangGraph `interrupt()` not pausing

Ensure `ENABLE_HITL=true` in `.env` **and** `--hitl` flag is passed to CLI.
The graph must be compiled with `interrupt_before=["human_review"]`.

### Hallucination score is low (< 0.5)

This is expected in mock mode since mock research data has minimal overlap
with a generated draft. With live OpenAI + Tavily, the score typically reaches 0.6–0.9.

### Langfuse traces not appearing

1. Check `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set
2. Verify the `LANGFUSE_HOST` URL is reachable
3. Check `ENABLE_TRACING=true`
4. Look for `INFO: Langfuse tracing enabled → host=...` in logs
5. Flush may be needed: add `flush_langfuse()` at end of your script

### `resource` module error on Windows

The `resource` module (UNIX-only) is used for CPU limiting in the local sandbox.
The code handles this gracefully — the `except Exception: pass` in the sandbox
means it still works on Windows, just without resource limits.

---

## License

MIT License — see `LICENSE` for details.

---

## Acknowledgements

Built with:
- [LangGraph](https://langchain-ai.github.io/langgraph/) — Agent orchestration
- [LangChain](https://python.langchain.com/) — LLM toolkit
- [OpenAI](https://openai.com/) — LLM backend
- [Tavily](https://tavily.com/) — Real-time web search
- [matplotlib](https://matplotlib.org/) — Chart/visualisation generation in the code sandbox
- [Langfuse](https://langfuse.com/) — LLM observability
- [Pydantic](https://docs.pydantic.dev/) — Data validation
- [FastAPI](https://fastapi.tiangolo.com/) — API framework
