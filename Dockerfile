# Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Multi-Agent Blog Generator — single-image build.
#
# Runs `python main.py serve`, which starts the FastAPI app (Web UI +
# API) on 0.0.0.0:8000. There is no separate worker/Redis/Celery
# container here on purpose: even though `config/settings.py` and
# `requirements.txt` reference Celery/Redis, the actual API
# (`main.py::create_fastapi_app`) uses an in-memory task store +
# FastAPI `BackgroundTasks` for async generation — Celery/Redis are not
# wired up anywhere in the code, so shipping a Redis container would be
# dead weight. If that changes later, add a `redis` service + worker
# stage to docker-compose.yml.
#
# NOTE: the ~400MB spaCy model (`en_core_web_lg`) used for advanced PII
# detection is intentionally NOT downloaded during this build — the
# output guardrail already falls back to regex-based PII redaction when
# it's absent (see guardrails/output_guard.py). Download it inside the
# running container only if you actually want the upgrade:
#   docker compose exec app python -m spacy download en_core_web_lg
# ─────────────────────────────────────────────────────────────────────────────

FROM node:20-alpine AS web
WORKDIR /web
COPY web/package.json ./
RUN npm install
COPY web/ ./
RUN npm run build

FROM python:3.11-slim

# Build-time OS deps: gcc/g++ are needed for a couple of packages in
# requirements.txt (presidio/spacy transitively, ragas) that don't always
# ship prebuilt wheels for every platform. Removed after pip install to
# keep the final image lean.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so this layer is cached across
# code-only changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the application source.
COPY . .
COPY --from=web /web/dist /app/web/dist

# `blog_posts/` is where generated posts + assets live — mounted as a
# volume in docker-compose.yml so they survive container restarts and
# are browsable directly on the host filesystem.
RUN mkdir -p blog_posts data

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

# Shell-form CMD so ${API_PORT} is expanded against the *container's*
# runtime environment (injected by docker-compose's `env_file: .env`),
# not baked in at build time — this stays correct even if you change
# API_PORT in .env without rebuilding the image. Falls back to 8000 if
# API_PORT isn't set at all. (docker-compose.yml also defines its own
# `healthcheck:` block, which takes precedence when running via compose;
# this one only matters for a bare `docker run`.)
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:${API_PORT:-8000}/health || exit 1

# settings.api_host defaults to 0.0.0.0 and settings.api_port to 8000,
# so no extra flags are needed here — see config/settings.py.
CMD ["python", "main.py", "serve"]
