# ── Content Recommendation Backend ──────────────────────────────
# Multi-stage build: slim Python image with only runtime deps.
# Deploys backend + ML artifacts together (~35 MB of data files).
# Render sets $PORT at runtime; see render.yaml for service config.
# ────────────────────────────────────────────────────────────────

# ---------- stage 1: build / install deps ----------
FROM python:3.13-slim AS builder

WORKDIR /build

COPY backend/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt \
    && rm -rf /install/lib/python3.13/site-packages/pip \
    && rm -rf /install/lib/python3.13/site-packages/pip-* \
    && find /install -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true \
    && find /install -name '*.pyc' -delete 2>/dev/null || true

# ---------- stage 2: runtime ----------
FROM python:3.13-slim

# Non-root user for security
RUN groupadd -r app && useradd -r -g app -d /app app

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# ---- Application code ----
COPY --chown=app:app backend/ /app/backend/

# ---- ML artifacts (embeddings + metadata + features + config) ----
COPY --chown=app:app ml/artifacts/content_based_anime_v1/ /app/ml/artifacts/content_based_anime_v1/

# ---- Source data for metadata enrichment (Image URL, etc.) ----
COPY --chown=app:app data/processed/anime.csv /app/data/processed/anime.csv

# ---- Environment defaults (overridable at runtime) ----
ENV ARTIFACT_DIR=ml/artifacts/content_based_anime_v1 \
    ANIME_SOURCE_DATA_PATH=data/processed/anime.csv \
    ALLOWED_ORIGINS=* \
    PORT=10000 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Render expects the app to listen on 0.0.0.0:$PORT
EXPOSE ${PORT}

# Switch to non-root user
USER app

# PROJECT_ROOT resolves to /app (2 parents up from /app/backend/app/main.py)
WORKDIR /app/backend

# Gunicorn + Uvicorn workers for production ASGI serving.
# --preload loads artifacts once in the master process, shared via fork.
CMD ["sh", "-c", "gunicorn app.main:app \
  --worker-class uvicorn.workers.UvicornWorker \
  --workers 1 \
  --bind 0.0.0.0:${PORT} \
  --timeout 120 \
  --preload"]
