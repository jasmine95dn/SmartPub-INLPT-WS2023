# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Build tools needed to compile native extensions (e.g. tokenizers, ctransformers)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry and keep the venv inside the project
RUN pip install --no-cache-dir poetry \
    && poetry config virtualenvs.in-project true

# Copy dependency manifests first for layer caching
COPY pyproject.toml ./

# Install runtime dependencies only (no dev extras)
RUN poetry install --no-root --no-interaction --without dev

# ── Stage 2: final ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS final

WORKDIR /app

# Runtime system deps (git needed for ReFinED URL install at run time if missing)
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Copy the pre-built virtualenv from builder
COPY --from=builder /app/.venv .venv

# Copy application source
COPY model/ model/
COPY src/   src/
COPY app.py store_index.py template.py ./

# Activate the venv for all subsequent commands
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8080

CMD ["python", "app.py"]
