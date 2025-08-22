# ───────────────────────────────
# Base image (shared setup)
# ───────────────────────────────
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV UV_NO_CACHE=1

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* ./

# ───────────────────────────────
# Development stage
# ───────────────────────────────
FROM base AS dev

RUN uv sync --all-groups

COPY . .

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# ───────────────────────────────
# Test stage
# ───────────────────────────────
FROM base AS test

RUN uv sync --all-groups

COPY . .

# test command by default
CMD ["uv", "run", "pytest", "-q"]

# ───────────────────────────────
# Production stage
# ───────────────────────────────
FROM base AS prod

RUN uv sync   # only prod deps

COPY . .

EXPOSE 8000

CMD ["uv", "run", "gunicorn", "-k", "uvicorn.workers.UvicornWorker", "src.api:app", "--bind", "0.0.0.0:8000"]
