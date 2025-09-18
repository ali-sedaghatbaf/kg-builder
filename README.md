KG Builder
=================================

End-to-end app for turning documents into a knowledge graph with an LLM-powered chat workflow. Upload a PDF, iteratively define or refine a schema in chat, extract a knowledge graph, and persist it to Neo4j — all with optional Langfuse tracing and OpenTelemetry instrumentation.

Features
- Chat-driven KG pipeline: intent discovery → schema proposal/refinement → KG extraction.
- PDF parsing via LlamaParse (Llama Cloud).
- FastAPI backend with streaming responses.
- Neo4j persistence for entities and relations.
- Redis-backed state and LM cache.
- Optional Langfuse + OTEL/Jaeger tracing.
- Vite + React frontend to upload files and visualize/edit schema.

Architecture
- Backend `src/app/api.py:1` exposes `POST /upload/` and streaming `POST /chat`.
- Pipeline in `src/app/lm/agents.py:1` orchestrates summarization, schema, extraction, normalization.
- Models in `src/app/lm/models.py:1` define `Schema`, `Entity`, `Relation`, `KnowledgeGraph`.
- Neo4j access in `src/app/db/neo4j.py:1` persists graphs.
- PDF parsing in `src/app/doc/parser.py:1` using LlamaParse.
- Frontend chat + schema viewer in `frontend/src/App.jsx:1`.
- Docker Compose stack wires API, Redis, Neo4j, Jaeger, and frontend (`docker-compose.yml:1`).

Prerequisites
- Docker and Docker Compose
- Node 18+ if developing the frontend locally
- Optional (local dev without Docker): Python 3.11 and `uv` (https://github.com/astral-sh/uv)

Quick Start (Docker Compose)
1) Create `.env` (example values below). Do not commit real secrets.

   Required
   - `OPENAI_API_KEY` — for LLM calls
   - `LLAMA_CLOUD_API_KEY` — for PDF parsing
   - `APP_NEO4J_PASSWORD` — Neo4j auth

   Common defaults
   - `REDIS_HOST=redis`, `REDIS_PORT=6379`, `REDIS_AUTH=redis123`
   - `APP_NEO4J_URI=bolt://neo4j:7687`, `APP_NEO4J_USER=neo4j`
   - `OPENAI_MODEL=gpt-4o`
   - To use a local OpenAI-compatible server: `USE_LOCAL_LLM=true`, `LOCAL_LLM_URL=http://host.docker.internal:12434/v1`, `LOCAL_LLM_MODEL=ai/smollm2`

   Minimal example
   ```env
   OPENAI_API_KEY="<your-openai-key>"
   OPENAI_MODEL="gpt-4o"
   LLAMA_CLOUD_API_KEY="<your-llama-cloud-key>"
   APP_NEO4J_URI="bolt://neo4j:7687"
   APP_NEO4J_USER="neo4j"
   APP_NEO4J_PASSWORD="<choose-a-password>"
   REDIS_HOST="redis"
   REDIS_PORT=6379
   REDIS_AUTH="redis123"
   ```

2) Start the stack
- `make up` (starts the frontend and its dependencies)
- Or `docker compose up -d frontend`

3) Open the app
- Frontend: `http://localhost:5173`
- API (dev): `http://localhost:8000`
- Jaeger UI: `http://localhost:16686`
- Neo4j Browser: `http://localhost:7474` (user `neo4j` / password from `.env`)

4) Try it out
- Upload a PDF in the UI and follow the chat prompts.
- Or call the API directly:
  - `GET /` → health: `{ "Hello": "World" }`
  - `POST /upload/` (multipart form, field `file`, optional `conversation_id`)
  - `POST /chat` with JSON `{ "message": "...", "conversation_id": "..." }` to stream responses

Local Development (without Docker)
- Install `uv` and sync deps:
  - `uv sync --all-groups`
- Run API with auto-reload:
  - `uv run uvicorn src.app.api:app --host 0.0.0.0 --port 8000 --reload`
- Frontend dev server:
  - `cd frontend && npm install && npm run dev`

Make Targets
- `make up` — start frontend + backend + deps
- `make down` — stop
- `make down-v` — stop and remove volumes
- `make logs` — follow logs
- `make shell` — shell into the selected service (`SERVICE=dev make shell`)
