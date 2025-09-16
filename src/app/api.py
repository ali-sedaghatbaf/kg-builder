import asyncio
from contextlib import asynccontextmanager
import json
import logging
from pathlib import Path
import shutil
from typing import Any, AsyncGenerator

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from httpx import ConnectError
from langfuse import get_client
from pydantic import BaseModel
from redis.asyncio import Redis

from .config import get_settings
from .db.neo4j import Neo4j
from .doc.parser import PDFParser
from .lm.agents import (
    IntentDiscoveryAgent as IntentDiscoveryModule,
    KGGenerator,
    SchemaExtractor,
    SchemaProposalAgent as SchemaProposalModule,
)
from .lm.setup import setup_dspy
from .utils import setup_logging

setup_logging()


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    langfuse_client = get_client()
    app.state.langfuse = langfuse_client
    try:
        if langfuse_client.auth_check():
            logger.info("Langfuse client is authenticated and ready!")
        else:
            logger.warning(
                "Langfuse authentication failed. Please check your credentials and host."
            )
    except ConnectError:
        logger.warning("Langfuse connection failud. Please make sure it's running.")

    neo4j = Neo4j()
    app.state.neo4j = neo4j
    redis_client = Redis(
        host=settings.REDIS_HOST,
        port=settings.REDIS_PORT,
        db=settings.REDIS_DB,
        password=(
            settings.REDIS_AUTH.get_secret_value() if settings.REDIS_AUTH else None
        ),
        decode_responses=True,
    )
    app.state.redis = redis_client
    setup_dspy(redis_client=redis_client)
    app.state.intent_agent = IntentDiscoveryModule()
    app.state.schema_proposal_agent = SchemaProposalModule()
    app.state.schema_analyzer = SchemaExtractor()
    app.state.kg_generator = KGGenerator()
    app.state.parser = PDFParser()
    yield
    # Cleanup resources on shutdown
    await redis_client.aclose()
    await neo4j.close()


app = FastAPI(title="KG Builder API", version="1.0.0", lifespan=lifespan)

# Enable CORS for the frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/upload/")
async def upload_file(
    file: UploadFile = File(...),
    conversation_id: str | None = Form(None),
) -> dict[str, Any]:
    # Save the uploaded file to disk
    logger.info(f"Uploading file: {file.filename}")
    filename = file.filename if file.filename is not None else "uploaded_file"
    with open(
        Path(__file__).parent.parent.parent / "uploads" / filename, "wb"
    ) as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Record upload metadata in Redis under the conversation-specific key
    try:
        conv_id = conversation_id or "default"
        await app.state.redis.set(
            f"file:{conv_id}",
            json.dumps(
                {
                    "filename": file.filename,
                    "content_type": file.content_type,
                }
            ),
        )

    except Exception:
        # Non-fatal if Redis is unavailable
        pass

    return {"filename": file.filename, "content_type": file.content_type}


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


async def _stream_text(text: str, delay: float = 0.01) -> AsyncGenerator[str, None]:
    for ch in text:
        yield ch
        if delay:
            await asyncio.sleep(delay)


async def _stream_reply(conv_id: str, user_message: str) -> AsyncGenerator[str, None]:
    redis: Redis = app.state.redis

    # Require an uploaded file (keeps UX aligned with the flow)
    file_key = f"file:{conv_id}"
    file_state = await redis.get(file_key)
    if not file_state:
        async for ch in _stream_text(
            "Please upload a document first by clicking the 'Upload File' button.\n"
        ):
            yield ch
        return
    file_info = json.loads(file_state)
    file_content = file_info.get("content", "")
    if not file_content:
        filename = file_info.get("filename", "unknown")
        try:
            parser: PDFParser = app.state.parser
            file_content: str = await parser.aparse(file_name=filename)

            # Mark file as processed
            file_info["content"] = file_content
            await redis.set(file_key, json.dumps(file_info))
        except Exception as e:
            async for ch in _stream_text(f"Error processing uploaded file: {e}\n"):
                yield ch
            return

    # If we are awaiting schema extraction confirmation, handle that first
    pending_key = f"schema_pending:{conv_id}"
    pending_raw = await redis.get(pending_key)
    if pending_raw:
        try:
            pending = json.loads(pending_raw)
        except Exception:
            pending = None
        proposal = (pending or {}).get("proposal")
        if proposal:
            lower_msg = (user_message or "").strip().lower()
            proceed_markers = [
                "extract schema",
                "schema extraction",
                "proceed with extraction",
                "proceed with schema",
                "run extraction",
                "go ahead with schema",
                "generate schema",
            ]
            cancel_markers = ["cancel", "stop", "abort", "not now", "later"]
            affirmative = any(m in lower_msg for m in proceed_markers)
            negative = any(m in lower_msg for m in cancel_markers)
            if affirmative:
                extractor: SchemaExtractor = app.state.schema_analyzer
                async for ch in _stream_text(
                    "Extracting the schema from the document...\n"
                ):
                    yield ch
                try:
                    extracted = await extractor.acall(
                        text=file_content, proposed_schema=proposal
                    )
                    if isinstance(extracted, dict) and "error" in extracted:
                        raise RuntimeError(
                            extracted.get("details") or extracted["error"]
                        )
                    extracted_json = (
                        extracted
                        if isinstance(extracted, dict)
                        else (
                            extracted.dict()
                            if hasattr(extracted, "dict")
                            else extracted
                        )
                    )
                    # Persist extracted schema for later steps
                    await redis.set(
                        f"schema_extracted:{conv_id}", json.dumps(extracted_json)
                    )
                    await redis.delete(pending_key)
                    async for ch in _stream_text("Here is the extracted schema:\n"):
                        yield ch
                    async for ch in _stream_text(
                        json.dumps(extracted_json, indent=2) + "\n"
                    ):
                        yield ch
                except Exception as e:
                    await redis.delete(pending_key)
                    async for ch in _stream_text(f"Sorry, extraction failed: {e}\n"):
                        yield ch
                return
            elif negative:
                await redis.delete(pending_key)
                # Acknowledge cancellation without prompting
                async for ch in _stream_text("Understood. Extraction canceled.\n"):
                    yield ch
                return
            else:
                # No prompt; just let the user decide when to proceed in a later message
                return

    # Load discovery state
    disc_key = f"discovery:{conv_id}"
    state_raw = await redis.get(disc_key)
    state: dict[str, Any] = {"goal": "", "qa": [], "awaiting": None}
    if state_raw:
        try:
            state = json.loads(state_raw)
        except Exception:
            state = {"goal": "", "qa": [], "awaiting": None}

    # Initialize goal from the user's first message, or attach as an answer/context
    if not state.get("goal"):
        state["goal"] = user_message.strip()
    else:
        awaiting = state.get("awaiting")
        if awaiting:
            state.setdefault("qa", []).append(
                {"q": awaiting, "a": user_message.strip()}
            )
            state["awaiting"] = None
        else:
            state.setdefault("qa", []).append(
                {"q": "(user note)", "a": user_message.strip()}
            )

    # Helper to build gathered_info for the LLM
    def build_gathered_info(s: dict[str, Any]) -> str:
        lines = [f"Goal: {s.get('goal', '').strip()}\n"]
        for i, qa in enumerate(s.get("qa", []), start=1):
            lines.append(f"Q{i}: {qa.get('q', '')}\nA{i}: {qa.get('a', '')}\n")
        return "".join(lines)

    # Intent discovery loop: ask one question at a time until DONE
    discovery = app.state.intent_agent  # IntentDiscoveryModule instance
    while True:
        gathered_info = build_gathered_info(state)
        res = await discovery.acall(
            user_goal=state["goal"],
            gathered_info=gathered_info,
            file_content=file_content,
        )
        q = (getattr(res, "question", None) or "").strip()
        if not q:
            q = "Could you clarify your primary use case for the KG?"
        if q.upper() == "DONE":
            break
        # Ask exactly one question and persist awaiting state
        state["awaiting"] = q
        await redis.set(disc_key, json.dumps(state))
        async for ch in _stream_text(q + "\n"):
            yield ch
        return

    # Discovery complete: produce a schema proposal
    proposer = app.state.schema_proposal_agent  # SchemaProposalModule instance
    gathered_info = build_gathered_info(state)
    try:
        prop = await proposer.acall(gathered_info=gathered_info)
        schema_obj = getattr(prop, "proposed_schema", None)
        if not schema_obj:
            raise ValueError("No proposed_schema returned")
        await redis.set(disc_key, json.dumps(state))
        async for ch in _stream_text("Great! Here's a draft schema proposal:\n"):
            yield ch
        # Pretty print proposal for the UI to render
        proposal_json = schema_obj.dict() if hasattr(schema_obj, "dict") else schema_obj
        pretty = json.dumps(proposal_json, indent=2)
        async for ch in _stream_text(pretty + "\n", delay=0):
            yield ch
        # Persist proposal and wait for explicit user request to extract
        await redis.set(pending_key, json.dumps({"proposal": proposal_json}))
        # Do not ask a question; user can send a message like "extract schema" to proceed
        return
    except Exception as e:
        async for ch in _stream_text(
            f"Sorry, I couldn't complete schema extraction: {e}\n"
        ):
            yield ch


@app.post("/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """Streams a textual response for a chat message.
    Frontend consumes via fetch + ReadableStream.
    """
    conv_id = req.conversation_id or "default"
    return StreamingResponse(
        _stream_reply(conv_id, req.message), media_type="text/plain"
    )


def __store_intent(intent, awaiting_field: str | None = None) -> str:
    return json.dumps(
        {
            "intent": intent.model_dump(),
            "awaiting_field": awaiting_field,
        }
    )
