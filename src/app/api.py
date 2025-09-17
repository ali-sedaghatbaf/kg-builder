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
from .lm.agents import KGPipeline
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
    app.state.pipeline = KGPipeline(redis=redis_client, parser=PDFParser())
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
    messages = await app.state.pipeline.acall(conv_id, user_message)
    if not messages:
        return
    for msg in messages:
        try:
            json.loads(msg)
            yield msg  # If it's valid JSON, yield as-is
        except (ValueError, TypeError):
            async for ch in _stream_text(msg):
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
