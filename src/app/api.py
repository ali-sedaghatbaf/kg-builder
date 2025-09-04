from contextlib import asynccontextmanager
import json
import logging
import shutil
from typing import Any

from fastapi import FastAPI, File, UploadFile
from httpx import ConnectError
from langfuse import get_client
from redis.asyncio import Redis

from app.config import settings
from app.db.neo4j import Neo4j
from app.doc.parser import PDFParser
from app.lm.agents import (
    ContractClassifier,
    ContractContentAnalyzer,
    KnowledgeGraphExtractor,
    OntologyAnalyzer,
)
from app.lm.setup import setup_dspy
from app.utils import setup_logging

setup_logging()


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
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
        decode_responses=True,
    )
    setup_dspy()

    yield
    # Cleanup resources on shutdown
    await redis_client.aclose()
    await neo4j.close()


app = FastAPI(title="KG Builder API", version="1.0.0", lifespan=lifespan)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)) -> dict[str, Any]:
    # Save the uploaded file to disk
    logger.info(f"Uploading file: {file.filename}")
    with open(f"uploads/{file.filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": file.filename, "content_type": file.content_type}


@app.get("/classification/{contract_filename}")
async def classify_contract(contract_filename: str) -> dict[str, Any]:
    parser: PDFParser = PDFParser()
    file_content: str = await parser.parse_async(
        file_name=contract_filename, page_count=2
    )
    classifier: ContractClassifier = ContractClassifier()
    result: dict = await classifier.forward(contract_text=file_content)
    if "error" in result:
        return result
    return result


@app.get("/content_analysis/{contract_filename}")
async def analyze_contract(contract_filename: str) -> dict[str, Any]:
    classification_result: dict = await classify_contract(contract_filename)
    if "error" in classification_result:
        return classification_result

    parser: PDFParser = PDFParser()
    file_content: str = await parser.parse_async(
        file_name=contract_filename, page_count=10
    )
    contract_type: str = classification_result.get("contract_type", "")
    content_analyzer: ContractContentAnalyzer = ContractContentAnalyzer()
    analysis_result: dict = await content_analyzer.forward(
        file_content, contract_type=contract_type
    )

    if "error" in analysis_result:
        return analysis_result

    await app.state.redis.set(
        contract_filename,
        json.dumps(
            {
                "classification": classification_result,
                "content_analysis": analysis_result,
            },
            default=str,
        ),
    )
    return analysis_result


@app.get("/content_storage/{contract_filename}")
async def store_content(contract_filename: str) -> dict[str, Any]:
    content_analysis_result = None
    classification_result = None
    cached_results = await app.state.redis.get(contract_filename)
    if cached_results is not None:
        cached_data = json.loads(cached_results)
        if "storage" in cached_data:
            logger.info("Using cached kg result")
            return cached_data["storage"]
        if "content_analysis" in cached_data:
            logger.info("Using cached content analysis result")
            content_analysis_result = cached_data.get("content_analysis")

            classification_result = cached_data.get("classification")

    if not content_analysis_result:
        content_analysis_result = await analyze_contract(contract_filename)
        if "error" in content_analysis_result:
            return content_analysis_result
        cached_results = await app.state.redis.get(contract_filename)
        cached_data = json.loads(cached_results)
        classification_result = cached_data.get("classification")

    db = app.state.neo4j
    result = await db.generate_contract_kg(
        contract_data=content_analysis_result,
        contract_path=f"uploads/{contract_filename}",
    )
    if "error" in result:
        return result

    await app.state.redis.set(
        contract_filename,
        json.dumps(
            {
                "classification": classification_result,
                "content_analysis": content_analysis_result,
                "storage": result,
            }
        ),
    )
    return result


@app.get("/ontology_analysis/{contract_filename}")
async def analyze_ontology(contract_filename: str):
    parser: PDFParser = PDFParser()
    file_content: str = await parser.parse_async(file_name=contract_filename)
    analyzer: OntologyAnalyzer = OntologyAnalyzer()
    results = await analyzer.forward(text=file_content)
    return results


@app.get("/kg_extraction/{contract_filename}")
async def extract_knowledge_graph(contract_filename: str):
    parser: PDFParser = PDFParser()
    file_content: str = await parser.parse_async(file_name=contract_filename)
    analyzer: OntologyAnalyzer = OntologyAnalyzer()
    ontology = await analyzer.forward(file_content)
    if "error" in ontology:
        return ontology
    kg_extractor: KnowledgeGraphExtractor = KnowledgeGraphExtractor()
    result = await kg_extractor.forward(text=file_content, ontology=ontology)
    return result
