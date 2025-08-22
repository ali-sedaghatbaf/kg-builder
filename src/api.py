import json
import logging
import os
import shutil
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from redis.asyncio import Redis

from src.db.neo4j import Neo4j
from src.doc.parser import PDFParser
from src.lm.agents import ContractClassifier, ContractContentAnalyzer
from src.lm.config import setup_dspy
from src.lm.models import ContractType
from src.utils import EnumEncoder, setup_logging

load_dotenv()
setup_logging()
setup_dspy()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    neo4j = Neo4j()
    app.state.neo4j = neo4j
    redis_client = Redis(
        host=os.getenv("REDIS_HOST"),
        port=int(os.getenv("REDIS_PORT")),
        db=0,
        decode_responses=True,
    )
    app.state.redis = redis_client

    yield
    # Cleanup resources on shutdown
    await redis_client.close()
    await neo4j.close()


app = FastAPI(title="KG Builder API", version="1.0.0", lifespan=lifespan)


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Save the uploaded file to disk
    logger.info(f"Uploading file: {file.filename}")
    with open(f"uploads/{file.filename}", "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filename": file.filename, "content_type": file.content_type}


@app.get("/classification/{contract_filename}")
async def classify_contract(contract_filename: str):
    cached_results: str = await app.state.redis.get(contract_filename)
    if cached_results is not None:
        cache_data: dict = json.loads(cached_results)
        if "classification" in cache_data:
            logger.info("Using cached classification result")
            return cache_data.get("classification")

    parser: PDFParser = PDFParser()
    file_content: str = await parser.parse_async(
        file_name=contract_filename, page_count=2
    )
    classifier: ContractClassifier = ContractClassifier()
    result: dict = await classifier(contract_text=file_content)
    if "error" in result:
        return result
    await app.state.redis.set(contract_filename, json.dumps({"classification": result}))
    return result


@app.get("/content_analysis/{contract_filename}")
async def analyze_contract(contract_filename: str):
    classification_result: dict = None

    cached_results: str = await app.state.redis.get(contract_filename)
    if cached_results is not None:
        cached_data: dict = json.loads(cached_results)

        if "content_analysis" in cached_data:
            logger.info("Using cached content analysis result")
            return cached_data.get("content_analysis")
        if "classification" in cached_data:
            logger.info("Using cached classification result")
            classification_result = cached_data.get("classification")

    if not classification_result:
        classification_result = await classify_contract(contract_filename)
        if "error" in classification_result:
            return classification_result

    parser: PDFParser = PDFParser()
    file_content: str = await parser.parse_async(
        file_name=contract_filename, page_count=10
    )

    content_analyzer: ContractContentAnalyzer = ContractContentAnalyzer()
    analysis_result: dict = await content_analyzer(
        file_content, contract_type=classification_result.get("contract_type")
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


@app.get("/kg/{contract_filename}")
async def generate_kg(contract_filename: str):
    content_analysis_result = None
    classification_result = None
    cached_results = await app.state.redis.get(contract_filename)
    if cached_results is not None:
        cached_data = json.loads(cached_results)
        if "kg" in cached_data:
            logger.info("Using cached kg result")
            return cached_data["kg"]
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
    print(content_analysis_result["agreement_date"])

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
                "kg": result,
            }
        ),
    )
    return result
