import logging

from dspy import ChainOfThought, Module, Predict

from .signatures import (
    KGGeneration,
    KGIntentQuestion,
    KGSchemaProposal,
    SchemaExtraction,
)

logger = logging.getLogger(__name__)


class IntentDiscoveryAgent(Module):
    """Interacts with the user to discover their intent for knowledge graph generation and gather information."""

    def __init__(self):
        super().__init__()
        self.question_gen = ChainOfThought(KGIntentQuestion)

    async def aforward(
        self, user_goal: str, file_content: str, gathered_info: str = ""
    ):
        return await self.question_gen.acall(
            user_goal=user_goal, gathered_info=gathered_info, file_content=file_content
        )


class SchemaProposalAgent(Module):
    def __init__(self):
        super().__init__()
        self.schema_gen = ChainOfThought(KGSchemaProposal)

    async def aforward(self, gathered_info: str):
        return await self.schema_gen.acall(gathered_info=gathered_info)


class SchemaExtractor(Module):
    """Refines/extracts a schema from text given a proposed schema scaffold."""

    def __init__(self):
        self.module = Predict(signature=SchemaExtraction)

    async def aforward(self, text: str, proposed_schema: dict) -> dict:
        logger.info("Extracting schema from text with proposed scaffold.")
        try:
            result = await self.module.acall(text=text, proposed_schema=proposed_schema)
            extracted = getattr(result, "extracted_schema", None)
            if extracted is None:
                raise ValueError("SchemaExtraction returned no extracted_schema")
            return extracted.dict() if hasattr(extracted, "dict") else extracted
        except Exception as e:
            logger.error("Error during schema extraction: %s", e, exc_info=True)
            return {"error": "Failed to extract schema.", "details": str(e)}


class KGGenerator(Module):
    """Extracts a knowledge graph from a text document."""

    def __init__(self):
        self.module = Predict(signature=KGGeneration)

    async def aforward(self, text: str, graph_schema: dict) -> dict:
        logger.info("Extracting knowledge graph from text.")
        try:
            result = await self.module.acall(text=text, graph_schema=graph_schema)
            return result.knowledge_graph.dict()
        except Exception as e:
            logger.error(
                "Error during knowledge graph extraction: %s", e, exc_info=True
            )
            return {
                "error": "Failed to extract knowledge graph.",
                "details": str(e),
            }
