import logging

from dspy import ChainOfThought, Module, Predict

from .signatures import (
    ContractClassification,
    ContractContentExtraction,
    KnowledgeGraphExtraction,
    OntologyAnalysis,
)

logger = logging.getLogger(__name__)


class ContractClassifier(Module):
    """
    Given a list of available contract types, this module classifies a contract text into one of those types.

    """

    def __init__(self):
        self.module = ChainOfThought(signature=ContractClassification)

    async def forward(self, contract_text: str) -> dict:
        logger.info("Classifying contract text.")
        try:
            result = await self.module.acall(contract_text=contract_text)
            return {
                "contract_type": result.contract_type.value,
                "reasons": result.thoughts,
            }
        except Exception as e:
            logger.error("Error during contract classification: %s", e, exc_info=True)
            return {
                "error": "Failed to classify contract.",
                "details": str(e),
            }


class ContractContentAnalyzer(Module):
    """
    Analyzes the content of a contract based on its type and extracts relevant information.
    """

    def __init__(self):
        self.module = Predict(signature=ContractContentExtraction)

    async def forward(self, contract_text: str, contract_type: str) -> dict:
        logger.info("Analyzing contract of type: %s", contract_type)
        try:
            result = await self.module.acall(
                contract_text=contract_text, contract_type=contract_type
            )
            return result.extracted_information.dict()
        except Exception as e:
            logger.error("Error during contract content analysis: %s", e, exc_info=True)
            return {
                "error": "Failed to analyze contract content.",
                "details": str(e),
            }


class OntologyAnalyzer(Module):
    """Analyzes the content of a text document to extract an ontology."""

    def __init__(self):
        self.module = Predict(signature=OntologyAnalysis)

    async def forward(self, text: str) -> dict:
        logger.info("Analyzing ontology from text.")
        try:
            result = await self.module.acall(text=text)
            return result.ontology.dict()
        except Exception as e:
            logger.error("Error during ontology analysis: %s", e, exc_info=True)
            return {
                "error": "Failed to analyze ontology.",
                "details": str(e),
            }


class KnowledgeGraphExtractor(Module):
    """Extracts a knowledge graph from a text document."""

    def __init__(self):
        self.module = Predict(signature=KnowledgeGraphExtraction)

    async def forward(self, text: str, ontology: dict) -> dict:
        logger.info("Extracting knowledge graph from text.")
        try:
            result = await self.module.acall(text=text, ontology=ontology)
            return result.knowledge_graph.dict()
        except Exception as e:
            logger.error(
                "Error during knowledge graph extraction: %s", e, exc_info=True
            )
            return {
                "error": "Failed to extract knowledge graph.",
                "details": str(e),
            }
