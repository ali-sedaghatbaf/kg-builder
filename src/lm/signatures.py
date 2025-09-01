from dspy import InputField, OutputField, Signature

from src.lm.models import (
    AffiliateAgreement,
    CoBranding,
    ContractType,
    Ontology,
    KnowledgeGraph,
)


class ContractClassification(Signature):
    contract_text: str = InputField(description="The text of the contract to classify.")
    contract_type: ContractType = OutputField(description="The type of the contract")
    thoughts: str = OutputField(description="The reasoning behind the classification")


class ContractContentExtraction(Signature):
    contract_text: str = InputField(description="The text of the contract to analyze.")
    contract_type: ContractType = InputField(description="The type of the contract")
    extracted_information: AffiliateAgreement | CoBranding = OutputField(
        description="The extracted information from the contract"
    )


class OntologyAnalysis(Signature):
    text: str = InputField(description="The text to extract the ontology from.")
    ontology: Ontology = OutputField(
        description="The extracted ontology in a JSON format. "
    )


class KnowledgeGraphExtraction(Signature):
    text: str = InputField(description="The text to extract the knowledge graph from.")
    ontology: Ontology = InputField(description="The ontology of the knowledge graph. ")
    knowledge_graph: KnowledgeGraph = OutputField(
        description="The extracted knowledge graph. "
    )
