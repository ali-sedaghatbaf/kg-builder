from dspy import InputField, OutputField, Signature

from .models import (
    KnowledgeGraph,
    Schema,
)


class KGIntentQuestion(Signature):
    """Ask the next clarifying question based on what we already know"""

    user_goal = InputField(
        desc="The user's high-level goal for generating a knowledge graph"
    )
    file_content = InputField(
        desc="The content of the uploaded document (if any), to help tailor questions"
    )
    gathered_info = InputField(desc="The information gathered so far from the user")
    question = OutputField(
        desc="The next question to ask OR 'DONE' if enough info is gathered"
    )


class KGSchemaProposal(Signature):
    """Generate a draft KG schema based on clarified user answers"""

    gathered_info = InputField()
    proposed_schema: Schema = OutputField(
        description="A draft schema in JSON format, including entity and relation types"
    )


class SchemaExtraction(Signature):
    text: str = InputField(description="The text to extract the schema from.")
    proposed_schema: Schema = InputField(
        description="A draft schema in JSON format, including entity and relation types"
    )
    extracted_schema: Schema = OutputField(
        description="The extracted schema in a JSON format."
    )


class KGGeneration(Signature):
    text: str = InputField(description="The text to extract the knowledge graph from.")
    graph_schema: Schema = InputField(description="The schema of the knowledge graph. ")
    knowledge_graph: KnowledgeGraph = OutputField(
        description="The extracted knowledge graph. "
    )


class ReplyClassification(Signature):
    """Classify a short user reply as affirmative, negative, or neutral."""

    text: str = InputField(description="The user's reply")
    label: str = OutputField(description="One of: affirmative, negative, neutral")
