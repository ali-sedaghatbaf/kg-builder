from dspy import InputField, OutputField, Signature

from .models import (
    KnowledgeGraph,
    Schema,
)


class DocumentOverview(Signature):
    summaries: list[str] = InputField(description="The summaries of all text chunks.")
    overview: str = OutputField(description="A concise overview of the whole text.")


class ChunkSummary(Signature):
    text: str = InputField(description="The text chunk to summarize.")
    summary: str = OutputField(description="A concise summary of the text chunk.")


class KGIntentQuestion(Signature):
    """Ask one concise KG-extraction question, or output 'DONE'."""

    user_goal = InputField(
        desc="The user's high-level goal for generating a knowledge graph (KG) from the uploaded document"
    )
    file_overview = InputField(
        desc="The overview of the uploaded document (or excerpt), to tailor KG-specific questions"
    )
    gathered_info = InputField(
        desc="The information gathered so far; do not ask about items already answered"
    )
    question = OutputField(
        desc="A single concise question specific to KG extraction, OR 'DONE' if enough info is gathered"
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
    """Classify a short user reply as affirmative, negative, or change_request (for schema changes)."""

    text: str = InputField(description="The user's reply")
    label: str = OutputField(
        description="One of: affirmative, negative, change_request"
    )


class SchemaRefinement(Signature):
    """Revise an existing schema according to user instructions."""

    instruction: str = InputField(
        description="Natural language instructions describing desired schema changes"
    )
    current_schema: Schema = InputField(
        description="The current proposed schema to be revised"
    )
    proposed_schema: Schema = OutputField(
        description="The updated proposed schema after applying the changes"
    )
