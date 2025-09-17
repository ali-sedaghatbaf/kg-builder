import json
import logging
from typing import Any

from dspy import ChainOfThought, Module, Predict
from redis.asyncio import Redis

from ..doc.parser import PDFParser
from .signatures import (
    KGGeneration,
    KGIntentQuestion,
    KGSchemaProposal,
    ReplyClassification,
    SchemaExtraction,
    SchemaRefinement,
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


class KGPipeline(Module):
    """End-to-end conversational pipeline for KG building.

    Handles:
    - Ensuring a file is uploaded and parsed
    - Intent discovery Q&A loop
    - Schema proposal and confirmation
    - Schema extraction
    """

    def __init__(self, redis: Redis, parser: PDFParser | None = None):
        super().__init__()
        self.redis = redis
        self.parser = parser or PDFParser()
        # Internal agents
        self.intent_agent = IntentDiscoveryAgent()
        self.schema_proposal_agent = SchemaProposalAgent()
        self.schema_extractor = SchemaExtractor()
        self.kg_generator = KGGenerator()
        self.reply_classifier = Predict(signature=ReplyClassification)
        self.schema_refiner = Predict(signature=SchemaRefinement)

    @staticmethod
    def _looks_like_change_request(text: str) -> bool:
        t = (text or "").strip().lower()
        if not t:
            return False
        change_keywords = [
            "add ",
            "remove ",
            "delete ",
            "drop ",
            "rename ",
            "change ",
            "update ",
            "modify ",
            "field",
            "property",
            "relation",
            "relationship",
            "entity",
            "edge",
            "connect",
            "link",
            "type",
            "attributes",
        ]
        return any(k in t for k in change_keywords)

    async def aforward(self, conv_id: str, user_message: str) -> list[str]:
        redis = self.redis
        out: list[str] = []

        # Require an uploaded file (keeps UX aligned with the flow)
        file_key = f"file:{conv_id}"
        file_state = await redis.get(file_key)
        if not file_state:
            out.append(
                "Please upload a document first by clicking the 'Upload File' button.\n"
            )
            return out
        file_info = json.loads(file_state)
        file_content = file_info.get("content", "")
        if not file_content:
            filename = file_info.get("filename", "unknown")
            try:
                file_content = await self.parser.aparse(
                    file_name=filename, page_count=1
                )

                file_info["content"] = file_content
                await redis.set(file_key, json.dumps(file_info))
            except Exception as e:
                out.append(f"Error processing uploaded file: {e}\n")
                return out

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
                try:
                    cls = await self.reply_classifier.acall(text=(user_message or ""))
                    decision = getattr(cls, "label", "neutral") or "neutral"
                    decision = str(decision).strip().lower()
                    if decision not in {"affirmative", "negative", "neutral"}:
                        decision = "neutral"
                except Exception:
                    decision = "neutral"
                if decision == "affirmative":
                    extractor = self.schema_extractor
                    out.append("Extracting the schema from the document...\n")
                    try:
                        full_text = await self.parser.aparse(
                            file_name=file_info.get("filename"), page_count=None
                        )
                        extracted = await extractor.acall(
                            text=full_text, proposed_schema=proposal
                        )
                        if isinstance(extracted, dict) and "error" in extracted:
                            raise RuntimeError(
                                extracted.get("details") or extracted["error"]
                            )
                        extracted_json = (
                            extracted if isinstance(extracted, dict) else {}
                        )
                        # Persist extracted schema for later steps
                        await redis.set(
                            f"schema_extracted:{conv_id}", json.dumps(extracted_json)
                        )
                        await redis.delete(pending_key)
                        out.append("Here is the extracted schema:\n")
                        out.append(json.dumps(extracted_json, indent=2) + "\n")
                    except Exception as e:
                        await redis.delete(pending_key)
                        out.append(f"Sorry, extraction failed: {e}\n")
                    return out
                elif decision == "negative":
                    await redis.delete(pending_key)
                    out.append("Understood. Extraction canceled.\n")
                    return out
                else:
                    # Potential schema change request
                    if self._looks_like_change_request(user_message):
                        try:
                            refined = await self.schema_refiner.acall(
                                instruction=user_message, current_schema=proposal
                            )
                            refined_schema = getattr(refined, "proposed_schema", None)
                            if refined_schema is not None:
                                refined_json = (
                                    refined_schema.dict()
                                    if hasattr(refined_schema, "dict")
                                    else refined_schema
                                )
                                await redis.set(
                                    pending_key, json.dumps({"proposal": refined_json})
                                )
                                out.append("Updated schema based on your request:\n")
                                out.append(json.dumps(refined_json, indent=2) + "\n")
                                out.append(
                                    "Let me know if you'd like to proceed with extracting schema from the document.\n"
                                )
                                return out
                        except Exception:
                            pass
                    # Neutral/no-op: wait for clearer instruction
                    return out

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
        while True:
            gathered_info = build_gathered_info(state)
            res = await self.intent_agent.acall(
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
            out.append(q + "\n")
            return out

        # Discovery complete: produce a schema proposal
        gathered_info = build_gathered_info(state)
        try:
            prop = await self.schema_proposal_agent.acall(gathered_info=gathered_info)
            schema_obj = getattr(prop, "proposed_schema", None)
            if not schema_obj:
                raise ValueError("No proposed_schema returned")
            await redis.set(disc_key, json.dumps(state))
            out.append("Great! Here's a draft schema proposal:\n")
            proposal_json = (
                schema_obj.dict() if hasattr(schema_obj, "dict") else schema_obj
            )
            pretty = json.dumps(proposal_json, indent=2)
            out.append(pretty + "\n")
            # Persist proposal and wait for explicit user request to extract
            await redis.set(pending_key, json.dumps({"proposal": proposal_json}))
            out.append(
                "Let me know if you'd like to proceed with extracting this schema from the document.\n"
            )
            return out
        except Exception as e:
            out.append(f"Sorry, I couldn't complete schema extraction: {e}\n")
            return out
