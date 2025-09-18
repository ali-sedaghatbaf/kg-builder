import json
import logging
import re
from typing import Any

from dspy import ChainOfThought, Module, Predict
from redis.asyncio import Redis

from ..doc.parser import PDFParser
from ..utils import safe_sent_tokenize
from .models import EntityType, KnowledgeGraph, RelationType, Schema
from .signatures import (
    ChunkSummary,
    DocumentOverview,
    KGGeneration,
    KGIntentQuestion,
    KGSchemaProposal,
    ReplyClassification,
    SchemaExtraction,
    SchemaRefinement,
)

logger = logging.getLogger(__name__)


class SkimSummarizer(Module):
    def __init__(self):
        super().__init__()
        self.local_summary = Predict(signature=ChunkSummary)
        self.overview = Predict(signature=DocumentOverview)

    async def aforward(self, document):
        skimmed = self.skim_chunks(document)
        local_summaries = [
            (await self.local_summary.acall(text=chunk)).summary for chunk in skimmed
        ]
        overview = (await self.overview.acall(summaries=local_summaries)).overview
        return overview

    def skim_chunks(self, text, chunk_size=500):
        sentences = safe_sent_tokenize(text)
        chunks = [
            " ".join(sentences[i : i + chunk_size])
            for i in range(0, len(sentences), chunk_size)
        ]
        skimmed = []
        for c in chunks:
            sents = safe_sent_tokenize(c)
            if len(sents) > 1:
                skimmed.append(sents[0] + " " + sents[-1])
            else:
                skimmed.append(sents[0])
        return skimmed


class IntentDiscoveryAgent(Module):
    """Conversationally elicits KG-extraction requirements and missing details."""

    def __init__(self):
        super().__init__()
        self.question_gen = ChainOfThought(KGIntentQuestion)

    async def aforward(
        self, user_goal: str, file_overview: str, gathered_info: str = ""
    ):
        # Build guidance: focus on intent and business rules, not file content
        guidance_points = [
            "Ask one concise question.",
            "Focus only on the user's intent for KG generation and any business rules/constraints to consider before generation.",
            "Do NOT ask about or reference the uploaded file's content or structure.",
            "If enough info to draft a schema, answer 'DONE'.",
            "Avoid repeating answered items.",
        ]
        guidance = "\n".join(["Guidelines:"] + [f"- {g}" for g in guidance_points])

        # Augment gathered_info with guidance (without changing signature)
        augmented_info = f"{gathered_info}\n\n{guidance}\n"

        return await self.question_gen.acall(
            user_goal=user_goal,
            gathered_info=augmented_info,
            file_content="N/A — do not ask about file content; focus on intent and business rules.",
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
    """Extracts a knowledge graph from a text document.

    Adds lightweight retry and repair to improve robustness of LLM outputs.
    """

    def __init__(self, max_retries: int = 2, enable_repair: bool = True):
        self.module = Predict(signature=KGGeneration)
        self.max_retries = max(0, int(max_retries))
        self.enable_repair = bool(enable_repair)

    @staticmethod
    def _normalize_schema(schema_like: dict | Schema | None) -> Schema | None:
        if schema_like is None:
            return None
        if isinstance(schema_like, Schema):
            return schema_like
        try:
            return (
                Schema.model_validate(schema_like)
                if hasattr(Schema, "model_validate")
                else Schema(**schema_like)
            )
        except Exception:
            return None

    @staticmethod
    def _coerce_kg(kg_like: dict | Any, schema: Schema | None) -> dict:
        # Start from a dict shape
        if kg_like is None:
            kg = {}
        elif isinstance(kg_like, dict):
            kg = dict(kg_like)
        else:
            # dspy may return pydantic-like objects
            kg = getattr(kg_like, "dict", lambda: {})()

        entities = kg.get("entities") or []
        relations = kg.get("relations") or []

        # Build quick lookups from schema
        ent_type_map: dict[str, EntityType] = {}
        rel_type_map: dict[str, RelationType] = {}
        if schema and schema.entity_types:
            for et in schema.entity_types:
                if isinstance(et, dict):
                    try:
                        et_obj = EntityType(**et)
                    except Exception:
                        continue
                else:
                    et_obj = et
                ent_type_map[et_obj.name] = et_obj
        if schema and schema.relation_types:
            for rt in schema.relation_types:
                if isinstance(rt, dict):
                    try:
                        rt_obj = RelationType(**rt)
                    except Exception:
                        continue
                else:
                    rt_obj = rt
                rel_type_map[rt_obj.name] = rt_obj

        # Normalize entities
        norm_entities: list[dict] = []
        by_name: dict[str, dict] = {}
        for e in entities if isinstance(entities, list) else []:
            if not isinstance(e, dict):
                continue
            name = e.get("name") or e.get("id") or e.get("label")
            et = e.get("type")
            # Convert type to object-like dict
            if isinstance(et, str):
                et_obj = ent_type_map.get(et) or {"name": et}
            elif isinstance(et, dict):
                et_obj = et
            else:
                et_obj = None
            props = e.get("properties")
            if not isinstance(props, dict):
                props = {}
            ne = {
                "name": name or "",
                "type": et_obj or {"name": "Unknown"},
                "properties": props,
            }
            norm_entities.append(ne)
            if name:
                by_name[name] = ne

        # Normalize relations
        norm_relations: list[dict] = []
        for r in relations if isinstance(relations, list) else []:
            if not isinstance(r, dict):
                continue
            name = r.get("name") or r.get("type") or ""
            rt = r.get("type")
            if isinstance(rt, str):
                rt_obj = rel_type_map.get(rt) or {"name": rt}
            elif isinstance(rt, dict):
                rt_obj = rt
            else:
                rt_obj = {"name": name or "UnknownRelation"}

            subj = r.get("subject") or r.get("from") or r.get("source")
            obj = r.get("object") or r.get("to") or r.get("target")

            def _resolve_entity(x):
                if isinstance(x, dict):
                    n = x.get("name") or x.get("id") or x.get("label")
                    if isinstance(n, str):
                        return by_name.get(n) or {
                            "name": n or "",
                            "type": {"name": "Unknown"},
                            "properties": {},
                        }
                if isinstance(x, str):
                    return by_name.get(x) or {
                        "name": x,
                        "type": {"name": "Unknown"},
                        "properties": {},
                    }
                return {"name": "", "type": {"name": "Unknown"}, "properties": {}}

            subj_norm = _resolve_entity(subj)
            obj_norm = _resolve_entity(obj)
            props = r.get("properties")
            if not isinstance(props, dict):
                props = {}
            nr = {
                "name": name,
                "type": rt_obj,
                "subject": subj_norm,
                "object": obj_norm,
                "properties": props,
            }
            norm_relations.append(nr)

        # Final dict
        return {"entities": norm_entities, "relations": norm_relations}

    async def aforward(
        self,
        text: str,
        graph_schema: dict,
        *,
        max_retries: int | None = None,
        enable_repair: bool | None = None,
    ) -> dict:
        logger.info("Extracting knowledge graph from text.")
        attempts = 0
        max_r = self.max_retries if max_retries is None else max(0, int(max_retries))
        do_repair = self.enable_repair if enable_repair is None else bool(enable_repair)
        schema_norm = self._normalize_schema(graph_schema)

        while True:
            try:
                result = await self.module.acall(text=text, graph_schema=graph_schema)
                # Prefer raw dict if available
                kg_candidate = getattr(result, "knowledge_graph", None)
                if kg_candidate is None:
                    raise ValueError("KGGeneration returned no knowledge_graph")
                kg_raw = (
                    kg_candidate.dict()
                    if hasattr(kg_candidate, "dict")
                    else (kg_candidate if isinstance(kg_candidate, dict) else None)
                )
                if kg_raw is None:
                    # Attempt to coerce
                    kg_raw = self._coerce_kg(kg_candidate, schema_norm)
                # Try strict validation to ensure downstream compatibility
                try:
                    kg_model = KnowledgeGraph(**kg_raw)
                    return (
                        kg_model.model_dump()
                        if hasattr(kg_model, "model_dump")
                        else kg_model.dict()
                    )
                except Exception as ve:
                    if do_repair:
                        repaired = self._coerce_kg(kg_raw, schema_norm)
                        kg_model = KnowledgeGraph(**repaired)
                        return (
                            kg_model.model_dump()
                            if hasattr(kg_model, "model_dump")
                            else kg_model.dict()
                        )
                    raise ve
            except Exception as e:
                if attempts >= max_r:
                    logger.error(
                        "Error during knowledge graph extraction: %s", e, exc_info=True
                    )
                    return {
                        "error": "Failed to extract knowledge graph.",
                        "details": str(e),
                    }
                attempts += 1
                logger.info(
                    "KG extraction failed (attempt %d/%d). Retrying...", attempts, max_r
                )


class EntityNormalizationAgent(Module):
    """Normalizes entities in a knowledge graph (deduplicate names, unify references).

    Heuristics:
    - Canonical key: lowercase alphanumeric with single spaces.
    - Keep the first-seen cleaned name as the display name; merge properties.
    - Ensure relations reference the normalized entity dicts.
    - Optionally uses schema to keep type names consistent.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _key(name: str) -> str:
        s = (name or "").lower()
        s = re.sub(r"[^a-z0-9]+", " ", s).strip()
        return s

    @staticmethod
    def _clean_display(name: str) -> str:
        # Trim extra whitespace; keep original casing
        return re.sub(r"\s+", " ", (name or "").strip())

    @staticmethod
    def _as_schema_obj(x, cls):
        if x is None:
            return None
        if isinstance(x, cls):
            return x
        if isinstance(x, dict):
            try:
                return cls(**x)
            except Exception:
                return None
        if isinstance(x, str) and cls is EntityType:
            return EntityType(name=x)
        if isinstance(x, str) and cls is RelationType:
            return RelationType(name=x)
        return None

    async def aforward(
        self,
        knowledge_graph: dict | KnowledgeGraph,
        schema: dict | Schema | None = None,
    ) -> dict:
        # Normalize inputs
        if isinstance(knowledge_graph, KnowledgeGraph):
            kg_in = (
                knowledge_graph.model_dump()
                if hasattr(knowledge_graph, "model_dump")
                else knowledge_graph.dict()
            )
        else:
            kg_in = dict(knowledge_graph or {})

        schema_obj = None
        if schema is not None:
            try:
                schema_obj = (
                    schema
                    if isinstance(schema, Schema)
                    else (
                        Schema.model_validate(schema)
                        if hasattr(Schema, "model_validate")
                        else Schema(**schema)
                    )
                )
            except Exception:
                schema_obj = None

        ent_type_by_name: dict[str, EntityType] = {}
        if schema_obj and schema_obj.entity_types:
            for et in schema_obj.entity_types:
                et_obj = self._as_schema_obj(et, EntityType)
                if et_obj and isinstance(et_obj, EntityType):
                    ent_type_by_name[et_obj.name] = et_obj

        # Build canonical entities
        in_entities = kg_in.get("entities") or []
        canonical_map: dict[str, str] = {}
        canonical_entities: dict[str, dict] = {}

        for e in in_entities if isinstance(in_entities, list) else []:
            if not isinstance(e, dict):
                continue
            raw_name = e.get("name") or e.get("id") or e.get("label") or ""
            key = self._key(raw_name)
            display = self._clean_display(raw_name)
            et = e.get("type")
            et_obj = (
                self._as_schema_obj(et, EntityType)
                or ent_type_by_name.get(
                    str(et)
                    if isinstance(et, str)
                    else str(et.get("name"))
                    if isinstance(et, dict) and et.get("name") is not None
                    else ""
                )
                or EntityType(
                    name=(
                        et
                        if isinstance(et, str) and et is not None
                        else (
                            str(et.get("name"))
                            if isinstance(et, dict) and et.get("name") is not None
                            else "Unknown"
                        )
                    )
                )
            )
            props = e.get("properties") if isinstance(e.get("properties"), dict) else {}

            if key not in canonical_entities:
                canonical_entities[key] = {
                    "name": display,
                    "type": et_obj.model_dump()
                    if hasattr(et_obj, "model_dump")
                    else et_obj.model_dump(),
                    "properties": dict(props or {}),
                }
            else:
                # Merge properties (existing wins on conflict)
                merged = canonical_entities[key]
                merged_props = dict(props or {})
                merged_props.update(merged.get("properties") or {})
                merged["properties"] = merged_props
            canonical_map[raw_name] = canonical_entities[key]["name"]

        # Helper to resolve entity by various shapes
        def resolve_entity(x):
            if isinstance(x, dict):
                n = x.get("name") or x.get("id") or x.get("label") or ""
            elif isinstance(x, str):
                n = x
            else:
                n = ""
            k = self._key(n)
            ent = canonical_entities.get(k)
            if ent is None:
                ent = {
                    "name": self._clean_display(n),
                    "type": {"name": "Unknown"},
                    "properties": {},
                }
                canonical_entities[k] = ent
            return ent

        # Rebuild relations against canonical entities
        in_relations = kg_in.get("relations") or []
        out_relations: list[dict] = []
        for r in in_relations if isinstance(in_relations, list) else []:
            if not isinstance(r, dict):
                continue
            rt = r.get("type")
            rt_name = (
                rt
                if isinstance(rt, str)
                else (rt.get("name") if isinstance(rt, dict) else (r.get("name") or ""))
            )
            rt_obj = self._as_schema_obj(rt_name, RelationType) or RelationType(
                name=(rt_name or "UnknownRelation")
            )
            subj = resolve_entity(r.get("subject") or r.get("from") or r.get("source"))
            obj = resolve_entity(r.get("object") or r.get("to") or r.get("target"))
            props = r.get("properties") if isinstance(r.get("properties"), dict) else {}
            out_relations.append(
                {
                    "name": r.get("name") or rt_name or "",
                    "type": rt_obj.model_dump()
                    if hasattr(rt_obj, "model_dump")
                    else rt_obj.dict(),
                    "subject": subj,
                    "object": obj,
                    "properties": props,
                }
            )

        out_entities = list(canonical_entities.values())
        return {"entities": out_entities, "relations": out_relations}


class KGPipeline(Module):
    """End-to-end conversational pipeline for KG building.

    Handles:
    - Ensuring a file is uploaded and parsed
    - Intent discovery Q&A loop
    - Schema proposal and confirmation
    - Schema extraction
    - KG building and Neo4j storage
    """

    def __init__(
        self, redis: Redis, parser: PDFParser | None = None, neo4j: Any | None = None
    ):
        super().__init__()
        self.redis = redis
        self.parser = parser or PDFParser()
        self.neo4j = neo4j
        # Internal agents
        self.intent_agent = IntentDiscoveryAgent()
        self.schema_proposal_agent = SchemaProposalAgent()
        self.schema_extractor = SchemaExtractor()
        self.kg_generator = KGGenerator()
        self.entity_normalizer = EntityNormalizationAgent()
        self.reply_classifier = Predict(signature=ReplyClassification)
        self.schema_refiner = Predict(signature=SchemaRefinement)
        self.skim_summarizer = SkimSummarizer()

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
        file_overview = file_info.get("overview", "")
        if not file_content:
            filename = file_info.get("filename", "unknown")
            try:
                file_content = await self.parser.aparse(file_name=filename)
                file_overview = await self.skim_summarizer.acall(document=file_content)
                file_info["content"] = file_content
                file_info["overview"] = file_overview
                await redis.set(file_key, json.dumps(file_info))
                out.append(f"An overview of the uploaded document:\n{file_overview}\n")
            except Exception as e:
                out.append(f"Error processing uploaded file: {e}\n")

                return out
        # If we are awaiting KG extraction approval, handle that next
        kg_pending_key = f"kg_pending:{conv_id}"
        kg_pending_raw = await redis.get(kg_pending_key)
        if kg_pending_raw:
            try:
                kg_pending = json.loads(kg_pending_raw)
            except Exception:
                kg_pending = None
            schema_for_kg = (kg_pending or {}).get("schema")
            if schema_for_kg:
                try:
                    cls = await self.reply_classifier.acall(text=(user_message or ""))
                    decision = (getattr(cls, "label", "") or "").strip().lower()
                except Exception:
                    decision = ""

                if decision == "affirmative":
                    try:
                        full_text = await self.parser.aparse(
                            file_name=file_info.get("filename"), page_count=None
                        )
                        kg_dict = await self.kg_generator.acall(
                            text=full_text, graph_schema=schema_for_kg
                        )
                        if "error" in kg_dict:
                            raise RuntimeError(
                                kg_dict.get("details") or kg_dict["error"]
                            )
                        # Normalize entities before validation/storage
                        try:
                            normalized = await self.entity_normalizer.acall(
                                knowledge_graph=kg_dict, schema=schema_for_kg
                            )
                        except Exception:
                            normalized = kg_dict
                        try:
                            kg_model = KnowledgeGraph(**normalized)
                        except Exception as ve:
                            raise RuntimeError(
                                f"Invalid KG structure after normalization: {ve}"
                            )

                        if self.neo4j:
                            store_res = await self.neo4j.store_knowledge_graph(kg_model)
                            if isinstance(store_res, dict) and store_res.get("error"):
                                out.append(
                                    f"Warning: storing KG in Neo4j failed: {store_res.get('details') or store_res['error']}\n"
                                )
                            else:
                                out.append(
                                    f"Knowledge graph stored in Neo4j.\n{store_res}"
                                )
                        else:
                            out.append(
                                "Note: Neo4j not configured; skipped storing the knowledge graph.\n"
                            )
                    except Exception as e:
                        out.append(f"Warning: KG extraction/storage failed: {e}\n")
                    finally:
                        await redis.delete(kg_pending_key)
                    return out
                elif decision == "negative":
                    await redis.delete(kg_pending_key)
                    out.append("Understood. KG extraction canceled.\n")
                    return out
                elif decision == "change_request" or self._looks_like_change_request(
                    user_message
                ):
                    # Refine the current schema before KG extraction
                    try:
                        refined = await self.schema_refiner.acall(
                            instruction=user_message, current_schema=schema_for_kg
                        )
                        refined_schema = getattr(refined, "proposed_schema", None)
                        if refined_schema is not None:
                            refined_json = (
                                refined_schema.dict()
                                if hasattr(refined_schema, "dict")
                                else refined_schema
                            )
                            # Keep in kg_pending with the refined schema and ask again
                            await redis.set(
                                kg_pending_key, json.dumps({"schema": refined_json})
                            )
                            out.append("Updated schema based on your request:\n")
                            out.append(json.dumps(refined_json, indent=2) + "\n")
                            out.append(
                                "Proceed to extract the knowledge graph with the updated schema? (yes/no)\n"
                            )
                            return out
                    except Exception:
                        pass
                    # If we couldn't refine, ask for a clearer instruction
                    return out
                else:
                    # Unrecognized; wait for clearer instruction
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
                    decision = (getattr(cls, "label", "") or "").strip().lower()
                except Exception:
                    decision = ""
                if decision == "affirmative":
                    extractor = self.schema_extractor

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

                        out.append("Here is the extracted schema:\n")
                        out.append(json.dumps(extracted_json, indent=2) + "\n")
                        # Ask explicit approval before extracting the knowledge graph
                        await redis.delete(pending_key)
                        kg_pending_key = f"kg_pending:{conv_id}"
                        await redis.set(
                            kg_pending_key, json.dumps({"schema": extracted_json})
                        )
                        out.append(
                            "Would you like me to extract the knowledge graph now? (yes/no)\n"
                        )

                    except Exception as e:
                        out.append(f"Sorry, extraction failed: {e}\n")
                    return out
                elif decision == "negative":
                    out.append("Understood. Extraction canceled.\n")
                    return out
                elif decision == "change_request" or self._looks_like_change_request(
                    user_message
                ):
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
                    # Wait for clearer instruction
                    return out

        # Load discovery state
        disc_key = f"discovery:{conv_id}"
        state_raw = await redis.get(disc_key)
        state: dict[str, Any] = {
            "goal": "",
            "qa": [],
            "awaiting": None,
            "asked_count": 0,
        }
        if state_raw:
            try:
                state = json.loads(state_raw)
            except Exception:
                state = {"goal": "", "qa": [], "awaiting": None, "asked_count": 0}

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

        # Intent discovery loop: ask one question at a time until DONE, max 3 questions
        max_questions = 3
        asked_count = int(state.get("asked_count") or 0)
        while True:
            # Enforce hard cap
            if asked_count >= max_questions:
                break
            gathered_info = build_gathered_info(state)
            res = await self.intent_agent.acall(
                user_goal=state["goal"],
                gathered_info=gathered_info,
                file_overview=file_overview,
            )
            q = (getattr(res, "question", None) or "").strip()
            if not q:
                q = "Could you clarify your primary use case for the KG?"
            if q.upper() == "DONE":
                break
            # Ask exactly one question and persist awaiting state
            state["awaiting"] = q
            asked_count += 1
            state["asked_count"] = asked_count
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
                "Let me know if you'd like to proceed with extracting schema from the document.\n"
            )
            return out
        except Exception as e:
            out.append(f"Sorry, I couldn't complete schema extraction: {e}\n")
            return out
