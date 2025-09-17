import logging
from typing import Any

from neo4j import AsyncGraphDatabase

from ..config import get_settings
from ..lm.models import KnowledgeGraph

logger = logging.getLogger(__name__)


class Neo4j:
    async def initialize_db(self):
        """
        Initializes the Neo4j database connection.
        Raises:
            ValueError: If required environment variables are not set.
        """
        settings = get_settings()
        uri = settings.APP_NEO4J_URI
        user = settings.APP_NEO4J_USER
        password = (
            settings.APP_NEO4J_PASSWORD.get_secret_value()
            if settings.APP_NEO4J_PASSWORD
            else ""
        )

        if not all((uri, user, password)):
            logger.error("APP_NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set.")
            raise ValueError(
                "APP_NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set."
            )
        logger.info("Connecting to Neo4j database at %s", uri)
        self.driver = AsyncGraphDatabase.driver(
            uri=uri.encoded_string(), auth=(user, password)
        )
        if not self.driver.session():
            logger.error("Failed to connect to Neo4j database.")
            raise ConnectionError("Failed to connect to Neo4j database.")

    async def close(self):
        """
        Closes the Neo4j database connection.
        """
        await self.driver.close()

    async def store_knowledge_graph(self, kg: KnowledgeGraph) -> dict[str, Any]:
        """Persist a KnowledgeGraph into Neo4j.

        Creates/merges Entity nodes with a generic label `Entity` and properties:
        - name (string)
        - type (string; the EntityType name)
        - properties (map; flattened via SET +=)

        Creates/merges relationships with type `RELATED_TO` between subject and object
        nodes. Relationship properties include:
        - name (string)
        - type (string; the RelationType name)
        - plus any provided relation properties (SET +=)
        """
        # Prepare payloads safe for Cypher params
        entities_payload = []
        for e in kg.entities:
            ent_type = getattr(e.type, "name", None) or (
                e.type if isinstance(e.type, str) else None
            )
            props = e.properties or {}
            entities_payload.append(
                {
                    "name": e.name,
                    "type": ent_type,
                    "props": props,
                }
            )

        relations_payload = []
        for r in kg.relations:
            rel_type = getattr(r.type, "name", None) or (
                r.type if isinstance(r.type, str) else None
            )
            rel_props = r.properties or {}
            relations_payload.append(
                {
                    "name": r.name,
                    "type": rel_type,
                    "props": rel_props,
                    "subject": {
                        "name": r.subject.name,
                        "type": getattr(r.subject.type, "name", None),
                    },
                    "object": {
                        "name": r.object.name,
                        "type": getattr(r.object.type, "name", None),
                    },
                }
            )

        cypher = """
        // Upsert entities
        UNWIND $entities AS ent
        MERGE (e:Entity {name: ent.name, type: ent.type})
        SET e += ent.props

        // Upsert relations
        WITH $relations AS rels
        UNWIND rels AS rel
        MATCH (s:Entity {name: rel.subject.name, type: rel.subject.type})
        MATCH (o:Entity {name: rel.object.name, type: rel.object.type})
        MERGE (s)-[r:RELATED_TO {name: rel.name, type: rel.type}]->(o)
        SET r += rel.props
        """

        logger.info(
            "Storing knowledge graph with %d entities and %d relations",
            len(entities_payload),
            len(relations_payload),
        )
        try:
            result = await self.driver.execute_query(
                cypher,
                entities=entities_payload,
                relations=relations_payload,
            )
            return dict(vars(result.summary.counters))
        except Exception as e:
            logger.error("Failed to store knowledge graph: %s", e, exc_info=True)
            return {"error": "Failed to store knowledge graph.", "details": str(e)}
