import logging
import os

from neo4j import AsyncGraphDatabase

logger = logging.getLogger(__name__)


class Neo4j:
    def __init__(self):
        self.initialize_db()

    def initialize_db(self):
        """
        Initializes the Neo4j database connection.
        Raises:
            ValueError: If required environment variables are not set.
        """
        uri = os.getenv("APP_NEO4J_URI")
        user = os.getenv("APP_NEO4J_USER")
        password = os.getenv("APP_NEO4J_PASSWORD")
        if not all((uri, user, password)):
            logger.error("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set.")
            raise ValueError("NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set.")
        logger.info("Connecting to Neo4j database at %s", uri)
        self.driver = AsyncGraphDatabase.driver(uri=uri, auth=(user, password))
        if not self.driver.session():
            logger.error("Failed to connect to Neo4j database.")
            raise ConnectionError("Failed to connect to Neo4j database.")

    async def generate_contract_kg(self, contract_data: dict, contract_path: str):
        import_query = """
        WITH $contract AS contract
        MERGE (c:Contract {path: $path})
        //SET c += apoc.map.clean(contract, ["parties", "agreement_date", "effective_date", "expiration_date"], [])
        // Cast to date
        SET c.agreement_date = CASE 
            WHEN c.agreement_date IS NULL THEN NULL 
            ELSE date(c.agreement_date) 
            END, 
            c.effective_date =     
            CASE 
            WHEN c.effective_date IS NULL THEN NULL 
            ELSE date(contract.effective_date)
            END,
            c.expiration_date = 
            CASE 
            WHEN c.expiration_date IS NULL THEN NULL 
            ELSE date(contract.expiration_date)
            END


        // Create parties with their locations
        WITH c, contract
        UNWIND coalesce(contract.parties, []) AS party
        MERGE (p:Party {name: party.name})
        MERGE (c)-[:HAS_PARTY]->(p)

        // Create location nodes and link to parties
        WITH p, party
        WHERE party.location IS NOT NULL
        MERGE (p)-[:HAS_LOCATION]->(l:Location)
        SET l += party.location
        """
        logger.info("Importing contract data into Neo4j: %s", contract_path)
        response = await self.driver.execute_query(
            import_query, contract=contract_data, path=contract_path
        )
        if not response:
            logger.error("Failed to import contract data into Neo4j.")
            return {"error": "Failed to import contract data into Neo4j."}
        return dict(vars(response.summary.counters))

    async def close(self):
        """
        Closes the Neo4j database connection.
        """
        await self.driver.close()
