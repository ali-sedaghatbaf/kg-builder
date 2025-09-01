from pydantic import AnyUrl, HttpUrl, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # FastAPI
    API_PORT: int = 8000
    API_HOST: str = "0.0.0.0"

    # DSPy
    ANTHROPIC_API_KEY: SecretStr | None = None
    OPENAI_API_KEY: SecretStr | None = None
    LLAMA_CLOUD_API_KEY: SecretStr | None = None
    OPENAI_MODEL: str = "gpt-4o"

    # Langfuse
    LANGFUSE_PUBLIC_KEY: SecretStr | None = None
    LANGFUSE_SECRET_KEY: SecretStr | None = None
    LANGFUSE_HOST: HttpUrl = HttpUrl("https://cloud.langfuse.com")
    LANGFUSE_ENABLED: bool = False

    # Neo4j
    NEO4J_URI: AnyUrl = AnyUrl("bolt://localhost:7687")
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: SecretStr | None = None

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_URL: str = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

    # Colbert
    COLBERT_URL: HttpUrl = HttpUrl("http://localhost:8000")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def check_required_secrets(self) -> "Settings":
        """Ensures that required secrets are provided in the environment."""
        required_secrets = ["OPENAI_API_KEY", "NEO4J_PASSWORD", "LLAMA_CLOUD_API_KEY"]
        for secret in required_secrets:
            if not getattr(self, secret):
                raise ValueError(f"{secret} is not set in the environment.")
        return self


settings = Settings()
