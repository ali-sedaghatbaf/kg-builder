from functools import lru_cache

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
    APP_NEO4J_URI: AnyUrl = AnyUrl("bolt://localhost:7687")
    APP_NEO4J_USER: str = "neo4j"
    APP_NEO4J_PASSWORD: SecretStr | None = None

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_AUTH: SecretStr | None = None
    # Allow overriding via env; if not provided, compute in validator below
    REDIS_URL: str | None = None

    # LlamaIndex
    COLBERT_URL: HttpUrl = HttpUrl("http://localhost:8000")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @model_validator(mode="after")
    def check_required_secrets(self) -> "Settings":
        """Ensures that required secrets are provided in the environment."""
        required_secrets = [
            "OPENAI_API_KEY",
            "APP_NEO4J_PASSWORD",
            "LLAMA_CLOUD_API_KEY",
        ]
        for secret in required_secrets:
            if not getattr(self, secret):
                raise ValueError(f"{secret} is not set in the environment.")
        # Compute REDIS_URL if not provided explicitly, using current env-backed values
        if not self.REDIS_URL:
            auth_part = (
                f":{self.REDIS_AUTH.get_secret_value()}@" if self.REDIS_AUTH else ""
            )
            self.REDIS_URL = f"redis://{auth_part}{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
