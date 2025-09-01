from hashlib import sha256
import json
from typing import Any, Optional

import dspy
from dspy.clients import Cache
import redis
from redis import Redis

from app.config import settings


class RedisCache(Cache):
    """
    A DSPy cache that stores LM call results in Redis.
    Compatible with the interface described in DSPy's cache tutorial:
    - cache_key(request, ignored_args_for_cache_key)
    - get(request, ignored_args_for_cache_key)
    - put(request, value, ignored_args_for_cache_key, enable_memory_cache=True)
    """

    def __init__(
        self,
        redis_client: Optional[Redis] = None,
        namespace: str = "dspy:cache:",
        ttl_seconds: Optional[int] = 3600,
        # accept the common flags so you can swap this in without breaking code
        enable_memory_cache: bool = True,
        enable_disk_cache: bool = False,
        disk_cache_dir: str = "",
        **kwargs,
    ):
        super().__init__(
            enable_disk_cache=enable_disk_cache,
            enable_memory_cache=enable_memory_cache,
            disk_cache_dir=disk_cache_dir,
            **kwargs,
        )

        self.r = redis_client or redis.from_url(settings.REDIS_URL)
        self.namespace = namespace
        self.ttl = ttl_seconds

    # --- Keying strategy  ---
    def cache_key(
        self,
        request: dict[str, Any],
        ignored_args_for_cache_key: Optional[list[str]] = None,
    ) -> str:
        """
        Produce a stable key from the request. We follow the tutorial’s pattern:
        hash only the 'messages' (chat) payload by default, ignoring creds, etc.
        If you need to key on other fields (e.g., model name, temperature),
        include them here.
        """
        ignored = set(ignored_args_for_cache_key or [])

        # Default: prefer messages; if absent, fall back to the full request minus ignored keys
        if "messages" in request:
            basis = {"messages": request["messages"]}
        else:
            basis = {k: v for k, v in request.items() if k not in ignored}

        blob = json.dumps(basis, sort_keys=True, ensure_ascii=False)
        digest = sha256(blob.encode("utf-8")).hexdigest()
        return f"{self.namespace}{digest}"

    # --- Read from Redis ---
    def get(
        self,
        request: dict[str, Any],
        ignored_args_for_cache_key: Optional[list[str]] = None,
    ) -> Any:
        key = self.cache_key(request, ignored_args_for_cache_key)
        raw = self.r.get(key)
        if not raw:
            return None
        try:
            return json.loads(str(raw))
        except json.JSONDecodeError:
            # if someone manually wrote non-json, just return raw
            return raw

    # --- Write to Redis ---
    def put(
        self,
        request: dict[str, Any],
        value: Any,
        ignored_args_for_cache_key: Optional[list[str]] = None,
        enable_memory_cache: bool = True,  # kept for signature compatibility
    ) -> None:
        key = self.cache_key(request, ignored_args_for_cache_key)
        payload = json.dumps(value, ensure_ascii=False)
        if self.ttl and self.ttl > 0:
            self.r.set(key, payload, ex=self.ttl)
        else:
            self.r.set(key, payload)


def setup_dspy(redis_client: Redis | None = None) -> None:
    """Sets up the dspy configuration using environment variables.
    Raises:
        ValueError: If required environment variables are not set.
    """
    if not settings.OPENAI_MODEL or not settings.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_MODEL and OPENAI_API_KEY must be set as environment variables."
        )

    openai_model = settings.OPENAI_MODEL
    if not str(openai_model).startswith("openai/"):
        openai_model = "openai/" + str(openai_model)

    dspy.settings.configure(
        lm=dspy.LM(
            model=openai_model,
            api_key=settings.OPENAI_API_KEY.get_secret_value(),
            temperature=0.0,
        )
    )
    dspy.cache = RedisCache()
