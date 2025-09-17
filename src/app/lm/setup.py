from hashlib import sha256
import json
import logging
import threading
from typing import Any, Optional

import dspy
from dspy.clients import Cache
import redis
from redis.asyncio import Redis as AsyncRedis

from ..config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


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
        redis_client: Optional[redis.Redis] = None,
        redis_async_client: Optional[AsyncRedis] = None,
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

        # Use a plain (sync) Redis client for DSPy cache operations.
        # Ensure decoded string responses to avoid bytes/JSON issues.
        # Sync client for compatibility with DSPy's sync Cache API
        self.r: redis.Redis = redis_client or redis.from_url(
            settings.REDIS_URL, decode_responses=True
        )
        # Optional async client if the host app wants to provide one
        self.ra: Optional[AsyncRedis] = redis_async_client
        self.namespace = namespace
        self.ttl = ttl_seconds
        self._mem: dict[str, Any] = {}

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
        # If we're inside an async loop, avoid blocking: use memory cache only.
        try:
            import asyncio

            asyncio.get_running_loop()
            if key in self._mem:
                return self._mem[key]
            return None
        except RuntimeError:
            try:
                self.r.delete(key)
            except Exception:
                pass
            return None

    # --- Write to Redis ---
    def put(
        self,
        request: dict[str, Any],
        value: Any,
        ignored_args_for_cache_key: Optional[list[str]] = None,
        enable_memory_cache: bool = True,  # kept for signature compatibility
    ) -> None:
        key = self.cache_key(request, ignored_args_for_cache_key)
        # Always populate local memory cache to keep async paths responsive
        self._mem[key] = value
        # Try to serialize for Redis persistence; if it fails, skip Redis write gracefully
        try:
            payload = json.dumps(value, ensure_ascii=False)
        except TypeError:
            payload = None

        def _write():
            if payload is None:
                return
            try:
                if self.ttl and self.ttl > 0:
                    self.r.set(key, payload, ex=self.ttl)
                else:
                    self.r.set(key, payload)
            except Exception:
                pass

        try:
            import asyncio

            asyncio.get_running_loop()
            threading.Thread(target=_write, daemon=True).start()
        except RuntimeError:
            # No running loop: safe to write synchronously
            _write()


def setup_dspy(redis_client: AsyncRedis | None = None) -> None:
    """Sets up the dspy configuration using environment variables.
    Raises:
        ValueError: If required environment variables are not set.
    """
    if not settings.OPENAI_MODEL or not settings.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_MODEL and OPENAI_API_KEY must be set as environment variables."
        )

    # Prefer a local LLM model id if provided; otherwise use OPENAI_MODEL

    provider_model = settings.OPENAI_MODEL
    if not str(provider_model).startswith("openai/"):
        provider_model = "openai/" + str(provider_model)

    base_url = str(settings.LOCAL_LLM_URL) if settings.USE_LOCAL_LLM else None
    if base_url:
        provider_model = settings.LOCAL_LLM_MODEL
    logger.info("Configuring LM: model=%s base_url=%s", provider_model, base_url)

    dspy.settings.configure(
        model=provider_model,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
        temperature=0.0,
        base_url=base_url,
    )

    dspy.cache = RedisCache(redis_async_client=redis_client)
