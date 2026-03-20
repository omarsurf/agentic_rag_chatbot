"""Session Store Abstraction for GRI RAG.

Provides pluggable session persistence:
- InMemorySessionStore: Default, for dev/single-instance
- RedisSessionStore: For distributed deployments
- PostgresSessionStore: For persistent sessions across restarts

Usage:
    from src.core.session_store import get_session_store

    store = get_session_store()
    await store.save_session(session_id, memory)
    memory = await store.load_session(session_id)
"""

import json
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Any, cast

import structlog

from src.core.config import settings
from src.core.memory import ConversationTurn, GRIMemory

log = structlog.get_logger()


class SessionStore(ABC):
    """Abstract base class for session storage."""

    @abstractmethod
    async def create_session(self, session_id: str) -> GRIMemory:
        """Create a new session with the given ID.

        Args:
            session_id: Unique session identifier

        Returns:
            New GRIMemory instance with session_id set
        """
        pass

    async def get_or_create_session(self, session_id: str) -> GRIMemory:
        """Get existing session or create new one.

        Args:
            session_id: Unique session identifier

        Returns:
            GRIMemory instance (existing or new)
        """
        memory = await self.load_session(session_id)
        if memory is None:
            memory = await self.create_session(session_id)
        return memory

    def get_session(self, session_id: str) -> GRIMemory | None:
        """Synchronous get_session for test compatibility.

        Args:
            session_id: Unique session identifier

        Returns:
            GRIMemory instance or None if not found
        """
        import asyncio

        try:
            asyncio.get_running_loop()
            # We're in an async context, create a task
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.load_session(session_id))
                return future.result()
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            return asyncio.run(self.load_session(session_id))

    @abstractmethod
    async def save_session(self, session_id: str, memory: GRIMemory) -> bool:
        """Save session memory.

        Args:
            session_id: Unique session identifier
            memory: GRIMemory instance to persist

        Returns:
            True if saved successfully
        """
        pass

    @abstractmethod
    async def load_session(self, session_id: str) -> GRIMemory | None:
        """Load session memory, or None if not found/expired.

        Args:
            session_id: Unique session identifier

        Returns:
            GRIMemory instance or None if not found
        """
        pass

    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Delete a session.

        Args:
            session_id: Unique session identifier

        Returns:
            True if deleted successfully
        """
        pass

    @abstractmethod
    async def cleanup_expired(self) -> int:
        """Clean up expired sessions.

        Returns:
            Count of removed sessions
        """
        pass


class InMemorySessionStore(SessionStore):
    """In-memory session store (default, for dev/single-instance)."""

    def __init__(self, ttl_seconds: int | None = None, default_ttl: int | None = None) -> None:
        self._sessions: dict[str, tuple[GRIMemory, float]] = {}
        self._ttl = default_ttl or ttl_seconds or settings.session_ttl_seconds

    async def create_session(self, session_id: str) -> GRIMemory:
        """Create a new session."""
        memory = GRIMemory(session_id=session_id)
        self._sessions[session_id] = (memory, time.time())
        return memory

    async def save_session(self, session_id: str, memory: GRIMemory) -> bool:
        self._sessions[session_id] = (memory, time.time())
        return True

    async def load_session(self, session_id: str) -> GRIMemory | None:
        if session_id not in self._sessions:
            return None

        memory, last_access = self._sessions[session_id]

        # Check expiry
        if time.time() - last_access > self._ttl:
            del self._sessions[session_id]
            return None

        # Update last access time
        self._sessions[session_id] = (memory, time.time())
        return memory

    async def delete_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False

    async def cleanup_expired(self) -> int:
        now = time.time()
        expired = [
            sid for sid, (_, last_access) in self._sessions.items() if now - last_access > self._ttl
        ]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)


class RedisSessionStore(SessionStore):
    """Redis-backed session store for distributed deployments."""

    def __init__(
        self,
        redis_url: str | None = None,
        ttl_seconds: int | None = None,
        default_ttl: int | None = None,
    ) -> None:
        self._redis_url: str = redis_url or settings.redis_url or ""
        self._ttl = default_ttl or ttl_seconds or settings.session_ttl_seconds
        self._redis: Any = None
        self._key_prefix = "gri:session:"

    async def create_session(self, session_id: str) -> GRIMemory:
        """Create a new session in Redis."""
        memory = GRIMemory(session_id=session_id)
        await self.save_session(session_id, memory)
        return memory

    async def _get_redis(self) -> Any:
        if self._redis is None:
            import redis.asyncio as redis

            self._redis = redis.from_url(self._redis_url)
        return self._redis

    def _serialize_memory(self, memory: GRIMemory) -> str:
        """Serialize GRIMemory to JSON."""
        turns_data = [turn.to_dict() for turn in memory.turns]
        return json.dumps(
            {
                "session_id": memory.session_id,
                "turns": turns_data,
                "max_turns": memory._max_turns,
            }
        )

    def _deserialize_memory(self, data: str) -> GRIMemory:
        """Deserialize JSON to GRIMemory."""
        parsed = json.loads(data)
        memory = GRIMemory(
            session_id=parsed.get("session_id"),
            max_turns=parsed.get("max_turns", 10),
        )

        for turn_data in parsed.get("turns", []):
            turn = ConversationTurn(
                query=turn_data["query"],
                answer=turn_data["answer"],
                timestamp=datetime.fromisoformat(turn_data["timestamp"]),
                intent=turn_data.get("intent"),
                cycle=turn_data.get("cycle"),
                tool_calls=turn_data.get("tool_calls", []),
                citations=turn_data.get("citations", []),
            )
            memory._turns.append(turn)

        return memory

    async def save_session(self, session_id: str, memory: GRIMemory) -> bool:
        try:
            redis = await self._get_redis()
            key = f"{self._key_prefix}{session_id}"
            data = self._serialize_memory(memory)
            await redis.setex(key, self._ttl, data)
            return True
        except Exception as e:
            log.error("redis_session.save_failed", session_id=session_id, error=str(e))
            return False

    async def load_session(self, session_id: str) -> GRIMemory | None:
        try:
            redis = await self._get_redis()
            key = f"{self._key_prefix}{session_id}"
            data = await redis.get(key)

            if data is None:
                return None

            # Refresh TTL on access
            await redis.expire(key, self._ttl)

            return self._deserialize_memory(data.decode("utf-8"))
        except Exception as e:
            log.error("redis_session.load_failed", session_id=session_id, error=str(e))
            return None

    async def delete_session(self, session_id: str) -> bool:
        try:
            redis = await self._get_redis()
            key = f"{self._key_prefix}{session_id}"
            result = await redis.delete(key)
            return cast(int, result) > 0
        except Exception as e:
            log.error("redis_session.delete_failed", session_id=session_id, error=str(e))
            return False

    async def cleanup_expired(self) -> int:
        # Redis handles TTL automatically
        return 0


class PostgresSessionStore(SessionStore):
    """PostgreSQL-backed session store for persistent sessions."""

    def __init__(
        self,
        dsn: str | None = None,
        ttl_seconds: int | None = None,
        default_ttl: int | None = None,
    ) -> None:
        self._dsn: str = dsn or settings.postgres_dsn or ""
        self._ttl = default_ttl or ttl_seconds or settings.session_ttl_seconds
        self._pool: Any = None

    async def create_session(self, session_id: str) -> GRIMemory:
        """Create a new session in PostgreSQL."""
        memory = GRIMemory(session_id=session_id)
        await self.save_session(session_id, memory)
        return memory

    async def _get_pool(self) -> Any:
        if self._pool is None:
            import asyncpg

            self._pool = await asyncpg.create_pool(self._dsn, min_size=2, max_size=10)
        return self._pool

    def _serialize_memory(self, memory: GRIMemory) -> dict[str, Any]:
        """Serialize GRIMemory to JSON-compatible dict."""
        return {
            "session_id": memory.session_id,
            "turns": [turn.to_dict() for turn in memory.turns],
            "max_turns": memory._max_turns,
        }

    def _deserialize_memory(self, data: dict[str, Any]) -> GRIMemory:
        """Deserialize dict to GRIMemory."""
        memory = GRIMemory(
            session_id=data.get("session_id"),
            max_turns=data.get("max_turns", 10),
        )

        for turn_data in data.get("turns", []):
            turn = ConversationTurn(
                query=turn_data["query"],
                answer=turn_data["answer"],
                timestamp=datetime.fromisoformat(turn_data["timestamp"]),
                intent=turn_data.get("intent"),
                cycle=turn_data.get("cycle"),
                tool_calls=turn_data.get("tool_calls", []),
                citations=turn_data.get("citations", []),
            )
            memory._turns.append(turn)

        return memory

    async def save_session(self, session_id: str, memory: GRIMemory) -> bool:
        try:
            pool = await self._get_pool()
            data = self._serialize_memory(memory)
            expires_at = datetime.utcnow() + timedelta(seconds=self._ttl)

            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO sessions (session_id, memory_data, expires_at)
                    VALUES ($1, $2, $3)
                    ON CONFLICT (session_id)
                    DO UPDATE SET
                        memory_data = $2,
                        updated_at = CURRENT_TIMESTAMP,
                        expires_at = $3
                    """,
                    session_id,
                    json.dumps(data),
                    expires_at,
                )

            return True
        except Exception as e:
            log.error("postgres_session.save_failed", session_id=session_id, error=str(e))
            return False

    async def load_session(self, session_id: str) -> GRIMemory | None:
        try:
            pool = await self._get_pool()

            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT memory_data FROM sessions
                    WHERE session_id = $1 AND expires_at > CURRENT_TIMESTAMP
                    """,
                    session_id,
                )

                if row is None:
                    return None

                # Update expiry on access
                expires_at = datetime.utcnow() + timedelta(seconds=self._ttl)
                await conn.execute(
                    """
                    UPDATE sessions
                    SET updated_at = CURRENT_TIMESTAMP, expires_at = $2
                    WHERE session_id = $1
                    """,
                    session_id,
                    expires_at,
                )

                data = json.loads(row["memory_data"])
                return self._deserialize_memory(data)
        except Exception as e:
            log.error("postgres_session.load_failed", session_id=session_id, error=str(e))
            return None

    async def delete_session(self, session_id: str) -> bool:
        try:
            pool = await self._get_pool()

            async with pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM sessions WHERE session_id = $1", session_id
                )
                return "DELETE 1" in cast(str, result)
        except Exception as e:
            log.error("postgres_session.delete_failed", session_id=session_id, error=str(e))
            return False

    async def cleanup_expired(self) -> int:
        try:
            pool = await self._get_pool()

            async with pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM sessions WHERE expires_at < CURRENT_TIMESTAMP"
                )
                # Parse "DELETE N" to get count
                count = int(result.split()[1]) if result else 0
                return count
        except Exception as e:
            log.error("postgres_session.cleanup_failed", error=str(e))
            return 0


# Factory function
_session_store: SessionStore | None = None


def get_session_store() -> SessionStore:
    """Get the configured session store singleton.

    Returns:
        SessionStore instance based on configuration
    """
    global _session_store

    if _session_store is None:
        backend = getattr(settings, "session_backend", "memory")

        if backend == "redis" and hasattr(settings, "redis_url") and settings.redis_url:
            _session_store = RedisSessionStore()
            log.info("session_store.init", backend="redis")
        elif backend == "postgres" and hasattr(settings, "postgres_dsn") and settings.postgres_dsn:
            _session_store = PostgresSessionStore()
            log.info("session_store.init", backend="postgres")
        else:
            _session_store = InMemorySessionStore()
            log.info("session_store.init", backend="memory")

    return _session_store


def reset_session_store() -> None:
    """Reset the session store singleton (for testing)."""
    global _session_store
    _session_store = None
