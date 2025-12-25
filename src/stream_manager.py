"""Manages streaming sessions using Redis streams."""

import json
from datetime import datetime
from typing import Any, AsyncGenerator
from uuid import uuid4

import redis.asyncio as aioredis
from redis.asyncio import Redis

from src import create_logger
from src.config import app_config, app_settings
from src.schemas.types import EventsType

logger = create_logger("stream_manager")


class StreamSessionManager:
    """Manages streaming sessions using Redis streams. Redis streams are used to store and
    retrieve streaming events associated with unique session IDs.

    Methods
    -------
    acreate_session() -> str
        Creates a new streaming session and returns its unique session ID.
    asession_exists(session_id: str) -> bool
        Checks if a streaming session exists.
    asend_event(session_id: str, event_type: str, data: Any, metadata: dict[str, Any] | None = None) -> bool
        Sends an event to the Redis stream for the given session.
    aread_events(session_id: str, last_id: str = "0", block: int = 5_000, count: int = 100)
    -> AsyncGenerator[dict[str, Any], None]
        Reads events from the Redis stream for the given session.
    add_connection(session_id: str) -> None
        Adds a new active connection for the given session ID.
    remove_connection(session_id: str) -> None
        Removes an active connection for the given session ID.
    aclose_session(session_id: str) -> None
        Closes the streaming session and cleans up resources.
    acleanup_old_sessions() -> None
        Cleans up old sessions that have exceeded their TTL.
    aclose() -> None
        Closes the Redis client connection.
    """

    def __init__(self, redis_client: Redis | None = None) -> None:
        self._redis = redis_client
        self._is_initialized: bool = False
        self._active_connecton: dict[str, Any] = {}

    async def _aget_redis(self) -> Redis:
        """Lazy initialization of Redis client.

        Returns
        -------
        Redis
            An instance of the Redis client.
        """
        if self._redis is None and not self._is_initialized:
            # Initialize Redis client
            # Use the constructed redis URL from settings (includes password if present)
            self._redis = await aioredis.from_url(
                app_settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            self._is_initialized = True

        if self._redis is None:
            raise RuntimeError("Redis client is not initialized")

        return self._redis

    def _get_stream_key(self, session_id: str) -> str:
        """Constructs the Redis stream key for a given session ID.

        Parameters
        ----------
        session_id : str
            The unique identifier for the session.

        Returns
        -------
        str
            The Redis stream key.
        """
        return f"{app_config.stream_config.stream_prefix}{session_id}"

    def _get_metadata_key(self, session_id: str) -> str:
        """Constructs the Redis metadata key for a given session ID.

        Parameters
        ----------
        session_id : str
            The unique identifier for the session.

        Returns
        -------
        str
            The Redis metadata key.
        """
        return f"{self._get_stream_key(session_id)}:metadata"

    def _get_user_sessions_key(self, user_id: str) -> str:
        """Constructs the Redis key for tracking user's sessions.

        Parameters
        ----------
        user_id : str
            The unique identifier for the user.

        Returns
        -------
        str
            The Redis key for user's session set.
        """
        return f"user:{user_id}:sessions"

    async def acreate_session(
        self, session_id: str | None = None, user_id: str | None = None
    ) -> str:
        """Creates a new streaming session.

        Parameters
        ----------
        session_id : str | None, optional
            The unique identifier for the session. If None, a new UUID will be generated, by default None
        user_id : str | None, optional
            The unique identifier for the user. If provided, session is indexed for the user, by default None

        Returns
        -------
        str
            The unique identifier for the newly created session.
        """
        # Generate session ID if not provided
        session_id: str = session_id if session_id is not None else str(uuid4())  # type: ignore
        redis = await self._aget_redis()
        # Session metadata
        metadata: dict[str, Any] = {
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "last_activity": datetime.now().isoformat(),
        }
        if user_id:
            metadata["user_id"] = user_id
        metadata_key: str = self._get_metadata_key(session_id)
        # Store metadata in a hash
        await redis.hset(metadata_key, mapping=metadata)  # type: ignore
        # Set TTL for metadata
        await redis.expire(metadata_key, app_config.stream_config.stream_ttl)

        # Index session for user if user_id provided
        if user_id:
            user_sessions_key: str = self._get_user_sessions_key(user_id)
            # Add session ID to user's session set
            await redis.sadd(user_sessions_key, session_id)  # type: ignore
            await redis.expire(user_sessions_key, app_config.stream_config.stream_ttl)

        self._active_connecton[session_id] = 0
        logger.info(
            f"Created new session with ID: {session_id}"
            + (f" for user {user_id}" if user_id else "")
        )
        return session_id

    async def asession_exists(self, session_id: str) -> bool:
        """Checks if a streaming session exists.

        Parameters
        ----------
        session_id : str
            The unique identifier for the session.

        Returns
        -------
        bool
            True if the session exists, False otherwise.
        """
        redis = await self._aget_redis()
        metadata_key: str = self._get_metadata_key(session_id)
        exists: bool = await redis.exists(metadata_key) > 0
        return exists

    async def aget_user_sessions(self, user_id: str) -> list[dict[str, Any]]:
        """Get all sessions for a user with metadata.

        Parameters
        ----------
        user_id : str
            The unique identifier for the user.

        Returns
        -------
        list[dict[str, Any]]
            List of session metadata dicts, sorted by creation time (newest first).
        """
        redis = await self._aget_redis()
        user_sessions_key: str = self._get_user_sessions_key(user_id)
        session_ids: set[str] = await redis.smembers(user_sessions_key)  # type: ignore

        sessions: list[dict[str, Any]] = []
        for session_id in session_ids:
            metadata_key: str = self._get_metadata_key(session_id)
            metadata: dict[str, Any] = await redis.hgetall(metadata_key)  # type: ignore
            if metadata:
                metadata["session_id"] = session_id
                sessions.append(metadata)

        # Sort by created_at descending (newest first)
        sessions.sort(
            key=lambda x: x.get("created_at", ""),
            reverse=True,
        )
        return sessions

    async def asend_event(
        self,
        session_id: str,
        event_type: str,
        data: Any,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """Sends an event to the Redis stream for the given session.

        Parameters
        ----------
        session_id : str
            The unique identifier for the session.
        event_type : str
            The type of event being sent.
        data : Any
            The data associated with the event.
        metadata : dict[str, Any] | None, optional
            Additional metadata for the event, by default None

        Returns
        -------
        bool
            True if the event was successfully sent, False otherwise.
        """
        redis = await self._aget_redis()
        stream_key: str = self._get_stream_key(session_id)
        # Prepare message
        message: dict[str, Any] = {
            "event_type": event_type,
            "data": json.dumps(data) if not isinstance(data, str) else data,
            "timestamp": datetime.now().isoformat(),
        }
        if metadata:
            message["metadata"] = json.dumps(metadata)

        try:
            # Add message to stream
            message_id: str = await redis.xadd(
                stream_key,
                message,  # type: ignore
                maxlen=app_config.stream_config.max_stream_length,
                approximate=True,
            )
            # Update session activity
            metadata_key: str = self._get_metadata_key(session_id)
            await redis.hset(metadata_key, "last_activity", datetime.now().isoformat())  # type: ignore
            # Extend TTL for metadata and stream
            await redis.expire(metadata_key, app_config.stream_config.stream_ttl)
            await redis.expire(stream_key, app_config.stream_config.stream_ttl)
            logger.debug(f"Added message ID {message_id} to stream {stream_key}")
            return True

        except Exception as e:
            logger.error(
                f"Failed to send event with session ID {session_id} to stream {stream_key}: {e}"
            )
            return False

    async def aread_events(
        self,
        session_id: str,
        last_id: str = "0",
        block: int = 5_000,
        count: int = 100,
        stop_on_session_end: bool = True,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Reads events from the Redis stream for the given session.

        Parameters
        ----------
        session_id : str
            The unique identifier for the session.
        last_id : str, optional
            The ID of the last event read, by default "0"
        block : int, optional
            The number of milliseconds to block while waiting for new events, by default 5_000
        count : int, optional
            The maximum number of events to read at once, by default 100
        stop_on_session_end : bool, optional
            If True, stop reading when the first SESSION_ENDED is seen. For full
            history replays (last_id != "$"), set to False so subsequent tasks in the
            same session are also streamed.

        Returns
        -------
        AsyncGenerator[dict[str, Any], None]
            Stream of events from Redis.
        """
        redis = await self._aget_redis()
        stream_key: str = self._get_stream_key(session_id)
        current_id: str = last_id

        while True:
            try:
                # Try to read new messages from the stream
                response = await redis.xread(
                    {stream_key: current_id}, block=block, count=count
                )
                if not response:
                    # No new messages; yield keep-alive
                    yield {
                        "event_type": EventsType.KEEPALIVE,
                        "data": "",
                        "timestamp": datetime.now().isoformat(),
                    }
                    continue

                # Process messages
                for _, messages in response:
                    for message_id, message_data in messages:
                        event: dict[str, Any] = {
                            "id": message_id,  # Include Redis message ID for client reconnection
                            "event_type": message_data.get("event_type"),
                            "data": json.loads(message_data.get("data"))
                            if message_data.get("data")
                            else None,
                            "timestamp": message_data.get("timestamp"),
                        }
                        if "metadata" in message_data:
                            event["metadata"] = json.loads(message_data.get("metadata"))
                        # Update current ID
                        current_id = message_id
                        yield event

                        # Check for completion event
                        if (
                            stop_on_session_end
                            and event["event_type"] == EventsType.SESSION_ENDED
                        ):
                            logger.info(f"Stream completed for session ID {session_id}")
                            # Stop reading further events
                            return

            except aioredis.ConnectionError as e:
                logger.error(
                    f"Redis connection error while reading events for session ID {session_id}: {e}"
                )
                yield {
                    "event_type": EventsType.ERROR,
                    "data": "Redis connection error.",
                    "timestamp": datetime.now().isoformat(),
                }
                return
            except Exception as e:
                logger.error(
                    f"Unexpected error while reading events for session ID {session_id}: {e}"
                )
                yield {
                    "event_type": EventsType.ERROR,
                    "data": "An unexpected error occurred.",
                    "timestamp": datetime.now().isoformat(),
                }
                return

    def add_connection(self, session_id: str) -> None:
        """Adds a new active connection for the given session ID.

        Parameters
        ----------
        session_id : str
            The unique identifier for the session.
        """
        self._active_connecton[session_id] = (
            self._active_connecton.get(session_id, 0) + 1
        )
        logger.info(
            f"Added connection for session ID {session_id}. "
            f"Total connections: {self._active_connecton[session_id]}"
        )

    def remove_connection(self, session_id: str) -> None:
        """Removes an active connection for the given session ID by decrementing the count.

        If multiple clients are connected to the same session, only remove one connection.

        Parameters
        ----------
        session_id : str
            The unique identifier for the session.
        """
        if (
            self._is_initialized
            and session_id in self._active_connecton
            and self._active_connecton[session_id] > 0
        ):
            self._active_connecton[session_id] -= 1
            logger.info(
                f"Removed connection for session ID {session_id}. "
                f"Total connections: {self._active_connecton[session_id]}"
            )

    async def aclose_session(self, session_id: str) -> None:
        """Closes the streaming session and cleans up resources.

        Parameters
        ----------
        session_id : str
            The unique identifier for the session.
        """
        redis = await self._aget_redis()
        stream_key: str = self._get_stream_key(session_id)
        metadata_key: str = self._get_metadata_key(session_id)

        # Gracefully delete stream and metadata by setting expiration
        await redis.expire(stream_key, time=60)
        await redis.expire(metadata_key, time=60)

        # Remove from active connections
        if session_id in self._active_connecton:
            del self._active_connecton[session_id]

        logger.info(f"Closed session with ID: {session_id}")

    async def acleanup_old_sessions(self) -> None:
        """Cleans up old sessions that have exceeded their TTL."""
        redis = await self._aget_redis()
        pattern: str = f"{app_config.stream_config.stream_prefix}*"
        cleaned: int = 0

        async for key in redis.scan_iter(match=pattern):
            ttl = await redis.ttl(key)
            if ttl == -1:  # Key does not exist
                await redis.delete(key)
                # Extract session ID from key
                session_id: str = key.replace(
                    app_config.stream_config.stream_prefix, ""
                ).replace(":metadata", "")

                if session_id in self._active_connecton:
                    await self.aclose_session(session_id)
                logger.info(f"Cleaned up old stream with key: {key}")
                cleaned += 1

    async def aclose(self) -> None:
        """Closes the Redis client connection."""
        if self._redis and self._is_initialized:
            self._is_initialized = False
            await self._redis.close()
            self._redis = None
            logger.info("Closed Redis client connection.")
