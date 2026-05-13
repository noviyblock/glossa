import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import orjson
import redis.asyncio as aioredis
from redis.asyncio.client import Redis
from tenacity import retry, stop_after_attempt, wait_exponential

from ..logging import get_logger
from .events import GlossaEvent

logger = get_logger(__name__)


class RedisStreamPublisher:
    def __init__(self, redis: Redis, max_len: int = 10_000) -> None:
        self._redis = redis
        self._max_len = max_len

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=5))
    async def publish(self, event: GlossaEvent) -> str:
        stream = event.stream_name()
        payload = orjson.dumps(event.model_dump(mode="json")).decode()
        msg_id: str = await self._redis.xadd(  # type: ignore[assignment]
            stream,
            {"data": payload, "event_type": event.event_type},
            maxlen=self._max_len,
            approximate=True,
        )
        logger.debug("event_published", stream=stream, event_type=event.event_type, msg_id=msg_id)
        return msg_id


class RedisStreamConsumer:
    def __init__(
        self,
        redis: Redis,
        stream: str,
        group: str,
        consumer_name: str,
        batch_size: int = 10,
        block_ms: int = 1000,
    ) -> None:
        self._redis = redis
        self._stream = stream
        self._group = group
        self._consumer = consumer_name
        self._batch_size = batch_size
        self._block_ms = block_ms

    async def ensure_group(self) -> None:
        try:
            await self._redis.xgroup_create(
                self._stream, self._group, id="0", mkstream=True
            )
        except aioredis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    async def consume(self) -> AsyncIterator[tuple[str, GlossaEvent]]:
        await self.ensure_group()
        while True:
            try:
                results: list[Any] = await self._redis.xreadgroup(
                    self._group,
                    self._consumer,
                    {self._stream: ">"},
                    count=self._batch_size,
                    block=self._block_ms,
                )
                if not results:
                    await asyncio.sleep(0)
                    continue

                for _stream, messages in results:
                    for msg_id, fields in messages:
                        try:
                            data = orjson.loads(fields[b"data"])
                            event = GlossaEvent.model_validate(data)
                            yield msg_id, event
                        except Exception:
                            logger.exception(
                                "failed_to_parse_event", msg_id=msg_id, fields=fields
                            )
                            await self._redis.xack(self._stream, self._group, msg_id)
            except aioredis.ConnectionError:
                logger.warning("redis_connection_lost", stream=self._stream)
                await asyncio.sleep(1)

    async def ack(self, msg_id: str) -> None:
        await self._redis.xack(self._stream, self._group, msg_id)


@asynccontextmanager
async def create_redis_pool(url: str) -> AsyncIterator[Redis]:
    pool = aioredis.from_url(
        url,
        encoding="utf-8",
        decode_responses=False,
        max_connections=20,
        socket_connect_timeout=5,
        socket_timeout=5,
        retry_on_timeout=True,
    )
    try:
        await pool.ping()
        logger.info("redis_connected", url=url)
        yield pool
    finally:
        await pool.aclose()
        logger.info("redis_disconnected")
