from typing import Annotated

import httpx
import redis.asyncio as aioredis
from fastapi import Depends, Request

from glossa_common.messaging import RedisStreamPublisher

from .config import GatewaySettings, get_settings


def get_http_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.http_client  # type: ignore[no-any-return]


def get_redis(request: Request) -> aioredis.Redis:
    return request.app.state.redis  # type: ignore[no-any-return]


def get_publisher(request: Request) -> RedisStreamPublisher:
    return request.app.state.publisher  # type: ignore[no-any-return]


SettingsDep = Annotated[GatewaySettings, Depends(get_settings)]
HttpClientDep = Annotated[httpx.AsyncClient, Depends(get_http_client)]
RedisDep = Annotated[aioredis.Redis, Depends(get_redis)]
PublisherDep = Annotated[RedisStreamPublisher, Depends(get_publisher)]
