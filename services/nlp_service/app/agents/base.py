"""Base agent — retry logic, latency tracking, confidence scoring."""

from __future__ import annotations

import time
from abc import abstractmethod
from typing import Any

from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from glossa_common.logging import get_logger

from ..domain.entities import AgentResult, AgentStatus
from ..domain.interfaces import AgentPort

logger = get_logger(__name__)


class BaseAgent(AgentPort):
    """Common plumbing for all graph agents.

    Sub-classes implement `_execute()` only.  This base class handles:
    - AsyncRetrying with exponential backoff
    - Latency measurement
    - Structured AgentResult construction
    - Fallback on final failure
    """

    MAX_RETRIES    = 2
    MIN_WAIT_S     = 0.1
    MAX_WAIT_S     = 1.0
    FALLBACK_OUTPUT = ""

    def __init__(self, llm, prompt_version: str = "1.0") -> None:
        self._llm = llm
        self._prompt_version = prompt_version

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def _execute(self, state: dict[str, Any]) -> AgentResult: ...

    async def run(self, state: dict[str, Any]) -> AgentResult:
        t0 = time.perf_counter()
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.MAX_RETRIES + 1),
                wait=wait_exponential(min=self.MIN_WAIT_S, max=self.MAX_WAIT_S),
                retry=retry_if_exception_type(Exception),
                reraise=False,
            ):
                with attempt:
                    result = await self._execute(state)
                    result.latency_ms = (time.perf_counter() - t0) * 1000
                    logger.debug(
                        "agent_success",
                        agent=self.name,
                        latency_ms=round(result.latency_ms, 1),
                        confidence=round(result.confidence, 3),
                    )
                    return result
        except Exception as exc:
            logger.exception("agent_failed_all_retries", agent=self.name)
            return AgentResult(
                agent=self.name,
                status=AgentStatus.FALLBACK,
                output=self.FALLBACK_OUTPUT,
                confidence=0.0,
                latency_ms=(time.perf_counter() - t0) * 1000,
                error=str(exc),
            )

    def _build_messages(self, system: str, user: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ]
