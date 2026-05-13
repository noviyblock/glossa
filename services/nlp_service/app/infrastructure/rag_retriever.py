"""Qdrant RAG retriever — multilingual embeddings (BGE-M3) + domain filtering."""

from __future__ import annotations

import asyncio
from typing import Any

from glossa_common.logging import get_logger
from glossa_common.schemas.nlp import RAGDocument

logger = get_logger(__name__)


class QdrantRAGRetriever:
    """Retrieves RSL/domain documents from Qdrant vector store.

    Uses BGE-M3 for multilingual embeddings that handle Russian + gloss sequences.
    Supports optional domain filtering via Qdrant payload filters.
    """

    def __init__(
        self,
        url: str,
        collection: str,
        embedding_model: str = "BAAI/bge-m3",
        top_k: int = 5,
        score_threshold: float = 0.65,
        cache_embeddings: bool = True,
    ) -> None:
        self._url = url
        self._collection = collection
        self._embedding_model = embedding_model
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._cache_embeddings = cache_embeddings
        self._client = None
        self._encoder = None
        self._embed_cache: dict[str, list[float]] = {}  # LRU in real usage

    async def ensure_initialized(self) -> None:
        if self._client is None:
            await asyncio.to_thread(self._initialize)

    def _initialize(self) -> None:
        from qdrant_client import QdrantClient
        from sentence_transformers import SentenceTransformer

        self._client = QdrantClient(url=self._url, timeout=10)
        self._encoder = SentenceTransformer(self._embedding_model)
        logger.info(
            "rag_initialized",
            collection=self._collection,
            embedding_model=self._embedding_model,
        )

    async def retrieve(self, query: str, domain: str | None = None) -> list[RAGDocument]:
        await self.ensure_initialized()
        if not query.strip():
            return []
        try:
            return await asyncio.to_thread(self._search_sync, query, domain)
        except Exception:
            logger.exception("rag_retrieval_failed", query=query[:80])
            return []

    def _search_sync(self, query: str, domain: str | None) -> list[RAGDocument]:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Embedding with cache
        embedding = self._get_embedding(query)

        # Optional domain filter
        query_filter = None
        if domain and domain != "general":
            query_filter = Filter(
                must=[FieldCondition(key="domain", match=MatchValue(value=domain))]
            )

        results = self._client.search(
            collection_name=self._collection,
            query_vector=embedding,
            limit=self._top_k,
            score_threshold=self._score_threshold,
            with_payload=True,
            query_filter=query_filter,
        )

        return [
            RAGDocument(
                id=str(r.id),
                text=r.payload.get("text", "") if r.payload else "",
                score=float(r.score),
                metadata=r.payload or {},
            )
            for r in results
        ]

    def _get_embedding(self, text: str) -> list[float]:
        if self._cache_embeddings and text in self._embed_cache:
            return self._embed_cache[text]
        embedding = self._encoder.encode(text, normalize_embeddings=True).tolist()
        if self._cache_embeddings:
            if len(self._embed_cache) > 512:  # simple eviction
                oldest = next(iter(self._embed_cache))
                del self._embed_cache[oldest]
            self._embed_cache[text] = embedding
        return embedding

    def format_context(self, docs: list[RAGDocument]) -> str:
        if not docs:
            return ""
        parts = []
        for doc in docs:
            gloss = doc.metadata.get("gloss", "")
            translation = doc.metadata.get("translation", doc.text)
            parts.append(f"• {gloss}: {translation}" if gloss else f"• {doc.text}")
        return "\n".join(parts)
