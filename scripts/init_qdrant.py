#!/usr/bin/env python3
"""Initialize Qdrant collections for RSL glossary RAG."""
import asyncio
import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff, OptimizersConfigDiff

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "rsl_glossary")
VECTOR_SIZE = 1024  # BGE-M3 embedding dimension


def init_collections(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}

    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            hnsw_config=HnswConfigDiff(m=16, ef_construct=200),
            optimizers_config=OptimizersConfigDiff(memmap_threshold=20_000),
        )
        print(f"Created collection: {COLLECTION}")
    else:
        print(f"Collection already exists: {COLLECTION}")

    # Create payload index for fast filtering by domain
    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="domain",
        field_schema="keyword",
    )
    print("Payload indexes created.")


def seed_sample_glossary(client: QdrantClient) -> None:
    """Seed a small sample glossary for development."""
    from qdrant_client.models import PointStruct
    import hashlib

    sample_entries = [
        {"gloss": "ВРАЧ", "translation": "врач, доктор", "domain": "medical"},
        {"gloss": "БОЛЬ", "translation": "боль, болит", "domain": "medical"},
        {"gloss": "ДАВЛЕНИЕ", "translation": "артериальное давление", "domain": "medical"},
        {"gloss": "БАНК", "translation": "банк, банковский", "domain": "banking"},
        {"gloss": "ДЕНЬГИ", "translation": "деньги, денежные средства", "domain": "banking"},
        {"gloss": "КРЕДИТ", "translation": "кредит, займ", "domain": "banking"},
        {"gloss": "СПАСИБО", "translation": "спасибо, благодарю", "domain": "general"},
        {"gloss": "ПОМОЩЬ", "translation": "помощь, помогите", "domain": "general"},
    ]

    # Use fake embeddings for seed data (replace with real encoder in production)
    import random
    random.seed(42)

    points = []
    for i, entry in enumerate(sample_entries):
        uid = int(hashlib.md5(entry["gloss"].encode()).hexdigest()[:8], 16)
        vector = [random.uniform(-1, 1) for _ in range(VECTOR_SIZE)]
        # Normalize
        norm = sum(v**2 for v in vector) ** 0.5
        vector = [v / norm for v in vector]
        points.append(
            PointStruct(
                id=uid,
                vector=vector,
                payload={
                    "gloss": entry["gloss"],
                    "translation": entry["translation"],
                    "text": f"{entry['gloss']}: {entry['translation']}",
                    "domain": entry["domain"],
                },
            )
        )

    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Seeded {len(points)} glossary entries.")


def main() -> None:
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    client = QdrantClient(url=QDRANT_URL, timeout=30)

    try:
        client.get_collections()
        print("Connected to Qdrant.")
    except Exception as e:
        print(f"Failed to connect to Qdrant: {e}", file=sys.stderr)
        sys.exit(1)

    init_collections(client)

    if "--seed" in sys.argv:
        seed_sample_glossary(client)

    print("Qdrant initialization complete.")


if __name__ == "__main__":
    main()
