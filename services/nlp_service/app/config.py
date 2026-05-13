from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from glossa_common.config import BaseServiceSettings


class NLPSettings(BaseServiceSettings):
    model_config = SettingsConfigDict(env_prefix="NLP_", env_file=".env", extra="ignore")

    service_name: str = "nlp-service"
    port: int = Field(default=8003, ge=1, le=65535)

    # LLM backend
    model_path: str = "/models/Qwen2-1.5B-Instruct"
    device: str = "auto"          # auto | cpu | cuda | mps
    max_new_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    context_window: int = Field(default=4096, ge=512)
    load_in_4bit: bool = False
    load_in_8bit: bool = False

    # RAG / Qdrant
    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "rsl_glossary"
    rag_top_k: int = Field(default=5, ge=1, le=20)
    rag_score_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    embedding_model: str = "BAAI/bge-m3"
    rag_enabled: bool = True

    # Safety & quality guards
    safety_llm_enabled: bool = True
    hallucination_check_enabled: bool = True

    # Conversation memory
    memory_backend: Literal["memory", "redis"] = "memory"
    redis_url: str = "redis://redis:6379/1"
    session_ttl_active: int = Field(default=7200, ge=60)    # seconds
    session_ttl_disconnected: int = Field(default=300, ge=30)
    summarize_after_turns: int = Field(default=10, ge=2)

    # Prompt versioning
    default_prompt_version: str = "1.0"


@lru_cache
def get_settings() -> NLPSettings:
    return NLPSettings()
