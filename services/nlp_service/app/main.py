"""FastAPI application entry point — wires all agents into the NLPOrchestrator."""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import make_asgi_app

from glossa_common.logging import configure_logging, get_logger
from glossa_common.telemetry import instrument_fastapi, setup_telemetry

from .agents import (
    BankingSimplifierAgent,
    DialogueManagerAgent,
    GlossTranslatorAgent,
    HallucinationCheckerAgent,
    IntentClassifierAgent,
    MedicalSimplifierAgent,
    RAGRetrieverAgent,
    ResponseGeneratorAgent,
    SafetyFilterAgent,
)
from .config import get_settings
from .graphs.orchestrator import NLPOrchestrator
from .infrastructure.qwen_runner import QwenRunner
from .infrastructure.rag_retriever import QdrantRAGRetriever
from .memory.conversation import InMemoryConversationStore, RedisConversationStore
from .routers.v1 import admin, chat, health

logger = get_logger(__name__)
settings = get_settings()


def _build_memory_store(cfg):
    if cfg.memory_backend == "redis":
        return RedisConversationStore(redis_url=cfg.redis_url)
    return InMemoryConversationStore()


def _build_agents(llm, retriever, memory_store, cfg) -> dict:
    v = cfg.default_prompt_version
    dialogue_mgr = DialogueManagerAgent(
        llm=llm,
        memory_store=memory_store,
        prompt_version=v,
    )
    return {
        "intent_classifier":      IntentClassifierAgent(llm=llm, prompt_version=v),
        "gloss_translator":       GlossTranslatorAgent(llm=llm, prompt_version=v),
        "medical_simplifier":     MedicalSimplifierAgent(llm=llm, prompt_version=v),
        "banking_simplifier":     BankingSimplifierAgent(llm=llm, prompt_version=v),
        "dialogue_manager":       dialogue_mgr,
        "rag_retriever":          RAGRetrieverAgent(llm=llm, retriever=retriever, prompt_version=v),
        "safety_filter":          SafetyFilterAgent(llm=llm, prompt_version=v),
        "hallucination_checker":  HallucinationCheckerAgent(llm=llm, prompt_version=v),
        "response_generator":     ResponseGeneratorAgent(llm=llm, prompt_version=v),
    }


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging(
        level=settings.log_level,
        fmt=settings.log_format,
        service_name=settings.service_name,
    )
    logger.info("starting_nlp_service", service=settings.service_name)

    setup_telemetry(
        service_name=settings.service_name,
        service_version=settings.service_version,
        otlp_endpoint=settings.otel_exporter_otlp_endpoint,
        sample_rate=settings.otel_traces_sampler_arg,
    )

    # LLM backend
    runner = QwenRunner(
        model_path=settings.model_path,
        device=settings.device,
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        top_p=settings.top_p,
        load_in_4bit=settings.load_in_4bit,
        load_in_8bit=settings.load_in_8bit,
    )
    await runner.ensure_loaded()
    logger.info("llm_ready", model=settings.model_path, device=settings.device)

    # RAG retriever
    retriever = QdrantRAGRetriever(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        embedding_model=settings.embedding_model,
        top_k=settings.rag_top_k,
        score_threshold=settings.rag_score_threshold,
    )
    if settings.rag_enabled:
        await retriever.ensure_initialized()
        logger.info("rag_ready", collection=settings.qdrant_collection)

    # Conversation memory
    memory_store = _build_memory_store(settings)
    await memory_store.start()
    logger.info("memory_store_ready", backend=settings.memory_backend)

    # Assemble agents and orchestrator
    agents = _build_agents(runner, retriever, memory_store, settings)
    orchestrator = NLPOrchestrator(
        agents=agents,
        dialogue_manager=agents["dialogue_manager"],
    )

    app.state.orchestrator = orchestrator
    app.state.memory_store = memory_store
    app.state.runner = runner
    app.state.retriever = retriever
    app.state.start_time = time.time()

    logger.info("nlp_service_ready", agents=list(agents.keys()))
    yield

    logger.info("nlp_service_stopping")
    await memory_store.stop()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Glossa NLP Service",
        description="Qwen2-1.5B + LangGraph multi-agent pipeline for RSL translation",
        version=settings.service_version,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url=None,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, tags=["Health"])
    app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
    app.include_router(admin.router, prefix="/api/v1", tags=["Admin"])

    app.mount("/metrics", make_asgi_app())
    instrument_fastapi(app)

    return app


app = create_app()
