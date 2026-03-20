"""API FastAPI pour le système RAG GRI.

Endpoints:
- POST /query         : Query standard (JSON response)
- POST /query/stream  : Query avec SSE streaming
- GET  /health        : Healthcheck
- GET  /stats         : Statistiques de l'index
- GET  /metrics       : Prometheus metrics
- POST /feedback      : Feedback utilisateur

Usage:
    uvicorn src.api.main:app --reload --port 8000
"""

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager, suppress
from datetime import datetime
from typing import Any, cast

import structlog
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response
from starlette.types import ASGIApp

from src.agents.orchestrator import GRIOrchestrator
from src.api.auth import is_auth_required, verify_token
from src.api.models import (
    Citation,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    Source,
    StatsResponse,
    ToolCallInfo,
)
from src.api.streaming import stream_query_response
from src.core.config import settings
from src.core.memory import GRIMemory
from src.core.session_store import (
    SessionStore,
    get_session_store,
    reset_session_store,
)
from src.core.vector_store import GRIHybridStore

log = structlog.get_logger()

# === Rate Limiter ===
limiter = Limiter(key_func=get_remote_address)

# === Global State ===
_store: GRIHybridStore | None = None
_session_store: SessionStore | None = None
_start_time: float = time.time()
_query_count: int = 0
_total_latency: float = 0.0
_cleanup_task: asyncio.Task | None = None


def get_store() -> GRIHybridStore:
    """Récupère le store global."""
    if _store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not initialized",
        )
    return _store


async def get_or_create_memory(session_id: str | None) -> GRIMemory:
    """Récupère ou crée une mémoire de session via SessionStore."""
    if session_id is None:
        return GRIMemory()

    if _session_store is None:
        # Fallback if session store not initialized
        return GRIMemory(session_id=session_id)

    return await _session_store.get_or_create_session(session_id)


async def _cleanup_expired_sessions() -> None:
    """Periodically clean up expired sessions via SessionStore."""
    while True:
        await asyncio.sleep(60)  # Check every minute
        if _session_store is not None:
            try:
                expired_count = await _session_store.cleanup_expired()
                if expired_count > 0:
                    log.info("api.sessions_cleanup", expired_count=expired_count)
            except Exception as e:
                log.warning("api.sessions_cleanup_failed", error=str(e))


# === Lifespan ===


@asynccontextmanager
async def lifespan(app_instance: FastAPI) -> AsyncIterator[None]:
    """Gestion du cycle de vie de l'application."""
    global _store, _session_store, _cleanup_task
    _ = app_instance

    log.info("api.startup", port=settings.api_port, auth_required=is_auth_required())

    # Repartir d'un store propre à chaque cycle de vie de l'app.
    reset_session_store()

    # Initialiser le session store
    try:
        _session_store = get_session_store()
        log.info("api.session_store_initialized")
    except Exception as e:
        log.error("api.session_store_init_failed", error=str(e))

    # Initialiser le vector store
    try:
        _store = GRIHybridStore()
        log.info("api.store_initialized")
    except Exception as e:
        log.error("api.store_init_failed", error=str(e))
        # Continue anyway - health endpoint will report degraded status

    # Start session cleanup task
    _cleanup_task = asyncio.create_task(_cleanup_expired_sessions())
    log.info("api.session_cleanup_started", ttl_seconds=settings.session_ttl_seconds)

    yield

    # Cleanup
    log.info("api.shutdown")
    if _cleanup_task:
        _cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await _cleanup_task
    if _store is not None:
        _store.close()
    _store = None
    reset_session_store()
    _session_store = None


# === Timeout Middleware ===


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Middleware pour timeout des requêtes."""

    def __init__(self, app: ASGIApp, timeout_seconds: int = 60) -> None:
        super().__init__(app)
        self.timeout = timeout_seconds

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        try:
            return await asyncio.wait_for(call_next(request), timeout=self.timeout)
        except TimeoutError:
            return JSONResponse(
                status_code=504,
                content={
                    "error": "Gateway Timeout",
                    "detail": f"Request timeout after {self.timeout}s",
                },
            )


# === App ===

app = FastAPI(
    title="GRI RAG API",
    description="API pour interroger le Guide de Référence pour l'Innovation (GRI)",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# === Rate Limiter Configuration ===
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded,
    cast(
        Callable[[Request, Exception], Response | Awaitable[Response]],
        _rate_limit_exceeded_handler,
    ),
)

# === Prometheus Instrumentation ===
# Note: expose() est appelé à la fin du fichier après tous les endpoints
instrumentator = Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
    should_respect_env_var=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=["/health", "/metrics"],
    inprogress_name="gri_rag_requests_inprogress",
    inprogress_labels=True,
)

# === Middleware ===

# Timeout middleware (60 seconds)
app.add_middleware(TimeoutMiddleware, timeout_seconds=60)

# CORS configuration - parse allowed origins from config
_cors_origins = [
    origin.strip() for origin in settings.cors_allowed_origins.split(",") if origin.strip()
]
# Security: cannot use credentials with wildcard origins
_allow_credentials = "*" not in _cors_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_allow_credentials,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)


@app.middleware("http")
async def log_requests(request: Request, call_next: RequestResponseEndpoint) -> Response:
    """Log toutes les requêtes."""
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id

    start = time.time()

    log.info(
        "api.request.start",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
    )

    try:
        response = await call_next(request)
        latency = (time.time() - start) * 1000

        log.info(
            "api.request.done",
            request_id=request_id,
            status_code=response.status_code,
            latency_ms=f"{latency:.0f}",
        )

        response.headers["X-Request-ID"] = request_id
        return response

    except Exception as e:
        log.error(
            "api.request.error",
            request_id=request_id,
            error=str(e),
        )
        raise


# === Exception Handlers ===


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handler pour les exceptions HTTP."""
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.__class__.__name__,
            message=exc.detail,
            request_id=request_id,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handler pour les exceptions non gérées."""
    request_id = getattr(request.state, "request_id", None)
    log.error(
        "api.unhandled_exception",
        request_id=request_id,
        error=str(exc),
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="InternalServerError",
            message="Une erreur inattendue s'est produite",
            detail=str(exc) if settings.log_level == "DEBUG" else None,
            request_id=request_id,
        ).model_dump(),
    )


# === Endpoints ===


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Vérifie l'état de santé du service.

    Returns:
        HealthResponse avec le statut et les infos de connexion
    """
    collections: dict[str, int] = {}
    qdrant_connected = False

    if _store is not None:
        try:
            # Vérifier Qdrant
            stats = await _store.get_collection_stats()
            collections = {name: info.get("vectors_count", 0) for name, info in stats.items()}
            qdrant_connected = True
        except Exception as e:
            log.warning("health.qdrant_check_failed", error=str(e))

    status_val = "healthy" if qdrant_connected else "degraded"
    if _store is None:
        status_val = "unhealthy"

    return HealthResponse(
        status=status_val,
        version="0.1.0",
        qdrant_connected=qdrant_connected,
        collections=collections,
        timestamp=datetime.utcnow(),
    )


@app.get("/stats", response_model=StatsResponse, tags=["System"])
async def stats(_token: str | None = Depends(verify_token)) -> StatsResponse:
    """Retourne les statistiques du système.

    Returns:
        StatsResponse avec les métriques
    """
    global _query_count, _total_latency

    collections: dict[str, dict[str, Any]] = {}
    total_docs = 0

    if _store is not None:
        try:
            coll_stats = await _store.get_collection_stats()
            for name, info in coll_stats.items():
                count = info.get("vectors_count", 0)
                collections[name] = {
                    "vectors_count": count,
                    "status": info.get("status", "unknown"),
                }
                total_docs += count
        except Exception as e:
            log.warning("stats.collection_stats_failed", error=str(e))

    avg_latency = (_total_latency / _query_count) if _query_count > 0 else None
    uptime = time.time() - _start_time

    return StatsResponse(
        total_documents=total_docs,
        collections=collections,
        avg_latency_ms=avg_latency,
        queries_today=_query_count,
        uptime_seconds=uptime,
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
@limiter.limit("30/minute")
async def query(
    request: Request,
    query_request: QueryRequest,
    _token: str | None = Depends(verify_token),
) -> QueryResponse:
    """Exécute une question sur le système RAG GRI.

    Args:
        request: FastAPI Request (pour rate limiting)
        query_request: QueryRequest avec la question et les options
        _token: Bearer token (validated by dependency)

    Returns:
        QueryResponse avec la réponse et les sources
    """
    global _query_count, _total_latency
    _ = request

    store = get_store()
    memory = await get_or_create_memory(query_request.session_id)

    # Créer l'orchestrateur
    orchestrator = GRIOrchestrator(store=store, memory=memory)

    # Exécuter la query
    try:
        result = await orchestrator.run(query_request.query)
    except Exception as e:
        log.error("query.orchestrator_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors du traitement de la question: {str(e)}",
        )

    # Mettre à jour les stats
    _query_count += 1
    _total_latency += result.latency_ms

    if query_request.session_id and _session_store is not None:
        saved = await _session_store.save_session(query_request.session_id, memory)
        if not saved:
            log.warning(
                "api.session_save_failed",
                session_id=query_request.session_id,
            )

    # Construire la réponse
    query_id = str(uuid.uuid4())

    # Convertir les citations
    citations = [Citation(text=c, section=c.strip("[]")) for c in result.citations]

    # Récupérer les sources si demandé
    sources = None
    if query_request.include_sources and result.collected_chunks:
        sources = [
            Source(
                chunk_id=chunk.get("chunk_id", f"chunk_{idx}"),
                content=chunk.get("content", "")[:500],  # Tronquer pour la réponse
                section_type=chunk.get("section_type", "unknown"),
                score=chunk.get("score", 0.0),
                metadata={
                    k: v
                    for k, v in chunk.items()
                    if k not in ("content", "section_type", "score", "chunk_id")
                },
            )
            for idx, chunk in enumerate(result.collected_chunks[: query_request.max_chunks])
        ]

    # Convertir les tool calls
    tool_calls = [
        ToolCallInfo(
            tool=tc["tool"],
            input=tc.get("input", {}),
            iteration=tc.get("iteration", 0),
            success=tc.get("success", True),
        )
        for tc in result.tool_calls
    ]

    return QueryResponse(
        query_id=query_id,
        answer=result.answer,
        intent=result.intent,
        cycle=result.cycle,
        citations=citations,
        sources=sources,
        tool_calls=tool_calls,
        latency_ms=result.latency_ms,
        iterations=result.iterations,
        warning=result.warning,
        created_at=datetime.utcnow(),
    )


@app.post("/query/stream", tags=["Query"])
@limiter.limit("30/minute")
async def query_stream(
    request: Request,
    query_request: QueryRequest,
    _token: str | None = Depends(verify_token),
) -> EventSourceResponse:
    """Exécute une question avec streaming SSE.

    Retourne un flux d'événements SSE:
    - routing: Intent et cycle détectés
    - tool_call: Appel de tool en cours
    - tool_result: Résultat d'un tool
    - chunk: Partie de la réponse
    - done: Réponse complète
    - error: En cas d'erreur

    Args:
        request: FastAPI Request (pour rate limiting)
        query_request: QueryRequest

    Returns:
        EventSourceResponse (SSE stream)
    """
    _ = request
    store = get_store()
    memory = await get_or_create_memory(query_request.session_id)

    return EventSourceResponse(
        stream_query_response(
            query=query_request.query,
            store=store,
            memory=memory,
            session_store=_session_store,
            include_sources=query_request.include_sources,
            max_chunks=query_request.max_chunks,
        )
    )


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
@limiter.limit("30/minute")
async def submit_feedback(
    request: Request,
    feedback: FeedbackRequest,
    _token: str | None = Depends(verify_token),
) -> FeedbackResponse:
    """Soumet un feedback utilisateur sur une réponse.

    Args:
        request: FastAPI Request (pour rate limiting)
        feedback: FeedbackRequest avec le rating et commentaires
        _token: Bearer token (validated by dependency)

    Returns:
        FeedbackResponse
    """
    log.info(
        "api.feedback",
        query_id=feedback.query_id,
        rating=feedback.rating,
        has_comment=feedback.comment is not None,
        has_incorrect_info=feedback.incorrect_info is not None,
    )

    # Persister le feedback
    from src.core.feedback_store import save_feedback

    request_id = getattr(request.state, "request_id", None)

    success = await save_feedback(
        query_id=feedback.query_id,
        rating=feedback.rating,
        comment=feedback.comment,
        incorrect_info=feedback.incorrect_info,
        metadata={"request_id": request_id} if request_id else None,
    )

    if success:
        return FeedbackResponse(
            success=True,
            message="Merci pour votre feedback !",
        )
    else:
        return FeedbackResponse(
            success=False,
            message="Erreur lors de l'enregistrement du feedback.",
        )


@app.get("/feedback/stats", tags=["Feedback"])
@limiter.limit("10/minute")
async def get_feedback_statistics(
    request: Request,
    _token: str | None = Depends(verify_token),
) -> dict[str, Any]:
    """Retourne les statistiques de feedback.

    Args:
        request: FastAPI Request (pour rate limiting)
        _token: Bearer token (validated by dependency)

    Returns:
        Statistiques de feedback
    """
    _ = request
    from src.core.feedback_store import get_feedback_stats

    stats = await get_feedback_stats()
    return stats


@app.delete("/sessions/{session_id}", tags=["Sessions"])
@limiter.limit("10/minute")
async def clear_session(
    request: Request,
    session_id: str,
    _token: str | None = Depends(verify_token),
) -> dict[str, str]:
    """Efface la mémoire d'une session.

    Args:
        request: FastAPI Request (pour rate limiting)
        session_id: ID de la session à effacer
        _token: Bearer token (validated by dependency)

    Returns:
        Message de confirmation
    """
    _ = request
    if _session_store is not None and await _session_store.delete_session(session_id):
        log.info("api.session_cleared", session_id=session_id)
        return {"message": f"Session {session_id} cleared"}

    # Return 404 consistently (don't reveal which sessions exist for security)
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Session not found",
    )


# === Prometheus Metrics Exposure ===
# Instrument l'app
instrumentator.instrument(app)


@app.get("/metrics", tags=["System"], include_in_schema=True)
async def metrics(_token: str | None = Depends(verify_token)) -> Response:
    """Expose les métriques Prometheus."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# === Entry point ===

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        workers=settings.api_workers,
    )
