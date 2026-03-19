"""API FastAPI avec SSE streaming.

Ce module expose l'API REST et SSE pour le système RAG GRI/FAR.

Endpoints:
- POST /query         : Query standard (JSON)
- POST /query/stream  : Query avec SSE streaming
- GET  /health        : Healthcheck
- GET  /stats         : Statistiques
- POST /feedback      : Feedback utilisateur

Usage:
    uvicorn src.api.main:app --reload --port 8000
"""

from src.api.main import app
from src.api.models import (
    Citation,
    CycleType,
    ErrorResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    IntentType,
    QueryRequest,
    QueryResponse,
    Source,
    StatsResponse,
    ToolCallInfo,
)
from src.api.streaming import stream_query_response

__all__ = [
    "app",
    "stream_query_response",
    "QueryRequest",
    "QueryResponse",
    "HealthResponse",
    "StatsResponse",
    "FeedbackRequest",
    "FeedbackResponse",
    "ErrorResponse",
    "Citation",
    "Source",
    "ToolCallInfo",
    "CycleType",
    "IntentType",
]
