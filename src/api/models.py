"""Modèles Pydantic pour l'API FastAPI GRI RAG.

Ces modèles définissent les schemas de request/response pour tous
les endpoints de l'API.

Usage:
    from src.api.models import QueryRequest, QueryResponse
"""

import re
from datetime import datetime
from enum import Enum
from typing import Annotated, Any

from pydantic import AfterValidator, BaseModel, Field

# UUID v4 pattern for session_id validation
UUID_V4_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def validate_session_id(v: str | None) -> str | None:
    """Validate session_id is a valid UUID v4 format."""
    if v is None:
        return None
    if not UUID_V4_PATTERN.match(v):
        raise ValueError("session_id must be a valid UUID v4")
    return v.lower()  # Normalize to lowercase


SessionId = Annotated[str | None, AfterValidator(validate_session_id)]


class CycleType(str, Enum):
    """Type de cycle GRI ou CIR."""

    GRI = "GRI"
    CIR = "CIR"
    AUTO = "AUTO"


class IntentType(str, Enum):
    """Intent détecté par le query router."""

    DEFINITION = "DEFINITION"
    PROCESSUS = "PROCESSUS"
    JALON = "JALON"
    PHASE_COMPLETE = "PHASE_COMPLETE"
    COMPARAISON = "COMPARAISON"
    CIR = "CIR"
    UNKNOWN = "UNKNOWN"


# === Request Models ===


class QueryRequest(BaseModel):
    """Requête de question au système RAG.

    Attributes:
        query: Question utilisateur (obligatoire)
        cycle: Type de cycle à cibler (AUTO détecte automatiquement)
        include_sources: Inclure les chunks sources dans la réponse
        max_chunks: Nombre maximum de chunks à retourner
        session_id: ID de session pour la mémoire conversationnelle
    """

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Question utilisateur",
        examples=["Quels sont les critères du CDR ?"],
    )
    cycle: CycleType = Field(
        default=CycleType.AUTO,
        description="Cycle cible : GRI (7 phases), CIR (4 phases), ou AUTO",
    )
    include_sources: bool = Field(
        default=True,
        description="Inclure les chunks sources dans la réponse",
    )
    max_chunks: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Nombre maximum de chunks à retourner",
    )
    session_id: SessionId = Field(
        default=None,
        description="ID de session pour la mémoire conversationnelle (UUID v4)",
    )


class FeedbackRequest(BaseModel):
    """Feedback utilisateur sur une réponse.

    Attributes:
        query_id: ID de la query originale
        rating: Note de 1 à 5
        comment: Commentaire optionnel
        incorrect_info: Information incorrecte signalée
    """

    query_id: str = Field(..., description="ID de la query originale")
    rating: int = Field(..., ge=1, le=5, description="Note de 1 à 5")
    comment: str | None = Field(default=None, max_length=1000)
    incorrect_info: str | None = Field(
        default=None,
        description="Information incorrecte signalée par l'utilisateur",
    )


# === Response Models ===


class Citation(BaseModel):
    """Une citation vers une source GRI.

    Attributes:
        text: Texte de la citation (ex: "[GRI > Phase 3 > ...]")
        section: Section référencée
        chunk_id: ID du chunk source (optionnel)
    """

    text: str = Field(..., description="Citation formatée")
    section: str = Field(..., description="Section GRI/CIR référencée")
    chunk_id: str | None = Field(default=None)


class Source(BaseModel):
    """Un chunk source utilisé pour la réponse.

    Attributes:
        chunk_id: Identifiant unique du chunk
        content: Contenu textuel du chunk
        score: Score de pertinence (0-1)
        section_type: Type de section (definition, milestone, etc.)
        metadata: Métadonnées complètes du chunk
    """

    chunk_id: str = Field(..., description="ID unique du chunk")
    content: str = Field(..., description="Contenu du chunk")
    score: float = Field(..., ge=0.0, le=1.0, description="Score de pertinence")
    section_type: str = Field(..., description="Type de section")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Métadonnées du chunk",
    )


class ToolCallInfo(BaseModel):
    """Information sur un appel de tool.

    Attributes:
        tool: Nom du tool appelé
        input: Paramètres d'entrée
        iteration: Numéro d'itération de la boucle ReAct
        success: Si l'appel a réussi
    """

    tool: str
    input: dict[str, Any] = Field(default_factory=dict)
    iteration: int
    success: bool


class QueryResponse(BaseModel):
    """Réponse complète à une question.

    Attributes:
        query_id: ID unique de cette query
        answer: Réponse générée
        intent: Intent détecté
        cycle: Cycle identifié (GRI/CIR)
        citations: Liste des citations
        sources: Chunks sources (si include_sources=True)
        tool_calls: Historique des appels de tools
        latency_ms: Temps de réponse en ms
        iterations: Nombre d'itérations ReAct
        warning: Avertissement éventuel
        created_at: Timestamp de création
    """

    query_id: str = Field(..., description="ID unique de la query")
    answer: str = Field(..., description="Réponse générée")
    intent: str = Field(..., description="Intent détecté")
    cycle: str = Field(..., description="Cycle GRI ou CIR")
    citations: list[Citation] = Field(default_factory=list)
    sources: list[Source] | None = Field(default=None)
    tool_calls: list[ToolCallInfo] = Field(default_factory=list)
    latency_ms: float = Field(..., description="Latence en millisecondes")
    iterations: int = Field(..., description="Nombre d'itérations ReAct")
    warning: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Réponse du healthcheck.

    Attributes:
        status: État du service (healthy/degraded/unhealthy)
        version: Version de l'API
        qdrant_connected: Connexion Qdrant OK
        collections: État des collections
        timestamp: Timestamp
    """

    status: str = Field(..., description="État du service")
    version: str = Field(default="0.1.0")
    qdrant_connected: bool = Field(default=False)
    collections: dict[str, int] = Field(
        default_factory=dict,
        description="Nombre de documents par collection",
    )
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class StatsResponse(BaseModel):
    """Statistiques du système.

    Attributes:
        total_documents: Nombre total de documents indexés
        collections: Stats par collection
        avg_latency_ms: Latence moyenne
        queries_today: Nombre de queries aujourd'hui
        uptime_seconds: Temps de fonctionnement
    """

    total_documents: int = Field(default=0)
    collections: dict[str, dict[str, Any]] = Field(default_factory=dict)
    avg_latency_ms: float | None = Field(default=None)
    queries_today: int = Field(default=0)
    uptime_seconds: float = Field(default=0.0)


class FeedbackResponse(BaseModel):
    """Réponse après soumission de feedback."""

    success: bool
    message: str


class ErrorResponse(BaseModel):
    """Réponse d'erreur standardisée.

    Attributes:
        error: Type d'erreur
        message: Message d'erreur
        detail: Détails supplémentaires
        request_id: ID de la requête
    """

    error: str = Field(..., description="Type d'erreur")
    message: str = Field(..., description="Message d'erreur")
    detail: str | None = Field(default=None)
    request_id: str | None = Field(default=None)


# === SSE Event Models ===


class SSEEventType(str, Enum):
    """Types d'événements SSE."""

    ROUTING = "routing"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    CHUNK = "chunk"
    DONE = "done"
    ERROR = "error"


class SSERoutingEvent(BaseModel):
    """Événement SSE : routing détecté."""

    event: str = "routing"
    intent: str
    cycle: str
    confidence: float


class SSEToolCallEvent(BaseModel):
    """Événement SSE : appel de tool."""

    event: str = "tool_call"
    tool: str
    input: dict[str, Any]
    iteration: int


class SSEToolResultEvent(BaseModel):
    """Événement SSE : résultat de tool."""

    event: str = "tool_result"
    tool: str
    success: bool
    n_results: int


class SSEChunkEvent(BaseModel):
    """Événement SSE : chunk de réponse."""

    event: str = "chunk"
    text: str


class SSEDoneEvent(BaseModel):
    """Événement SSE : réponse terminée.

    Inclut les mêmes métadonnées que QueryResponse pour parité.
    """

    event: str = "done"
    query_id: str
    answer: str
    intent: str
    cycle: str
    citations: list[str]
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[dict[str, Any]] | None = None
    latency_ms: float
    iterations: int


class SSEErrorEvent(BaseModel):
    """Événement SSE : erreur."""

    event: str = "error"
    error: str
    message: str
