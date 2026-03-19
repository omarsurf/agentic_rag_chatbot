"""Tool retrieve_gri_chunks - Recherche hybride dans la base GRI.

Ce tool effectue une recherche hybride (dense + BM25) dans l'index GRI
avec possibilité de filtrer par type de section, cycle, et phase.

Usage:
    from src.tools.retrieve_gri import retrieve_gri_chunks

    result = await retrieve_gri_chunks(
        query="processus de vérification",
        section_type="process",
        n_results=5,
        store=store,
    )
"""

from typing import TYPE_CHECKING, Any, Literal

import structlog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.core.vector_store import GRIHybridStore

log = structlog.get_logger()


class RetrieveGRIInput(BaseModel):
    """Input pour le tool retrieve_gri_chunks."""

    query: str = Field(..., description="La question ou le concept à rechercher")
    section_type: Literal[
        "definition",
        "principle",
        "phase",
        "milestone",
        "process",
        "cir",
        "table",
        "content",
    ] | None = Field(default=None, description="Filtrer par type de section GRI")
    cycle: Literal["GRI", "CIR", "BOTH"] = Field(
        default="GRI", description="Filtrer par cycle"
    )
    phase_num: int | None = Field(
        default=None, ge=1, le=7, description="Filtrer par numéro de phase (1-7)"
    )
    n_results: int = Field(
        default=5, ge=1, le=15, description="Nombre de résultats à retourner"
    )


class RetrieveChunk(BaseModel):
    """Un chunk récupéré."""

    id: str
    content: str
    score: float
    section_type: str | None = None
    cycle: str | None = None
    milestone_id: str | None = None
    phase_num: int | None = None
    context_prefix: str | None = None


class RetrieveGRIOutput(BaseModel):
    """Output du tool retrieve_gri_chunks."""

    chunks: list[RetrieveChunk] = Field(default_factory=list)
    query: str
    n_results: int
    filters_applied: dict[str, Any] = Field(default_factory=dict)
    has_results: bool = False
    avg_score: float = 0.0
    max_score: float = 0.0


async def retrieve_gri_chunks(
    query: str,
    store: "GRIHybridStore",
    section_type: str | None = None,
    cycle: str = "GRI",
    phase_num: int | None = None,
    n_results: int = 5,
) -> RetrieveGRIOutput:
    """Recherche des chunks dans la base GRI.

    Args:
        query: Query de recherche
        store: Vector store GRI
        section_type: Filtre par type de section
        cycle: Filtre par cycle (GRI/CIR/BOTH)
        phase_num: Filtre par numéro de phase
        n_results: Nombre de résultats

    Returns:
        RetrieveGRIOutput avec les chunks trouvés
    """
    log.info(
        "tool.retrieve_gri_chunks.start",
        query=query[:80],
        section_type=section_type,
        cycle=cycle,
        phase_num=phase_num,
        n_results=n_results,
    )

    # Construire les filtres
    filters: dict[str, Any] = {}

    if section_type:
        filters["section_type"] = section_type

    if cycle and cycle != "BOTH":
        filters["cycle"] = cycle

    if phase_num:
        filters["phase_num"] = phase_num

    # Déterminer l'alpha RRF selon le type de contenu
    alpha = 0.6  # Défaut hybride
    if section_type == "definition":
        alpha = 0.3  # Favoriser sparse pour exact match
    elif section_type == "process":
        alpha = 0.6  # Hybride standard

    # Recherche hybride
    try:
        results = await store.hybrid_search(
            query=query,
            collection="main",
            n_results=n_results,
            filters=filters if filters else None,
            alpha=alpha,
        )
    except Exception as e:
        log.error("tool.retrieve_gri_chunks.search_failed", error=str(e))
        return RetrieveGRIOutput(
            chunks=[],
            query=query,
            n_results=n_results,
            filters_applied=filters,
            has_results=False,
        )

    # Convertir les résultats
    chunks = [
        RetrieveChunk(
            id=r.id,
            content=r.content,
            score=r.score,
            section_type=r.section_type,
            cycle=r.cycle,
            milestone_id=r.milestone_id,
            phase_num=r.phase_num,
            context_prefix=r.context_prefix,
        )
        for r in results
    ]

    # Calculer les stats
    scores = [c.score for c in chunks]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    max_score = max(scores) if scores else 0.0

    log.info(
        "tool.retrieve_gri_chunks.done",
        n_chunks=len(chunks),
        avg_score=f"{avg_score:.3f}",
        max_score=f"{max_score:.3f}",
    )

    return RetrieveGRIOutput(
        chunks=chunks,
        query=query,
        n_results=len(chunks),
        filters_applied=filters,
        has_results=len(chunks) > 0,
        avg_score=avg_score,
        max_score=max_score,
    )


async def execute(
    input_data: dict[str, Any],
    store: "GRIHybridStore",
) -> dict[str, Any]:
    """Point d'entrée pour l'executor.

    Args:
        input_data: Paramètres du tool
        store: Vector store

    Returns:
        Résultat sous forme de dict
    """
    # Valider l'input
    validated = RetrieveGRIInput(**input_data)

    # Exécuter
    result = await retrieve_gri_chunks(
        query=validated.query,
        store=store,
        section_type=validated.section_type,
        cycle=validated.cycle,
        phase_num=validated.phase_num,
        n_results=validated.n_results,
    )

    return result.model_dump()
