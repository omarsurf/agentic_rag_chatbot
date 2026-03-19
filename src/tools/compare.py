"""Tool compare_approaches - Comparaison de deux éléments GRI.

Ce tool effectue un retrieval parallèle sur deux entités du GRI
et structure les résultats pour faciliter la comparaison.

Usage:
    from src.tools.compare import compare_approaches

    result = await compare_approaches(
        entity_a="GRI standard",
        entity_b="CIR",
        store=store,
    )
"""

import asyncio
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.core.vector_store import GRIHybridStore

log = structlog.get_logger()


class CompareApproachesInput(BaseModel):
    """Input pour le tool compare_approaches."""

    entity_a: str = Field(..., description="Premier élément à comparer")
    entity_b: str = Field(..., description="Deuxième élément à comparer")
    comparison_dimensions: list[str] = Field(
        default_factory=list,
        description="Dimensions de comparaison (optionnel)",
    )


class EntityInfo(BaseModel):
    """Informations sur une entité."""

    name: str
    chunks: list[dict[str, Any]] = Field(default_factory=list)
    n_chunks: int = 0
    avg_score: float = 0.0
    cycle: str | None = None
    phase_num: int | None = None
    summary: str = ""


class CompareApproachesOutput(BaseModel):
    """Output du tool compare_approaches."""

    entity_a: EntityInfo
    entity_b: EntityInfo
    comparison_dimensions: list[str] = Field(default_factory=list)
    combined_context: str = ""
    has_sufficient_data: bool = False
    citations: list[str] = Field(default_factory=list)


async def compare_approaches(
    entity_a: str,
    entity_b: str,
    store: "GRIHybridStore",
    comparison_dimensions: list[str] | None = None,
    n_results_per_entity: int = 5,
) -> CompareApproachesOutput:
    """Compare deux entités du GRI.

    Effectue un retrieval parallèle sur les deux entités et
    structure les résultats pour faciliter la comparaison.

    Args:
        entity_a: Première entité
        entity_b: Deuxième entité
        store: Vector store GRI
        comparison_dimensions: Dimensions de comparaison
        n_results_per_entity: Résultats par entité

    Returns:
        CompareApproachesOutput avec le contexte de comparaison
    """
    log.info(
        "tool.compare_approaches.start",
        entity_a=entity_a,
        entity_b=entity_b,
    )

    comparison_dimensions = comparison_dimensions or []

    # Retrieval parallèle sur les deux entités
    results_a, results_b = await asyncio.gather(
        _retrieve_entity_info(entity_a, store, n_results_per_entity),
        _retrieve_entity_info(entity_b, store, n_results_per_entity),
    )

    # Construire le contexte combiné
    combined_context = _build_combined_context(results_a, results_b, comparison_dimensions)

    # Extraire les citations
    citations = _extract_citations(results_a, results_b)

    # Vérifier si on a assez de données
    has_sufficient_data = results_a.n_chunks > 0 and results_b.n_chunks > 0

    log.info(
        "tool.compare_approaches.done",
        n_chunks_a=results_a.n_chunks,
        n_chunks_b=results_b.n_chunks,
        has_sufficient_data=has_sufficient_data,
    )

    return CompareApproachesOutput(
        entity_a=results_a,
        entity_b=results_b,
        comparison_dimensions=comparison_dimensions,
        combined_context=combined_context,
        has_sufficient_data=has_sufficient_data,
        citations=citations,
    )


async def _retrieve_entity_info(
    entity: str,
    store: "GRIHybridStore",
    n_results: int,
) -> EntityInfo:
    """Récupère les informations sur une entité.

    Args:
        entity: Nom de l'entité
        store: Vector store
        n_results: Nombre de résultats

    Returns:
        EntityInfo avec les chunks
    """
    # Déterminer le cycle si mentionné explicitement
    cycle = None
    entity_lower = entity.lower()
    if "cir" in entity_lower:
        cycle = "CIR"
    elif "gri" in entity_lower:
        cycle = "GRI"

    # Déterminer si c'est une phase
    phase_num = None
    import re

    phase_match = re.search(r"phase\s+(\d+)", entity_lower)
    if phase_match:
        phase_num = int(phase_match.group(1))

    # Construire les filtres
    filters: dict[str, Any] = {}
    if cycle and cycle != "BOTH":
        filters["cycle"] = cycle
    if phase_num:
        filters["phase_num"] = phase_num

    # Recherche
    try:
        results = await store.hybrid_search(
            query=entity,
            collection="main",
            n_results=n_results,
            filters=filters if filters else None,
            alpha=0.6,
        )

        chunks = [
            {
                "content": r.content,
                "score": r.score,
                "section_type": r.section_type,
                "cycle": r.cycle,
                "phase_num": r.phase_num,
                "context_prefix": r.context_prefix,
            }
            for r in results
        ]

        scores = [c["score"] for c in chunks]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # Construire un résumé
        summary = _build_entity_summary(entity, chunks)

        return EntityInfo(
            name=entity,
            chunks=chunks,
            n_chunks=len(chunks),
            avg_score=avg_score,
            cycle=cycle,
            phase_num=phase_num,
            summary=summary,
        )

    except Exception as e:
        log.error(
            "tool.compare_approaches.retrieve_failed",
            entity=entity,
            error=str(e),
        )
        return EntityInfo(
            name=entity,
            chunks=[],
            n_chunks=0,
            summary=f"Erreur lors de la récupération : {str(e)}",
        )


def _build_entity_summary(entity: str, chunks: list[dict]) -> str:
    """Construit un résumé des chunks d'une entité.

    Args:
        entity: Nom de l'entité
        chunks: Chunks récupérés

    Returns:
        Résumé textuel
    """
    if not chunks:
        return f"Aucune information trouvée pour '{entity}'."

    # Prendre le premier chunk comme base du résumé
    first_chunk = chunks[0]
    content_preview = first_chunk["content"][:500]
    if len(first_chunk["content"]) > 500:
        content_preview += "..."

    return content_preview


def _build_combined_context(
    entity_a: EntityInfo,
    entity_b: EntityInfo,
    dimensions: list[str],
) -> str:
    """Construit le contexte combiné pour la comparaison.

    Args:
        entity_a: Info entité A
        entity_b: Info entité B
        dimensions: Dimensions de comparaison

    Returns:
        Contexte formaté
    """
    lines = []

    # Section entité A
    lines.append(f"## {entity_a.name}")
    lines.append("")

    if entity_a.cycle:
        lines.append(f"**Cycle :** {entity_a.cycle}")
    if entity_a.phase_num:
        lines.append(f"**Phase :** {entity_a.phase_num}")

    lines.append("")

    for i, chunk in enumerate(entity_a.chunks, 1):
        prefix = chunk.get("context_prefix", "")
        content = chunk.get("content", "")
        score = chunk.get("score", 0)

        lines.append(f"### Source {i} (score: {score:.2f})")
        if prefix:
            lines.append(f"*{prefix}*")
        lines.append(content)
        lines.append("")

    # Section entité B
    lines.append("---")
    lines.append(f"## {entity_b.name}")
    lines.append("")

    if entity_b.cycle:
        lines.append(f"**Cycle :** {entity_b.cycle}")
    if entity_b.phase_num:
        lines.append(f"**Phase :** {entity_b.phase_num}")

    lines.append("")

    for i, chunk in enumerate(entity_b.chunks, 1):
        prefix = chunk.get("context_prefix", "")
        content = chunk.get("content", "")
        score = chunk.get("score", 0)

        lines.append(f"### Source {i} (score: {score:.2f})")
        if prefix:
            lines.append(f"*{prefix}*")
        lines.append(content)
        lines.append("")

    # Dimensions de comparaison
    if dimensions:
        lines.append("---")
        lines.append("## Dimensions de comparaison demandées")
        for dim in dimensions:
            lines.append(f"- {dim}")

    return "\n".join(lines)


def _extract_citations(entity_a: EntityInfo, entity_b: EntityInfo) -> list[str]:
    """Extrait les citations des deux entités.

    Args:
        entity_a: Info entité A
        entity_b: Info entité B

    Returns:
        Liste de citations
    """
    citations = []

    for chunk in entity_a.chunks:
        prefix = chunk.get("context_prefix")
        if prefix:
            citations.append(prefix)

    for chunk in entity_b.chunks:
        prefix = chunk.get("context_prefix")
        if prefix:
            citations.append(prefix)

    # Dédupliquer
    return list(set(citations))


def format_comparison_for_response(output: CompareApproachesOutput) -> str:
    """Formate la comparaison pour inclusion dans une réponse.

    Args:
        output: Résultat du tool

    Returns:
        Texte formaté suggérant une structure de réponse
    """
    if not output.has_sufficient_data:
        missing = []
        if output.entity_a.n_chunks == 0:
            missing.append(output.entity_a.name)
        if output.entity_b.n_chunks == 0:
            missing.append(output.entity_b.name)

        return (
            f"Données insuffisantes pour la comparaison. "
            f"Informations manquantes pour : {', '.join(missing)}"
        )

    lines = [
        f"## Comparaison : {output.entity_a.name} vs {output.entity_b.name}",
        "",
        "### Vue d'ensemble",
        f"- **{output.entity_a.name}** : {output.entity_a.n_chunks} sources, "
        f"score moyen {output.entity_a.avg_score:.2f}",
        f"- **{output.entity_b.name}** : {output.entity_b.n_chunks} sources, "
        f"score moyen {output.entity_b.avg_score:.2f}",
        "",
    ]

    if output.comparison_dimensions:
        lines.append("### Dimensions de comparaison")
        lines.append("")
        lines.append(f"| Dimension | {output.entity_a.name} | {output.entity_b.name} |")
        lines.append("|-----------|-----------|-----------|")
        for dim in output.comparison_dimensions:
            lines.append(f"| {dim} | ... | ... |")
        lines.append("")

    lines.append("### Contexte détaillé")
    lines.append("")
    lines.append(output.combined_context)

    return "\n".join(lines)


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
    validated = CompareApproachesInput(**input_data)

    # Exécuter
    result = await compare_approaches(
        entity_a=validated.entity_a,
        entity_b=validated.entity_b,
        store=store,
        comparison_dimensions=validated.comparison_dimensions,
    )

    return result.model_dump()
