"""Tool get_phase_summary - Résumé structuré d'une phase GRI/CIR.

Ce tool utilise le Parent Document Retriever pour récupérer
les objectifs, activités, livrables et jalons d'une phase complète.

Usage:
    from src.tools.phases import get_phase_summary

    result = await get_phase_summary(
        phase_num=3,
        cycle="GRI",
        store=store,
    )
"""

import contextlib
from typing import TYPE_CHECKING, Any, Literal

import structlog
from pydantic import BaseModel, Field

from src.core.config import CIR_PHASES, GRI_PHASES

if TYPE_CHECKING:
    from src.core.vector_store import GRIHybridStore

log = structlog.get_logger()


# Noms des phases GRI
GRI_PHASE_NAMES: dict[int, str] = {
    1: "Idéation / Préparation",
    2: "Faisabilité / Définition préliminaire",
    3: "Conception et Développement",
    4: "Intégration et Vérification",
    5: "Qualification",
    6: "Production et Déploiement",
    7: "Utilisation et Soutien / Retrait",
}

# Noms des phases CIR
CIR_PHASE_NAMES: dict[int, str] = {
    1: "Lancement et Cadrage",
    2: "Conception et Prototypage",
    3: "Intégration et Tests",
    4: "Livraison et Clôture",
}

# Jalons par phase GRI
GRI_PHASE_MILESTONES: dict[int, list[str]] = {
    1: ["M0", "M1"],
    2: ["M2", "M3"],
    3: ["M4"],
    4: ["M5", "M6"],
    5: ["M7"],
    6: ["M8"],
    7: ["M9"],
}

# Jalons par phase CIR
CIR_PHASE_MILESTONES: dict[int, list[str]] = {
    1: ["J1"],
    2: ["J2"],
    3: ["J3", "J4"],
    4: ["J5", "J6"],
}


class GetPhaseSummaryInput(BaseModel):
    """Input pour le tool get_phase_summary."""

    phase_num: int = Field(..., ge=1, le=7, description="Numéro de phase (1-7 GRI, 1-4 CIR)")
    cycle: Literal["GRI", "CIR"] = Field(default="GRI", description="Cycle de vie")
    include_deliverables: bool = Field(default=True, description="Inclure les livrables")
    include_milestone_criteria: bool = Field(
        default=False,
        description="Inclure les critères de jalons (réponse plus longue)",
    )


class PhaseObjective(BaseModel):
    """Objectif d'une phase."""

    type: str  # "general" | "specific"
    text: str


class PhaseDeliverable(BaseModel):
    """Livrable d'une phase."""

    name: str
    description: str | None = None


class GetPhaseSummaryOutput(BaseModel):
    """Output du tool get_phase_summary."""

    found: bool = False
    phase_num: int
    phase_name: str
    cycle: str = "GRI"
    objectives: list[PhaseObjective] = Field(default_factory=list)
    activities: list[str] = Field(default_factory=list)
    deliverables: list[PhaseDeliverable] = Field(default_factory=list)
    entry_milestone: str | None = None
    exit_milestone: str | None = None
    milestones: list[str] = Field(default_factory=list)
    content: str = ""
    citations: list[str] = Field(default_factory=list)


async def get_phase_summary(
    phase_num: int,
    store: "GRIHybridStore",
    cycle: str = "GRI",
    include_deliverables: bool = True,
    include_milestone_criteria: bool = False,
) -> GetPhaseSummaryOutput:
    """Récupère un résumé structuré d'une phase.

    Args:
        phase_num: Numéro de la phase
        store: Vector store GRI
        cycle: Cycle (GRI ou CIR)
        include_deliverables: Inclure les livrables
        include_milestone_criteria: Inclure les critères de jalons

    Returns:
        GetPhaseSummaryOutput avec le résumé de la phase
    """
    _ = include_milestone_criteria
    log.info(
        "tool.get_phase_summary.start",
        phase_num=phase_num,
        cycle=cycle,
    )

    # Valider la phase
    valid_phases = list(GRI_PHASES) if cycle == "GRI" else list(CIR_PHASES)
    if phase_num not in valid_phases:
        max_phase = max(valid_phases)
        return GetPhaseSummaryOutput(
            found=False,
            phase_num=phase_num,
            phase_name="",
            cycle=cycle,
            content=f"Phase {phase_num} invalide pour le cycle {cycle}. "
            f"Phases valides : 1 à {max_phase}.",
        )

    # Récupérer le nom et les jalons
    phase_names = GRI_PHASE_NAMES if cycle == "GRI" else CIR_PHASE_NAMES
    phase_milestones = GRI_PHASE_MILESTONES if cycle == "GRI" else CIR_PHASE_MILESTONES

    phase_name = phase_names.get(phase_num, f"Phase {phase_num}")
    milestones = phase_milestones.get(phase_num, [])

    # Recherche des chunks de la phase
    filters = {
        "section_type": "phase",
        "phase_num": phase_num,
    }
    if cycle != "BOTH":
        filters["cycle"] = cycle

    try:
        results = await store.hybrid_search(
            query=f"Phase {phase_num} {phase_name} objectifs activités livrables",
            collection="main",
            n_results=10,
            filters=filters,
            alpha=0.7,  # Favoriser dense pour la sémantique
        )
    except Exception as e:
        log.error(
            "tool.get_phase_summary.search_failed",
            error=str(e),
        )
        return GetPhaseSummaryOutput(
            found=False,
            phase_num=phase_num,
            phase_name=phase_name,
            cycle=cycle,
            content=f"Erreur lors de la recherche : {str(e)}",
        )

    if not results:
        # Essayer sans le filtre phase_num (peut être manquant dans les metadata)
        with contextlib.suppress(Exception):
            results = await store.hybrid_search(
                query=f"Phase {phase_num} {phase_name} {cycle}",
                collection="main",
                n_results=8,
                filters={"section_type": "phase"} if cycle == "BOTH" else {"cycle": cycle},
                alpha=0.7,
            )

    if not results:
        return GetPhaseSummaryOutput(
            found=False,
            phase_num=phase_num,
            phase_name=phase_name,
            cycle=cycle,
            milestones=milestones,
            content=f"Aucune information trouvée pour la Phase {phase_num} ({cycle}).",
        )

    # Extraire les informations structurées
    objectives = _extract_objectives(results)
    activities = _extract_activities(results)
    deliverables = _extract_deliverables(results) if include_deliverables else []

    # Construire le contenu complet
    content = _build_phase_content(results)

    # Extraire les citations
    citations = [r.context_prefix for r in results if r.context_prefix]
    citations = list(set(citations))

    # Jalons d'entrée et sortie
    entry_milestone = milestones[0] if milestones else None
    exit_milestone = milestones[-1] if len(milestones) > 1 else entry_milestone

    log.info(
        "tool.get_phase_summary.done",
        phase_num=phase_num,
        n_chunks=len(results),
        n_objectives=len(objectives),
    )

    return GetPhaseSummaryOutput(
        found=True,
        phase_num=phase_num,
        phase_name=phase_name,
        cycle=cycle,
        objectives=objectives,
        activities=activities,
        deliverables=deliverables,
        entry_milestone=entry_milestone,
        exit_milestone=exit_milestone,
        milestones=milestones,
        content=content,
        citations=citations,
    )


def _extract_objectives(results: list) -> list[PhaseObjective]:
    """Extrait les objectifs des chunks.

    Args:
        results: Résultats de recherche

    Returns:
        Liste d'objectifs
    """
    import re

    objectives = []
    seen = set()

    for r in results:
        content = r.content

        # Chercher les objectifs généraux
        general_match = re.search(
            r"[Oo]bjectif(?:s)?\s+g[ée]n[ée]ra(?:l|ux)[^:]*:\s*(.+?)(?=\n\n|\n[A-Z]|$)",
            content,
            re.DOTALL,
        )
        if general_match:
            text = general_match.group(1).strip()
            if text and text not in seen:
                seen.add(text)
                objectives.append(PhaseObjective(type="general", text=text))

        # Chercher les objectifs spécifiques
        specific_pattern = r"[Oo]bjectif(?:s)?\s+sp[ée]cifiques?[^:]*:\s*(.+?)(?=\n\n|\n[A-Z]|$)"
        specific_match = re.search(specific_pattern, content, re.DOTALL)
        if specific_match:
            text = specific_match.group(1).strip()
            # Séparer si plusieurs objectifs numérotés
            items = re.split(r"\n\d+\.\s*", text)
            for item in items:
                item = item.strip()
                if item and item not in seen:
                    seen.add(item)
                    objectives.append(PhaseObjective(type="specific", text=item))

    return objectives


def _extract_activities(results: list) -> list[str]:
    """Extrait les activités des chunks.

    Args:
        results: Résultats de recherche

    Returns:
        Liste d'activités
    """
    import re

    activities = []
    seen = set()

    for r in results:
        content = r.content

        # Chercher les activités
        activity_match = re.search(
            r"[Aa]ctivit[ée]s?(?:\s+principales?)?[^:]*:\s*(.+?)(?=\n\n|\n[A-Z]|$)",
            content,
            re.DOTALL,
        )
        if activity_match:
            text = activity_match.group(1).strip()
            items = re.split(r"\n[-•]\s*", text)
            for item in items:
                item = item.strip()
                if item and item not in seen and len(item) > 10:
                    seen.add(item)
                    activities.append(item)

    return activities[:10]  # Max 10 activités


def _extract_deliverables(results: list) -> list[PhaseDeliverable]:
    """Extrait les livrables des chunks.

    Args:
        results: Résultats de recherche

    Returns:
        Liste de livrables
    """
    import re

    deliverables = []
    seen = set()

    for r in results:
        content = r.content

        # Chercher les livrables / produits
        deliverable_match = re.search(
            r"(?:[Ll]ivrables?|[Pp]roduits?(?:\s+de\s+la\s+phase)?)[^:]*:\s*(.+?)(?=\n\n|\n[A-Z]|$)",
            content,
            re.DOTALL,
        )
        if deliverable_match:
            text = deliverable_match.group(1).strip()
            items = re.split(r"\n[-•]\s*", text)
            for item in items:
                item = item.strip()
                if item and item not in seen and len(item) > 5:
                    seen.add(item)
                    deliverables.append(PhaseDeliverable(name=item))

    return deliverables[:15]  # Max 15 livrables


def _build_phase_content(results: list) -> str:
    """Construit le contenu complet de la phase.

    Args:
        results: Résultats de recherche

    Returns:
        Contenu formaté
    """
    lines = []

    for i, r in enumerate(results, 1):
        lines.append(f"### Source {i}")
        if r.context_prefix:
            lines.append(f"*{r.context_prefix}*")
        lines.append("")
        lines.append(r.content)
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def format_phase_for_response(output: GetPhaseSummaryOutput) -> str:
    """Formate le résumé de phase pour inclusion dans une réponse.

    Args:
        output: Résultat du tool

    Returns:
        Texte formaté
    """
    if not output.found:
        return output.content

    lines = []

    # Titre
    lines.append(f"## {output.cycle} Phase {output.phase_num} — {output.phase_name}")
    lines.append("")

    # Jalons
    if output.milestones:
        milestones_str = ", ".join(output.milestones)
        lines.append(f"**Jalons :** {milestones_str}")
        if output.entry_milestone and output.exit_milestone:
            lines.append(
                f"- Entrée : {output.entry_milestone}"
            )
            lines.append(
                f"- Sortie : {output.exit_milestone}"
            )
        lines.append("")

    # Objectifs
    if output.objectives:
        lines.append("### Objectifs")
        lines.append("")
        general = [o for o in output.objectives if o.type == "general"]
        specific = [o for o in output.objectives if o.type == "specific"]

        if general:
            lines.append("**Objectif général :**")
            for o in general:
                lines.append(f"- {o.text}")
            lines.append("")

        if specific:
            lines.append("**Objectifs spécifiques :**")
            for i, o in enumerate(specific, 1):
                lines.append(f"{i}. {o.text}")
            lines.append("")

    # Activités
    if output.activities:
        lines.append("### Activités clés")
        lines.append("")
        for a in output.activities:
            lines.append(f"- {a}")
        lines.append("")

    # Livrables
    if output.deliverables:
        lines.append("### Livrables")
        lines.append("")
        for d in output.deliverables:
            lines.append(f"- {d.name}")
        lines.append("")

    # Citations
    if output.citations:
        lines.append("### Sources")
        for c in output.citations[:3]:  # Max 3 citations
            lines.append(f"- {c}")

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
    validated = GetPhaseSummaryInput(**input_data)

    # Exécuter
    result = await get_phase_summary(
        phase_num=validated.phase_num,
        store=store,
        cycle=validated.cycle,
        include_deliverables=validated.include_deliverables,
        include_milestone_criteria=validated.include_milestone_criteria,
    )

    return result.model_dump()
