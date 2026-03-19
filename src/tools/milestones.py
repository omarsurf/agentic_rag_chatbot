"""Tool get_milestone_criteria - Critères complets d'un jalon GRI/CIR.

Ce tool retourne la checklist COMPLÈTE des critères de passage d'un jalon.
Les critères ne sont jamais fragmentés ou tronqués.

Pour les jalons CIR (J1-J6), le mapping vers les jalons GRI équivalents
est automatiquement inclus.

Usage:
    from src.tools.milestones import get_milestone_criteria

    result = await get_milestone_criteria(
        milestone_id="M4",  # ou "CDR"
        store=store,
    )
"""

from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, Field

from src.core.config import CIR_GRI_MAPPING, VALID_MILESTONES
from src.core.milestone_retriever import (
    MILESTONE_NAMES,
    GRIMilestoneRetriever,
    MilestoneResult,
)

if TYPE_CHECKING:
    from src.core.vector_store import GRIHybridStore

log = structlog.get_logger()


# Mapping des acronymes vers les IDs de jalons (aligné avec milestone_retriever.py)
MILESTONE_ALIASES: dict[str, str] = {
    "ASR": "M0",
    "MNS": "M0",
    "SRR": "M1",
    "SFR": "M2",
    "PDR": "M2",
    "CDR": "M3",
    "IRR": "M4",
    "TRR": "M5",
    "SAR": "M6",
    "ORR": "M7",
    "MNR": "M8",
}


class GetMilestoneInput(BaseModel):
    """Input pour le tool get_milestone_criteria."""

    milestone_id: str = Field(
        ...,
        description="ID du jalon (M0-M9, J1-J6) ou acronyme (CDR, PDR, etc.)",
    )
    include_gri_mapping: bool = Field(
        default=True,
        description="Pour les jalons CIR, inclure les jalons GRI équivalents",
    )


class MilestoneCriterion(BaseModel):
    """Un critère de passage de jalon."""

    number: int
    text: str
    category: str | None = None


class GetMilestoneOutput(BaseModel):
    """Output du tool get_milestone_criteria."""

    found: bool = False
    milestone_id: str
    milestone_name: str | None = None
    cycle: str = "GRI"
    is_cir: bool = False
    criteria: list[MilestoneCriterion] = Field(default_factory=list)
    criteria_count: int = 0
    content: str = ""
    gri_equivalents: list[str] = Field(default_factory=list)
    gri_mapping_info: str | None = None
    citation: str | None = None


def normalize_milestone_id(milestone_id: str) -> str:
    """Normalise un ID de jalon (gère les aliases).

    Args:
        milestone_id: ID ou alias du jalon

    Returns:
        ID normalisé (M0-M9 ou J1-J6)
    """
    normalized = milestone_id.upper().strip()

    # Vérifier les aliases
    if normalized in MILESTONE_ALIASES:
        return MILESTONE_ALIASES[normalized]

    return normalized


async def get_milestone_criteria(
    milestone_id: str,
    store: "GRIHybridStore",
    include_gri_mapping: bool = True,
) -> GetMilestoneOutput:
    """Récupère les critères complets d'un jalon.

    Args:
        milestone_id: ID du jalon (M0-M9, J1-J6, ou alias)
        store: Vector store GRI
        include_gri_mapping: Inclure le mapping GRI pour les jalons CIR

    Returns:
        GetMilestoneOutput avec les critères
    """
    # Normaliser l'ID
    normalized_id = normalize_milestone_id(milestone_id)

    log.info(
        "tool.get_milestone_criteria.start",
        milestone_id=milestone_id,
        normalized_id=normalized_id,
    )

    # Valider l'ID
    if normalized_id not in VALID_MILESTONES:
        log.warning(
            "tool.get_milestone_criteria.invalid_id",
            milestone_id=milestone_id,
            normalized_id=normalized_id,
        )
        return GetMilestoneOutput(
            found=False,
            milestone_id=milestone_id,
            content=f"Jalon '{milestone_id}' non reconnu. "
            f"Jalons valides : M0-M9 (GRI) ou J1-J6 (CIR).",
        )

    # Utiliser le milestone retriever
    retriever = GRIMilestoneRetriever(store)
    result: MilestoneResult = await retriever.get_milestone(
        normalized_id,
        include_gri_mapping=include_gri_mapping,
    )

    if not result.found:
        log.warning(
            "tool.get_milestone_criteria.not_found",
            milestone_id=normalized_id,
        )
        return GetMilestoneOutput(
            found=False,
            milestone_id=normalized_id,
            milestone_name=MILESTONE_NAMES.get(normalized_id),
            cycle=result.cycle,
            is_cir=result.is_cir,
            content=f"Critères du jalon {normalized_id} non trouvés dans l'index.",
        )

    # Extraire le contenu et les critères
    content_parts = []
    criteria: list[MilestoneCriterion] = []

    for chunk in result.chunks:
        content_parts.append(chunk.content)
        # Extraire les critères numérotés du contenu
        chunk_criteria = _extract_criteria_from_text(chunk.content)
        criteria.extend(chunk_criteria)

    # Construire le contenu principal
    main_content = "\n\n".join(content_parts)

    # Ajouter info mapping GRI si CIR
    gri_mapping_info = None
    if result.is_cir and result.gri_equivalents:
        gri_mapping_info = (
            f"Équivalents GRI : {', '.join(result.gri_equivalents)}\n"
            f"Mapping : {normalized_id} → {' + '.join(result.gri_equivalents)}"
        )

        # Ajouter le contenu des équivalents GRI
        if result.gri_chunks:
            main_content += "\n\n---\n**Critères GRI équivalents :**\n"
            for chunk in result.gri_chunks:
                main_content += f"\n{chunk.content}"

    # Construire la citation
    citation = _build_milestone_citation(normalized_id, result.cycle)

    log.info(
        "tool.get_milestone_criteria.found",
        milestone_id=normalized_id,
        n_chunks=len(result.chunks),
        n_criteria=len(criteria),
        is_cir=result.is_cir,
    )

    return GetMilestoneOutput(
        found=True,
        milestone_id=normalized_id,
        milestone_name=MILESTONE_NAMES.get(normalized_id),
        cycle=result.cycle,
        is_cir=result.is_cir,
        criteria=criteria,
        criteria_count=len(criteria),
        content=main_content,
        gri_equivalents=result.gri_equivalents,
        gri_mapping_info=gri_mapping_info,
        citation=citation,
    )


def _extract_criteria_from_text(text: str) -> list[MilestoneCriterion]:
    """Extrait les critères numérotés d'un texte.

    Args:
        text: Texte contenant les critères

    Returns:
        Liste de critères extraits
    """
    import re

    criteria = []

    # Pattern pour les critères numérotés (1., 2., etc.)
    pattern = r"^(\d+)\.\s*(.+?)(?=\n\d+\.|$)"
    matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)

    for num, content in matches:
        criteria.append(
            MilestoneCriterion(
                number=int(num),
                text=content.strip(),
            )
        )

    # Pattern alternatif pour les tirets ou puces
    if not criteria:
        pattern = r"^[-•]\s*(.+)$"
        matches = re.findall(pattern, text, re.MULTILINE)
        for i, content in enumerate(matches, 1):
            criteria.append(
                MilestoneCriterion(
                    number=i,
                    text=content.strip(),
                )
            )

    return criteria


def _build_milestone_citation(milestone_id: str, cycle: str) -> str:
    """Construit la citation pour un jalon.

    Args:
        milestone_id: ID du jalon
        cycle: Cycle (GRI ou CIR)

    Returns:
        Citation formatée
    """
    milestone_name = MILESTONE_NAMES.get(milestone_id, milestone_id)

    if cycle == "CIR":
        gri_equiv = CIR_GRI_MAPPING.get(milestone_id, [])
        if gri_equiv:
            equiv_str = " + ".join(gri_equiv)
            return f"[CIR > Jalon {milestone_id} ({milestone_name}) — Équivalent GRI : {equiv_str}]"
        return f"[CIR > Jalon {milestone_id} ({milestone_name})]"

    return f"[GRI > Jalon {milestone_id} ({milestone_name})]"


def format_milestone_for_response(output: GetMilestoneOutput) -> str:
    """Formate les critères de jalon pour inclusion dans une réponse.

    Args:
        output: Résultat du tool

    Returns:
        Texte formaté pour la réponse
    """
    if not output.found:
        return output.content

    lines = []

    # Titre
    title = f"## Jalon {output.milestone_id}"
    if output.milestone_name:
        title += f" — {output.milestone_name}"
    lines.append(title)

    # Cycle
    lines.append(f"\n**Cycle :** {output.cycle}")

    # Mapping GRI si CIR
    if output.gri_mapping_info:
        lines.append(f"\n**{output.gri_mapping_info}**")

    # Critères
    if output.criteria:
        lines.append("\n### Critères de passage\n")
        for c in output.criteria:
            lines.append(f"{c.number}. {c.text}")
    else:
        # Contenu brut si pas de critères structurés
        lines.append("\n### Contenu\n")
        lines.append(output.content)

    # Citation
    if output.citation:
        lines.append(f"\n{output.citation}")

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
    validated = GetMilestoneInput(**input_data)

    # Exécuter
    result = await get_milestone_criteria(
        milestone_id=validated.milestone_id,
        store=store,
        include_gri_mapping=validated.include_gri_mapping,
    )

    return result.model_dump()
