"""Tool lookup_gri_glossary - Définitions exactes ISO du glossaire GRI.

Ce tool recherche les définitions normatives des termes ISO/GRI.
Les définitions sont retournées mot pour mot (aucune paraphrase).

Usage:
    from src.tools.glossary import lookup_gri_glossary

    result = await lookup_gri_glossary(
        term="artefact",
        store=store,
    )
"""

from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.core.vector_store import GRIHybridStore

log = structlog.get_logger()


class LookupGlossaryInput(BaseModel):
    """Input pour le tool lookup_gri_glossary."""

    term: str = Field(..., description="Le terme à définir (FR ou EN)")
    return_both_languages: bool = Field(
        default=True, description="Retourner la définition en FR ET EN"
    )


class GlossaryDefinition(BaseModel):
    """Définition d'un terme GRI."""

    term_fr: str
    term_en: str | None = None
    definition_fr: str
    definition_en: str | None = None
    standard_ref: str | None = None
    source: str = "GRI"
    context_prefix: str | None = None


class LookupGlossaryOutput(BaseModel):
    """Output du tool lookup_gri_glossary."""

    found: bool = False
    term_searched: str
    definition: GlossaryDefinition | None = None
    alternatives: list[str] = Field(default_factory=list)
    citation: str | None = None


async def lookup_gri_glossary(
    term: str,
    store: "GRIHybridStore",
    return_both_languages: bool = True,
) -> LookupGlossaryOutput:
    """Recherche une définition dans le glossaire GRI.

    Args:
        term: Terme à rechercher
        store: Vector store GRI
        return_both_languages: Inclure FR et EN

    Returns:
        LookupGlossaryOutput avec la définition trouvée
    """
    log.info("tool.lookup_gri_glossary.start", term=term)

    try:
        # Lookup dans le glossaire
        result = await store.glossary_lookup(term)

        if result is None:
            log.info("tool.lookup_gri_glossary.not_found", term=term)

            # Essayer une recherche alternative
            alternatives = await _find_alternatives(term, store)

            return LookupGlossaryOutput(
                found=False,
                term_searched=term,
                definition=None,
                alternatives=alternatives,
            )

        # Extraire les métadonnées
        metadata = result.metadata

        # Construire la définition
        definition = GlossaryDefinition(
            term_fr=metadata.get("term_fr", term),
            term_en=metadata.get("term_en") if return_both_languages else None,
            definition_fr=metadata.get("definition_fr", result.content),
            definition_en=metadata.get("definition_en") if return_both_languages else None,
            standard_ref=metadata.get("standard_ref"),
            source=metadata.get("source", "GRI"),
            context_prefix=result.context_prefix,
        )

        # Construire la citation
        citation = _build_citation(definition)

        log.info(
            "tool.lookup_gri_glossary.found",
            term=term,
            term_fr=definition.term_fr,
            has_en=definition.term_en is not None,
        )

        return LookupGlossaryOutput(
            found=True,
            term_searched=term,
            definition=definition,
            citation=citation,
        )

    except Exception as e:
        log.error("tool.lookup_gri_glossary.error", term=term, error=str(e))
        return LookupGlossaryOutput(
            found=False,
            term_searched=term,
            definition=None,
        )


async def _find_alternatives(
    term: str,
    store: "GRIHybridStore",
    max_alternatives: int = 3,
) -> list[str]:
    """Recherche des termes alternatifs similaires.

    Args:
        term: Terme recherché
        store: Vector store
        max_alternatives: Nombre max d'alternatives

    Returns:
        Liste de termes alternatifs
    """
    try:
        # Recherche sémantique dans le glossaire
        results = await store.hybrid_search(
            query=term,
            collection="glossary",
            n_results=max_alternatives,
            alpha=0.5,
        )

        alternatives = []
        for r in results:
            term_fr = r.metadata.get("term_fr")
            if term_fr and term_fr.lower() != term.lower():
                alternatives.append(term_fr)

        return alternatives

    except Exception:
        return []


def _build_citation(definition: GlossaryDefinition) -> str:
    """Construit la citation au format GRI.

    Args:
        definition: Définition du terme

    Returns:
        Citation formatée
    """
    parts = ["GRI", "Terminologie", f"'{definition.term_fr}'"]

    if definition.standard_ref:
        parts.append(definition.standard_ref)

    return "[" + " > ".join(parts) + "]"


def format_definition_for_response(output: LookupGlossaryOutput) -> str:
    """Formate la définition pour inclusion dans une réponse.

    Args:
        output: Résultat du lookup

    Returns:
        Texte formaté pour la réponse
    """
    if not output.found or not output.definition:
        if output.alternatives:
            alts = ", ".join(output.alternatives)
            return (
                f"Le terme '{output.term_searched}' n'est pas défini dans le glossaire GRI. "
                f"Termes similaires : {alts}"
            )
        return f"Le terme '{output.term_searched}' n'est pas défini dans le glossaire GRI."

    d = output.definition
    lines = []

    # Titre avec terme bilingue
    if d.term_en:
        lines.append(f"**{d.term_fr}** ({d.term_en})")
    else:
        lines.append(f"**{d.term_fr}**")

    # Définition FR
    lines.append(f"\n{d.definition_fr}")

    # Définition EN si disponible
    if d.definition_en:
        lines.append(f"\n*EN: {d.definition_en}*")

    # Source
    if d.standard_ref:
        lines.append(f"\n[Source : {d.standard_ref}]")

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
    validated = LookupGlossaryInput(**input_data)

    # Exécuter
    result = await lookup_gri_glossary(
        term=validated.term,
        store=store,
        return_both_languages=validated.return_both_languages,
    )

    return result.model_dump()
