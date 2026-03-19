"""Term Expander - Enrichissement pré-retrieval avec termes GRI.

Avant la retrieval principale, détecte et injecte les définitions GRI
connues dans le contexte LLM. Cela améliore la compréhension des termes
ISO/GRI spécifiques.

Usage:
    from src.core.term_expander import GRITermExpander, expand_query_with_terms

    expander = GRITermExpander(store)
    query, term_context = await expander.expand("Quels sont les critères du SEMP ?")
    # term_context contient la définition de SEMP
"""

import re
from typing import TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from src.core.vector_store import GRIHybridStore

log = structlog.get_logger()


# Termes GRI détectables dans les queries (subset des 200+)
GRI_KEY_TERMS: list[str] = [
    # Acronymes courants
    "artefact",
    "artifact",
    "CONOPS",
    "SEMP",
    "SRR",
    "PDR",
    "CDR",
    "IRR",
    "TRR",
    "SAR",
    "ORR",
    "MNR",
    "CIR",
    "TRL",
    "MRL",
    "IRL",
    "FRR",
    "PRR",
    # Termes d'ingénierie système
    "exigence système",
    "system requirement",
    "architecture système",
    "system architecture",
    "vérification",
    "verification",
    "validation",
    "intégration",
    "integration",
    "jalon",
    "milestone",
    "cycle de vie",
    "lifecycle",
    "life cycle",
    "ingénierie système",
    "systems engineering",
    "parties prenantes",
    "stakeholders",
    "traçabilité",
    "traceability",
    # Processus IS 15288
    "processus de vérification",
    "processus de validation",
    "processus d'intégration",
    "processus d'acquisition",
    "processus de conception",
    "processus de définition",
    # Livrables
    "spécification technique",
    "plan de management",
    "dossier de définition",
    "dossier justificatif",
    # Approches
    "modèle en V",
    "DevSecOps",
    "approche séquentielle",
    "approche incrémentale",
    "approche agile",
]

# Patterns regex pour détecter les termes
GRI_TERM_PATTERNS: list[tuple[str, str]] = [
    (r"\b(artefact|artifact)s?\b", "artefact"),
    (r"\b(CONOPS)\b", "CONOPS"),
    (r"\b(SEMP)\b", "SEMP"),
    (r"\b(SRR)\b", "SRR"),
    (r"\b(PDR)\b", "PDR"),
    (r"\b(CDR)\b", "CDR"),
    (r"\b(IRR)\b", "IRR"),
    (r"\b(TRR)\b", "TRR"),
    (r"\b(SAR)\b", "SAR"),
    (r"\b(ORR)\b", "ORR"),
    (r"\b(MNR)\b", "MNR"),
    (r"\b(CIR)\b", "CIR"),
    (r"\b(TRL)\b", "TRL"),
    (r"\b(MRL)\b", "MRL"),
    (r"\b(IRL)\b", "IRL"),
    (r"\b(FRR)\b", "FRR"),
    (r"\b(PRR)\b", "PRR"),
    (r"cycle\s+de\s+vie", "cycle de vie"),
    (r"life\s*cycle", "cycle de vie"),
    (r"ingénierie\s+système", "ingénierie système"),
    (r"systems?\s+engineering", "ingénierie système"),
    (r"exigences?\s+système", "exigence système"),
    (r"system\s+requirements?", "exigence système"),
    (r"architecture\s+système", "architecture système"),
    (r"system\s+architecture", "architecture système"),
    (r"parties?\s+prenantes?", "parties prenantes"),
    (r"stakeholders?", "parties prenantes"),
    (r"traçabilité", "traçabilité"),
    (r"traceability", "traçabilité"),
    (r"vérification", "vérification"),
    (r"validation", "validation"),
    (r"intégration", "intégration"),
]


class TermDefinition(BaseModel):
    """Définition d'un terme GRI."""

    term_fr: str
    term_en: str | None = None
    definition_fr: str
    definition_en: str | None = None
    standard_ref: str | None = None
    source: str = "GRI"


class ExpansionResult(BaseModel):
    """Résultat de l'expansion de query."""

    original_query: str
    detected_terms: list[str] = Field(default_factory=list)
    definitions: list[TermDefinition] = Field(default_factory=list)
    term_context: str = ""
    has_expansions: bool = False


class GRITermExpander:
    """Enrichissement de queries avec définitions GRI.

    Détecte les termes ISO/GRI dans une query et récupère leurs définitions
    depuis l'index glossaire. Le contexte terminologique est ensuite injecté
    dans le system prompt du LLM.

    Attributes:
        store: GRIHybridStore pour le lookup glossaire
        max_terms: Nombre maximum de termes à injecter
    """

    def __init__(
        self,
        store: "GRIHybridStore",
        max_terms: int = 3,
    ) -> None:
        """Initialise l'expander.

        Args:
            store: Vector store pour le lookup glossaire
            max_terms: Nombre max de définitions injectées
        """
        self.store = store
        self.max_terms = max_terms

        log.info("term_expander.init", max_terms=max_terms)

    def detect_terms(self, query: str) -> list[str]:
        """Détecte les termes GRI dans une query.

        Args:
            query: Query utilisateur

        Returns:
            Liste de termes détectés (normalisés)
        """
        detected: set[str] = set()
        query_lower = query.lower()

        # Vérifier les patterns
        for pattern, normalized_term in GRI_TERM_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                detected.add(normalized_term)

        # Vérifier les termes exacts
        for term in GRI_KEY_TERMS:
            if term.lower() in query_lower:
                detected.add(term)

        log.info("term_expander.detected", terms=list(detected), query=query[:50])
        return list(detected)

    async def expand(self, query: str) -> ExpansionResult:
        """Enrichit une query avec les définitions des termes GRI détectés.

        Args:
            query: Query utilisateur

        Returns:
            ExpansionResult avec query originale et contexte terminologique
        """
        detected_terms = self.detect_terms(query)

        if not detected_terms:
            return ExpansionResult(
                original_query=query,
                detected_terms=[],
                definitions=[],
                term_context="",
                has_expansions=False,
            )

        # Récupérer les définitions (max N)
        definitions: list[TermDefinition] = []

        for term in detected_terms[: self.max_terms]:
            try:
                result = await self.store.glossary_lookup(term)
                if result:
                    definitions.append(
                        TermDefinition(
                            term_fr=result.metadata.get("term_fr", term),
                            term_en=result.metadata.get("term_en"),
                            definition_fr=result.metadata.get(
                                "definition_fr", result.content
                            ),
                            definition_en=result.metadata.get("definition_en"),
                            standard_ref=result.metadata.get("standard_ref"),
                            source=result.metadata.get("source", "GRI"),
                        )
                    )
            except Exception as e:
                log.warning("term_expander.lookup_failed", term=term, error=str(e))

        # Construire le contexte terminologique
        term_context = self._build_term_context(definitions)

        return ExpansionResult(
            original_query=query,
            detected_terms=detected_terms,
            definitions=definitions,
            term_context=term_context,
            has_expansions=len(definitions) > 0,
        )

    def _build_term_context(self, definitions: list[TermDefinition]) -> str:
        """Construit le contexte terminologique formaté.

        Args:
            definitions: Liste de définitions

        Returns:
            Contexte formaté pour injection dans le prompt
        """
        if not definitions:
            return ""

        lines = ["## Définitions GRI applicables"]

        for defn in definitions:
            term_line = f"• **{defn.term_fr}**"
            if defn.term_en:
                term_line += f" ({defn.term_en})"
            term_line += f": {defn.definition_fr}"

            if defn.standard_ref:
                term_line += f" [{defn.standard_ref}]"

            lines.append(term_line)

        return "\n".join(lines)


# Fonction helper pour usage simple
async def expand_query_with_terms(
    query: str,
    store: "GRIHybridStore",
    max_terms: int = 3,
) -> tuple[str, str]:
    """Enrichit une query avec les définitions GRI.

    Args:
        query: Query utilisateur
        store: Vector store
        max_terms: Nombre max de définitions

    Returns:
        Tuple (query_originale, contexte_terminologique)
    """
    expander = GRITermExpander(store, max_terms=max_terms)
    result = await expander.expand(query)
    return result.original_query, result.term_context


def detect_gri_terms(query: str) -> list[str]:
    """Détecte les termes GRI dans une query (sans lookup).

    Args:
        query: Query utilisateur

    Returns:
        Liste de termes détectés
    """
    detected: set[str] = set()
    query_lower = query.lower()

    for pattern, normalized_term in GRI_TERM_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            detected.add(normalized_term)

    for term in GRI_KEY_TERMS:
        if term.lower() in query_lower:
            detected.add(term)

    return list(detected)
