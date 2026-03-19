"""Utilitaires centralisés pour l'extraction des IDs de jalons.

Ce module fournit une fonction unique d'extraction utilisée par:
- src/ingestion/chunker.py
- src/ingestion/parser.py
- src/ingestion/table_extractor.py

Évite la duplication de logique et la dérive entre modules.
"""

import re

from src.core.config import MILESTONE_ACRONYM_TO_ID, VALID_MILESTONES

# Pattern case-insensitive pour M0-M9 et J1-J6
# Utilise lookbehind/lookahead pour éviter les faux positifs
# Couvre: (M4), M4-CDR, M4:, "jalon m3", etc.
MILESTONE_PATTERN = re.compile(r"(?<!\w)([MJ]\d)(?!\w)", re.IGNORECASE)

# Pattern pour les acronymes de revue (ASR, SRR, PDR, CDR, etc.)
ACRONYM_PATTERN = re.compile(
    r"\b(" + "|".join(MILESTONE_ACRONYM_TO_ID.keys()) + r")\b",
    re.IGNORECASE,
)


def extract_milestone_id(
    text: str,
    hierarchy: list[str] | None = None,
) -> str | None:
    """Extrait l'ID du jalon depuis le texte et optionnellement la hiérarchie.

    Ordre de priorité:
    1. Pattern M/J direct dans le texte (case-insensitive)
    2. Acronymes de revue dans le texte (CDR -> M3, PDR -> M2, etc.)
    3. Pattern M/J dans la hiérarchie
    4. Acronymes dans la hiérarchie

    Args:
        text: Texte à analyser (typiquement title + content)
        hierarchy: Chemin hiérarchique de la section (optionnel)

    Returns:
        Milestone ID normalisé (M0-M9, J1-J6) ou None si non trouvé

    Examples:
        >>> extract_milestone_id("Critères du jalon m3")
        'M3'
        >>> extract_milestone_id("Critical Design Review (CDR)")
        'M3'
        >>> extract_milestone_id("Objectifs", hierarchy=["GRI", "Jalon M4"])
        'M4'
    """
    result = _search_in_text(text)
    if result:
        return result

    # Fallback: chercher dans la hiérarchie
    if hierarchy:
        for level in hierarchy:
            result = _search_in_text(level)
            if result:
                return result

    return None


def _search_in_text(text: str) -> str | None:
    """Cherche un milestone ID dans un texte donné.

    Args:
        text: Texte à analyser

    Returns:
        Milestone ID normalisé ou None
    """
    if not text:
        return None

    # Priorité 1: Pattern M/J direct
    match = MILESTONE_PATTERN.search(text)
    if match:
        milestone_id = match.group(1).upper()
        if milestone_id in VALID_MILESTONES:
            return milestone_id

    # Priorité 2: Acronyme de revue
    match = ACRONYM_PATTERN.search(text)
    if match:
        acronym = match.group(1).upper()
        return MILESTONE_ACRONYM_TO_ID.get(acronym)

    return None


def normalize_milestone_id(milestone_id: str) -> str | None:
    """Normalise un ID de jalon ou acronyme vers le format standard.

    Args:
        milestone_id: ID brut (M3, m3, CDR, pdr, etc.)

    Returns:
        ID normalisé (M0-M9, J1-J6) ou None si invalide

    Examples:
        >>> normalize_milestone_id("m3")
        'M3'
        >>> normalize_milestone_id("CDR")
        'M3'
        >>> normalize_milestone_id("invalid")
        None
    """
    if not milestone_id:
        return None

    upper = milestone_id.upper().strip()

    # Déjà un format M/J valide
    if upper in VALID_MILESTONES:
        return upper

    # Essayer le mapping acronyme
    return MILESTONE_ACRONYM_TO_ID.get(upper)
