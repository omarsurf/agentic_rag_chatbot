"""Post-processing des réponses GRI.

Ce module effectue les vérifications post-génération spécifiques au GRI :
- Validation des jalons cités (M0-M9, J1-J6)
- Validation des phases citées (GRI: 1-7, CIR: 1-4)
- Extraction et validation des citations
- Détection des hallucinations potentielles

Usage:
    from src.generation.postprocessor import postprocess_gri_answer, extract_citations

    processed = postprocess_gri_answer(answer, GRIResponseType.MILESTONE)
    citations = extract_citations(answer)
"""

import re
from typing import Any

import structlog

from src.generation.prompts import GRIResponseType

log = structlog.get_logger()


# Jalons valides
VALID_GRI_MILESTONES = {f"M{i}" for i in range(10)}  # M0-M9
VALID_CIR_MILESTONES = {f"J{i}" for i in range(1, 7)}  # J1-J6
VALID_MILESTONES = VALID_GRI_MILESTONES | VALID_CIR_MILESTONES

# Phases valides
MAX_GRI_PHASE = 7
MAX_CIR_PHASE = 4

# Patterns de citation GRI/CIR
CITATION_PATTERNS = [
    r"\[GRI\s*>\s*[^\]]+\]",
    r"\[CIR\s*>\s*[^\]]+\]",
    r"\[Source\s*:\s*[^\]]+\]",
]


def postprocess_gri_answer(
    answer: str,
    response_type: GRIResponseType,
    context_chunks: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Post-traitement complet d'une réponse GRI.

    Args:
        answer: Réponse générée
        response_type: Type de réponse
        context_chunks: Chunks de contexte (optionnel, pour validation)

    Returns:
        Dict avec:
        - answer: Réponse post-traitée
        - citations: Liste des citations extraites
        - warnings: Liste des avertissements
        - has_normative_content: bool
        - validation: Dict de résultats de validation
    """
    _ = context_chunks
    warnings: list[str] = []
    validation: dict[str, Any] = {}

    # 1. Valider les jalons cités
    milestone_validation = validate_milestones(answer)
    if milestone_validation["invalid"]:
        invalid_str = ", ".join(milestone_validation["invalid"])
        warnings.append(f"Jalons non reconnus dans le GRI : {invalid_str}")
    validation["milestones"] = milestone_validation

    # 2. Valider les phases citées
    phase_validation = validate_phases(answer)
    if phase_validation["invalid"]:
        invalid_str = ", ".join(str(p) for p in phase_validation["invalid"])
        warnings.append(f"Phases invalides : {invalid_str}")
    validation["phases"] = phase_validation

    # 3. Extraire les citations
    citations = extract_citations(answer)
    validation["has_citations"] = len(citations) > 0

    # 4. Vérifications spécifiques par type
    if response_type == GRIResponseType.DEFINITION:
        # Vérifier que c'est une définition (pas une reformulation)
        if not _looks_like_definition(answer):
            warnings.append(
                "La réponse ne semble pas suivre le format de définition ISO attendu."
            )

    elif response_type == GRIResponseType.MILESTONE:
        # Compter les critères numérotés (avec espaces possibles en début de ligne)
        criteria_count = len(re.findall(r"^\s*\d+\.", answer, re.MULTILINE))
        validation["criteria_count"] = criteria_count
        if criteria_count == 0:
            warnings.append(
                "Aucun critère numéroté trouvé. Format attendu : 1. [critère]"
            )

    # 5. Ajouter les warnings à la réponse si nécessaire
    processed_answer = answer
    if warnings:
        warning_section = "\n\n---\n**Avertissements :**\n"
        for w in warnings:
            warning_section += f"- {w}\n"
        processed_answer = answer + warning_section

    # 6. Déterminer si contenu normatif
    has_normative = response_type in (
        GRIResponseType.DEFINITION,
        GRIResponseType.MILESTONE,
    )

    log.info(
        "postprocess.complete",
        response_type=response_type.value,
        n_citations=len(citations),
        n_warnings=len(warnings),
    )

    return {
        "answer": processed_answer,
        "citations": citations,
        "warnings": warnings,
        "has_normative_content": has_normative,
        "validation": validation,
    }


def validate_milestones(text: str) -> dict[str, Any]:
    """Valide les jalons cités dans le texte.

    Args:
        text: Texte à analyser

    Returns:
        Dict avec cited, valid, invalid
    """
    # Trouver tous les jalons cités (M0-M9, J1-J6, et invalides comme M15)
    cited = set(re.findall(r"\b([MJ]\d+)\b", text.upper()))

    valid = cited & VALID_MILESTONES
    invalid = cited - VALID_MILESTONES

    return {
        "cited": list(cited),
        "valid": list(valid),
        "invalid": list(invalid),
    }


def validate_phases(text: str) -> dict[str, Any]:
    """Valide les phases citées dans le texte.

    Args:
        text: Texte à analyser

    Returns:
        Dict avec cited, valid, invalid
    """
    # Trouver toutes les phases citées
    phase_matches = re.findall(r"[Pp]hase\s+(\d+)", text)
    cited = {int(p) for p in phase_matches}

    # Déterminer le cycle (CIR ou GRI) selon le contexte
    is_cir = "cir" in text.lower()
    max_phase = MAX_CIR_PHASE if is_cir else MAX_GRI_PHASE

    valid = {p for p in cited if 1 <= p <= max_phase}
    invalid = cited - valid

    return {
        "cited": list(cited),
        "valid": list(valid),
        "invalid": list(invalid),
        "assumed_cycle": "CIR" if is_cir else "GRI",
    }


def extract_citations(text: str) -> list[str]:
    """Extrait les citations GRI/CIR du texte.

    Args:
        text: Texte à analyser

    Returns:
        Liste de citations uniques
    """
    citations = []

    for pattern in CITATION_PATTERNS:
        matches = re.findall(pattern, text)
        citations.extend(matches)

    # Dédupliquer tout en préservant l'ordre
    seen = set()
    unique = []
    for c in citations:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    return unique


def validate_citations_against_context(
    citations: list[str],
    context_chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Valide que les citations correspondent au contexte fourni.

    Args:
        citations: Citations extraites de la réponse
        context_chunks: Chunks de contexte utilisés

    Returns:
        Dict avec grounded, ungrounded, coverage
    """
    if not context_chunks:
        return {
            "grounded": [],
            "ungrounded": citations,
            "coverage": 0.0,
        }

    # Construire un ensemble de références du contexte
    context_refs = set()
    for chunk in context_chunks:
        if chunk.get("context_prefix"):
            context_refs.add(chunk["context_prefix"].lower())
        if chunk.get("milestone_id"):
            context_refs.add(chunk["milestone_id"].lower())
        if chunk.get("cycle"):
            context_refs.add(chunk["cycle"].lower())

    grounded = []
    ungrounded = []

    for citation in citations:
        citation_lower = citation.lower()
        # Vérifier si la citation est supportée par le contexte
        is_grounded = any(ref in citation_lower for ref in context_refs)
        if is_grounded:
            grounded.append(citation)
        else:
            ungrounded.append(citation)

    coverage = len(grounded) / len(citations) if citations else 1.0

    return {
        "grounded": grounded,
        "ungrounded": ungrounded,
        "coverage": coverage,
    }


def _looks_like_definition(text: str) -> bool:
    """Vérifie si le texte ressemble à une définition ISO.

    Args:
        text: Texte à analyser

    Returns:
        True si ressemble à une définition
    """
    # Patterns typiques d'une définition
    patterns = [
        r"\*\*[A-Za-zÀ-ÿ\s]+\*\*\s*\([A-Za-z\s]+\)\s*:",  # **Terme FR** (Term EN) :
        r"définition\s*:",
        r"ISO/IEC",
        r"15288",
        r"selon\s+(?:le\s+)?GRI",
    ]

    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def clean_response(text: str) -> str:
    """Nettoie la réponse générée.

    - Supprime les espaces multiples
    - Corrige la ponctuation
    - Normalise les sauts de ligne

    Args:
        text: Texte à nettoyer

    Returns:
        Texte nettoyé
    """
    # Normaliser les sauts de ligne
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Supprimer les espaces multiples (mais pas les sauts de ligne)
    text = re.sub(r"[^\S\n]+", " ", text)

    # Supprimer les espaces en début/fin de ligne
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Supprimer les espaces en début/fin
    text = text.strip()

    return text


def add_source_footer(
    answer: str,
    citations: list[str],
    response_type: GRIResponseType,
) -> str:
    """Ajoute un pied de page avec les sources si absent.

    Args:
        answer: Réponse
        citations: Citations extraites
        response_type: Type de réponse

    Returns:
        Réponse avec footer
    """
    _ = response_type
    # Vérifier si un footer existe déjà
    if re.search(r"\*\*Sources?\s*:\*\*", answer, re.IGNORECASE):
        return answer

    if re.search(r"^Sources?\s*:", answer, re.MULTILINE | re.IGNORECASE):
        return answer

    # Ajouter le footer si on a des citations
    if citations:
        footer = "\n\n**Sources :**\n"
        for citation in citations[:5]:  # Max 5 citations
            footer += f"- {citation}\n"
        return answer + footer

    return answer


def format_criteria_list(text: str) -> str:
    """Reformate une liste de critères pour cohérence.

    Assure que chaque critère est sur sa propre ligne
    avec une numérotation cohérente.

    Args:
        text: Texte contenant des critères

    Returns:
        Texte reformaté
    """
    # Trouver la section des critères
    lines = text.split("\n")
    in_criteria = False
    criteria_lines = []
    other_lines = []
    current_criterion = ""

    for line in lines:
        # Détecter le début d'un critère
        if re.match(r"^\d+\.", line.strip()):
            if current_criterion:
                criteria_lines.append(current_criterion)
            current_criterion = line.strip()
            in_criteria = True
        elif in_criteria and line.strip() and not line.strip().startswith("#"):
            # Continuation du critère courant
            current_criterion += " " + line.strip()
        else:
            if current_criterion:
                criteria_lines.append(current_criterion)
                current_criterion = ""
            in_criteria = False
            other_lines.append(line)

    if current_criterion:
        criteria_lines.append(current_criterion)

    # Renuméroter si nécessaire
    formatted_criteria = []
    for i, criterion in enumerate(criteria_lines, 1):
        # Enlever l'ancien numéro
        criterion = re.sub(r"^\d+\.\s*", "", criterion)
        formatted_criteria.append(f"{i}. {criterion}")

    # Reconstruire le texte
    # (simplifié - dans une vraie implémentation, on préserverait la structure)
    return text  # Pour l'instant, retourner tel quel
