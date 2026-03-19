"""Formatage du contexte GRI pour le LLM.

Ce module formate les chunks récupérés pour maximiser la compréhension du LLM.
Ordre : du plus pertinent au moins pertinent.
Préserver les context prefixes — ils aident le LLM à localiser les infos.

Usage:
    from src.generation.context_formatter import format_gri_context, check_context_sufficiency

    context = format_gri_context(chunks)
    sufficiency = check_context_sufficiency(chunks, GRIResponseType.DEFINITION)
"""

from typing import Any

import structlog

from src.generation.prompts import GRIResponseType

log = structlog.get_logger()


# Seuils de qualité
MIN_SCORE_THRESHOLD = 0.35
MIN_CHUNKS_FOR_COMPARISON = 2


def format_gri_context(chunks: list[dict[str, Any]]) -> str:
    """Formate les chunks récupérés pour le contexte LLM.

    Ordre : du plus pertinent au moins pertinent.
    Préserve les context prefixes pour la localisation.

    Args:
        chunks: Liste de chunks avec content, score, metadata

    Returns:
        Contexte formaté pour le prompt
    """
    if not chunks:
        return "Aucune source disponible dans la base GRI pour cette query."

    formatted = []

    for i, chunk in enumerate(chunks, 1):
        # Extraire les métadonnées
        cycle_tag = chunk.get("cycle", "GRI")
        section_type = chunk.get("section_type", "content")
        score = chunk.get("score", 0)

        # Construire le header
        header = f"[SOURCE {i}] {cycle_tag} · {section_type.upper()} · Score: {score:.2f}"

        if chunk.get("milestone_id"):
            header += f" · Jalon: {chunk['milestone_id']}"
        if chunk.get("phase_num"):
            header += f" · Phase {chunk['phase_num']}"
        if chunk.get("context_prefix"):
            header += f"\n{chunk['context_prefix']}"

        # Contenu
        content = chunk.get("content", "")

        formatted.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(formatted)


def format_comparison_context(
    chunks_a: list[dict[str, Any]],
    chunks_b: list[dict[str, Any]],
    entity_a: str,
    entity_b: str,
) -> tuple[str, str]:
    """Formate les contextes pour une comparaison.

    Args:
        chunks_a: Chunks pour l'entité A
        chunks_b: Chunks pour l'entité B
        entity_a: Nom de l'entité A
        entity_b: Nom de l'entité B

    Returns:
        Tuple (context_a, context_b)
    """
    context_a = format_gri_context(chunks_a) if chunks_a else f"Aucune source pour {entity_a}"
    context_b = format_gri_context(chunks_b) if chunks_b else f"Aucune source pour {entity_b}"

    return context_a, context_b


def check_context_sufficiency(
    chunks: list[dict[str, Any]],
    response_type: GRIResponseType,
) -> dict[str, Any]:
    """Vérifie si le contexte est suffisant pour la génération.

    Args:
        chunks: Chunks récupérés
        response_type: Type de réponse demandé

    Returns:
        Dict avec:
        - sufficient: bool
        - reason: str (si insufficient)
        - message: str (message utilisateur si insufficient)
    """
    # Pas de chunks
    if not chunks:
        log.warning("context.insufficient", reason="no_chunks")
        return {
            "sufficient": False,
            "reason": "no_chunks",
            "message": (
                "Aucune source pertinente trouvée dans le GRI pour cette query. "
                "Reformuler avec des termes du GRI ou vérifier que le document est bien indexé."
            ),
        }

    # Scores trop faibles
    max_score = max(c.get("score", 0) for c in chunks)
    if max_score < MIN_SCORE_THRESHOLD:
        log.warning("context.insufficient", reason="low_scores", max_score=max_score)
        return {
            "sufficient": False,
            "reason": "low_scores",
            "message": (
                f"Les sources récupérées ont un score faible (max: {max_score:.2f}). "
                "La question est peut-être hors du périmètre du GRI."
            ),
        }

    # Vérifications spécifiques par type
    if response_type == GRIResponseType.MILESTONE:
        has_milestone_chunk = any(c.get("section_type") == "milestone" for c in chunks)
        if not has_milestone_chunk:
            log.warning("context.insufficient", reason="missing_milestone_chunk")
            return {
                "sufficient": False,
                "reason": "missing_milestone_chunk",
                "message": (
                    "Critères de jalon non trouvés dans le contexte. "
                    "Appeler get_milestone_criteria() directement."
                ),
            }

    if response_type == GRIResponseType.DEFINITION:
        # Pour les définitions, vérifier qu'on a du contenu glossaire
        has_definition_content = any(
            c.get("section_type") in ("definition", "glossary")
            or "définition" in c.get("content", "").lower()
            for c in chunks
        )
        if not has_definition_content:
            log.warning("context.insufficient", reason="no_definition_content")
            return {
                "sufficient": False,
                "reason": "no_definition_content",
                "message": (
                    "Aucune définition trouvée dans les sources. "
                    "Appeler lookup_gri_glossary() pour une recherche directe."
                ),
            }

    if response_type == GRIResponseType.PHASE_COMPLETE:
        has_phase_content = any(c.get("section_type") == "phase" for c in chunks)
        if not has_phase_content:
            log.warning("context.insufficient", reason="no_phase_content")
            return {
                "sufficient": False,
                "reason": "no_phase_content",
                "message": (
                    "Contenu de phase non trouvé. "
                    "Appeler get_phase_summary() pour une recherche directe."
                ),
            }

    return {"sufficient": True}


def extract_context_variables(
    chunks: list[dict[str, Any]],
    response_type: GRIResponseType,
    query: str,
) -> dict[str, Any]:
    """Extrait les variables de contexte depuis les chunks.

    Utile pour préremplir les variables des prompts.

    Args:
        chunks: Chunks récupérés
        response_type: Type de réponse
        query: Query originale

    Returns:
        Dict de variables pour le prompt
    """
    import re

    variables: dict[str, Any] = {
        "context": format_gri_context(chunks),
        "query": query,
    }

    if response_type == GRIResponseType.DEFINITION:
        # Extraire le terme de la query
        term_patterns = [
            r"(?:définition|definition)\s+(?:de|d'|du|de la)\s+['\"]?(\w+)['\"]?",
            r"qu'est[- ]ce\s+qu['\"]?(?:un|une)\s+(\w+)",
            r"c'est\s+quoi\s+(?:un|une)\s+(\w+)",
        ]
        for pattern in term_patterns:
            match = re.search(pattern, query.lower())
            if match:
                variables["term"] = match.group(1)
                break
        if "term" not in variables:
            # Fallback : dernier mot significatif
            words = [w for w in query.split() if len(w) > 3]
            variables["term"] = words[-1] if words else "terme"

    elif response_type == GRIResponseType.MILESTONE:
        # Extraire l'ID du jalon
        milestone_match = re.search(r"\b([MJ]\d)\b", query.upper())
        if milestone_match:
            variables["milestone_id"] = milestone_match.group(1)
        else:
            # Chercher dans les chunks
            for chunk in chunks:
                if chunk.get("milestone_id"):
                    variables["milestone_id"] = chunk["milestone_id"]
                    break

        # Nom du jalon (simpliste)
        milestone_names = {
            "M0": "LANCEMENT",
            "M1": "SRR",
            "M2": "SFR",
            "M3": "PDR",
            "M4": "CDR",
            "M5": "IRR",
            "M6": "SVR",
            "M7": "QR",
            "M8": "MISE EN SERVICE",
            "M9": "RETRAIT",
            "J1": "Lancement CIR",
            "J2": "Validation conception",
            "J3": "Livraison intermédiaire",
            "J4": "SAR",
            "J5": "SAR Final",
            "J6": "Clôture",
        }
        mid = variables.get("milestone_id", "")
        variables["milestone_name"] = milestone_names.get(mid, mid)

    elif response_type == GRIResponseType.PHASE_COMPLETE:
        # Extraire le numéro de phase
        phase_match = re.search(r"phase\s+(\d)", query.lower())
        if phase_match:
            variables["phase_num"] = int(phase_match.group(1))

        # Détecter le cycle
        variables["cycle"] = "CIR" if "cir" in query.lower() else "GRI"

        # Phase titles et milestones
        gri_phases = {
            1: ("Idéation / Préparation", "M0", "M1"),
            2: ("Faisabilité / Définition préliminaire", "M1", "M3"),
            3: ("Conception et Développement", "M3", "M4"),
            4: ("Intégration et Vérification", "M4", "M6"),
            5: ("Qualification", "M6", "M7"),
            6: ("Production et Déploiement", "M7", "M8"),
            7: ("Utilisation et Soutien / Retrait", "M8", "M9"),
        }
        cir_phases = {
            1: ("Lancement et Cadrage", "J1", "J1"),
            2: ("Conception et Prototypage", "J1", "J2"),
            3: ("Intégration et Tests", "J2", "J4"),
            4: ("Livraison et Clôture", "J4", "J6"),
        }

        phases = cir_phases if variables["cycle"] == "CIR" else gri_phases
        phase_num = variables.get("phase_num", 1)
        if phase_num in phases:
            title, entry, exit_ = phases[phase_num]
            variables["phase_title"] = title
            variables["entry_milestone"] = entry
            variables["exit_milestone"] = exit_

    elif response_type == GRIResponseType.PROCESS:
        # Extraire le nom du processus
        process_patterns = [
            r"processus\s+(?:de\s+)?['\"]?(.+?)['\"]?(?:\s+selon|\s+dans|\?|$)",
            r"(?:IS\s*15288|15288)[^:]*:\s*(.+?)(?:\?|$)",
        ]
        for pattern in process_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                variables["process_name"] = match.group(1).strip()
                break
        if "process_name" not in variables:
            variables["process_name"] = "processus"

    elif response_type == GRIResponseType.COMPARISON:
        # Extraire les deux entités
        comparison_patterns = [
            r"(?:comparer?|différence)\s+(?:entre\s+)?['\"]?(.+?)['\"]?\s+(?:et|vs|versus)\s+['\"]?(.+?)['\"]?(?:\?|$)",
            r"(.+?)\s+(?:vs|versus|contre)\s+(.+?)(?:\?|$)",
        ]
        for pattern in comparison_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                variables["entity_a"] = match.group(1).strip()
                variables["entity_b"] = match.group(2).strip()
                break

        # Contextes séparés si disponibles
        if "entity_a" in variables and "entity_b" in variables:
            # TODO: Séparer les chunks par entité
            variables["context_a"] = variables["context"]
            variables["context_b"] = variables["context"]

    return variables


def truncate_context(context: str, max_chars: int = 8000) -> str:
    """Tronque le contexte si nécessaire.

    Préserve les chunks complets (ne coupe pas au milieu d'un chunk).

    Args:
        context: Contexte formaté
        max_chars: Limite de caractères

    Returns:
        Contexte tronqué
    """
    if len(context) <= max_chars:
        return context

    # Séparer par chunks
    chunks = context.split("\n\n---\n\n")

    # Garder autant de chunks que possible
    truncated = []
    current_length = 0

    for chunk in chunks:
        if current_length + len(chunk) + 7 > max_chars:  # +7 pour le séparateur
            break
        truncated.append(chunk)
        current_length += len(chunk) + 7

    result = "\n\n---\n\n".join(truncated)

    if len(truncated) < len(chunks):
        omitted = len(chunks) - len(truncated)
        result += f"\n\n[... {omitted} source(s) omise(s) pour respecter la limite de contexte]"

    log.info(
        "context.truncated",
        original_chunks=len(chunks),
        kept_chunks=len(truncated),
        original_chars=len(context),
        final_chars=len(result),
    )

    return result
