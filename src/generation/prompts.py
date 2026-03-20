"""Prompts de génération par type de contenu GRI.

Ce module définit les prompts spécialisés pour chaque type de réponse :
- DEFINITION : Fidélité normative absolue (temperature=0.0)
- MILESTONE : Critères exhaustifs (temperature=0.0)
- PROCESS : Description structurée (temperature=0.1)
- PHASE_COMPLETE : Résumé de phase (temperature=0.1)
- COMPARISON : Tableau comparatif (temperature=0.1)
- GENERAL : Réponse libre grounded (temperature=0.1)

Usage:
    from src.generation.prompts import get_prompt, GRIResponseType

    prompt = get_prompt(GRIResponseType.DEFINITION, term="artefact")
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class GRIResponseType(StrEnum):
    """Types de réponses GRI avec contraintes spécifiques."""

    DEFINITION = "definition"
    MILESTONE = "milestone"
    PROCESS = "process"
    PHASE_COMPLETE = "phase_complete"
    COMPARISON = "comparison"
    GENERAL = "general"


# === Configuration par type ===

TEMPERATURE_MAP: dict[GRIResponseType, float] = {
    GRIResponseType.DEFINITION: 0.0,  # Fidélité normative absolue
    GRIResponseType.MILESTONE: 0.0,  # Critères exhaustifs
    GRIResponseType.PROCESS: 0.1,
    GRIResponseType.PHASE_COMPLETE: 0.1,
    GRIResponseType.COMPARISON: 0.1,
    GRIResponseType.GENERAL: 0.1,
}

MAX_TOKENS_MAP: dict[GRIResponseType, int] = {
    GRIResponseType.DEFINITION: 256,
    GRIResponseType.MILESTONE: 1024,
    GRIResponseType.PROCESS: 1536,
    GRIResponseType.PHASE_COMPLETE: 2048,
    GRIResponseType.COMPARISON: 2048,
    GRIResponseType.GENERAL: 1024,
}


# === Prompts par type ===

DEFINITION_PROMPT = """Fournis la définition exacte du terme "{term}" selon le GRI des FAR.

Contexte source :
{context}

Instructions :
- Reproduire la définition FR telle qu'elle apparaît dans le GRI (aucune paraphrase)
- Inclure l'équivalent EN entre parenthèses
- Indiquer la référence normative (ISO/IEC/IEEE 15288:2023 ou autre)
- Format : **Terme FR** (Term EN) : [définition exacte] [Source normative]

Si le terme n'est pas dans le contexte fourni, dire explicitement :
"Ce terme n'est pas défini dans les sources disponibles du GRI."
"""

MILESTONE_PROMPT = """Liste les critères de passage du jalon "{milestone_id} — {milestone_name}".

Contexte source :
{context}

Instructions :
- Lister TOUS les critères, aucun n'est optionnel
- Format numéroté : 1. [critère exact du GRI]
- Si CIR : afficher d'abord les critères CIR, puis section "Équivalents GRI : [M_IDs]"
- Terminer par : "Source : [GRI > Jalon {milestone_id} ({milestone_name})]"
- Ne PAS résumer ou regrouper les critères — les lister individuellement

Si les critères sont incomplets dans le contexte : indiquer le nombre manquant et
suggérer d'appeler get_milestone_criteria("{milestone_id}") pour la version complète.
"""

PROCESS_PROMPT = """Décris le processus IS 15288 "{process_name}" selon le GRI.

Contexte source :
{context}

Structure de réponse :
## Objectif du processus
[1-2 phrases max, tirées du GRI]

## Activités principales
[Liste numérotée des activités, dans l'ordre GRI]

## Données d'entrée (Inputs)
[Liste des inputs définis dans le GRI]

## Données de sortie (Outputs / Livrables)
[Liste des outputs/livrables définis dans le GRI]

## Liens avec d'autres processus
[Si mentionné dans le contexte]

**Source : [GRI > Processus IS 15288 > {process_name}]**
"""

PHASE_PROMPT = """Résume la {cycle} Phase {phase_num} — "{phase_title}".

Contexte source :
{context}

Structure de réponse :
## Objectif général
[Tiré du GRI]

## Objectifs spécifiques
[Liste numérotée, tirée du GRI]

## Activités clés
[3-5 activités principales]

## Produits / Livrables de la phase
[Liste des livrables définis dans le GRI]

## Jalons encadrant cette phase
- Entrée : Jalon {entry_milestone} — [nom]
- Sortie : Jalon {exit_milestone} — [nom]

**Source : [GRI > Phase {phase_num} : {phase_title}]**
"""

COMPARISON_PROMPT = """Compare "{entity_a}" et "{entity_b}" selon le GRI des FAR.

Contexte source A ({entity_a}) :
{context_a}

Contexte source B ({entity_b}) :
{context_b}

Structure de réponse :
## Vue d'ensemble
[1 phrase par entité]

## Tableau comparatif

| Dimension | {entity_a} | {entity_b} |
|-----------|-----------|-----------|
| [dim 1]   | ...       | ...       |
| [dim 2]   | ...       | ...       |

## Points communs
[Issu du GRI]

## Différences clés
[Issu du GRI]

## Quand utiliser l'un plutôt que l'autre
[Si mentionné dans le GRI]

**Sources :**
- {entity_a} : [GRI > ...]
- {entity_b} : [GRI > ...]
"""

GENERAL_PROMPT = """Réponds à la question suivante en utilisant uniquement les sources GRI fournies.

Question : {query}

Contexte source :
{context}

Instructions :
- Baser la réponse uniquement sur le contexte fourni
- Citer les sources au format [GRI > Section > ...]
- Si l'information n'est pas disponible, le dire explicitement
- Structurer la réponse de manière claire et concise
"""


# === System prompts par type ===

BASE_SYSTEM_PROMPT = """Tu es un expert en ingénierie système selon le GRI des FAR (ISO/IEC/IEEE 15288:2023).

RÈGLES DE GÉNÉRATION :
1. Utilise UNIQUEMENT les informations des SOURCES fournies
2. Citations obligatoires au format [GRI > Section > ...] ou [CIR > Phase N > ...]
3. Termes ISO : reproduire le libellé exact du GRI (aucune paraphrase)
4. Si l'information n'est pas dans les sources : dire "Non disponible dans les sources GRI fournies"
5. Ne jamais inventer de critères, jalons, ou livrables"""

DEFINITION_SYSTEM_ADDON = """
6. DÉFINITIONS : reproduire mot pour mot, aucune reformulation autorisée"""

MILESTONE_SYSTEM_ADDON = """
6. JALONS : lister TOUS les critères, aucun ne peut être omis ou résumé"""


class PromptConfig(BaseModel):
    """Configuration d'un prompt pour un type de réponse."""

    template: str
    system_addon: str = ""
    required_vars: list[str]


PROMPT_CONFIGS: dict[GRIResponseType, PromptConfig] = {
    GRIResponseType.DEFINITION: PromptConfig(
        template=DEFINITION_PROMPT,
        system_addon=DEFINITION_SYSTEM_ADDON,
        required_vars=["term", "context"],
    ),
    GRIResponseType.MILESTONE: PromptConfig(
        template=MILESTONE_PROMPT,
        system_addon=MILESTONE_SYSTEM_ADDON,
        required_vars=["milestone_id", "milestone_name", "context"],
    ),
    GRIResponseType.PROCESS: PromptConfig(
        template=PROCESS_PROMPT,
        system_addon="",
        required_vars=["process_name", "context"],
    ),
    GRIResponseType.PHASE_COMPLETE: PromptConfig(
        template=PHASE_PROMPT,
        system_addon="",
        required_vars=[
            "cycle",
            "phase_num",
            "phase_title",
            "entry_milestone",
            "exit_milestone",
            "context",
        ],
    ),
    GRIResponseType.COMPARISON: PromptConfig(
        template=COMPARISON_PROMPT,
        system_addon="",
        required_vars=["entity_a", "entity_b", "context_a", "context_b"],
    ),
    GRIResponseType.GENERAL: PromptConfig(
        template=GENERAL_PROMPT,
        system_addon="",
        required_vars=["query", "context"],
    ),
}


def get_prompt(response_type: GRIResponseType, **kwargs: Any) -> str:
    """Récupère le prompt formaté pour un type de réponse.

    Args:
        response_type: Type de réponse GRI
        **kwargs: Variables à insérer dans le template

    Returns:
        Prompt formaté

    Raises:
        ValueError: Si une variable requise est manquante
    """
    config = PROMPT_CONFIGS[response_type]

    # Vérifier les variables requises
    missing = [v for v in config.required_vars if v not in kwargs]
    if missing:
        raise ValueError(f"Variables manquantes pour {response_type.value}: {missing}")

    return config.template.format(**kwargs)


def get_system_prompt(response_type: GRIResponseType) -> str:
    """Récupère le system prompt pour un type de réponse.

    Args:
        response_type: Type de réponse GRI

    Returns:
        System prompt complet
    """
    config = PROMPT_CONFIGS[response_type]
    return BASE_SYSTEM_PROMPT + config.system_addon


def get_temperature(response_type: GRIResponseType) -> float:
    """Récupère la température pour un type de réponse.

    Args:
        response_type: Type de réponse GRI

    Returns:
        Température (0.0 ou 0.1)
    """
    return TEMPERATURE_MAP[response_type]


def get_max_tokens(response_type: GRIResponseType) -> int:
    """Récupère le max_tokens pour un type de réponse.

    Args:
        response_type: Type de réponse GRI

    Returns:
        Nombre max de tokens
    """
    return MAX_TOKENS_MAP[response_type]


def intent_to_response_type(intent: str) -> GRIResponseType:
    """Convertit un intent GRI en type de réponse.

    Args:
        intent: Intent du query router (DEFINITION, JALON, etc.)

    Returns:
        GRIResponseType correspondant
    """
    mapping = {
        "DEFINITION": GRIResponseType.DEFINITION,
        "JALON": GRIResponseType.MILESTONE,
        "PROCESSUS": GRIResponseType.PROCESS,
        "PHASE_COMPLETE": GRIResponseType.PHASE_COMPLETE,
        "COMPARAISON": GRIResponseType.COMPARISON,
        "CIR": GRIResponseType.MILESTONE,  # CIR = jalons CIR
        "GENERAL": GRIResponseType.GENERAL,
    }
    return mapping.get(intent.upper(), GRIResponseType.GENERAL)
