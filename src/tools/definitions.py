"""Tool Definitions - Schemas JSON des 5 tools GRI pour le LLM.

Ces définitions sont utilisées par l'orchestrateur pour présenter
les outils disponibles au LLM. Chaque tool a :
- name: identifiant unique
- description: usage et contraintes pour le LLM
- input_schema: schema JSON des paramètres

Usage:
    from src.tools.definitions import TOOLS, get_tool_by_name

    # Tous les tools
    for tool in TOOLS:
        print(tool["name"])

    # Un tool spécifique
    tool = get_tool_by_name("retrieve_gri_chunks")
"""

from typing import Any


# === Tool 1: retrieve_gri_chunks ===
RETRIEVE_GRI_CHUNKS = {
    "name": "retrieve_gri_chunks",
    "description": """Recherche des passages pertinents dans la base GRI.
Utilise la recherche hybride (dense + BM25) avec filtres par type de section.

QUAND utiliser :
- Pour toute question sur des processus, phases, principes, approches
- Après lookup_gri_glossary pour le contexte élargi

NE PAS utiliser :
- Pour les définitions de termes -> utiliser lookup_gri_glossary
- Pour les critères de jalons complets -> utiliser get_milestone_criteria
- Pour résumer une phase entière -> utiliser get_phase_summary

Scores de confiance : > 0.7 très pertinent, 0.4-0.7 pertinent, < 0.4 insuffisant""",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "La question ou le concept à rechercher. Être spécifique.",
            },
            "section_type": {
                "type": "string",
                "enum": [
                    "definition",
                    "principle",
                    "phase",
                    "milestone",
                    "process",
                    "cir",
                    "table",
                    "content",
                ],
                "description": "Filtrer par type de section GRI",
            },
            "cycle": {
                "type": "string",
                "enum": ["GRI", "CIR", "BOTH"],
                "default": "GRI",
                "description": "Filtrer par cycle",
            },
            "phase_num": {
                "type": "integer",
                "minimum": 1,
                "maximum": 7,
                "description": "Filtrer par numéro de phase GRI (1-7)",
            },
            "n_results": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "maximum": 15,
                "description": "Nombre de résultats à retourner",
            },
        },
        "required": ["query"],
    },
}


# === Tool 2: lookup_gri_glossary ===
LOOKUP_GRI_GLOSSARY = {
    "name": "lookup_gri_glossary",
    "description": """Cherche la définition exacte d'un terme dans le glossaire GRI.
Retourne la définition normative FR + EN selon ISO/IEC/IEEE 15288:2023.

TOUJOURS utiliser EN PREMIER quand :
- La question contient un terme technique GRI (artefact, CONOPS, SEMP, TRL, etc.)
- La question demande explicitement une définition
- Le terme utilisé pourrait avoir un sens différent du sens courant

La définition retournée DOIT être citée telle quelle dans la réponse finale (terme normé).""",
    "input_schema": {
        "type": "object",
        "properties": {
            "term": {
                "type": "string",
                "description": "Le terme à définir (en français de préférence, ou anglais ISO)",
            },
            "return_both_languages": {
                "type": "boolean",
                "default": True,
                "description": "Retourner la définition en FR ET EN",
            },
        },
        "required": ["term"],
    },
}


# === Tool 3: get_milestone_criteria ===
GET_MILESTONE_CRITERIA = {
    "name": "get_milestone_criteria",
    "description": """Récupère la checklist COMPLÈTE des critères de passage d'un jalon.
Pour le GRI standard : M0 à M9.
Pour le CIR : J1 à J6 (avec mapping automatique vers les jalons GRI équivalents).

TOUJOURS utiliser pour toute question sur les critères d'un jalon spécifique.
Retourne les critères structurés sous forme de liste, jamais tronqués.

Mapping CIR -> GRI automatique :
- J1 -> M0 + M1
- J2 -> M2 + M3 + M4
- J3 -> M5 + M6
- J4, J5 -> SAR
- J6 -> M8""",
    "input_schema": {
        "type": "object",
        "properties": {
            "milestone_id": {
                "type": "string",
                "description": "ID du jalon : M0-M9 (GRI) ou J1-J6 (CIR). Accepte aussi CDR, PDR, SRR, etc.",
                "pattern": "^[MJ][0-9]$|^(CDR|PDR|SRR|IRR|TRR|SAR|ORR|MNR|MNS)$",
            },
            "include_gri_mapping": {
                "type": "boolean",
                "default": True,
                "description": "Pour les jalons CIR, inclure automatiquement les jalons GRI équivalents",
            },
        },
        "required": ["milestone_id"],
    },
}


# === Tool 4: compare_approaches ===
COMPARE_APPROACHES = {
    "name": "compare_approaches",
    "description": """Compare deux éléments du GRI (phases, approches, cycles, processus).
Effectue un retrieval parallèle sur les deux entités et prépare un contexte comparatif structuré.

QUAND utiliser :
- "Différence entre approche séquentielle et CIR"
- "Comparer le modèle en V et DevSecOps"
- "GRI standard vs Cycle d'Innovation Rapide"
- "Phase 2 vs Phase 3"

Retourne les informations des deux entités avec les points de convergence et de divergence.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "entity_a": {
                "type": "string",
                "description": "Premier élément à comparer (ex: 'approche séquentielle', 'Phase 3')",
            },
            "entity_b": {
                "type": "string",
                "description": "Deuxième élément à comparer (ex: 'CIR', 'Phase 4')",
            },
            "comparison_dimensions": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Dimensions de comparaison (ex: ['durée', 'jalons', 'livrables', 'risques'])",
                "default": [],
            },
        },
        "required": ["entity_a", "entity_b"],
    },
}


# === Tool 5: get_phase_summary ===
GET_PHASE_SUMMARY = {
    "name": "get_phase_summary",
    "description": """Récupère un résumé structuré complet d'une phase du GRI ou du CIR.
Utilise le Parent Document Retriever pour retourner les objectifs, activités, livrables et jalons.

QUAND utiliser :
- "Résume la Phase 3 du GRI"
- "Quels sont les objectifs de la phase d'idéation ?"
- "Qu'est-ce que la phase d'intégration et vérification ?"

Retourne : objectifs généraux + objectifs spécifiques + produits de la phase + critères de jalons.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "phase_num": {
                "type": "integer",
                "minimum": 1,
                "maximum": 7,
                "description": "Numéro de phase : 1-7 pour GRI, 1-4 pour CIR",
            },
            "cycle": {
                "type": "string",
                "enum": ["GRI", "CIR"],
                "default": "GRI",
                "description": "Cycle de vie concerné",
            },
            "include_deliverables": {
                "type": "boolean",
                "default": True,
                "description": "Inclure la liste des livrables",
            },
            "include_milestone_criteria": {
                "type": "boolean",
                "default": False,
                "description": "Inclure les critères de jalons (rend la réponse plus longue)",
            },
        },
        "required": ["phase_num"],
    },
}


# === Liste complète des tools ===
TOOLS: list[dict[str, Any]] = [
    RETRIEVE_GRI_CHUNKS,
    LOOKUP_GRI_GLOSSARY,
    GET_MILESTONE_CRITERIA,
    COMPARE_APPROACHES,
    GET_PHASE_SUMMARY,
]


# === Helpers ===
def get_tool_by_name(name: str) -> dict[str, Any] | None:
    """Retourne un tool par son nom.

    Args:
        name: Nom du tool

    Returns:
        Définition du tool ou None si non trouvé
    """
    for tool in TOOLS:
        if tool["name"] == name:
            return tool
    return None


def get_tool_names() -> list[str]:
    """Retourne la liste des noms de tools disponibles."""
    return [tool["name"] for tool in TOOLS]


def format_tools_for_prompt() -> str:
    """Formate les tools pour injection dans un prompt.

    Returns:
        Texte formaté décrivant les tools disponibles
    """
    lines = ["## Outils disponibles\n"]

    for tool in TOOLS:
        lines.append(f"### {tool['name']}")
        lines.append(tool["description"])
        lines.append("")

        # Paramètres
        schema = tool["input_schema"]
        props = schema.get("properties", {})
        required = schema.get("required", [])

        lines.append("**Paramètres :**")
        for param_name, param_def in props.items():
            req_marker = "(requis)" if param_name in required else "(optionnel)"
            param_type = param_def.get("type", "any")
            param_desc = param_def.get("description", "")
            lines.append(f"- `{param_name}` ({param_type}) {req_marker}: {param_desc}")

        lines.append("")

    return "\n".join(lines)
