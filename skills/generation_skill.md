---
name: rag-generation-gri
description: Génération de réponses grounded pour le GRI/FAR avec citations normatives ISO, formatage des définitions ISO/IEC/IEEE 15288 sans paraphrase, gestion des réponses par type de contenu GRI (jalons, processus, phases), et vérification anti-hallucination spécifique au domaine défense/IS. Déclencher pour toute mention de "génération GRI", "synthèse réponse", "citations ISO", "réponse grounded", "formatage jalons", "anti-hallucination GRI", ou "prompt engineering GRI".
---

# Generation Skill — GRI/FAR

## Principe Fondamental

Pour le GRI, la génération est soumise à une contrainte plus forte qu'un RAG classique :
les définitions ISO/IEC/IEEE 15288:2023 et les critères de jalons sont **normatifs**.
Toute paraphrase est une erreur réglementaire dans un contexte défense.

```
Règle d'or GRI :
  Définitions ISO → Copier mot pour mot (temperature=0.0)
  Critères jalons → Lister exhaustivement (ne rien omettre)
  Synthèses       → Grounded avec citations (temperature=0.1)
  Comparaisons    → Structuré par dimension (temperature=0.1)
```

## Prompts par Type de Contenu GRI

### Type DEFINITION

```python
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
# temperature=0.0 OBLIGATOIRE pour ce type
```

### Type JALON

```python
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
# temperature=0.0 OBLIGATOIRE
```

### Type PROCESSUS

```python
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

Temperature=0.1. Citer les termes techniques IS 15288 tels qu'ils apparaissent dans le GRI.
"""
```

### Type PHASE_COMPLETE

```python
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
- Entrée : Jalon {jalon_entree} — [nom]
- Sortie : Jalon {jalon_sortie} — [nom]

**Source : [GRI > Phase {phase_num} : {phase_title}]**
"""
# temperature=0.1
```

### Type COMPARAISON

```python
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
# temperature=0.1
```

## Sélection Automatique du Prompt

```python
# src/agents/generation_agent.py
from huggingface_hub import InferenceClient
import os
from enum import Enum

# Modèle HF pour la génération (doit être capable de suivre des instructions complexes)
HF_GENERATION_MODEL = os.getenv("HF_GENERATION_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")

class GRIResponseType(str, Enum):
    DEFINITION    = "definition"
    MILESTONE     = "milestone"
    PROCESS       = "process"
    PHASE_COMPLETE = "phase_complete"
    COMPARISON    = "comparison"
    GENERAL       = "general"

TEMPERATURE_MAP = {
    GRIResponseType.DEFINITION:     0.0,  # Fidélité normative absolue
    GRIResponseType.MILESTONE:      0.0,  # Critères exhaustifs
    GRIResponseType.PROCESS:        0.1,
    GRIResponseType.PHASE_COMPLETE: 0.1,
    GRIResponseType.COMPARISON:     0.1,
    GRIResponseType.GENERAL:        0.1,
}

MAX_TOKENS_MAP = {
    GRIResponseType.DEFINITION:     256,
    GRIResponseType.MILESTONE:      1024,
    GRIResponseType.PROCESS:        1536,
    GRIResponseType.PHASE_COMPLETE: 2048,
    GRIResponseType.COMPARISON:     2048,
    GRIResponseType.GENERAL:        1024,
}

async def generate_gri_answer(
    query: str,
    chunks: list[dict],
    response_type: GRIResponseType,
    context_vars: dict = None,
) -> dict:
    client = InferenceClient(token=os.getenv("HF_API_KEY"))

    prompt = _select_prompt(response_type, context_vars or {})
    context = _format_gri_context(chunks)
    system_prompt = _build_generation_system(response_type)
    user_content = prompt.format(context=context, **context_vars or {})

    # Construire le prompt complet pour HF (format chat)
    full_prompt = f"<s>[INST] {system_prompt}\n\n{user_content} [/INST]"

    response = await client.text_generation(
        prompt=full_prompt,
        model=HF_GENERATION_MODEL,
        max_new_tokens=MAX_TOKENS_MAP[response_type],
        temperature=TEMPERATURE_MAP[response_type],
        return_full_text=False,
    )

    answer = response  # HF retourne directement le texte généré
    return {
        "answer": answer,
        "response_type": response_type,
        "temperature_used": TEMPERATURE_MAP[response_type],
        "citations": _extract_gri_citations(answer),
        "has_normative_definitions": response_type == GRIResponseType.DEFINITION,
    }
```

## Formatage du Contexte GRI pour le LLM

```python
def _format_gri_context(chunks: list[dict]) -> str:
    """
    Formate les chunks récupérés pour maximiser la compréhension du LLM.
    Ordre : du plus pertinent au moins pertinent.
    Préserver les context prefixes — ils aident le LLM à localiser les infos.
    """
    if not chunks:
        return "⚠️ Aucune source disponible dans la base GRI pour cette query."

    formatted = []
    for i, chunk in enumerate(chunks, 1):
        cycle_tag = chunk.get("cycle", "GRI")
        section_type = chunk.get("section_type", "content")
        score = chunk.get("score", 0)

        header = f"[SOURCE {i}] {cycle_tag} · {section_type.upper()} · Score: {score:.2f}"
        if chunk.get("milestone_id"):
            header += f" · Jalon: {chunk['milestone_id']}"
        if chunk.get("phase_num"):
            header += f" · Phase {chunk['phase_num']}"

        formatted.append(f"{header}\n{chunk.get('content', '')}")

    return "\n\n---\n\n".join(formatted)


def _build_generation_system(response_type: GRIResponseType) -> str:
    base = """Tu es un expert en ingénierie système selon le GRI des FAR (ISO/IEC/IEEE 15288:2023).

RÈGLES DE GÉNÉRATION :
1. Utilise UNIQUEMENT les informations des SOURCES fournies
2. Citations obligatoires au format [GRI > Section > ...] ou [CIR > Phase N > ...]
3. Termes ISO : reproduire le libellé exact du GRI (aucune paraphrase)
4. Si l'information n'est pas dans les sources : dire "Non disponible dans les sources GRI fournies"
5. Ne jamais inventer de critères, jalons, ou livrables"""

    if response_type == GRIResponseType.DEFINITION:
        base += "\n6. DÉFINITIONS : reproduire mot pour mot, aucune reformulation autorisée"
    elif response_type == GRIResponseType.MILESTONE:
        base += "\n6. JALONS : lister TOUS les critères, aucun ne peut être omis ou résumé"

    return base
```

## Détection de Contexte Insuffisant

```python
INSUFFICIENT_CONTEXT_SIGNALS = [
    lambda chunks: len(chunks) == 0,
    lambda chunks: all(c.get("score", 0) < 0.35 for c in chunks),
    lambda chunks: all("definition" not in c.get("content", "").lower()
                       for c in chunks if c.get("section_type") == "definition"),
]

def check_context_sufficiency(chunks: list[dict], response_type: GRIResponseType) -> dict:
    if not chunks:
        return {
            "sufficient": False,
            "reason": "no_chunks",
            "message": "Aucune source pertinente trouvée dans le GRI pour cette query. "
                       "Reformuler avec des termes du GRI ou vérifier que le document est bien indexé."
        }

    max_score = max(c.get("score", 0) for c in chunks)
    if max_score < 0.35:
        return {
            "sufficient": False,
            "reason": "low_scores",
            "message": f"Les sources récupérées ont un score faible (max: {max_score:.2f}). "
                       "La question est peut-être hors du périmètre du GRI."
        }

    if response_type == GRIResponseType.MILESTONE:
        has_milestone_chunk = any(c.get("section_type") == "milestone" for c in chunks)
        if not has_milestone_chunk:
            return {
                "sufficient": False,
                "reason": "missing_milestone_chunk",
                "message": "Critères de jalon non trouvés. Appeler get_milestone_criteria() directement."
            }

    return {"sufficient": True}
```

## Post-Processing GRI

```python
def postprocess_gri_answer(answer: str, response_type: GRIResponseType) -> str:
    """Vérifications post-génération spécifiques au GRI."""
    import re

    # Vérifier que les jalons cités existent dans le GRI
    VALID_MILESTONES = {f"M{i}" for i in range(10)} | {f"J{i}" for i in range(1, 7)}
    cited_milestones = re.findall(r'\b([MJ]\d)\b', answer)
    invalid = set(cited_milestones) - VALID_MILESTONES
    if invalid:
        answer += f"\n\n⚠️ Jalons cités non reconnus dans le GRI : {invalid}. Vérifier."

    # Vérifier que les phases citées sont valides (GRI: 1-7, CIR: 1-4)
    cited_phases = re.findall(r'Phase (\d+)', answer)
    for phase in cited_phases:
        if int(phase) > 7:
            answer += f"\n\n⚠️ Phase {phase} invalide — le GRI standard a 7 phases maximum."

    return answer
```

## Tests Prioritaires

```python
# tests/test_generation_gri.py

async def test_definition_not_paraphrased():
    # La définition retournée doit correspondre mot pour mot au glossaire
    answer = await generate("Définition d'artefact", response_type=DEFINITION)
    glossary_def = glossary.get("artefact")
    assert glossary_def["definition_fr"] in answer["answer"]

async def test_milestone_criteria_exhaustive():
    # Tous les critères du M4 (CDR) doivent être dans la réponse
    answer = await generate("Critères CDR", response_type=MILESTONE)
    expected_criteria_count = 8  # Nombre réel dans le GRI
    found = len(re.findall(r'^\d+\.', answer["answer"], re.MULTILINE))
    assert found >= expected_criteria_count, f"Seulement {found}/{expected_criteria_count} critères"

async def test_citations_present_in_all_answers():
    for query in GOLDEN_QUERIES_SAMPLE:
        answer = await generate(query)
        assert len(answer["citations"]) > 0, f"Pas de citation pour : {query}"

async def test_low_score_chunks_acknowledged():
    # Si tous les scores < 0.35, la réponse doit signaler l'incertitude
    answer = await generate_with_low_score_chunks("Question hors GRI")
    assert "non disponible" in answer["answer"].lower() or \
           "sources" in answer["answer"].lower()

async def test_cir_gri_distinction_in_comparison():
    answer = await generate("Comparer CIR et GRI standard", response_type=COMPARISON)
    assert "CIR" in answer["answer"] and "GRI" in answer["answer"]
    assert "[CIR" in str(answer["citations"]) or "[GRI" in str(answer["citations"])
```
