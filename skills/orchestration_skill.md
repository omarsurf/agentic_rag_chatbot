---
name: rag-orchestration-gri
description: Orchestrateur agentique ReAct spécialisé pour le GRI des FAR. Couvre la boucle de raisonnement multi-étapes, les 5 tools domaine-spécifiques, la détection GRI vs CIR, la décomposition de questions complexes, et la gestion de la mémoire conversationnelle. Déclencher pour toute mention de "agent GRI", "ReAct loop", "orchestrateur", "multi-step retrieval GRI", "tools spécialisés", "boucle agent", "planification requête GRI", ou "décomposition question".
---

# Orchestration Skill — GRI/FAR

## Philosophie de l'Orchestrateur GRI

L'agent ne doit jamais répondre en one-shot. Pour le GRI, chaque question exige :
1. **Identifier le cycle** : GRI standard (7 phases, M0-M9) ou CIR (4 phases, J1-J6) ?
2. **Classifier l'intent** : parmi 6 types de questions GRI
3. **Planifier** : quels outils appeler, dans quel ordre ?
4. **Retriever** : de façon ciblée avec les bons filtres metadata
5. **Valider** : les chunks récupérés sont-ils suffisants et cohérents ?
6. **Synthétiser** : avec citations `[GRI Section X.Y]` obligatoires

## System Prompt Complet de l'Orchestrateur

```python
ORCHESTRATOR_SYSTEM = """Tu es un expert en ingénierie système et gestion de l'innovation selon le GRI des FAR.
Sources autorisées : GRI/FAR (ISO/IEC/IEEE 15288:2023) et INCOSE Systems Engineering Handbook 5e éd.

## PROCESSUS DE RAISONNEMENT OBLIGATOIRE

### ÉTAPE 1 : IDENTIFICATION DU CYCLE
- La question concerne-t-elle le GRI standard (7 phases, jalons M0-M9) ?
- Ou le CIR - Cycle d'Innovation Rapide (4 phases, jalons J1-J6) ?
- Ou les deux (question comparative) ?

### ÉTAPE 2 : CLASSIFICATION DE L'INTENT
Identifie parmi : DEFINITION / PROCESSUS / JALON / PHASE_COMPLETE / COMPARAISON / CIR

### ÉTAPE 3 : PLANIFICATION DES OUTILS
- DEFINITION → lookup_gri_glossary (toujours en premier pour les termes)
- PROCESSUS → retrieve_gri_chunks avec filter section_type='process'
- JALON → get_milestone_criteria (retourne la checklist complète)
- PHASE_COMPLETE → get_phase_summary (retourne objectifs + livrables + jalons)
- COMPARAISON → compare_approaches (multi-retrieve sur les deux entités)
- CIR → retrieve_gri_chunks avec cycle='CIR' + get_milestone_criteria pour le mapping

### ÉTAPE 4 : VALIDATION DES RÉSULTATS
Après chaque tool call, évalue :
- Les informations récupérées répondent-elles à la question ?
- Y a-t-il des contradictions entre sources ?
- Manque-t-il une information critique (ex: critères de jalon incomplets) ?
Si insuffisant → reformuler la query et recommencer (max 2 reformulations par tool)

### ÉTAPE 5 : SYNTHÈSE GROUNDED
- Utilise UNIQUEMENT les informations des chunks récupérés
- Citation obligatoire pour chaque affirmation : [GRI Section X.Y] ou [CIR Phase N, Jalon JN]
- Pour les définitions ISO : reproduire exactement le libellé du GRI (temperature=0 mentale)
- Pour les jalons : lister TOUS les critères, jamais en résumer certains

## RÈGLES ABSOLUES
1. Ne jamais inventer de critères, de jalons, ou de livrables non présents dans les sources
2. Les définitions ISO/IEC/IEEE 15288:2023 sont normatives : ne pas paraphraser
3. Si GRI et CIR sont mentionnés dans la même question → distinguer clairement les deux
4. Maximum {max_iter} itérations de retrieval par question
5. Si un terme GRI est utilisé sans définition connue → toujours appeler lookup_gri_glossary d'abord

## FORMAT DE CITATION
[GRI > Terminologie > Terme]
[GRI > Principe N°X > Titre]
[GRI > Phase N > Titre Phase > Sous-section]
[GRI > Jalon MN (Nom) > Critère #X]
[CIR > Phase N > Jalon JN > Critère]
[GRI > Processus IS 15288 > Nom Processus > Activité/Input/Output]
"""
```

## Les 5 Tools — Définitions Complètes

```python
# src/tools/definitions.py

TOOLS = [

    {
        "name": "retrieve_gri_chunks",
        "description": """Recherche des passages pertinents dans la base GRI.
Utilise la recherche hybride (dense + BM25) avec filtres par type de section.

QUAND utiliser :
- Pour toute question sur des processus, phases, principes, approches
- Après lookup_gri_glossary pour le contexte élargi

NE PAS utiliser :
- Pour les définitions de termes → utiliser lookup_gri_glossary
- Pour les critères de jalons complets → utiliser get_milestone_criteria
- Pour résumer une phase entière → utiliser get_phase_summary

Scores de confiance : > 0.7 très pertinent, 0.4-0.7 pertinent, < 0.4 insuffisant""",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "La question ou le concept à rechercher. Être spécifique."
                },
                "section_type": {
                    "type": "string",
                    "enum": ["definition", "principle", "phase", "milestone",
                             "process", "cir", "table", "content"],
                    "description": "Filtrer par type de section GRI"
                },
                "cycle": {
                    "type": "string",
                    "enum": ["GRI", "CIR", "BOTH"],
                    "default": "GRI"
                },
                "phase_num": {
                    "type": "integer",
                    "minimum": 1, "maximum": 7,
                    "description": "Filtrer par numéro de phase GRI (1-7)"
                },
                "n_results": {
                    "type": "integer",
                    "default": 5, "minimum": 1, "maximum": 15
                },
            },
            "required": ["query"]
        }
    },

    {
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
                    "description": "Le terme à définir (en français de préférence, ou anglais ISO)"
                },
                "return_both_languages": {
                    "type": "boolean",
                    "default": True,
                    "description": "Retourner la définition en FR ET EN"
                }
            },
            "required": ["term"]
        }
    },

    {
        "name": "get_milestone_criteria",
        "description": """Récupère la checklist COMPLÈTE des critères de passage d'un jalon.
Pour le GRI standard : M0 à M9.
Pour le CIR : J1 à J6 (avec mapping automatique vers les jalons GRI équivalents).

TOUJOURS utiliser pour toute question sur les critères d'un jalon spécifique.
Retourne les critères structurés sous forme de liste, jamais tronqués.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "milestone_id": {
                    "type": "string",
                    "description": "ID du jalon : M0, M1, M2, M3, M4, M5, M6, M7, M8, M9 (GRI) ou J1, J2, J3, J4, J5, J6 (CIR)",
                    "pattern": "^[MJ][0-9]$"
                },
                "include_gri_mapping": {
                    "type": "boolean",
                    "default": True,
                    "description": "Pour les jalons CIR, inclure automatiquement les jalons GRI équivalents"
                }
            },
            "required": ["milestone_id"]
        }
    },

    {
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
                    "description": "Premier élément à comparer (ex: 'approche séquentielle', 'Phase 3')"
                },
                "entity_b": {
                    "type": "string",
                    "description": "Deuxième élément à comparer (ex: 'CIR', 'Phase 4')"
                },
                "comparison_dimensions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Dimensions de comparaison (ex: ['durée', 'jalons', 'livrables', 'risques'])"
                }
            },
            "required": ["entity_a", "entity_b"]
        }
    },

    {
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
                    "description": "Numéro de phase : 1-7 pour GRI, 1-4 pour CIR"
                },
                "cycle": {
                    "type": "string",
                    "enum": ["GRI", "CIR"],
                    "default": "GRI"
                },
                "include_deliverables": {
                    "type": "boolean",
                    "default": True
                },
                "include_milestone_criteria": {
                    "type": "boolean",
                    "default": False,
                    "description": "Inclure les critères de jalons (rend la réponse plus longue)"
                }
            },
            "required": ["phase_num"]
        }
    },
]
```

## Boucle ReAct — Implémentation

```python
# src/agents/orchestrator.py
from huggingface_hub import InferenceClient
import asyncio
import json
import os
import structlog
from .query_router import route_query
from ..core.memory import GRIMemory
from ..tools import execute_tool

log = structlog.get_logger()

# Modèle HF principal pour l'orchestration (doit supporter function calling ou être assez capable)
HF_ORCHESTRATOR_MODEL = os.getenv("HF_ORCHESTRATOR_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")

class GRIOrchestrator:

    def __init__(self, store, memory: GRIMemory, max_iter: int = 5):
        self.client = InferenceClient(token=os.getenv("HF_API_KEY"))
        self.store = store
        self.memory = memory
        self.max_iter = max_iter
        self.model = HF_ORCHESTRATOR_MODEL

    async def run(self, query: str) -> dict:
        log.info("gri_agent.start", query=query[:120])

        # Pré-step : routing intent + enrichissement terminologique
        routing   = await route_query(query, self.client)
        term_ctx  = await self._get_term_context(query)
        conv_ctx  = self.memory.get_context()

        system = ORCHESTRATOR_SYSTEM.format(max_iter=self.max_iter)
        if term_ctx:
            system += f"\n\n## Contexte terminologique GRI\n{term_ctx}"
        if conv_ctx:
            system += f"\n\n{conv_ctx}"

        messages = [{"role": "user", "content": query}]
        stats = {"iterations": 0, "tool_calls": [], "tokens": 0, "routing": routing}

        for i in range(self.max_iter):
            stats["iterations"] += 1

            # Construire le prompt complet avec system + messages pour HF
            full_prompt = self._build_prompt(system, messages, TOOLS)
            response = await self.client.text_generation(
                prompt=full_prompt,
                model=self.model,
                max_new_tokens=4096,
                temperature=0.1,
                return_full_text=False,
            )

            stats["tokens"] += response.usage.input_tokens + response.usage.output_tokens
            log.info("gri_agent.step", iter=i, stop=response.stop_reason, tokens=stats["tokens"])

            # Réponse finale
            if response.stop_reason == "end_turn":
                answer = self._extract_text(response)
                citations = self._extract_citations(answer)
                self.memory.add_turn(query, answer)
                return {
                    "answer": answer,
                    "citations": citations,
                    "intent": routing["intent"],
                    "cycle": routing["cycle"],
                    **stats,
                }

            # Tool calls — en parallèle si indépendants
            tool_blocks = [b for b in response.content if b.type == "tool_use"]
            tool_results = await asyncio.gather(*[
                self._exec_tool(b) for b in tool_blocks
            ])

            stats["tool_calls"] += [
                {"tool": b.name, "input": b.input, "iter": i}
                for b in tool_blocks
            ]

            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(r)}
                    for b, r in zip(tool_blocks, tool_results)
                ]
            })

        # Max iterations
        log.warning("gri_agent.max_iter", query=query[:80])
        return {
            "answer": self._extract_text(response) or "Contexte insuffisant pour répondre.",
            "warning": "max_iterations_reached",
            **stats,
        }

    async def _exec_tool(self, block) -> dict:
        try:
            return await execute_tool(block.name, block.input, self.store)
        except Exception as e:
            log.error("tool.error", tool=block.name, error=str(e))
            return {"error": str(e), "tool": block.name}

    def _extract_text(self, response) -> str:
        return " ".join(b.text for b in response.content if hasattr(b, "text"))

    def _extract_citations(self, text: str) -> list[dict]:
        import re
        pattern = r'\[GRI[^\]]+\]|\[CIR[^\]]+\]'
        return [{"citation": m} for m in re.findall(pattern, text)]

    async def _get_term_context(self, query: str) -> str:
        from ..core.term_expander import expand_query_with_terms
        _, ctx = await expand_query_with_terms(query, self.store)
        return ctx
```

## Scénarios de Raisonnement

### Scénario 1 : Question sur un jalon
```
Query : "Quels sont les critères du CDR ?"
→ Route : JALON, GRI
→ Plan  : get_milestone_criteria(milestone_id="M4")
→ 1 tool call → réponse avec liste complète des critères
→ Citer : [GRI > Jalon M4 (CDR) > Critère #N]
```

### Scénario 2 : Définition + contexte
```
Query : "Comment le GRI définit le CONOPS et dans quelle phase l'utilise-t-on ?"
→ Route : DEFINITION + PHASE, GRI
→ Plan  : lookup_gri_glossary("CONOPS")
        → retrieve_gri_chunks("CONOPS utilisation phase", section_type="phase")
→ 2 tool calls séquentiels
→ Citer : [GRI > Terminologie > CONOPS] + [GRI > Phase 1 > Idéation]
```

### Scénario 3 : Comparaison GRI / CIR
```
Query : "Quelle est la différence entre le CDR du GRI et le jalon J2 du CIR ?"
→ Route : COMPARAISON, BOTH
→ Plan  : get_milestone_criteria("M4") en parallèle avec get_milestone_criteria("J2")
→ 2 tool calls parallèles
→ compare_approaches("M4 CDR", "J2 CIR") si insuffisant
→ Citer les deux cycles distinctement
```

### Scénario 4 : Question hors-GRI
```
Query : "Quel est le meilleur framework agile ?"
→ Route : hors-scope
→ Répondre : "Cette question dépasse le périmètre du GRI. Je peux vous informer
  sur les approches décrites dans le GRI (séquentielle, incrémentale, DevSecOps)."
→ Proposer : retrieve_gri_chunks("approches cycle de vie agile DevSecOps")
```

## Tests Prioritaires

```python
# tests/test_orchestrator_gri.py

async def test_jalon_question_one_tool_call():
    result = await agent.run("Critères du CDR M4")
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["tool"] == "get_milestone_criteria"
    assert result["tool_calls"][0]["input"]["milestone_id"] == "M4"

async def test_definition_first_tool_call():
    result = await agent.run("Qu'est-ce qu'un artefact selon le GRI ?")
    assert result["tool_calls"][0]["tool"] == "lookup_gri_glossary"

async def test_cir_includes_gri_mapping():
    result = await agent.run("Critères du jalon J2 du CIR")
    assert result["cycle"] in ["CIR", "BOTH"]
    assert any("M2" in str(tc) or "M3" in str(tc) for tc in result["tool_calls"])

async def test_citations_format():
    result = await agent.run("Objectifs de la Phase 3 du GRI")
    assert any("[GRI" in c["citation"] for c in result["citations"])

async def test_out_of_scope_refused():
    result = await agent.run("Quel est le meilleur framework Python ?")
    assert "GRI" in result["answer"] or "périmètre" in result["answer"].lower()

async def test_max_iter_not_exceeded():
    result = await agent.run("Question très vague sans réponse directe")
    assert result["iterations"] <= 5
```
