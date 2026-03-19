---
name: rag-evaluation-gri
description: Suite d'évaluation complète pour le RAG GRI/FAR — métriques RAGAS adaptées, Term Accuracy ISO spécifique au domaine normatif, golden dataset 50 questions GRI, LLM-as-judge avec contexte IS 15288, et quality gates production. Déclencher pour toute mention de "évaluer le RAG GRI", "golden dataset", "faithfulness GRI", "Term Accuracy", "RAGAS", "quality gates", "benchmark RAG", "mesurer la qualité", ou "LLM judge GRI".
---

# Evaluation Skill — GRI/FAR

## Métriques Spécifiques au GRI

En plus des 4 métriques RAGAS classiques, le GRI exige une 5e métrique critique.

| Métrique | Cible | Bloquant | Spécificité GRI |
|---------|-------|---------|----------------|
| **Faithfulness** | ≥ 0.85 | ✅ OUI | Claims vérifiés contre le GRI, pas internet |
| **Answer Relevance** | ≥ 0.80 | ✅ OUI | Réponse GRI-pertinente (pas générique) |
| **Context Recall** | ≥ 0.75 | ✅ OUI | Les chunks couvrent-ils la vraie réponse GRI ? |
| **Context Precision** | ≥ 0.70 | Non | % chunks GRI pertinents dans le top-K |
| **Term Accuracy** | ≥ 0.95 | ✅ OUI critique | Définitions ISO citées mot pour mot |
| Latency P95 | ≤ 8 000ms | ✅ OUI | Mesure end-to-end |

**Term Accuracy** est la métrique la plus importante pour ce projet : une définition ISO paraphrasée est une erreur réglementaire dans un contexte défense.

## Term Accuracy — Métrique Custom GRI

```python
# src/evaluation/term_accuracy.py
from huggingface_hub import InferenceClient
import os
import json

# Modèle HF pour l'évaluation (doit être capable d'analyse fine)
HF_EVAL_MODEL = os.getenv("HF_EVAL_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")

TERM_ACCURACY_PROMPT = """Vérifie si les définitions de termes ISO/GRI dans cette réponse 
correspondent exactement aux définitions normatives du glossaire GRI.

GLOSSAIRE DE RÉFÉRENCE (définitions normatives) :
{glossary_context}

RÉPONSE À ÉVALUER :
{answer}

Pour chaque terme ISO/GRI mentionné dans la réponse :
1. Identifier le terme et sa définition dans la réponse
2. Comparer avec la définition normative du glossaire
3. Évaluer : EXACT / APPROXIMATIF / INCORRECT / NON_TROUVÉ

Format JSON :
{{
  "term_evaluations": [
    {{
      "term": "...",
      "definition_in_answer": "...",
      "normative_definition": "...",
      "status": "EXACT|APPROXIMATIF|INCORRECT|NON_TROUVÉ",
      "severity": "CRITIQUE|MINEUR|OK"
    }}
  ],
  "term_accuracy_score": 0.0-1.0,
  "critical_errors": ["..."]
}}

Scoring :
- EXACT → 1.0 par terme
- APPROXIMATIF → 0.5 par terme (acceptable si le sens est préservé)
- INCORRECT → 0.0 par terme (sens différent de la norme)
- NON_TROUVÉ → N/A (terme présent dans la réponse mais pas dans le glossaire fourni)

Score final = (Σ scores) / (nb termes évalués hors NON_TROUVÉ)"""

async def compute_term_accuracy(
    answer: str,
    glossary_index,       # Index des définitions GRI
    client: InferenceClient = None,
) -> dict:
    if client is None:
        client = InferenceClient(token=os.getenv("HF_API_KEY"))
    # Extraire les termes potentiellement ISO dans la réponse
    iso_terms = _extract_iso_terms(answer)
    if not iso_terms:
        return {"term_accuracy_score": 1.0, "term_evaluations": [], "no_terms_found": True}

    # Récupérer les définitions normatives pour ces termes
    glossary_context = []
    for term in iso_terms[:10]:  # Max 10 termes par évaluation
        defn = await glossary_index.lookup(term)
        if defn:
            glossary_context.append(
                f"**{defn['term_fr']}** ({defn['term_en']}): {defn['definition_fr']}"
                f"\n  Source: {defn.get('standard_ref', 'GRI')}"
            )

    if not glossary_context:
        return {"term_accuracy_score": 1.0, "term_evaluations": [], "no_normative_terms": True}

    prompt_content = TERM_ACCURACY_PROMPT.format(
        glossary_context="\n".join(glossary_context),
        answer=answer
    )
    full_prompt = f"<s>[INST] {prompt_content} [/INST]"

    response = await client.text_generation(
        prompt=full_prompt,
        model=HF_EVAL_MODEL,
        max_new_tokens=1024,
        temperature=0.1,
        return_full_text=False,
    )

    return json.loads(response)


def _extract_iso_terms(text: str) -> list[str]:
    """Détecte les termes GRI/ISO dans le texte."""
    import re
    GRI_TERMS_PATTERN = [
        r'\b(artefact|artifact)\b',
        r'\b(CONOPS|SEMP|SRR|PDR|CDR|IRR|TRR|SAR|ORR|MNR)\b',
        r'\b(TRL|MRL|IRL)\b',
        r'\b(cycle de vie|lifecycle)\b',
        r'\b(jalons? décisionnel)\b',
        r'\b(vérification|validation|intégration)\b',
        r'\b(ingénierie système|systems? engineering)\b',
        r'\b(parties prenantes?|stakeholders?)\b',
        r'\b(traçabilité|traceability)\b',
        r'\b(exigences? système|system requirements?)\b',
        r'\b(architecture système|system architecture)\b',
    ]
    found = []
    for pattern in GRI_TERMS_PATTERN:
        matches = re.findall(pattern, text, re.IGNORECASE)
        found.extend([m if isinstance(m, str) else m[0] for m in matches])
    return list(set(found))
```

## Faithfulness Adaptée au GRI

```python
# src/evaluation/faithfulness_gri.py

FAITHFULNESS_GRI_PROMPT = """Évalue si cette réponse sur le GRI/FAR est factualmente supportée 
par les sources GRI fournies.

Sources GRI :
{context}

Réponse à évaluer :
{answer}

Pour chaque affirmation factuelle de la réponse :
1. Identifier l'affirmation
2. Chercher le support dans les sources GRI
3. Classifier : SUPPORTÉE / INFÉRÉE / INVENTÉE / HORS_PÉRIMÈTRE

Attention aux erreurs GRI-spécifiques :
- Jalon inventé (ex: citer "M10" qui n'existe pas)
- Critère de jalon non présent dans les sources
- Mauvais mapping CIR↔GRI (ex: dire que J3 équivaut à M4 alors que c'est M5+M6)
- Phase inexistante (ex: "Phase 8 du GRI")
- Durée inventée non mentionnée dans les sources

Format JSON :
{{
  "claims": [
    {{
      "claim": "...",
      "status": "SUPPORTÉE|INFÉRÉE|INVENTÉE|HORS_PÉRIMÈTRE",
      "evidence": "...",
      "gri_error_type": "jalon_inexistant|mauvais_mapping|phase_inexistante|durée_inventée|null"
    }}
  ],
  "faithfulness_score": 0.0-1.0,
  "gri_specific_errors": ["..."]
}}"""

async def compute_faithfulness_gri(answer: str, context_chunks: list[str], client: InferenceClient = None) -> dict:
    if client is None:
        client = InferenceClient(token=os.getenv("HF_API_KEY"))

    context = "\n---\n".join(context_chunks)
    prompt_content = FAITHFULNESS_GRI_PROMPT.format(context=context, answer=answer)
    full_prompt = f"<s>[INST] {prompt_content} [/INST]"

    response = await client.text_generation(
        prompt=full_prompt,
        model=HF_EVAL_MODEL,
        max_new_tokens=1024,
        temperature=0.1,
        return_full_text=False,
    )
    return json.loads(response)
```

## Golden Dataset GRI — Structure et Questions

```python
# data/golden_dataset.json — Minimum 50 questions

GOLDEN_DATASET_STRUCTURE = {
    "metadata": {
        "source_doc": "IRF20251211_last_FF.docx",
        "version": "1.0",
        "created_at": "2025-12-11",
        "n_questions": 50
    },
    "questions": [
        # ── DÉFINITIONS (10 questions) ─────────────────────────
        {
            "id": "DEF_001",
            "query": "Qu'est-ce qu'un artefact selon le GRI ?",
            "ground_truth": "Produit ou livrable élaboré et utilisé au cours d'un projet "
                           "pour capter et transmettre de l'information. "
                           "(Artifact: Work product that is produced and used during a project "
                           "to capture and convey information.) [ISO/IEC/IEEE 15288:2023]",
            "expected_intent": "DEFINITION",
            "expected_cycle": "GRI",
            "required_sources": ["definition"],
            "difficulty": "easy",
            "evaluation_focus": "term_accuracy"
        },
        {
            "id": "DEF_002",
            "query": "Définition de CONOPS dans le GRI",
            "ground_truth": "[Extraire du document GRI]",
            "expected_intent": "DEFINITION",
            "difficulty": "easy",
        },
        # ... 8 autres définitions

        # ── PROCESSUS IS 15288 (10 questions) ──────────────────
        {
            "id": "PROC_001",
            "query": "Quelles sont les activités du processus de vérification selon le GRI ?",
            "ground_truth": "[Extraire du document GRI — Process de vérification]",
            "expected_intent": "PROCESSUS",
            "expected_cycle": "GRI",
            "required_sources": ["process"],
            "difficulty": "medium",
            "evaluation_focus": "context_recall"
        },
        # ... 9 autres processus

        # ── JALONS (10 questions) ───────────────────────────────
        {
            "id": "JAL_001",
            "query": "Quels sont les critères de passage du CDR (M4) ?",
            "ground_truth": "[Extraire du document GRI — Tous les critères du M4]",
            "expected_intent": "JALON",
            "expected_cycle": "GRI",
            "required_milestone": "M4",
            "required_sources": ["milestone"],
            "difficulty": "medium",
            "evaluation_focus": "faithfulness",
            "critical_check": "all_criteria_listed"  # Vérifier exhaustivité
        },
        {
            "id": "JAL_002",
            "query": "Critères de passage du jalon J3 du CIR",
            "ground_truth": "[Extraire du document GRI — CIR J3 + mapping M5/M6]",
            "expected_intent": "JALON",
            "expected_cycle": "CIR",
            "required_milestone": "J3",
            "critical_check": "includes_gri_mapping"
        },
        # ... 8 autres jalons (mix GRI + CIR)

        # ── PHASES COMPLÈTES (8 questions) ─────────────────────
        {
            "id": "PHA_001",
            "query": "Quels sont les objectifs de la Phase 1 — Idéation du GRI ?",
            "ground_truth": "[Extraire — Phase 1 Objectifs]",
            "expected_intent": "PHASE_COMPLETE",
            "required_phase": 1,
            "difficulty": "medium",
        },
        # ... 7 autres phases

        # ── COMPARAISONS (7 questions) ──────────────────────────
        {
            "id": "COMP_001",
            "query": "Quelle est la différence entre le GRI standard et le CIR ?",
            "ground_truth": "[Synthèse des différences phases/jalons/durées/risques]",
            "expected_intent": "COMPARAISON",
            "expected_cycle": "BOTH",
            "difficulty": "hard",
            "evaluation_focus": "answer_relevance"
        },
        {
            "id": "COMP_002",
            "query": "Comparer l'approche séquentielle et l'approche DevSecOps selon le GRI",
            "ground_truth": "[Synthèse GRI Section Approches]",
            "expected_intent": "COMPARAISON",
            "difficulty": "hard",
        },
        # ... 5 autres comparaisons

        # ── CIR SPÉCIFIQUE (5 questions) ───────────────────────
        {
            "id": "CIR_001",
            "query": "Dans quel contexte le CIR s'applique-t-il selon le GRI ?",
            "ground_truth": "[Extraire — Contexte d'application du CIR]",
            "expected_intent": "CIR",
            "difficulty": "easy",
        },
        # ... 4 autres CIR
    ]
}
```

## Pipeline d'Évaluation Complète

```python
# src/evaluation/pipeline.py
import asyncio
import json
from dataclasses import dataclass, asdict
from datetime import datetime

@dataclass
class GRIEvalResult:
    question_id: str
    query: str
    answer: str
    faithfulness: float
    faithfulness_gri_errors: list
    answer_relevance: float
    context_recall: float
    context_precision: float
    term_accuracy: float
    term_critical_errors: list
    latency_ms: float
    intent_correct: bool
    cycle_correct: bool
    milestone_complete: bool   # Pour les questions de type JALON

class GRIEvaluator:

    QUALITY_GATES = {
        "faithfulness":     0.85,
        "answer_relevance": 0.80,
        "context_recall":   0.75,
        "context_precision":0.70,
        "term_accuracy":    0.95,  # CRITIQUE
        "latency_p95_ms":   8000,
    }

    async def evaluate_dataset(self, dataset: list[dict], rag_system) -> dict:
        results = []
        for item in dataset:
            result = await self._evaluate_one(item, rag_system)
            results.append(result)

        summary = self._aggregate(results)
        failures = self._check_quality_gates(summary)

        return {
            "timestamp": datetime.now().isoformat(),
            "n_evaluated": len(results),
            "summary": summary,
            "quality_gates_passed": len(failures) == 0,
            "failures": failures,
            "results": [asdict(r) for r in results],
        }

    async def _evaluate_one(self, item: dict, rag_system) -> GRIEvalResult:
        import time
        client = InferenceClient(token=os.getenv("HF_API_KEY"))

        start = time.time()
        response = await rag_system.run(item["query"])
        latency = (time.time() - start) * 1000

        answer = response["answer"]
        chunks_text = [c.get("content", "") for c in response.get("chunks_used", [])]

        # Métriques parallèles
        faithfulness, relevance, term_acc = await asyncio.gather(
            compute_faithfulness_gri(answer, chunks_text, client),
            compute_answer_relevance(item["query"], answer),
            compute_term_accuracy(answer, glossary_index, client),
        )

        recall = 0.0
        if item.get("ground_truth"):
            recall = await compute_context_recall(item["ground_truth"], chunks_text, client)

        return GRIEvalResult(
            question_id=item["id"],
            query=item["query"],
            answer=answer,
            faithfulness=faithfulness["faithfulness_score"],
            faithfulness_gri_errors=faithfulness.get("gri_specific_errors", []),
            answer_relevance=relevance,
            context_recall=recall,
            context_precision=0.0,  # Calculer séparément
            term_accuracy=term_acc["term_accuracy_score"],
            term_critical_errors=term_acc.get("critical_errors", []),
            latency_ms=latency,
            intent_correct=response.get("intent") == item.get("expected_intent"),
            cycle_correct=response.get("cycle") == item.get("expected_cycle"),
            milestone_complete=self._check_milestone_completeness(response, item),
        )

    def _check_milestone_completeness(self, response: dict, item: dict) -> bool:
        if item.get("expected_intent") != "JALON":
            return True  # N/A
        if item.get("critical_check") == "all_criteria_listed":
            # Vérifier que la réponse contient une liste numérotée de critères
            import re
            criteria = re.findall(r'^\d+\.', response["answer"], re.MULTILINE)
            return len(criteria) >= 3  # Au moins 3 critères
        return True

    def _aggregate(self, results: list[GRIEvalResult]) -> dict:
        import numpy as np
        metrics = ["faithfulness", "answer_relevance", "context_recall",
                   "context_precision", "term_accuracy"]
        summary = {}
        for m in metrics:
            values = [getattr(r, m) for r in results if getattr(r, m) > 0]
            if values:
                summary[m] = {
                    "mean": float(np.mean(values)),
                    "p25": float(np.percentile(values, 25)),
                    "p75": float(np.percentile(values, 75)),
                    "min": float(np.min(values)),
                }

        latencies = [r.latency_ms for r in results]
        summary["latency"] = {
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
        }
        summary["intent_accuracy"] = sum(r.intent_correct for r in results) / len(results)
        summary["cycle_accuracy"] = sum(r.cycle_correct for r in results) / len(results)
        summary["milestone_completeness"] = sum(r.milestone_complete for r in results) / len(results)
        return summary

    def _check_quality_gates(self, summary: dict) -> list[str]:
        failures = []
        for metric, threshold in self.QUALITY_GATES.items():
            if metric == "latency_p95_ms":
                val = summary.get("latency", {}).get("p95", 0)
                if val > threshold:
                    failures.append(f"Latency P95 trop élevée : {val:.0f}ms > {threshold}ms")
            else:
                val = summary.get(metric, {}).get("mean", 0)
                if val < threshold:
                    label = "🔴 CRITIQUE" if metric == "term_accuracy" else "⚠️"
                    failures.append(f"{label} {metric} : {val:.3f} < {threshold}")
        return failures
```

## Commandes d'Évaluation

```bash
# Évaluation complète sur le golden dataset
python -m src.evaluation.pipeline \
  --dataset data/golden_dataset.json \
  --output reports/eval_$(date +%Y%m%d_%H%M).json \
  --verbose

# Évaluation rapide (10 questions de smoke test)
python -m src.evaluation.pipeline \
  --dataset data/golden_dataset.json \
  --smoke-test \
  --n 10

# Comparer deux versions du RAG
python -m src.evaluation.compare \
  --baseline reports/eval_v1.json \
  --candidate reports/eval_v2.json \
  --focus term_accuracy faithfulness

# Vérifier quality gates uniquement
python -m src.evaluation.gates --report reports/eval_latest.json
```

## Alertes Production

```python
# Déclencher une alerte si en prod ces seuils sont dépassés
PRODUCTION_ALERT_THRESHOLDS = {
    "faithfulness_rolling_avg": 0.80,      # Fenêtre 100 dernières requêtes
    "term_accuracy_rolling_avg": 0.92,     # Plus strict en rolling que en batch
    "latency_p95_5min": 10_000,            # ms — fenêtre 5 minutes
    "error_rate_1min": 0.05,               # 5% max d'erreurs par minute
}
```
