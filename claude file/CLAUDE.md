# CLAUDE.md — Agentic RAG · GRI/FAR

## Projet
Système RAG agentique pour interroger le **Guide de Référence pour l'Innovation (GRI) des FAR**.
Document source : `IRF20251211_last_FF.docx` — 22 755 lignes · 7 phases · 11 principes · 50+ processus IS 15288 · CIR 4 phases.

---

## Stack Technique

| Composant | Technologie | Notes |
|-----------|------------|-------|
| Language | Python 3.11+ | async-first partout |
| API | FastAPI + SSE streaming | Pydantic v2 pour tous les schemas |
| Vector DB | Qdrant (prod) / ChromaDB (dev) | 2 collections : `gri_main` + `gri_glossary` |
| Embeddings | `paraphrase-multilingual-mpnet-base-v2` | FR+EN natif — **non négociable** |
| Sparse | `BM25Okapi` (rank-bm25) | Exact match termes ISO |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Après retrieval initiale |
| LLM | Hugging Face Inference API | SDK `huggingface_hub` async (modèles: `mistralai/Mistral-7B-Instruct-v0.3`, `meta-llama/Meta-Llama-3-8B-Instruct`, etc.) |
| Parsing | `python-docx` + `pandoc` | Conservation hiérarchie DOCX |
| Logging | `structlog` | Jamais `print()` |
| Tests | `pytest` + `pytest-asyncio` | Coverage > 80% sur le core |

---

## Configuration Hugging Face

### Variables d'environnement requises

```bash
# .env
HF_API_KEY=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Modèles par défaut (peuvent être surchargés)
HF_ROUTER_MODEL=mistralai/Mistral-7B-Instruct-v0.3      # Query routing (léger)
HF_ORCHESTRATOR_MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1  # Agent principal
HF_GENERATION_MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1    # Génération réponses
HF_EVAL_MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1          # Évaluation
```

### Modèles recommandés

| Usage | Modèle HF | Notes |
|-------|-----------|-------|
| Routing (rapide) | `mistralai/Mistral-7B-Instruct-v0.3` | 7B params, faible latence |
| Orchestration | `mistralai/Mixtral-8x7B-Instruct-v0.1` | MoE, bon suivi d'instructions |
| Génération | `mistralai/Mixtral-8x7B-Instruct-v0.1` | Qualité comparable à GPT-3.5 |
| Alternative FR | `croissantllm/CroissantLLMChat-v0.1` | Optimisé français |
| Alternative open | `meta-llama/Meta-Llama-3-8B-Instruct` | Très bon rapport qualité/coût |

### Client HF async

```python
from huggingface_hub import InferenceClient
import os

client = InferenceClient(token=os.getenv("HF_API_KEY"))

# Appel async
response = await client.text_generation(
    prompt="<s>[INST] Question ici [/INST]",
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    max_new_tokens=1024,
    temperature=0.1,
    return_full_text=False,
)
```

---

## Conventions Obligatoires

1. **Type hints stricts** sur toutes les fonctions publiques
2. **Pydantic v2** pour tous les modèles de données (pas de TypedDict)
3. **`async`/`await`** pour tout I/O — API Hugging Face, Qdrant, fichiers
4. **Credentials** via `os.environ` ou `.env` — jamais hardcodés
5. **Docstrings Google-style** sur classes et méthodes publiques
6. **Batch de 100 chunks** maximum par appel d'indexation
7. **Logging structuré** : `log.info("module.action", key=value)` partout

---

## Structure du Projet

```
agentic-rag-gri/
├── .claude/
│   └── CLAUDE.md                        ← ce fichier
│
├── skills/                              ← lire AVANT d'implémenter
│   ├── ingestion/SKILL.md               ← parsing + chunking GRI (345 lignes)
│   ├── retrieval/SKILL.md               ← hybrid search + query router (376 lignes)
│   ├── orchestration/SKILL.md           ← ReAct loop + 5 tools (439 lignes)
│   ├── generation/SKILL.md              ← synthesis + citations ISO (371 lignes)
│   └── evaluation/SKILL.md             ← RAGAS + Term Accuracy GRI (475 lignes)
│
├── src/
│   ├── agents/
│   │   ├── orchestrator.py              ← boucle ReAct principale
│   │   └── query_router.py             ← classification 6 intents GRI
│   ├── core/
│   │   ├── vector_store.py              ← Qdrant hybrid search + RRF
│   │   ├── reranker.py                  ← cross-encoder reranking
│   │   ├── memory.py                    ← mémoire conversationnelle
│   │   └── term_expander.py            ← enrichissement terminologique pre-retrieval
│   ├── tools/
│   │   ├── retrieve_gri.py              ← tool retrieve_gri_chunks
│   │   ├── glossary.py                  ← tool lookup_gri_glossary
│   │   ├── milestones.py               ← tool get_milestone_criteria
│   │   ├── compare.py                   ← tool compare_approaches
│   │   └── phases.py                    ← tool get_phase_summary
│   ├── ingestion/
│   │   ├── pipeline.py                  ← orchestration complète ingestion
│   │   ├── chunker.py                   ← 7 stratégies chunking GRI
│   │   └── table_extractor.py          ← extraction tables DOCX (critères jalons)
│   ├── evaluation/
│   │   ├── faithfulness_gri.py         ← faithfulness adaptée au domaine GRI
│   │   ├── term_accuracy.py            ← métrique custom ISO exactitude
│   │   └── pipeline.py                  ← pipeline d'évaluation complète
│   └── api/
│       └── main.py                      ← FastAPI endpoints + streaming SSE
│
├── data/
│   ├── raw/
│   │   └── IRF20251211_last_FF.docx    ← document source GRI
│   ├── golden_dataset.json             ← 50 questions annotées (à construire)
│   └── glossary_gri.json               ← 200+ termes ISO extraits
│
├── tests/
│   ├── test_ingestion_gri.py
│   ├── test_retrieval_gri.py
│   ├── test_orchestrator_gri.py
│   ├── test_generation_gri.py
│   └── test_evaluation_gri.py
│
├── reports/                             ← sorties des évaluations
├── configs/
│   └── config.yaml
└── pyproject.toml
```

---

## Skills — Quelle Skill Lire pour Quelle Tâche

**Lire la skill AVANT d'écrire le moindre code.**

| Si tu travailles sur... | Lis d'abord |
|------------------------|-------------|
| Parser le DOCX GRI / chunking / indexation | `skills/ingestion/SKILL.md` |
| Hybrid search / BM25 / reranking / query router | `skills/retrieval/SKILL.md` |
| Boucle ReAct / tool calls / orchestrateur | `skills/orchestration/SKILL.md` |
| Prompts de synthèse / citations ISO / temperature | `skills/generation/SKILL.md` |
| Évaluation RAGAS / Term Accuracy / golden dataset | `skills/evaluation/SKILL.md` |

---

## Règles de Chunking GRI (CRITIQUE)

### Règle N°1 — Context Prefix OBLIGATOIRE

Chaque chunk commence par son chemin hiérarchique. Sans ça, le LLM ne peut pas localiser l'information.

```python
# Format obligatoire — toujours
context_prefix = "[GRI > Phase 3 > Conception et Développement > Processus de Vérification]"
chunk_content  = f"{context_prefix}\n\n{section_text}"
```

### Règle N°2 — Stratégie par Type de Contenu

```
Type            | Taille        | Overlap | Boundary   | Metadata clé
----------------|---------------|---------|------------|---------------------------
definition      | 150-300 tok   | 0       | term       | term_fr, term_en, standard_ref
principle       | 400-600 tok   | 0       | section    | principle_num (1-11)
phase (enfant)  | 512 tok       | 64      | sentence   | phase_num, subsection_type
phase (parent)  | 2 048 tok     | 0       | phase      | phase_num (PDR)
milestone       | 600-900 tok   | 0       | milestone  | milestone_id — NE JAMAIS couper
process IS15288 | 400-600 tok   | 0       | activity   | process_name, inputs, outputs
cir             | 400-600 tok   | 0       | section    | cir_phase, gri_equivalent
table           | 200-400 tok   | 0       | row        | table_id, columns
```

### Règle N°3 — Metadata Obligatoire

```python
{
    "doc_id":         str,    # sha256(fichier)[:16]
    "source":         str,    # "GRI_FAR_2025"
    "section_type":   str,    # 'definition'|'principle'|'phase'|'milestone'|'process'|'cir'|'table'
    "hierarchy":      list,   # ['GRI', 'Phase 3', 'Conception', 'Vérification']
    "context_prefix": str,    # "[GRI > Phase 3 > ...]"
    "cycle":          str,    # 'GRI' | 'CIR'
    "phase_num":      int,    # 1-7 (GRI) ou 1-4 (CIR)
    "cir_phase":      int,    # 1-4 (CIR uniquement)
    "milestone_id":   str,    # 'M0'..'M9' ou 'J1'..'J6'
    "process_name":   str,    # Nom du processus IS 15288
    "principle_num":  int,    # 1-11
    "language":       "fr",
    "chunk_index":    int,
    "created_at":     str,    # ISO timestamp
    "parent_chunk_id":str,    # Pour Parent Document Retriever (phases)
}
```

---

## Query Router — 6 Intents GRI

Classifier la question **avant** toute retrieval.

```
Intent          | Triggers                                     | Stratégie            | Filtres
----------------|----------------------------------------------|----------------------|------------------
DEFINITION      | "qu'est-ce que", "définir", "signification"  | BM25 sur glossaire   | type=definition
PROCESSUS       | "activités de", "inputs/outputs", "processus"| Hybrid + parent      | type=process
JALON           | "critères de M4", "CDR", "passage du jalon"  | Exact + complet      | milestone_id=MN
PHASE_COMPLETE  | "phase 3", "idéation", "phase complète"      | Parent doc retrieval | phase_num=N
COMPARAISON     | "différence entre", "comparer", "vs"         | Multi-retrieve       | multi-filter
CIR             | "cycle rapide", "CIR", "J1 à J6"            | CIR + mapping GRI    | cycle=CIR
```

**Détection GRI vs CIR** — Toujours identifier le cycle avant de retriever :
- **GRI standard** : 7 phases, jalons M0 → M9
- **CIR** : 4 phases, jalons J1 → J6
- **Mapping** : J1=M0+M1 · J2=M2+M3+M4 · J3=M5+M6 · J4=SAR · J5=SAR · J6=M8

---

## Les 5 Tools de l'Agent

```python
TOOLS = [
    "retrieve_gri_chunks",      # Retrieval principal — hybrid search avec filtres section_type
    "lookup_gri_glossary",      # Définitions exactes ISO — appeler EN PREMIER pour tout terme
    "get_milestone_criteria",   # Checklist COMPLÈTE d'un jalon (M0-M9 ou J1-J6)
    "compare_approaches",       # Multi-retrieve parallèle pour comparaisons
    "get_phase_summary",        # Parent doc retrieval — objectifs + livrables + jalons
]
```

**Règle d'utilisation** :
- `lookup_gri_glossary` → **toujours en premier** si un terme ISO/GRI est dans la question
- `get_milestone_criteria` → **toujours** pour les critères de jalons (jamais `retrieve_gri_chunks`)
- `get_phase_summary` → **toujours** pour les résumés de phase (évite les réponses fragmentées)

---

## System Prompt de l'Orchestrateur

```
Tu es un expert en ingénierie système selon le GRI des FAR (ISO/IEC/IEEE 15288:2023).

ÉTAPE 1 — IDENTIFIER LE CYCLE : GRI standard (7 phases, M0-M9) ou CIR (4 phases, J1-J6) ?

ÉTAPE 2 — CLASSIFIER L'INTENT : DEFINITION / PROCESSUS / JALON / PHASE_COMPLETE / COMPARAISON / CIR

ÉTAPE 3 — PLANIFIER LES OUTILS selon le routing (voir CLAUDE.md section Query Router)

ÉTAPE 4 — VALIDER les chunks récupérés avant de synthétiser

ÉTAPE 5 — SYNTHÈSE avec citations [GRI > Section X.Y] ou [CIR > Phase N, Jalon JN]

RÈGLES ABSOLUES :
• Ne jamais inventer de critères, jalons, phases ou livrables
• Définitions ISO = reproduire mot pour mot (aucune paraphrase)
• Critères de jalons = lister exhaustivement, aucun ne peut être omis
• Si terme inconnu → lookup_gri_glossary AVANT de répondre
• Max 5 itérations de retrieval
```

---

## Paramètres LLM par Type de Réponse

```
Type            | Temperature | Max tokens | Chunks | Règle
----------------|-------------|------------|--------|---------------------------
DEFINITION      | 0.0         | 256        | 1-2    | Copier mot pour mot — ISO normatif
JALON           | 0.0         | 1 024      | 1      | Lister TOUS les critères
PROCESSUS       | 0.1         | 1 536      | 3-5    | Inclure inputs + outputs
PHASE_COMPLETE  | 0.1         | 2 048      | 5-8    | Via get_phase_summary
COMPARAISON     | 0.1         | 2 048      | 6-10   | Tableau structuré par dimension
```

---

## Format de Citation Standard

```
# Terminologie
[GRI > Terminologie > 'Artefact' — ISO/IEC/IEEE 15288:2023]

# Principes
[GRI > Principe N°8 : Capitalisation et gestion de la connaissance]

# Phases
[GRI > Phase 3 : Conception et Développement > Objectifs spécifiques]

# Jalons GRI
[GRI > Jalon M4 (CDR) > Critère #3]

# Jalons CIR
[CIR > Phase 2 > Jalon J3 — Équivalent GRI : M5 (IRR) + M6 (TRR)]

# Processus IS 15288
[GRI > Processus IS 15288 > Vérification > Activité : Planification]

# Approches
[GRI > Approches > DevSecOps > Avantages]
```

---

## Quality Gates Production

```
Métrique              | Seuil   | Bloquant    | Notes
----------------------|---------|-------------|-----------------------------------
faithfulness          | ≥ 0.85  | ✅ OUI      | Claims vérifiés contre le GRI
answer_relevance      | ≥ 0.80  | ✅ OUI      | Réponse GRI-pertinente
context_recall        | ≥ 0.75  | ✅ OUI      | Chunks couvrent la vraie réponse
term_accuracy         | ≥ 0.95  | ✅ CRITIQUE | Définitions ISO exactes (custom metric)
context_precision     | ≥ 0.70  | Non         |
latency P95           | ≤ 8 s   | ✅ OUI      |
intent_accuracy       | ≥ 0.90  | Non         | Routing correct
milestone_completeness| ≥ 0.95  | ✅ OUI      | Critères jalons exhaustifs
```

**`term_accuracy`** est la métrique la plus critique : une définition ISO paraphrasée est une erreur réglementaire dans un contexte défense.

---

## Commandes Fréquentes

```bash
# Ingérer le GRI depuis le DOCX
python -m src.ingestion.pipeline \
  --input data/raw/IRF20251211_last_FF.docx \
  --output data/processed/

# Vérifier l'index après ingestion
python -m src.core.vector_store --stats

# Tester le retrieval sur une query
python -m src.core.vector_store \
  --query "critères du CDR" \
  --section-type milestone \
  --n 5

# Lancer le RAG en mode interactif
python -m src.agents.orchestrator --interactive

# Évaluation complète sur le golden dataset
python -m src.evaluation.pipeline \
  --dataset data/golden_dataset.json \
  --output reports/eval_$(date +%Y%m%d).json

# Smoke test rapide (10 questions)
python -m src.evaluation.pipeline \
  --dataset data/golden_dataset.json \
  --smoke-test

# Démarrer l'API
uvicorn src.api.main:app --reload --port 8000

# Tests unitaires
pytest tests/ -v --asyncio-mode=auto

# Test ciblé
pytest tests/test_retrieval_gri.py -v
```

---

## 7 Pièges Critiques

```
#1  Chunking fixe 512 tokens    → Fracture les critères de jalons entre chunks sans contexte
#2  Embedding monolingue        → Rate les synonymes FR/EN (Artefact vs Artifact)
#3  Confondre GRI et CIR        → Jalons et délais incompatibles entre les deux cycles
#4  Ignorer les tables DOCX     → Perd les critères de passage des jalons (5 092 lignes)
#5  temperature > 0.1 sur ISO   → Paraphrase = erreur réglementaire dans un contexte défense
#6  Pas de cache sémantique     → Questions fréquentes (Phase 3, CDR...) payées à chaque fois
#7  Golden dataset générique    → Ne détecte pas les jalons inventés ni les mauvais mappings CIR↔GRI
```

---

## Golden Dataset — Répartition des 50 Questions

```
Catégorie          | Nb  | Exemples
-------------------|-----|--------------------------------------------
Définitions ISO    | 10  | "Qu'est-ce qu'un artefact ?" "Définir CONOPS"
Processus IS15288  | 10  | "Activités du processus de vérification ?"
Critères jalons    | 10  | "Critères du CDR (M4) ?" "Critères J3 CIR ?"
Phases complètes   |  8  | "Objectifs de la Phase 1 ?"
Comparaisons       |  7  | "GRI standard vs CIR ?" "Séquentiel vs DevSecOps ?"
CIR spécifique     |  5  | "Dans quel contexte le CIR s'applique-t-il ?"
```

---

*Basé sur l'analyse de IRF20251211_last_FF.docx (GRI/FAR)*
*Dernière mise à jour : Mars 2026*
