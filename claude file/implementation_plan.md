# Plan d'Implémentation Complet — Agentic RAG GRI/FAR

## Vue d'Ensemble du Projet

**Objectif** : Système RAG agentique pour interroger le **Guide de Référence pour l'Innovation (GRI) des FAR** — document normatif de 22 755 lignes couvrant 7 phases, 11 principes, 50+ processus IS 15288, et le CIR (4 phases).

**Contrainte principale** : Les définitions ISO/IEC/IEEE 15288:2023 et les critères de jalons sont **normatifs** — toute paraphrase est une erreur réglementaire dans un contexte défense.

---

## Architecture Globale

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              API FastAPI                                 │
│                         (SSE Streaming + REST)                           │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────┐
│                         ORCHESTRATEUR REACT                              │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌───────────────┐  │
│  │Query Router │→ │Term Expander │→ │ ReAct Loop  │→ │  Generator    │  │
│  │ (6 intents) │  │  (Glossaire) │  │ (5 Tools)   │  │ (Grounded)    │  │
│  └─────────────┘  └──────────────┘  └─────────────┘  └───────────────┘  │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────┐
│                           COUCHE RETRIEVAL                               │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────┐   │
│  │ Qdrant (Dense)   │  │  BM25 (Sparse)   │  │  Cross-Encoder       │   │
│  │ gri_main         │  │  Exact Match ISO │  │  Reranker            │   │
│  │ gri_glossary     │  │                  │  │                      │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────────┘   │
│                    └────────────┬────────────┘                          │
│                           RRF Fusion                                     │
│                         (α=0.6/0.4)                                      │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────────────┐
│                         PIPELINE INGESTION                               │
│  ┌───────────┐  ┌─────────────┐  ┌────────────────┐  ┌───────────────┐  │
│  │ DOCX      │→ │ Chunker     │→ │ Context Prefix │→ │ Double Index  │  │
│  │ Parser    │  │ (7 strats)  │  │ + Metadata     │  │               │  │
│  └───────────┘  └─────────────┘  └────────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Phases d'Implémentation

### PHASE 0 : Setup & Infrastructure (Jour 1-2)

#### 0.1 Structure du Projet
```bash
agentic-rag-gri/
├── .env.example
├── pyproject.toml
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   │   └── IRF20251211_last_FF.docx
│   ├── processed/
│   ├── golden_dataset.json
│   └── glossary_gri.json
├── src/
│   ├── __init__.py
│   ├── agents/
│   ├── core/
│   ├── tools/
│   ├── ingestion/
│   ├── evaluation/
│   └── api/
├── tests/
├── reports/
└── skills/
```

#### 0.2 Dépendances (pyproject.toml)
```toml
[project]
name = "agentic-rag-gri"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    # Core
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",

    # LLM & Embeddings
    "huggingface-hub>=0.20.0",
    "sentence-transformers>=2.2.2",

    # Vector Store
    "qdrant-client>=1.7.0",
    "chromadb>=0.4.22",  # dev only

    # Sparse Search
    "rank-bm25>=0.2.2",

    # Reranking
    "transformers>=4.36.0",

    # Document Processing
    "python-docx>=1.1.0",
    "pypandoc>=1.12",

    # Utilities
    "structlog>=24.1.0",
    "python-dotenv>=1.0.0",
    "aiofiles>=23.2.1",
    "httpx>=0.26.0",

    # Evaluation
    "ragas>=0.1.0",
    "numpy>=1.26.0",
    "pandas>=2.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.23.0",
    "pytest-cov>=4.1.0",
    "black>=24.1.0",
    "ruff>=0.1.14",
    "mypy>=1.8.0",
]
```

#### 0.3 Configuration
```yaml
# configs/config.yaml
project:
  name: "GRI-RAG"
  version: "0.1.0"

huggingface:
  api_key: ${HF_API_KEY}
  models:
    router: "mistralai/Mistral-7B-Instruct-v0.3"
    orchestrator: "mistralai/Mixtral-8x7B-Instruct-v0.1"
    generation: "mistralai/Mixtral-8x7B-Instruct-v0.1"
    evaluation: "mistralai/Mixtral-8x7B-Instruct-v0.1"

embeddings:
  model: "paraphrase-multilingual-mpnet-base-v2"
  dimension: 768

vector_store:
  provider: "qdrant"  # qdrant | chromadb
  url: "localhost:6333"
  collections:
    main: "gri_main"
    glossary: "gri_glossary"

retrieval:
  rrf_alpha: 0.6
  rrf_k: 60
  default_n_results: 5
  reranker_model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

generation:
  temperatures:
    definition: 0.0
    milestone: 0.0
    process: 0.1
    phase: 0.1
    comparison: 0.1
    general: 0.1

agent:
  max_iterations: 5
  max_reformulations_per_tool: 2

evaluation:
  quality_gates:
    faithfulness: 0.85
    answer_relevance: 0.80
    context_recall: 0.75
    context_precision: 0.70
    term_accuracy: 0.95
    latency_p95_ms: 8000
```

#### Livrables Phase 0
- [ ] Structure projet créée
- [ ] `pyproject.toml` configuré
- [ ] `.env.example` avec toutes les variables
- [ ] `config.yaml` paramétrable
- [ ] Docker Compose pour Qdrant
- [ ] Logging structuré (structlog) configuré

---

### PHASE 1 : Pipeline d'Ingestion (Jour 3-6)

#### 1.1 Parser DOCX avec Hiérarchie
```python
# src/ingestion/parser.py
# Objectif : Extraire le document en préservant la hiérarchie 7 niveaux
# - Détecter les sections ## et sous-sections ###
# - Identifier les types de contenu (définition, principe, phase, jalon, process, cir, table)
# - Construire l'arbre hiérarchique

Fichiers à créer :
├── src/ingestion/
│   ├── __init__.py
│   ├── parser.py           # Extraction DOCX avec python-docx
│   ├── hierarchy.py        # Construction de l'arbre hiérarchique
│   └── section_detector.py # Détection des types de sections
```

#### 1.2 Table Extractor
```python
# src/ingestion/table_extractor.py
# Objectif : Extraire les tables de critères de jalons (5 092 lignes critiques)
# - Identifier les tables de jalons (M0-M9, J1-J6)
# - Convertir en format structuré (row-by-row + summary)
# - Préserver les en-têtes de colonnes

Fichiers à créer :
├── src/ingestion/
│   └── table_extractor.py
```

#### 1.3 Chunker Multi-Stratégies
```python
# src/ingestion/chunker.py
# 7 stratégies de chunking selon le type de contenu :
#
# | Type        | Taille      | Overlap | Boundary   |
# |-------------|-------------|---------|------------|
# | definition  | 150-300 tok | 0       | term       |
# | principle   | 400-600 tok | 0       | section    |
# | phase       | 512/2048    | 64      | sentence   |  # Parent Doc Retriever
# | milestone   | 600-900 tok | 0       | milestone  |  # NE JAMAIS COUPER
# | process     | 400-600 tok | 0       | activity   |
# | cir         | 400-600 tok | 0       | section    |
# | table       | 200-400 tok | 0       | row        |

Fichiers à créer :
├── src/ingestion/
│   ├── chunker.py
│   └── strategies/
│       ├── __init__.py
│       ├── definition_strategy.py
│       ├── principle_strategy.py
│       ├── phase_strategy.py
│       ├── milestone_strategy.py
│       ├── process_strategy.py
│       ├── cir_strategy.py
│       └── table_strategy.py
```

#### 1.4 Context Prefix & Metadata
```python
# src/ingestion/metadata.py
# Chaque chunk DOIT avoir :
# - context_prefix: "[GRI > Phase 3 > Conception > Vérification]"
# - metadata complète (voir skill ingestion)

# Modèle Pydantic pour les chunks
@dataclass
class GRIChunk:
    content: str              # context_prefix + "\n\n" + text
    chunk_id: str             # sha256(content)[:16]
    metadata: GRIMetadata

@dataclass
class GRIMetadata:
    doc_id: str
    source: str               # "GRI_FAR_2025"
    section_type: Literal['definition', 'principle', 'phase', 'milestone', 'process', 'cir', 'table']
    hierarchy: list[str]
    context_prefix: str
    cycle: Literal['GRI', 'CIR']
    phase_num: int | None
    milestone_id: str | None
    process_name: str | None
    principle_num: int | None
    language: str             # 'fr'
    token_estimate: int
    created_at: str
    parent_chunk_id: str | None  # Pour Parent Doc Retriever
```

#### 1.5 Pipeline Complet
```python
# src/ingestion/pipeline.py
# Orchestration : Parser → Chunker → Metadata → Validation → Index

class GRIIngestionPipeline:
    def run(self, docx_path: str) -> dict:
        # 1. Parse DOCX
        # 2. Séparer flux : texte / tables / glossaire
        # 3. Chunk par type de contenu
        # 4. Ajouter context prefix + metadata
        # 5. Valider chaque chunk
        # 6. Indexer dans Qdrant + BM25
        pass
```

#### Tests Phase 1
```python
# tests/test_ingestion_gri.py
- test_all_milestones_indexed()        # M0-M9 + J1-J6 tous présents
- test_context_prefix_on_all_chunks()  # Prefix [GRI...] ou [CIR...]
- test_glossary_index_completeness()   # 150+ définitions
- test_cir_chunks_have_gri_mapping()   # J1-J6 mappés vers M0-M9
- test_milestone_chunks_not_split()    # Jalons non fragmentés
- test_no_empty_chunks()               # Min 80 chars
- test_metadata_completeness()         # Tous les champs requis
```

#### Livrables Phase 1
- [ ] Parser DOCX avec hiérarchie 7 niveaux
- [ ] Extracteur de tables de jalons
- [ ] 7 stratégies de chunking implémentées
- [ ] Context prefix automatique sur tous les chunks
- [ ] Pipeline d'ingestion complet
- [ ] Tests unitaires (coverage > 80%)
- [ ] Document indexé dans Qdrant (gri_main + gri_glossary)

---

### PHASE 2 : Couche Retrieval Hybride (Jour 7-10)

#### 2.1 Vector Store (Qdrant + BM25)
```python
# src/core/vector_store.py
# Double index : dense (Qdrant) + sparse (BM25)

class GRIHybridStore:
    EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"
    COLLECTIONS = {"main": "gri_main", "glossary": "gri_glossary"}
    RRF_ALPHA = 0.6
    RRF_K = 60

    def hybrid_search(self, query, collection, n_results, filters, alpha) -> list[dict]
    def glossary_lookup(self, term) -> dict | None
    def _rrf_fusion(self, dense_hits, sparse_hits, alpha, n_results) -> list[dict]
    def _build_filter(self, filters) -> Filter
```

#### 2.2 Query Router (6 Intents)
```python
# src/agents/query_router.py
# Classification avant retrieval

INTENTS = [
    "DEFINITION",      # "Qu'est-ce que...", "Définir..."
    "PROCESSUS",       # Processus IS 15288
    "JALON",           # Critères M0-M9 ou J1-J6
    "PHASE_COMPLETE",  # Phase entière (objectifs, livrables, jalons)
    "COMPARAISON",     # "Différence entre...", "vs"
    "CIR",             # Cycle d'Innovation Rapide
]

# Chaque intent a sa stratégie de retrieval :
ROUTING_TABLE = {
    "DEFINITION":     {"search_mode": "sparse", "primary_index": "glossary", ...},
    "PROCESSUS":      {"search_mode": "hybrid", "use_reranker": True, ...},
    "JALON":          {"return_complete": True, "use_reranker": False, ...},
    "PHASE_COMPLETE": {"use_parent": True, "use_mmr": True, ...},
    "COMPARAISON":    {"multi_query": True, ...},
    "CIR":            {"include_gri_mapping": True, ...},
}
```

#### 2.3 Term Expander (Pre-Retrieval)
```python
# src/core/term_expander.py
# Enrichissement query avec définitions GRI

GRI_KEY_TERMS = [
    "artefact", "CONOPS", "SEMP", "SRR", "PDR", "CDR",
    "IRR", "TRR", "SAR", "ORR", "MNR", "CIR", "TRL", "MRL", ...
]

async def expand_query_with_terms(query, store) -> tuple[str, str]:
    # Retourne (query_originale, contexte_terminologique)
    # Le contexte est injecté dans le system prompt
```

#### 2.4 Reranker
```python
# src/core/reranker.py
# Cross-encoder pour reranking final

class GRIReranker:
    MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def rerank(self, query, chunks, top_k) -> list[dict]
```

#### 2.5 Retrieval Spéciale Jalons
```python
# src/core/milestone_retriever.py
# Les jalons ont une logique spéciale : toujours retourner le jalon COMPLET

CIR_GRI_MAPPING = {
    "J1": ["M0", "M1"],
    "J2": ["M2", "M3", "M4"],
    "J3": ["M5", "M6"],
    "J4": ["SAR"],
    "J5": ["SAR"],
    "J6": ["M8"],
}

async def get_jalon_complet(milestone_id, store) -> dict:
    # Lookup direct + mapping CIR→GRI si applicable
```

#### Tests Phase 2
```python
# tests/test_retrieval_gri.py
- test_definition_lookup_exact()       # BM25 sur glossaire
- test_milestone_retrieval_complete()  # Jalon entier, pas fragment
- test_cir_includes_gri_mapping()      # J3 → M5+M6
- test_sparse_wins_on_iso_terms()      # "SEMP" → exact match
- test_phase_complete_uses_parent()    # Parent Document Retriever
- test_reranker_improves_precision()   # Score post-rerank
- test_rrf_fusion_balances_modes()     # α=0.6 dense
```

#### Livrables Phase 2
- [ ] GRIHybridStore avec Qdrant + BM25
- [ ] Query Router (6 intents GRI)
- [ ] Term Expander (enrichissement pré-retrieval)
- [ ] Reranker Cross-Encoder
- [ ] RRF Fusion implémenté
- [ ] Retrieval spéciale jalons (complet + mapping CIR)
- [ ] Tests unitaires

---

### PHASE 3 : Orchestrateur Agentique (Jour 11-16)

#### 3.1 Les 5 Tools
```python
# src/tools/
├── __init__.py
├── retrieve_gri.py          # retrieve_gri_chunks
├── glossary.py              # lookup_gri_glossary
├── milestones.py            # get_milestone_criteria
├── compare.py               # compare_approaches
└── phases.py                # get_phase_summary

# Chaque tool avec :
# - Input schema JSON
# - Description pour le LLM
# - Implémentation async
# - Gestion d'erreurs
```

#### 3.2 Boucle ReAct
```python
# src/agents/orchestrator.py

class GRIOrchestrator:
    MAX_ITER = 5

    async def run(self, query: str) -> dict:
        # 1. Query routing (intent + cycle)
        # 2. Term expansion (contexte terminologique)
        # 3. Conversation context (mémoire)
        # 4. Boucle ReAct :
        #    - LLM appelle les tools
        #    - Exécution parallèle si indépendants
        #    - Validation des résultats
        #    - Synthèse finale avec citations
        # 5. Retour : answer, citations, stats
```

#### 3.3 Mémoire Conversationnelle
```python
# src/core/memory.py

class GRIMemory:
    MAX_TURNS = 10

    def add_turn(self, query: str, answer: str)
    def get_context(self) -> str
    def clear(self)
```

#### 3.4 Tool Executor
```python
# src/tools/executor.py

async def execute_tool(tool_name: str, input: dict, store) -> dict:
    # Dispatch vers le bon tool
    # Gestion des erreurs
    # Logging structuré
```

#### 3.5 System Prompt Orchestrateur
```python
# src/agents/prompts.py

ORCHESTRATOR_SYSTEM = """
Tu es un expert en ingénierie système selon le GRI des FAR (ISO/IEC/IEEE 15288:2023).

## PROCESSUS DE RAISONNEMENT OBLIGATOIRE
1. IDENTIFIER LE CYCLE : GRI (7 phases, M0-M9) ou CIR (4 phases, J1-J6)
2. CLASSIFIER L'INTENT : DEFINITION / PROCESSUS / JALON / PHASE_COMPLETE / COMPARAISON / CIR
3. PLANIFIER LES OUTILS selon le routing
4. VALIDER les chunks récupérés
5. SYNTHÉTISER avec citations [GRI > Section] obligatoires

## RÈGLES ABSOLUES
- Ne jamais inventer de critères, jalons, ou livrables
- Définitions ISO = reproduire mot pour mot
- Critères de jalons = lister exhaustivement
- Max {max_iter} itérations
"""
```

#### Tests Phase 3
```python
# tests/test_orchestrator_gri.py
- test_jalon_question_one_tool_call()  # M4 → get_milestone_criteria
- test_definition_first_tool_call()     # lookup_gri_glossary en premier
- test_cir_includes_gri_mapping()       # J2 → M2+M3+M4
- test_citations_format()               # [GRI > ...]
- test_out_of_scope_refused()           # Question hors GRI
- test_max_iter_not_exceeded()          # Boucle terminée ≤5
- test_parallel_tool_calls()            # Comparaisons parallèles
```

#### Livrables Phase 3
- [ ] 5 tools implémentés avec schemas JSON
- [ ] Boucle ReAct avec HF Inference API
- [ ] Mémoire conversationnelle
- [ ] System prompt optimisé
- [ ] Validation des résultats inter-tool
- [ ] Gestion des erreurs et timeout
- [ ] Tests unitaires

---

### PHASE 4 : Génération Grounded (Jour 17-20)

#### 4.1 Prompts par Type de Réponse
```python
# src/agents/generation_agent.py

# 6 prompts spécialisés :
DEFINITION_PROMPT     # temperature=0.0, reproduire mot pour mot
MILESTONE_PROMPT      # temperature=0.0, lister TOUS les critères
PROCESS_PROMPT        # temperature=0.1, structure inputs/outputs
PHASE_PROMPT          # temperature=0.1, objectifs + livrables + jalons
COMPARISON_PROMPT     # temperature=0.1, tableau comparatif
GENERAL_PROMPT        # temperature=0.1, synthèse grounded
```

#### 4.2 Sélection Automatique du Prompt
```python
class GRIResponseType(Enum):
    DEFINITION = "definition"
    MILESTONE = "milestone"
    PROCESS = "process"
    PHASE_COMPLETE = "phase_complete"
    COMPARISON = "comparison"
    GENERAL = "general"

TEMPERATURE_MAP = {
    DEFINITION: 0.0,     # Fidélité normative absolue
    MILESTONE: 0.0,      # Critères exhaustifs
    ...
}

MAX_TOKENS_MAP = {
    DEFINITION: 256,
    MILESTONE: 1024,
    ...
}
```

#### 4.3 Formatage du Contexte GRI
```python
# src/agents/context_formatter.py

def format_gri_context(chunks: list[dict]) -> str:
    # Format :
    # [SOURCE 1] GRI · MILESTONE · Score: 0.87 · Jalon: M4
    # {content}
    # ---
    # [SOURCE 2] ...
```

#### 4.4 Détection Contexte Insuffisant
```python
# src/agents/sufficiency_checker.py

def check_context_sufficiency(chunks, response_type) -> dict:
    # Retourne {"sufficient": bool, "reason": str, "message": str}
    # Cas : no_chunks, low_scores, missing_milestone_chunk
```

#### 4.5 Post-Processing GRI
```python
# src/agents/postprocessor.py

def postprocess_gri_answer(answer: str, response_type) -> str:
    # Vérifier jalons valides (M0-M9, J1-J6)
    # Vérifier phases valides (1-7 GRI, 1-4 CIR)
    # Ajouter warnings si anomalies
```

#### Tests Phase 4
```python
# tests/test_generation_gri.py
- test_definition_not_paraphrased()      # Mot pour mot
- test_milestone_criteria_exhaustive()   # Tous les critères listés
- test_citations_present_in_all_answers()
- test_low_score_chunks_acknowledged()   # "Non disponible..."
- test_cir_gri_distinction_in_comparison()
- test_temperature_respected()           # 0.0 pour définitions
```

#### Livrables Phase 4
- [ ] 6 prompts spécialisés GRI
- [ ] Sélection automatique du type de réponse
- [ ] Formatage contexte pour le LLM
- [ ] Détection contexte insuffisant
- [ ] Post-processing avec validations
- [ ] Citations au format [GRI > Section > ...]
- [ ] Tests unitaires

---

### PHASE 5 : API FastAPI (Jour 21-23)

#### 5.1 Endpoints
```python
# src/api/main.py

# POST /query          → Query standard (JSON response)
# POST /query/stream   → Query avec SSE streaming
# GET  /health         → Healthcheck
# GET  /stats          → Stats index + latence
# POST /feedback       → Feedback utilisateur (amélioration continue)
```

#### 5.2 Modèles Pydantic
```python
# src/api/models.py

class QueryRequest(BaseModel):
    query: str
    cycle: Literal["GRI", "CIR", "AUTO"] = "AUTO"
    include_sources: bool = True
    max_chunks: int = 5

class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    intent: str
    cycle: str
    latency_ms: float
    sources: list[Source] | None

class Citation(BaseModel):
    text: str
    section: str

class Source(BaseModel):
    chunk_id: str
    content: str
    score: float
    section_type: str
    metadata: dict
```

#### 5.3 SSE Streaming
```python
# src/api/streaming.py

async def stream_response(query: str):
    # Yield events :
    # - routing: {"intent": "JALON", "cycle": "GRI"}
    # - tool_call: {"tool": "get_milestone_criteria", "input": {...}}
    # - chunk: {"text": "partial answer..."}
    # - done: {"answer": "...", "citations": [...]}
```

#### 5.4 Middleware & Error Handling
```python
# src/api/middleware.py
# - CORS
# - Rate limiting
# - Request logging
# - Error handling global
```

#### Tests Phase 5
```python
# tests/test_api.py
- test_query_endpoint_returns_answer()
- test_stream_endpoint_yields_events()
- test_health_endpoint()
- test_invalid_query_returns_400()
- test_rate_limiting()
```

#### Livrables Phase 5
- [ ] Endpoints REST implémentés
- [ ] SSE streaming fonctionnel
- [ ] Modèles Pydantic complets
- [ ] CORS + rate limiting
- [ ] Error handling robuste
- [ ] Tests d'intégration

---

### PHASE 6 : Suite d'Évaluation (Jour 24-28)

#### 6.1 Golden Dataset (50 questions)
```json
// data/golden_dataset.json
{
  "questions": [
    // 10 définitions
    {"id": "DEF_001", "query": "Qu'est-ce qu'un artefact ?", ...},
    // 10 processus IS 15288
    {"id": "PROC_001", "query": "Activités du processus de vérification ?", ...},
    // 10 jalons
    {"id": "JAL_001", "query": "Critères du CDR (M4) ?", ...},
    // 8 phases complètes
    {"id": "PHA_001", "query": "Objectifs de la Phase 1 ?", ...},
    // 7 comparaisons
    {"id": "COMP_001", "query": "Différence GRI vs CIR ?", ...},
    // 5 CIR spécifiques
    {"id": "CIR_001", "query": "Contexte d'application du CIR ?", ...},
  ]
}
```

#### 6.2 Métriques RAGAS Adaptées
```python
# src/evaluation/
├── faithfulness_gri.py      # Faithfulness avec détection erreurs GRI
├── answer_relevance.py      # Answer Relevance standard
├── context_recall.py        # Context Recall
├── context_precision.py     # Context Precision
└── term_accuracy.py         # CRITIQUE : définitions ISO exactes
```

#### 6.3 Term Accuracy (Métrique Custom)
```python
# src/evaluation/term_accuracy.py
# La métrique la plus importante pour ce projet

async def compute_term_accuracy(answer, glossary_index) -> dict:
    # 1. Extraire termes ISO de la réponse
    # 2. Lookup définitions normatives
    # 3. Comparer : EXACT / APPROXIMATIF / INCORRECT
    # 4. Score = (Σ scores) / nb_termes
    # Seuil : ≥ 0.95 (CRITIQUE)
```

#### 6.4 Pipeline d'Évaluation
```python
# src/evaluation/pipeline.py

class GRIEvaluator:
    QUALITY_GATES = {
        "faithfulness": 0.85,
        "answer_relevance": 0.80,
        "context_recall": 0.75,
        "term_accuracy": 0.95,  # CRITIQUE
        "latency_p95_ms": 8000,
    }

    async def evaluate_dataset(self, dataset, rag_system) -> dict:
        # Évalue toutes les questions
        # Agrège les métriques
        # Vérifie les quality gates
```

#### 6.5 CLI d'Évaluation
```bash
# Évaluation complète
python -m src.evaluation.pipeline \
  --dataset data/golden_dataset.json \
  --output reports/eval_$(date +%Y%m%d).json

# Smoke test (10 questions)
python -m src.evaluation.pipeline --smoke-test

# Comparer deux versions
python -m src.evaluation.compare --baseline v1.json --candidate v2.json
```

#### Tests Phase 6
```python
# tests/test_evaluation_gri.py
- test_term_accuracy_exact_match()
- test_faithfulness_detects_invented_milestone()
- test_quality_gates_fail_on_low_scores()
- test_full_pipeline_on_sample()
```

#### Livrables Phase 6
- [ ] Golden dataset 50 questions annotées
- [ ] 5 métriques implémentées (dont Term Accuracy)
- [ ] Pipeline d'évaluation complet
- [ ] Quality gates automatiques
- [ ] CLI d'évaluation
- [ ] Rapports JSON détaillés
- [ ] Tests unitaires

---

### PHASE 7 : Tests & Documentation (Jour 29-30)

#### 7.1 Tests d'Intégration E2E
```python
# tests/e2e/
├── test_full_pipeline.py    # Ingestion → Retrieval → Generation
├── test_api_scenarios.py    # Scénarios utilisateur complets
└── test_streaming.py        # SSE end-to-end
```

#### 7.2 Tests de Performance
```python
# tests/performance/
├── test_latency.py          # P50, P95, P99
├── test_throughput.py       # Requêtes/seconde
└── test_memory.py           # Utilisation mémoire
```

#### 7.3 Documentation
```markdown
# docs/
├── README.md                # Getting started
├── API.md                   # Documentation API
├── ARCHITECTURE.md          # Architecture détaillée
├── EVALUATION.md            # Guide d'évaluation
└── DEPLOYMENT.md            # Guide de déploiement
```

#### Livrables Phase 7
- [ ] Tests E2E complets
- [ ] Tests de performance
- [ ] Coverage > 80%
- [ ] Documentation complète
- [ ] README avec exemples

---

## Récapitulatif des Jalons

| Phase | Durée | Livrables Clés | Critère de Succès |
|-------|-------|----------------|-------------------|
| **Phase 0** | 2 jours | Setup & config | Projet initialisé, Qdrant up |
| **Phase 1** | 4 jours | Ingestion | 22K lignes indexées, tous jalons présents |
| **Phase 2** | 4 jours | Retrieval | Hybrid search + routing fonctionnels |
| **Phase 3** | 6 jours | Orchestrateur | 5 tools, boucle ReAct stable |
| **Phase 4** | 4 jours | Génération | Réponses grounded avec citations |
| **Phase 5** | 3 jours | API | Endpoints + streaming fonctionnels |
| **Phase 6** | 5 jours | Évaluation | Quality gates passés |
| **Phase 7** | 2 jours | Tests & Docs | Coverage > 80%, docs complètes |

**Total : 30 jours**

---

## Risques Identifiés

| Risque | Probabilité | Impact | Mitigation |
|--------|-------------|--------|------------|
| Chunking fracture les jalons | Haute | Critique | Stratégie dédiée "milestone" |
| Paraphrase des définitions ISO | Haute | Critique | temperature=0.0 + Term Accuracy |
| Confusion GRI vs CIR | Moyenne | Haute | Query router + metadata cycle |
| Latence HF API | Moyenne | Moyenne | Cache, retry, fallback local |
| Tables DOCX mal extraites | Moyenne | Haute | Tests sur vraies tables GRI |

---

## Commandes Utiles

```bash
# Démarrer l'environnement
docker compose up -d qdrant

# Ingestion
python -m src.ingestion.pipeline --input data/raw/IRF20251211_last_FF.docx

# Vérifier l'index
python -m src.core.vector_store --stats

# Mode interactif
python -m src.agents.orchestrator --interactive

# Lancer l'API
uvicorn src.api.main:app --reload --port 8000

# Tests
pytest tests/ -v --asyncio-mode=auto

# Évaluation
python -m src.evaluation.pipeline --dataset data/golden_dataset.json
```

---

*Document généré le 2026-03-17*
*Basé sur CLAUDE.md et les 5 skills GRI*
