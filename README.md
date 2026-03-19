# Agentic RAG — GRI

Système RAG agentique pour interroger le **Guide de Référence pour l'Innovation (GRI)**.

## Caractéristiques

- **RAG Hybride** : Dense (Qdrant) + Sparse (BM25) avec RRF Fusion (α=0.6)
- **Agent ReAct** : Boucle de raisonnement multi-étapes avec 5 tools spécialisés
- **Query Router** : Classification automatique en 6 intents GRI
- **Conformité ISO** : Définitions ISO/IEC/IEEE 15288:2023 reproduites mot pour mot (température=0.0)
- **Support GRI & CIR** : 7 phases GRI (M0-M9) + 4 phases CIR (J1-J6)
- **SSE Streaming** : Réponses en temps réel avec événements routing/tool_call/chunk/done
- **Session Memory** : Contexte conversationnel multi-tours avec TTL

## Stack Technique

| Composant | Technologie |
|-----------|-------------|
| Language | Python 3.11+ |
| API | FastAPI + SSE Streaming |
| Vector DB | Qdrant |
| Embeddings | `paraphrase-multilingual-mpnet-base-v2` (768-dim, FR+EN) |
| Sparse | BM25 (rank-bm25) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | Hugging Face Inference API (Mixtral-8x7B) |
| Evaluation | RAGAS + Term Accuracy custom |

## Quick Start

```bash
# Installation rapide avec Make
make install-dev

# Démarrer les services
make docker

# Lancer l'API
make run

# Tests
make test
```

## Installation

### Prérequis

- Python 3.11+
- Docker (pour Qdrant)
- Clé API Hugging Face

### Setup

```bash
# Cloner le repo
git clone <repo-url>
cd agentic-rag-gri

# Créer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Installer les dépendances (option 1: avec Make)
make install-dev

# Installer les dépendances (option 2: manuel)
pip install -e ".[dev]"
pre-commit install

# Configurer l'environnement
cp .env.example .env
# Éditer .env avec votre clé HF_API_KEY

# Démarrer Qdrant
make docker
# ou: docker compose up -d
```

## Usage

### 1. Ingestion du document GRI

```bash
# Placer le document dans data/raw/
python -m src.ingestion.pipeline --input data/raw/IRF20251211_last_FF.docx

# Vérifier l'index
python -m src.core.vector_store --stats
```

### 2. Mode interactif

```bash
python -m src.agents.orchestrator --interactive
```

### 3. API

```bash
# Démarrer l'API
uvicorn src.api.main:app --reload --port 8000

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Quels sont les critères du CDR ?"}'
```

### 4. Évaluation

```bash
# Évaluation complète
python -m src.evaluation.pipeline \
  --dataset data/golden_dataset.json \
  --output reports/eval_$(date +%Y%m%d).json

# Smoke test rapide
python -m src.evaluation.pipeline --smoke-test
```

## Structure du Projet

```
agentic-rag-gri/
├── src/
│   ├── agents/          # Orchestrateur ReAct + Query Router
│   ├── core/            # Vector Store, Reranker, Memory
│   ├── tools/           # 5 tools GRI
│   ├── ingestion/       # Pipeline d'ingestion
│   ├── evaluation/      # RAGAS + Term Accuracy
│   └── api/             # FastAPI endpoints
├── data/
│   ├── raw/             # Document source GRI
│   └── golden_dataset.json
├── configs/
│   └── config.yaml
├── tests/
└── reports/
```

## Les 5 Tools de l'Agent

| Tool | Usage |
|------|-------|
| `retrieve_gri_chunks` | Recherche hybride dans la base GRI |
| `lookup_gri_glossary` | Définitions ISO exactes |
| `get_milestone_criteria` | Critères complets d'un jalon |
| `compare_approaches` | Comparaison de deux éléments |
| `get_phase_summary` | Résumé structuré d'une phase |

## Quality Gates

| Métrique | Seuil |
|----------|-------|
| Faithfulness | ≥ 0.85 |
| Answer Relevance | ≥ 0.80 |
| Context Recall | ≥ 0.75 |
| **Term Accuracy** | **≥ 0.95** |
| Latency P95 | ≤ 8s |

## Tests

```bash
# Tests unitaires
make test

# Tous les tests avec coverage
make test-all

# Tests rapides (sans slow)
make test-fast

# Lint et type check
make lint

# Formater le code
make format
```

## Commandes Make Disponibles

```bash
make help           # Afficher toutes les commandes
make install-dev    # Installation dev + pre-commit hooks
make test           # Tests unitaires
make test-all       # Tous les tests + coverage
make lint           # Ruff + Black + MyPy
make format         # Formater le code
make docker         # Démarrer Qdrant
make run            # Lancer l'API
make clean          # Nettoyer les fichiers générés
```

## API Endpoints

| Endpoint | Méthode | Description |
|----------|---------|-------------|
| `/query` | POST | Question standard (JSON response) |
| `/query/stream` | POST | Question avec SSE streaming |
| `/health` | GET | Health check |
| `/stats` | GET | Statistiques de l'index |
| `/feedback` | POST | Soumettre un feedback |
| `/feedback/stats` | GET | Statistiques de feedback |
| `/sessions/{id}` | DELETE | Effacer une session |

### Exemple de requête

```bash
# Query standard
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Quels sont les critères du CDR ?",
    "cycle": "AUTO",
    "include_sources": true
  }'

# Avec streaming SSE
curl -N -X POST http://localhost:8000/query/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Définition d'\''artefact"}'

# Feedback
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "abc-123",
    "rating": 5,
    "comment": "Réponse précise"
  }'
```

## Documentation API

La documentation OpenAPI interactive est disponible à :
- Swagger UI : http://localhost:8000/docs
- ReDoc : http://localhost:8000/redoc

## CI/CD

Le projet utilise GitHub Actions pour :
- Lint (Ruff, Black, MyPy)
- Tests unitaires avec coverage
- Tests d'intégration (avec Qdrant)
- Build Docker

```bash
# Exécuter les hooks pre-commit manuellement
pre-commit run --all-files
```

