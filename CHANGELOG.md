# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-03-20

### Added

#### Core Features
- **Agentic RAG System** with ReAct-style orchestration (reasoning + acting loop)
- **Hybrid Retrieval** combining Qdrant dense vectors + BM25 sparse search with RRF fusion (alpha=0.6)
- **5 Specialized Tools** for GRI document queries:
  - `retrieve_gri_chunks` - Hybrid search with section filtering
  - `lookup_gri_glossary` - Exact ISO definitions (temperature=0)
  - `get_milestone_criteria` - Complete milestone checklists
  - `compare_approaches` - Multi-retrieve for comparisons
  - `get_phase_summary` - Structured phase overviews
- **Query Router** with 6 intent classifications (DEFINITION, PROCESSUS, JALON, PHASE_COMPLETE, COMPARAISON, CIR)
- **SSE Streaming API** with real-time events (routing, tool_call, chunk, done)
- **Session Memory** with multi-turn context and TTL support

#### Evaluation
- **Custom Term Accuracy metric** for ISO/IEC/IEEE 15288:2023 compliance (target: 95%+)
- **RAGAS integration** for standard RAG metrics (faithfulness, relevance, recall)
- **Quality gates** enforced: faithfulness 85%, relevance 80%, recall 75%

#### Infrastructure
- **FastAPI** with rate limiting, authentication, health checks
- **Multi-stage Docker build** with non-root user (security hardening)
- **docker-compose stack**: Qdrant + PostgreSQL + Redis
- **Prometheus metrics** endpoint for observability

#### CI/CD
- **4-job GitHub Actions pipeline**: lint, unit tests, integration tests, Docker build
- **80% coverage threshold** enforced
- **Pre-commit hooks** (7 hooks including secret detection)
- **Strict MyPy typing** with `disallow_untyped_defs`

### Known Limitations

- Requires Hugging Face API key for LLM inference (Mixtral-8x7B)
- French language support only (multilingual planned)
- API-only interface (no UI frontend)
- Single-tenant sessions (no multi-user isolation)

### Technical Debt

- Screenshots for documentation not yet captured
- Golden dataset limited to ~50 annotated questions
- No automated benchmark suite

---

## Future Releases

### [0.2.0] - Planned
- OAuth2/OIDC authentication
- Grafana observability dashboard
- Expanded golden dataset (100+ questions)

### [0.3.0] - Planned
- UI frontend (React/Next.js)
- Cloud deployment with Terraform
- Background ingestion jobs
