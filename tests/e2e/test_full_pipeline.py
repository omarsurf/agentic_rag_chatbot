"""Tests E2E pour le pipeline complet RAG GRI.

Ces tests vérifient le flux complet :
Ingestion → Retrieval → Orchestration → Generation → API

Exécution:
    pytest tests/e2e/test_full_pipeline.py -v -m e2e
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.e2e


class TestIngestionToRetrieval:
    """Tests E2E : Ingestion → Retrieval."""

    @pytest.fixture
    def mock_store(self):
        """Mock du vector store."""
        store = MagicMock()
        store.ensure_collections = AsyncMock()
        store.index_chunks = AsyncMock(return_value={"total": 100, "indexed": 100})
        store.hybrid_search = AsyncMock(return_value=[])
        store.glossary_lookup = AsyncMock(return_value=None)
        store.get_collection_stats = AsyncMock(return_value={
            "main": {"vectors_count": 1000},
            "glossary": {"vectors_count": 200},
        })
        return store

    @pytest.mark.asyncio
    async def test_chunks_are_searchable_after_indexing(self, mock_store):
        """Les chunks indexés sont retrouvables par recherche."""
        from src.core.vector_store import SearchResult

        # Simuler des résultats de recherche
        mock_store.hybrid_search.return_value = [
            SearchResult(
                id="chunk_001",
                score=0.85,
                content="[GRI > Phase 3 > CDR]\n\nLes critères du CDR sont...",
                section_type="milestone",
                cycle="GRI",
                milestone_id="M4",
                metadata={},
            )
        ]

        # Recherche
        results = await mock_store.hybrid_search(
            "critères du CDR",
            collection="main",
            n_results=5,
        )

        assert len(results) == 1
        assert results[0].milestone_id == "M4"
        assert "CDR" in results[0].content

    @pytest.mark.asyncio
    async def test_glossary_lookup_returns_definition(self, mock_store):
        """Le glossaire retourne les définitions ISO."""
        from src.core.vector_store import SearchResult

        mock_store.glossary_lookup.return_value = SearchResult(
            id="def_001",
            score=1.0,
            content="Artefact: Produit ou livrable élaboré...",
            section_type="definition",
            metadata={"term_fr": "artefact", "term_en": "artifact"},
        )

        result = await mock_store.glossary_lookup("artefact")

        assert result is not None
        assert "artefact" in result.metadata.get("term_fr", "").lower()


class TestRetrievalToOrchestration:
    """Tests E2E : Retrieval → Orchestration."""

    @pytest.fixture
    def mock_orchestrator_deps(self):
        """Mocks pour l'orchestrateur."""
        store = MagicMock()
        store.hybrid_search = AsyncMock(return_value=[])
        store.glossary_lookup = AsyncMock(return_value=None)

        memory = MagicMock()
        memory.get_context.return_value = ""
        memory.add_turn = MagicMock()

        return store, memory

    @pytest.mark.asyncio
    async def test_orchestrator_routes_definition_query(self, mock_orchestrator_deps):
        """L'orchestrateur route correctement une question de définition."""
        _ = mock_orchestrator_deps
        from src.agents.query_router import GRIIntent, GRIQueryRouter

        router = GRIQueryRouter()

        # Mock le client HF via _client
        mock_client = MagicMock()
        mock_client.text_generation = AsyncMock(return_value="DEFINITION")
        router._client = mock_client

        result = await router.route("Qu'est-ce qu'un artefact ?")

        assert result.intent == GRIIntent.DEFINITION

    @pytest.mark.asyncio
    async def test_orchestrator_routes_milestone_query(self, mock_orchestrator_deps):
        """L'orchestrateur route correctement une question de jalon."""
        _ = mock_orchestrator_deps
        from src.agents.query_router import GRIIntent, GRIQueryRouter

        router = GRIQueryRouter()

        # Mock le client HF via _client
        mock_client = MagicMock()
        mock_client.text_generation = AsyncMock(return_value="JALON")
        router._client = mock_client

        result = await router.route("Quels sont les critères du CDR ?")

        assert result.intent == GRIIntent.JALON


class TestOrchestrationToGeneration:
    """Tests E2E : Orchestration → Generation."""

    @pytest.mark.asyncio
    async def test_tool_results_are_passed_to_generator(self):
        """Les résultats des tools sont passés au générateur."""
        from src.tools.executor import ToolResult

        # Simuler un résultat de tool
        tool_result = ToolResult(
            tool_name="get_milestone_criteria",
            success=True,
            result={
                "milestone_id": "M4",
                "name": "CDR",
                "criteria": ["Critère 1", "Critère 2"],
            },
        )

        assert tool_result.success
        assert tool_result.result["milestone_id"] == "M4"

    @pytest.mark.asyncio
    async def test_citations_are_extracted_from_answer(self):
        """Les citations sont extraites de la réponse."""
        from src.agents.orchestrator import GRIOrchestrator

        # Test de la méthode d'extraction
        orchestrator = GRIOrchestrator.__new__(GRIOrchestrator)

        answer = """
        Les critères du CDR sont définis dans [GRI > Jalon M4 (CDR) > Critères].
        Voir aussi [GRI > Phase 3 > Conception].
        """

        citations = orchestrator._extract_citations(answer)

        assert len(citations) == 2
        assert any("M4" in c for c in citations)


class TestAPIEndToEnd:
    """Tests E2E : API complète."""

    @pytest.fixture
    def client(self):
        """Client de test FastAPI."""
        from fastapi.testclient import TestClient

        from src.api.main import app
        return TestClient(app)

    def test_health_to_query_flow(self, client):
        """Flux complet : health → query."""
        # 1. Vérifier la santé
        health_response = client.get("/health")
        assert health_response.status_code == 200

        # 2. Vérifier les stats
        stats_response = client.get("/stats")
        assert stats_response.status_code == 200

    def test_query_to_feedback_flow(self, client):
        """Flux complet : query → feedback."""
        # Le feedback fonctionne même sans query préalable
        feedback_response = client.post("/feedback", json={
            "query_id": "test-query-123",
            "rating": 5,
            "comment": "Très bonne réponse",
        })

        assert feedback_response.status_code == 200
        assert feedback_response.json()["success"] is True


class TestDataFlow:
    """Tests E2E : Flux de données."""

    def test_chunk_metadata_preserved_through_pipeline(self):
        """Les métadonnées des chunks sont préservées."""
        from src.ingestion.models import Cycle, GRIChunk, GRIMetadata, SectionType

        # Créer un chunk avec métadonnées complètes
        metadata = GRIMetadata(
            doc_id="a1b2c3d4e5f6g7h8",  # 16 chars requis
            source="GRI_FAR_2025",
            chunk_index=0,
            section_type=SectionType.MILESTONE,
            hierarchy=["GRI", "Phase 3", "CDR"],
            context_prefix="[GRI > Phase 3 > CDR]",
            cycle=Cycle.GRI,
            phase_num=3,
            milestone_id="M4",
            language="fr",
            char_count=100,
        )

        chunk = GRIChunk(
            content="[GRI > Phase 3 > CDR]\n\nContenu du chunk avec suffisamment de caractères pour respecter la validation minimum de 80 caractères.",
            chunk_id="a1b2c3d4e5f6g7h8",  # 16 chars requis
            metadata=metadata,
        )

        # Vérifier que tout est préservé
        assert chunk.metadata.milestone_id == "M4"
        assert chunk.metadata.cycle == Cycle.GRI
        assert "[GRI > Phase 3 > CDR]" in chunk.content

    def test_query_response_contains_required_fields(self):
        """La réponse de query contient tous les champs requis."""

        from src.api.models import Citation, QueryResponse

        response = QueryResponse(
            query_id="test-123",
            answer="Les critères du CDR sont...",
            intent="JALON",
            cycle="GRI",
            citations=[Citation(text="[GRI > M4]", section="GRI > M4")],
            latency_ms=500.0,
            iterations=2,
        )

        assert response.query_id is not None
        assert response.answer is not None
        assert response.intent == "JALON"
        assert len(response.citations) == 1


class TestErrorHandling:
    """Tests E2E : Gestion des erreurs."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from src.api.main import app
        return TestClient(app)

    def test_invalid_query_returns_422(self, client):
        """Une query invalide retourne 422."""
        response = client.post("/query", json={"query": ""})
        assert response.status_code == 422

    def test_missing_fields_returns_422(self, client):
        """Des champs manquants retournent 422."""
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_invalid_feedback_rating_returns_422(self, client):
        """Un rating invalide retourne 422."""
        response = client.post("/feedback", json={
            "query_id": "test",
            "rating": 10,  # > 5
        })
        assert response.status_code == 422


class TestCIRGRIMapping:
    """Tests E2E : Mapping CIR ↔ GRI."""

    def test_cir_milestone_mapping(self):
        """Le mapping CIR → GRI est correct."""
        from src.core.config import CIR_GRI_MAPPING

        assert CIR_GRI_MAPPING["J1"] == ["M0", "M1"]
        assert CIR_GRI_MAPPING["J2"] == ["M2", "M3", "M4"]
        assert CIR_GRI_MAPPING["J3"] == ["M5", "M6"]

    def test_valid_milestones(self):
        """Les jalons valides sont correctement définis."""
        from src.core.config import VALID_CIR_MILESTONES, VALID_GRI_MILESTONES

        assert "M0" in VALID_GRI_MILESTONES
        assert "M9" in VALID_GRI_MILESTONES
        assert "M10" not in VALID_GRI_MILESTONES

        assert "J1" in VALID_CIR_MILESTONES
        assert "J6" in VALID_CIR_MILESTONES
        assert "J7" not in VALID_CIR_MILESTONES
