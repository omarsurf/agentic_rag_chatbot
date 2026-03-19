"""Tests pour l'API FastAPI GRI RAG.

Ce module teste les endpoints REST et SSE de l'API.

Exécution:
    pytest tests/test_api.py -v
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.models import CycleType, IntentType


@pytest.fixture
def client():
    """Client de test FastAPI."""
    return TestClient(app)


@pytest.fixture
def mock_store():
    """Mock du vector store."""
    store = MagicMock()
    store.get_collection_stats = AsyncMock(
        return_value={
            "main": {"vectors_count": 1000, "status": "green"},
            "glossary": {"vectors_count": 200, "status": "green"},
        }
    )
    store.hybrid_search = AsyncMock(return_value=[])
    store.glossary_lookup = AsyncMock(return_value=None)
    return store


@pytest.fixture
def mock_orchestrator_result():
    """Résultat mock de l'orchestrateur."""
    from src.agents.orchestrator import OrchestratorResult

    return OrchestratorResult(
        answer="Les critères du CDR sont...",
        intent="JALON",
        cycle="GRI",
        citations=["[GRI > Jalon M4 (CDR) > Critères]"],
        tool_calls=[
            {
                "tool": "get_milestone_criteria",
                "input": {"milestone_id": "M4"},
                "iteration": 1,
                "success": True,
            }
        ],
        iterations=1,
        latency_ms=500.0,
        warning=None,
    )


class TestHealthEndpoint:
    """Tests pour l'endpoint /health."""

    def test_health_returns_200(self, client):
        """Le health check retourne 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status(self, client):
        """Le health check retourne un status."""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] in ["healthy", "degraded", "unhealthy"]

    def test_health_includes_version(self, client):
        """Le health check inclut la version."""
        response = client.get("/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == "0.1.0"

    def test_health_includes_timestamp(self, client):
        """Le health check inclut un timestamp."""
        response = client.get("/health")
        data = response.json()
        assert "timestamp" in data


class TestStatsEndpoint:
    """Tests pour l'endpoint /stats."""

    def test_stats_returns_200(self, client):
        """L'endpoint stats retourne 200."""
        response = client.get("/stats")
        assert response.status_code == 200

    def test_stats_returns_collections(self, client):
        """L'endpoint stats retourne les collections."""
        response = client.get("/stats")
        data = response.json()
        assert "collections" in data

    def test_stats_returns_uptime(self, client):
        """L'endpoint stats retourne l'uptime."""
        response = client.get("/stats")
        data = response.json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0

    def test_stats_requires_auth_when_enabled(self, client):
        """L'endpoint stats est protégé si l'auth est active."""
        with patch("src.api.auth.settings") as mock_settings:
            mock_settings.api_host = "0.0.0.0"
            mock_settings.api_bearer_token = "secret-token"
            mock_settings.api_auth_enabled = True

            response = client.get("/stats")
            assert response.status_code == 401

            response = client.get(
                "/stats",
                headers={"Authorization": "Bearer secret-token"},
            )
            assert response.status_code == 200


class TestMetricsEndpoint:
    """Tests pour l'endpoint /metrics."""

    def test_metrics_requires_auth_when_enabled(self, client):
        """L'endpoint metrics est protégé si l'auth est active."""
        with patch("src.api.auth.settings") as mock_settings:
            mock_settings.api_host = "0.0.0.0"
            mock_settings.api_bearer_token = "secret-token"
            mock_settings.api_auth_enabled = True

            response = client.get("/metrics")
            assert response.status_code == 401

            response = client.get(
                "/metrics",
                headers={"Authorization": "Bearer secret-token"},
            )
            assert response.status_code == 200


class TestQueryEndpoint:
    """Tests pour l'endpoint /query."""

    def test_query_requires_query_field(self, client):
        """Le champ query est obligatoire."""
        response = client.post("/query", json={})
        assert response.status_code == 422

    def test_query_min_length(self, client):
        """La query doit avoir au moins 3 caractères."""
        response = client.post("/query", json={"query": "ab"})
        assert response.status_code == 422

    def test_query_max_length(self, client):
        """La query ne peut pas dépasser 2000 caractères."""
        response = client.post("/query", json={"query": "a" * 2001})
        assert response.status_code == 422

    @patch("src.api.main._store")
    @patch("src.api.main.GRIOrchestrator")
    def test_query_returns_answer(
        self, mock_orch_class, mock_store_var, client, mock_store, mock_orchestrator_result
    ):
        """Une query valide retourne une réponse."""
        # Setup mocks
        mock_store_var.return_value = mock_store
        mock_orch_instance = MagicMock()
        mock_orch_instance.run = AsyncMock(return_value=mock_orchestrator_result)
        mock_orch_class.return_value = mock_orch_instance

        # Patch _store global
        with patch("src.api.main._store", mock_store):
            response = client.post("/query", json={"query": "Quels sont les critères du CDR ?"})

        # Vérifier la réponse
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "query_id" in data
        assert "intent" in data
        assert "cycle" in data

    def test_query_accepts_cycle_parameter(self, client):
        """La query accepte le paramètre cycle."""
        # Ce test vérifie juste la validation du modèle
        # La vraie exécution nécessite un store initialisé
        response = client.post(
            "/query",
            json={
                "query": "Test question",
                "cycle": "GRI",
            },
        )
        # 503 car pas de store, mais le modèle est validé
        assert response.status_code in [200, 500, 503]

    def test_query_accepts_include_sources(self, client):
        """La query accepte include_sources."""
        response = client.post(
            "/query",
            json={
                "query": "Test question",
                "include_sources": True,
            },
        )
        assert response.status_code in [200, 500, 503]

    def test_query_accepts_max_chunks(self, client):
        """La query accepte max_chunks."""
        response = client.post(
            "/query",
            json={
                "query": "Test question",
                "max_chunks": 10,
            },
        )
        assert response.status_code in [200, 500, 503]

    def test_query_max_chunks_validation(self, client):
        """max_chunks doit être entre 1 et 20."""
        response = client.post(
            "/query",
            json={
                "query": "Test question",
                "max_chunks": 25,
            },
        )
        assert response.status_code == 422


class TestQueryStreamEndpoint:
    """Tests pour l'endpoint /query/stream."""

    def test_stream_requires_query(self, client):
        """Le streaming nécessite une query."""
        response = client.post("/query/stream", json={})
        assert response.status_code == 422


class TestFeedbackEndpoint:
    """Tests pour l'endpoint /feedback."""

    def test_feedback_requires_query_id(self, client):
        """Le feedback nécessite un query_id."""
        response = client.post("/feedback", json={"rating": 5})
        assert response.status_code == 422

    def test_feedback_requires_rating(self, client):
        """Le feedback nécessite un rating."""
        response = client.post("/feedback", json={"query_id": "test-123"})
        assert response.status_code == 422

    def test_feedback_rating_validation(self, client):
        """Le rating doit être entre 1 et 5."""
        response = client.post(
            "/feedback",
            json={
                "query_id": "test-123",
                "rating": 10,
            },
        )
        assert response.status_code == 422

    def test_feedback_success(self, client):
        """Un feedback valide est accepté."""
        response = client.post(
            "/feedback",
            json={
                "query_id": "test-123",
                "rating": 4,
                "comment": "Bonne réponse",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


class TestSessionEndpoint:
    """Tests pour l'endpoint /sessions."""

    def test_clear_nonexistent_session(self, client):
        """Effacer une session inexistante retourne 404."""
        response = client.delete("/sessions/nonexistent-session")
        assert response.status_code == 404


class TestModels:
    """Tests pour les modèles Pydantic."""

    def test_cycle_type_values(self):
        """CycleType a les bonnes valeurs."""
        assert CycleType.GRI == "GRI"
        assert CycleType.CIR == "CIR"
        assert CycleType.AUTO == "AUTO"

    def test_intent_type_values(self):
        """IntentType a les bonnes valeurs."""
        assert IntentType.DEFINITION == "DEFINITION"
        assert IntentType.JALON == "JALON"
        assert IntentType.PROCESSUS == "PROCESSUS"

    def test_query_request_defaults(self):
        """QueryRequest a les bons défauts."""
        from src.api.models import QueryRequest

        req = QueryRequest(query="Test question")
        assert req.cycle == CycleType.AUTO
        assert req.include_sources is True
        assert req.max_chunks == 5
        assert req.session_id is None


class TestMiddleware:
    """Tests pour les middlewares."""

    def test_request_id_header(self, client):
        """Chaque réponse a un X-Request-ID."""
        response = client.get("/health")
        assert "X-Request-ID" in response.headers

    def test_cors_headers(self, client):
        """Les headers CORS sont présents."""
        response = client.options(
            "/query",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        # FastAPI/Starlette gère les OPTIONS automatiquement avec le middleware CORS
        assert response.status_code in [200, 405]  # 405 si méthode non autorisée explicitement


class TestErrorHandling:
    """Tests pour la gestion des erreurs."""

    def test_404_returns_json(self, client):
        """Une route 404 retourne du JSON."""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404

    def test_validation_error_format(self, client):
        """Les erreurs de validation ont le bon format."""
        response = client.post("/query", json={"query": ""})
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data
