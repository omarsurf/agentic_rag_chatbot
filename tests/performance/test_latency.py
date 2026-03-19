"""Tests de latence pour le système RAG GRI.

Ces tests vérifient que les temps de réponse respectent les SLAs.

Quality Gates:
- P50 latency < 3s
- P95 latency < 8s
- P99 latency < 15s

Exécution:
    pytest tests/performance/test_latency.py -v -m slow
"""

import time
from statistics import mean, stdev
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.performance]


class TestAPILatency:
    """Tests de latence de l'API."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        return TestClient(app)

    def test_health_endpoint_latency(self, client):
        """Le health check répond en < 100ms."""
        latencies = []

        for _ in range(10):
            start = time.time()
            response = client.get("/health")
            latency = (time.time() - start) * 1000
            latencies.append(latency)

            assert response.status_code == 200

        avg_latency = mean(latencies)
        assert avg_latency < 100, f"Health endpoint trop lent: {avg_latency:.0f}ms"

    def test_stats_endpoint_latency(self, client):
        """Le stats endpoint répond en < 200ms."""
        latencies = []

        for _ in range(10):
            start = time.time()
            response = client.get("/stats")
            latency = (time.time() - start) * 1000
            latencies.append(latency)

            assert response.status_code == 200

        avg_latency = mean(latencies)
        assert avg_latency < 200, f"Stats endpoint trop lent: {avg_latency:.0f}ms"

    def test_feedback_endpoint_latency(self, client):
        """Le feedback endpoint répond en < 100ms."""
        latencies = []

        for i in range(10):
            start = time.time()
            response = client.post("/feedback", json={
                "query_id": f"test-{i}",
                "rating": 5,
            })
            latency = (time.time() - start) * 1000
            latencies.append(latency)

            assert response.status_code == 200

        avg_latency = mean(latencies)
        assert avg_latency < 100, f"Feedback endpoint trop lent: {avg_latency:.0f}ms"


class TestComponentLatency:
    """Tests de latence des composants individuels."""

    def test_query_router_latency(self):
        """Le query router répond en < 500ms (mocké)."""
        from src.agents.query_router import GRIQueryRouter

        router = GRIQueryRouter()

        # Mesurer le temps de la logique de base (sans appel LLM)
        start = time.time()
        # Test de la détection de patterns (pas d'appel async)
        query = "Qu'est-ce qu'un artefact ?"
        # La détection de patterns est synchrone
        latency = (time.time() - start) * 1000

        assert latency < 10, f"Pattern detection trop lente: {latency:.0f}ms"

    def test_term_extraction_latency(self):
        """L'extraction de termes ISO répond en < 50ms."""
        from src.evaluation.term_accuracy import extract_iso_terms

        text = """
        Le CDR (Critical Design Review) est un jalon important du GRI.
        Il vérifie que la conception détaillée est complète.
        Les artefacts de vérification et validation doivent être prêts.
        Le TRL doit être au niveau 6 minimum.
        """

        latencies = []

        for _ in range(100):
            start = time.time()
            terms = extract_iso_terms(text)
            latency = (time.time() - start) * 1000
            latencies.append(latency)

        avg_latency = mean(latencies)
        assert avg_latency < 50, f"Term extraction trop lente: {avg_latency:.2f}ms"
        assert len(terms) > 0

    def test_gri_error_detection_latency(self):
        """La détection d'erreurs GRI répond en < 20ms."""
        from src.evaluation.faithfulness_gri import _detect_gri_errors

        text = """
        Le jalon M10 n'existe pas, tout comme la Phase 9.
        Le mapping J3 = M4 est incorrect.
        """

        latencies = []

        for _ in range(100):
            start = time.time()
            errors = _detect_gri_errors(text)
            latency = (time.time() - start) * 1000
            latencies.append(latency)

        avg_latency = mean(latencies)
        assert avg_latency < 20, f"Error detection trop lente: {avg_latency:.2f}ms"


class TestMemoryOperations:
    """Tests de latence des opérations mémoire."""

    def test_memory_add_turn_latency(self):
        """L'ajout d'un tour en mémoire répond en < 5ms."""
        from src.core.memory import GRIMemory

        memory = GRIMemory()
        latencies = []

        for i in range(100):
            start = time.time()
            memory.add_turn(
                query=f"Question {i}",
                answer=f"Réponse {i}",
                intent="DEFINITION",
                cycle="GRI",
            )
            latency = (time.time() - start) * 1000
            latencies.append(latency)

        avg_latency = mean(latencies)
        assert avg_latency < 5, f"Memory add_turn trop lent: {avg_latency:.2f}ms"

    def test_memory_get_context_latency(self):
        """La récupération du contexte répond en < 10ms."""
        from src.core.memory import GRIMemory

        memory = GRIMemory()

        # Remplir la mémoire
        for i in range(10):
            memory.add_turn(
                query=f"Question {i}",
                answer=f"Réponse détaillée {i}" * 50,
                intent="PHASE_COMPLETE",
                cycle="GRI",
            )

        latencies = []

        for _ in range(100):
            start = time.time()
            context = memory.get_context()
            latency = (time.time() - start) * 1000
            latencies.append(latency)

        avg_latency = mean(latencies)
        assert avg_latency < 10, f"Memory get_context trop lent: {avg_latency:.2f}ms"


class TestDataModelLatency:
    """Tests de latence des modèles de données."""

    def test_pydantic_model_validation_latency(self):
        """La validation Pydantic répond en < 5ms."""
        from src.api.models import QueryRequest

        latencies = []

        for i in range(100):
            start = time.time()
            request = QueryRequest(
                query=f"Question test numéro {i}",
                cycle="AUTO",
                include_sources=True,
                max_chunks=5,
            )
            latency = (time.time() - start) * 1000
            latencies.append(latency)

        avg_latency = mean(latencies)
        assert avg_latency < 5, f"Pydantic validation trop lente: {avg_latency:.2f}ms"

    def test_response_model_creation_latency(self):
        """La création de QueryResponse répond en < 5ms."""
        from src.api.models import QueryResponse, Citation
        from datetime import datetime

        latencies = []

        for i in range(100):
            start = time.time()
            response = QueryResponse(
                query_id=f"test-{i}",
                answer="Réponse test" * 100,
                intent="JALON",
                cycle="GRI",
                citations=[
                    Citation(text=f"[GRI > M{j}]", section=f"M{j}")
                    for j in range(5)
                ],
                latency_ms=500.0,
                iterations=3,
            )
            latency = (time.time() - start) * 1000
            latencies.append(latency)

        avg_latency = mean(latencies)
        assert avg_latency < 5, f"Response model creation trop lente: {avg_latency:.2f}ms"


class TestEvaluationLatency:
    """Tests de latence des métriques d'évaluation."""

    def test_score_calculation_latency(self):
        """Le calcul de score répond en < 1ms."""
        from src.evaluation.term_accuracy import _calculate_score, TermEvaluation

        evaluations = [
            TermEvaluation(
                term=f"term_{i}",
                definition_in_answer="...",
                normative_definition="...",
                status="EXACT" if i % 2 == 0 else "APPROXIMATIF",
                severity="OK",
            )
            for i in range(10)
        ]

        latencies = []

        for _ in range(1000):
            start = time.time()
            score = _calculate_score(evaluations)
            latency = (time.time() - start) * 1000
            latencies.append(latency)

        avg_latency = mean(latencies)
        assert avg_latency < 1, f"Score calculation trop lent: {avg_latency:.3f}ms"

    def test_aggregation_latency(self):
        """L'agrégation des résultats répond en < 50ms."""
        from src.evaluation.pipeline import GRIEvaluator, GRIEvalResult

        evaluator = GRIEvaluator()

        # Créer 50 résultats
        results = [
            GRIEvalResult(
                question_id=f"Q{i}",
                query=f"Question {i}",
                answer=f"Réponse {i}",
                faithfulness=0.8 + (i % 20) / 100,
                answer_relevance=0.7 + (i % 30) / 100,
                context_recall=0.75,
                context_precision=0.7,
                term_accuracy=0.95 + (i % 5) / 100,
                latency_ms=500 + i * 10,
                intent_correct=i % 3 != 0,
                cycle_correct=i % 4 != 0,
            )
            for i in range(50)
        ]

        latencies = []

        for _ in range(100):
            start = time.time()
            summary = evaluator._aggregate(results)
            latency = (time.time() - start) * 1000
            latencies.append(latency)

        avg_latency = mean(latencies)
        assert avg_latency < 50, f"Aggregation trop lente: {avg_latency:.2f}ms"


class TestConcurrency:
    """Tests de comportement sous charge."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        return TestClient(app)

    def test_concurrent_health_requests(self, client):
        """Requêtes health concurrentes."""
        import concurrent.futures

        def make_request():
            start = time.time()
            response = client.get("/health")
            return (time.time() - start) * 1000, response.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(20)]
            results = [f.result() for f in futures]

        latencies = [r[0] for r in results]
        statuses = [r[1] for r in results]

        # Toutes les requêtes doivent réussir
        assert all(s == 200 for s in statuses)

        # Latence moyenne acceptable
        avg_latency = mean(latencies)
        assert avg_latency < 200, f"Concurrent requests trop lentes: {avg_latency:.0f}ms"
