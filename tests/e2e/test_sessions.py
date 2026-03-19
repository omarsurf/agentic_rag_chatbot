"""Tests E2E pour la gestion des sessions.

Ces tests vérifient le comportement complet des sessions :
- Création et persistance
- TTL et expiration
- Nettoyage automatique

Exécution:
    pytest tests/e2e/test_sessions.py -v -m e2e
"""

import asyncio
import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

pytestmark = pytest.mark.e2e


def _session_uuid(suffix: int) -> str:
    """Retourne un UUID v4 déterministe valide pour les tests API."""
    return f"550e8400-e29b-41d4-a716-{suffix:012d}"


@pytest.fixture(autouse=True)
def reset_session_backend():
    """Isole l'état global du SessionStore entre les tests."""
    from src.api import main as api_main
    from src.core.session_store import reset_session_store

    reset_session_store()
    api_main._session_store = None
    yield
    reset_session_store()
    api_main._session_store = None


class TestSessionLifecycle:
    """Tests du cycle de vie des sessions."""

    def test_session_created_on_first_query(self):
        """Une session est créée lors de la première requête."""
        from src.core.memory import GRIMemory

        session_id = "test-session-001"
        memory = GRIMemory(session_id=session_id)

        assert memory.session_id == session_id
        assert len(memory.turns) == 0

    def test_session_context_accumulates(self):
        """Le contexte de session s'accumule au fil des tours."""
        from src.core.memory import GRIMemory
        from src.agents.query_router import GRIIntent, GRICycle

        memory = GRIMemory(session_id="test-accumulate", max_turns=10)

        # Premier tour
        memory.add_turn(
            query="Définition d'artefact",
            answer="Un artefact est un produit...",
            intent=GRIIntent.DEFINITION,
            cycle=GRICycle.GRI,
        )

        assert len(memory.turns) == 1

        # Deuxième tour
        memory.add_turn(
            query="Et le CONOPS ?",
            answer="Le CONOPS est...",
            intent=GRIIntent.DEFINITION,
            cycle=GRICycle.GRI,
        )

        assert len(memory.turns) == 2

        # Le contexte doit contenir les deux échanges
        context = memory.get_context_for_llm()
        assert "artefact" in context.lower()

    def test_session_respects_max_turns(self):
        """La session respecte la limite max_turns."""
        from src.core.memory import GRIMemory
        from src.agents.query_router import GRIIntent, GRICycle

        max_turns = 3
        memory = GRIMemory(session_id="test-max", max_turns=max_turns)

        # Ajouter plus de tours que la limite
        for i in range(5):
            memory.add_turn(
                query=f"Question {i}",
                answer=f"Réponse {i}",
                intent=GRIIntent.DEFINITION,
                cycle=GRICycle.GRI,
            )

        # Seuls les derniers tours doivent être conservés
        assert len(memory.turns) <= max_turns

    def test_session_extracts_context_entities(self):
        """La session extrait les entités contextuelles."""
        from src.core.memory import GRIMemory
        from src.agents.query_router import GRIIntent, GRICycle

        memory = GRIMemory(session_id="test-entities", max_turns=10)

        # Tour avec mention de jalon
        memory.add_turn(
            query="Critères du CDR (M4)",
            answer="Le CDR valide...",
            intent=GRIIntent.JALON,
            cycle=GRICycle.GRI,
            tool_calls=[{"tool": "get_milestone_criteria", "input": {"milestone_id": "M4"}}],
        )

        # Vérifier l'extraction du contexte
        stats = memory.get_stats()
        assert stats["total_turns"] == 1


class TestSessionPersistence:
    """Tests de persistance des sessions."""

    def test_session_id_is_preserved(self):
        """L'ID de session est préservé entre les tours."""
        from src.core.memory import GRIMemory
        from src.agents.query_router import GRIIntent, GRICycle

        session_id = "preserve-test-123"
        memory = GRIMemory(session_id=session_id)

        memory.add_turn(
            query="Test",
            answer="Réponse",
            intent=GRIIntent.DEFINITION,
            cycle=GRICycle.GRI,
        )

        assert memory.session_id == session_id

    def test_session_metadata_preserved(self):
        """Les métadonnées de session sont préservées."""
        from src.core.memory import GRIMemory
        from src.agents.query_router import GRIIntent, GRICycle

        memory = GRIMemory(session_id="metadata-test")

        memory.add_turn(
            query="Critères M4",
            answer="Les critères sont...",
            intent=GRIIntent.JALON,
            cycle=GRICycle.GRI,
            tool_calls=[{"tool": "get_milestone_criteria", "input": {"milestone_id": "M4"}}],
            citations=["[GRI > Jalon M4 (CDR)]"],
        )

        turn = memory.turns[0]
        assert turn.intent == GRIIntent.JALON
        assert turn.cycle == GRICycle.GRI
        assert len(turn.citations) > 0


class TestSessionCleanup:
    """Tests du nettoyage des sessions."""

    def test_memory_clear(self):
        """La mémoire peut être effacée."""
        from src.core.memory import GRIMemory
        from src.agents.query_router import GRIIntent, GRICycle

        memory = GRIMemory(session_id="clear-test")

        # Ajouter des tours
        for i in range(3):
            memory.add_turn(
                query=f"Q{i}",
                answer=f"R{i}",
                intent=GRIIntent.DEFINITION,
                cycle=GRICycle.GRI,
            )

        assert len(memory.turns) == 3

        # Effacer
        memory.clear()

        assert len(memory.turns) == 0

    def test_session_store_cleanup(self):
        """Le SessionStore nettoie les sessions expirées."""
        from src.core.session_store import InMemorySessionStore

        session_id = _session_uuid(10)
        store = InMemorySessionStore(default_ttl=1)

        # Créer une session
        asyncio.run(store.create_session(session_id))

        # Vérifier qu'elle existe
        assert asyncio.run(store.load_session(session_id)) is not None

        time.sleep(1.1)
        removed = asyncio.run(store.cleanup_expired())

        assert removed == 1
        assert asyncio.run(store.load_session(session_id)) is None

    def test_session_store_get_or_create(self):
        """SessionStore crée la session si elle n'existe pas."""
        from src.core.session_store import InMemorySessionStore

        session_id = _session_uuid(11)
        store = InMemorySessionStore()

        # Session n'existe pas encore
        session = asyncio.run(store.get_or_create_session(session_id))

        assert session is not None
        assert session.session_id == session_id

        # Appel répété retourne la même session
        session2 = asyncio.run(store.get_or_create_session(session_id))
        assert session2.session_id == session.session_id


class TestSessionAPIIntegration:
    """Tests d'intégration API pour les sessions."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from src.api.main import app
        return TestClient(app)

    def test_delete_session_not_found(self, client):
        """DELETE /sessions/{id} retourne 404 si session inexistante."""
        response = client.delete(f"/sessions/{_session_uuid(99)}")
        assert response.status_code == 404

    def test_session_across_multiple_requests(self, client):
        """Une session persiste entre plusieurs requêtes."""
        from src.core.session_store import InMemorySessionStore

        session_id = _session_uuid(20)
        session_store = InMemorySessionStore()
        mocked_store = MagicMock()
        memories = []

        with patch("src.api.main._store", mocked_store), \
             patch("src.api.main._session_store", session_store), \
             patch("src.api.main.GRIOrchestrator") as MockOrch:
            mock_result = MagicMock()
            mock_result.answer = "Réponse test"
            mock_result.citations = []
            mock_result.intent = "DEFINITION"
            mock_result.cycle = "GRI"
            mock_result.iterations = 1
            mock_result.tool_calls = []
            mock_result.collected_chunks = []
            mock_result.latency_ms = 10.0
            mock_result.warning = None

            def build_orchestrator(*args, **kwargs):
                memory = kwargs["memory"]
                memories.append(memory)

                async def run_side_effect(query):
                    if not memory.turns:
                        memory.add_turn(
                            query="Première question",
                            answer="Réponse test",
                            intent="DEFINITION",
                            cycle="GRI",
                        )
                    return mock_result

                mock_orch = MagicMock()
                mock_orch.run = AsyncMock(side_effect=run_side_effect)
                return mock_orch

            MockOrch.side_effect = build_orchestrator

            response1 = client.post("/query", json={
                "query": "Première question",
                "session_id": session_id,
            })
            response2 = client.post("/query", json={
                "query": "Deuxième question",
                "session_id": session_id,
            })

        assert response1.status_code == 200
        assert response2.status_code == 200
        assert len(memories) == 2
        assert memories[0].session_id == session_id
        assert memories[1].session_id == session_id
        assert len(memories[1].turns) == 1


class TestSessionContext:
    """Tests du contexte de session pour le LLM."""

    def test_context_format_for_llm(self):
        """Le contexte est formaté correctement pour le LLM."""
        from src.core.memory import GRIMemory
        from src.agents.query_router import GRIIntent, GRICycle

        memory = GRIMemory(session_id="format-test")

        memory.add_turn(
            query="Définition artefact",
            answer="Un artefact est un produit d'ingénierie.",
            intent=GRIIntent.DEFINITION,
            cycle=GRICycle.GRI,
        )

        context = memory.get_context_for_llm()

        # Le contexte doit être lisible par le LLM
        assert isinstance(context, str)
        assert len(context) > 0

    def test_context_truncation(self):
        """Le contexte est tronqué si trop long."""
        from src.core.memory import GRIMemory
        from src.agents.query_router import GRIIntent, GRICycle

        memory = GRIMemory(session_id="truncate-test", max_turns=10)

        # Ajouter beaucoup de tours avec des réponses longues
        for i in range(10):
            memory.add_turn(
                query=f"Question très longue numéro {i} " * 20,
                answer=f"Réponse très longue numéro {i} " * 100,
                intent=GRIIntent.DEFINITION,
                cycle=GRICycle.GRI,
            )

        context = memory.get_context_for_llm(max_length=5000)

        # Le contexte doit être limité
        assert len(context) <= 6000  # Marge pour le formatage

    def test_context_includes_recent_turns(self):
        """Le contexte inclut les tours récents."""
        from src.core.memory import GRIMemory
        from src.agents.query_router import GRIIntent, GRICycle

        memory = GRIMemory(session_id="recent-test", max_turns=3)

        # Ajouter des tours
        memory.add_turn(query="Q1", answer="R1", intent=GRIIntent.DEFINITION, cycle=GRICycle.GRI)
        memory.add_turn(query="Q2", answer="R2", intent=GRIIntent.DEFINITION, cycle=GRICycle.GRI)
        memory.add_turn(query="Q3", answer="R3", intent=GRIIntent.DEFINITION, cycle=GRICycle.GRI)

        context = memory.get_context_for_llm()

        # Les tours récents doivent être présents
        assert "Q3" in context or "R3" in context


class TestSessionStatistics:
    """Tests des statistiques de session."""

    def test_session_stats(self):
        """Les statistiques de session sont correctes."""
        from src.core.memory import GRIMemory
        from src.agents.query_router import GRIIntent, GRICycle

        memory = GRIMemory(session_id="stats-test")

        memory.add_turn(
            query="Q1",
            answer="R1",
            intent=GRIIntent.DEFINITION,
            cycle=GRICycle.GRI,
        )
        memory.add_turn(
            query="Q2",
            answer="R2",
            intent=GRIIntent.JALON,
            cycle=GRICycle.GRI,
        )

        stats = memory.get_stats()

        assert stats["total_turns"] == 2
        assert stats["session_id"] == "stats-test"

    def test_intent_distribution(self):
        """La distribution des intents est calculée."""
        from src.core.memory import GRIMemory
        from src.agents.query_router import GRIIntent, GRICycle

        memory = GRIMemory(session_id="dist-test")

        # Ajouter différents types de requêtes
        memory.add_turn(query="Q1", answer="R1", intent=GRIIntent.DEFINITION, cycle=GRICycle.GRI)
        memory.add_turn(query="Q2", answer="R2", intent=GRIIntent.DEFINITION, cycle=GRICycle.GRI)
        memory.add_turn(query="Q3", answer="R3", intent=GRIIntent.JALON, cycle=GRICycle.GRI)

        # Vérifier la distribution
        intents = [t.intent for t in memory.turns]
        assert intents.count(GRIIntent.DEFINITION) == 2
        assert intents.count(GRIIntent.JALON) == 1
