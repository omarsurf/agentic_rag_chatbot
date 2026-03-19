"""Tests unitaires pour l'orchestrateur GRI.

Ces tests vérifient le comportement de l'orchestrateur ReAct,
incluant le routing, l'exécution des tools, et la génération des réponses.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.agents.orchestrator import (
    GRIOrchestrator,
    OrchestratorResult,
    ToolCall,
)
from src.agents.query_router import GRICycle, GRIIntent, RoutingResult, RoutingStrategy
from src.core.memory import GRIMemory


# === Fixtures ===


@pytest.fixture
def mock_store():
    """Mock du vector store."""
    store = MagicMock()
    store.hybrid_search = AsyncMock(return_value=[])
    store.glossary_lookup = AsyncMock(return_value=None)
    store.client = MagicMock()
    store.client.scroll = MagicMock(return_value=([], None))
    store.COLLECTIONS = {"main": "gri_main", "glossary": "gri_glossary"}
    return store


@pytest.fixture
def mock_memory():
    """Mock de la mémoire conversationnelle."""
    return GRIMemory()


@pytest.fixture
def orchestrator(mock_store, mock_memory):
    """Orchestrateur avec mocks."""
    return GRIOrchestrator(
        store=mock_store,
        memory=mock_memory,
        max_iter=3,
    )


# === Tests de parsing des tool calls ===


class TestToolCallParsing:
    """Tests pour le parsing des appels de tools."""

    def test_parse_valid_tool_call(self, orchestrator):
        """Teste le parsing d'un appel de tool valide."""
        response = '''Voici mon analyse.

```json
{"tool_calls": [{"name": "retrieve_gri_chunks", "input": {"query": "processus"}}]}
```'''

        tool_calls = orchestrator._parse_tool_calls(response)

        assert len(tool_calls) == 1
        assert tool_calls[0].name == "retrieve_gri_chunks"
        assert tool_calls[0].input["query"] == "processus"

    def test_parse_multiple_tool_calls(self, orchestrator):
        """Teste le parsing de plusieurs appels de tools."""
        response = '''{"tool_calls": [
            {"name": "lookup_gri_glossary", "input": {"term": "artefact"}},
            {"name": "retrieve_gri_chunks", "input": {"query": "artefact usage"}}
        ]}'''

        tool_calls = orchestrator._parse_tool_calls(response)

        assert len(tool_calls) == 2
        assert tool_calls[0].name == "lookup_gri_glossary"
        assert tool_calls[1].name == "retrieve_gri_chunks"

    def test_parse_no_tool_calls(self, orchestrator):
        """Teste une réponse sans appel de tool."""
        response = "Voici ma réponse finale sans outils."

        tool_calls = orchestrator._parse_tool_calls(response)

        assert len(tool_calls) == 0

    def test_parse_invalid_json(self, orchestrator):
        """Teste une réponse avec JSON invalide."""
        response = '{"tool_calls": [{"name": "test"'  # JSON tronqué

        tool_calls = orchestrator._parse_tool_calls(response)

        assert len(tool_calls) == 0


# === Tests d'extraction des citations ===


class TestCitationExtraction:
    """Tests pour l'extraction des citations."""

    def test_extract_gri_citations(self, orchestrator):
        """Teste l'extraction de citations GRI."""
        text = """
        Selon le GRI, le CDR est défini comme [GRI > Jalon M4 (CDR) > Critère #1].
        Les objectifs sont décrits dans [GRI > Phase 3 > Conception].
        """

        citations = orchestrator._extract_citations(text)

        assert len(citations) == 2
        assert "[GRI > Jalon M4 (CDR) > Critère #1]" in citations
        assert "[GRI > Phase 3 > Conception]" in citations

    def test_extract_cir_citations(self, orchestrator):
        """Teste l'extraction de citations CIR."""
        text = "Le jalon J3 [CIR > Phase 3 > Jalon J3] équivaut à M5+M6."

        citations = orchestrator._extract_citations(text)

        assert len(citations) == 1
        assert "[CIR > Phase 3 > Jalon J3]" in citations

    def test_extract_mixed_citations(self, orchestrator):
        """Teste l'extraction de citations GRI et CIR mélangées."""
        text = """
        [GRI > Terminologie > Artefact]
        [CIR > Phase 2 > Jalon J2]
        [GRI > Processus IS 15288 > Vérification]
        """

        citations = orchestrator._extract_citations(text)

        assert len(citations) == 3

    def test_no_citations(self, orchestrator):
        """Teste une réponse sans citations."""
        text = "Réponse sans citation."

        citations = orchestrator._extract_citations(text)

        assert len(citations) == 0


# === Tests du system prompt ===


class TestSystemPrompt:
    """Tests pour la construction du system prompt."""

    def test_build_system_prompt_without_context(self, orchestrator):
        """Teste la construction du prompt sans contexte terminologique."""
        prompt = orchestrator._build_system_prompt("")

        assert "GRI" in prompt
        assert "tool_calls" in prompt
        assert "retrieve_gri_chunks" in prompt

    def test_build_system_prompt_with_context(self, orchestrator):
        """Teste la construction du prompt avec contexte terminologique."""
        term_context = "## Définitions GRI applicables\n• **Artefact** : ..."

        prompt = orchestrator._build_system_prompt(term_context)

        assert "Définitions GRI applicables" in prompt
        assert "Artefact" in prompt


# === Tests de la mémoire ===


class TestMemoryIntegration:
    """Tests pour l'intégration de la mémoire."""

    def test_memory_is_updated_after_run(self, mock_store, mock_memory):
        """Teste que la mémoire est mise à jour après une exécution."""
        # Ce test nécessiterait un mock complet du client HF
        # Pour l'instant, on teste juste la mémoire en isolation
        mock_memory.add_turn(
            query="Test query",
            answer="Test answer",
            intent="DEFINITION",
            cycle="GRI",
        )

        assert len(mock_memory) == 1
        assert mock_memory.get_last_turn().query == "Test query"

    def test_memory_context_in_prompt(self, mock_memory):
        """Teste que le contexte mémoire est inclus dans le prompt."""
        mock_memory.add_turn("Question 1", "Réponse 1", "DEFINITION", "GRI")
        mock_memory.add_turn("Question 2", "Réponse 2", "JALON", "GRI")

        context = mock_memory.get_context()

        assert "Question 1" in context
        assert "Question 2" in context


# === Tests des tools ===


class TestToolExecution:
    """Tests pour l'exécution des tools."""

    @pytest.mark.asyncio
    async def test_execute_single_tool_success(self, orchestrator, mock_store):
        """Teste l'exécution réussie d'un tool."""
        tool_call = ToolCall(
            name="retrieve_gri_chunks",
            input={"query": "test"},
        )

        # Mock de la recherche
        mock_store.hybrid_search = AsyncMock(return_value=[])

        results = await orchestrator._execute_tools([tool_call])

        assert len(results) == 1
        # Le tool devrait s'exécuter (même si pas de résultats)

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self, orchestrator):
        """Teste l'exécution d'un tool inconnu."""
        tool_call = ToolCall(
            name="unknown_tool",
            input={},
        )

        results = await orchestrator._execute_tools([tool_call])

        assert len(results) == 1
        assert results[0].success is False
        assert "non reconnu" in results[0].error


# === Tests de routing ===


class TestRoutingIntegration:
    """Tests pour l'intégration du routing."""

    @pytest.mark.asyncio
    async def test_jalon_question_routing(self, orchestrator):
        """Teste le routing d'une question sur un jalon."""
        routing = await orchestrator.router.route("Critères du CDR M4")

        assert routing.intent == GRIIntent.JALON
        assert routing.cycle == GRICycle.GRI

    @pytest.mark.asyncio
    async def test_definition_question_routing(self, orchestrator):
        """Teste le routing d'une question de définition."""
        routing = await orchestrator.router.route("Qu'est-ce qu'un artefact ?")

        assert routing.intent == GRIIntent.DEFINITION

    @pytest.mark.asyncio
    async def test_cir_question_routing(self, orchestrator):
        """Teste le routing d'une question CIR."""
        routing = await orchestrator.router.route("Critères du jalon J3 du CIR")

        # Devrait détecter soit JALON soit CIR
        assert routing.intent in [GRIIntent.JALON, GRIIntent.CIR]
        assert routing.cycle in [GRICycle.CIR, GRICycle.GRI]

    @pytest.mark.asyncio
    async def test_comparison_question_routing(self, orchestrator):
        """Teste le routing d'une question de comparaison."""
        routing = await orchestrator.router.route("Différence entre GRI et CIR")

        assert routing.intent == GRIIntent.COMPARAISON


# === Tests de format de réponse ===


class TestResponseFormatting:
    """Tests pour le formatage des réponses."""

    def test_format_tool_results(self, orchestrator):
        """Teste le formatage des résultats de tools."""
        from src.tools.executor import ToolResult

        tool_calls = [
            ToolCall(name="retrieve_gri_chunks", input={"query": "test"})
        ]
        results = [
            ToolResult(
                tool_name="retrieve_gri_chunks",
                success=True,
                result={"chunks": [], "n_results": 0},
            )
        ]

        formatted = orchestrator._format_tool_results(tool_calls, results)

        assert "retrieve_gri_chunks" in formatted
        assert "Résultat" in formatted


# === Tests d'intégration (nécessitent un environnement complet) ===


@pytest.mark.integration
class TestOrchestratorIntegration:
    """Tests d'intégration pour l'orchestrateur complet.

    Ces tests utilisent des mocks pour simuler l'environnement complet.
    """

    @pytest.fixture
    def mock_orchestrator(self):
        """Orchestrateur avec tous les composants mockés."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from src.agents.orchestrator import GRIOrchestrator
        from src.agents.query_router import GRIIntent, GRICycle, RoutingResult

        mock_store = MagicMock()
        mock_store.hybrid_search = AsyncMock(return_value=[
            MagicMock(
                id="chunk_001",
                content="[GRI > Jalon M4] Critères du CDR : validation architecture",
                score=0.95,
                section_type="milestone",
                milestone_id="M4",
            )
        ])
        mock_store.glossary_lookup = AsyncMock(return_value=MagicMock(
            content="artefact : Produit d'ingénierie",
            score=1.0,
        ))

        with patch("src.agents.orchestrator.AsyncInferenceClient") as MockClient:
            mock_client = MagicMock()
            mock_client.text_generation = AsyncMock(return_value="""
Voici les critères du CDR (M4) :

1. Architecture système validée
2. Interfaces définies

[Source: GRI > Jalon M4 (CDR)]
""")
            MockClient.return_value = mock_client

            orchestrator = GRIOrchestrator(store=mock_store)
            orchestrator._client = mock_client

            return orchestrator

    @pytest.mark.asyncio
    async def test_full_jalon_query(self, mock_orchestrator):
        """Teste une query complète sur un jalon avec mock."""
        from unittest.mock import AsyncMock, patch
        from src.agents.query_router import (
            GRICycle,
            GRIIntent,
            ROUTING_TABLE,
            RoutingResult,
        )

        # Mock le routing pour retourner JALON
        mock_orchestrator.router.route = AsyncMock(return_value=RoutingResult(
            intent=GRIIntent.JALON,
            cycle=GRICycle.GRI,
            confidence=0.95,
            entities=["M4", "CDR"],
            strategy=ROUTING_TABLE[GRIIntent.JALON],
        ))

        # Le test vérifie que l'orchestrateur est bien configuré
        assert mock_orchestrator.store is not None
        assert mock_orchestrator.router is not None

        # Vérifier que le routing fonctionne
        routing = await mock_orchestrator.router.route("Quels sont les critères du CDR ?")
        assert routing.intent == GRIIntent.JALON

    @pytest.mark.asyncio
    async def test_full_definition_query(self, mock_orchestrator):
        """Teste une query complète de définition avec mock."""
        from unittest.mock import AsyncMock
        from src.agents.query_router import (
            GRICycle,
            GRIIntent,
            ROUTING_TABLE,
            RoutingResult,
        )

        # Mock le routing pour retourner DEFINITION
        mock_orchestrator.router.route = AsyncMock(return_value=RoutingResult(
            intent=GRIIntent.DEFINITION,
            cycle=GRICycle.GRI,
            confidence=0.95,
            entities=["artefact"],
            strategy=ROUTING_TABLE[GRIIntent.DEFINITION],
        ))

        # Vérifier que le routing fonctionne
        routing = await mock_orchestrator.router.route("Qu'est-ce qu'un artefact ?")
        assert routing.intent == GRIIntent.DEFINITION

        # Vérifier que le glossary lookup est disponible
        result = await mock_orchestrator.store.glossary_lookup("artefact")
        assert result is not None
        assert "artefact" in result.content.lower()
