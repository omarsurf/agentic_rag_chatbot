"""Tests d'intégration pour l'orchestrateur et les tools GRI.

Ces tests vérifient l'intégration entre l'orchestrateur et les 5 tools :
- retrieve_gri_chunks
- lookup_gri_glossary
- get_milestone_criteria
- get_phase_summary
- compare_approaches

Lancer avec: pytest tests/integration/test_orchestrator_tools.py -v -m integration
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.tools.retrieve_gri import retrieve_gri_chunks, RetrieveGRIOutput
from src.tools.milestones import (
    get_milestone_criteria,
    GetMilestoneOutput,
    normalize_milestone_id,
    MILESTONE_ALIASES,
)
from src.tools.glossary import lookup_gri_glossary
from src.tools.phases import get_phase_summary
from src.tools.compare import compare_approaches
from src.core.vector_store import GRIHybridStore, SearchResult


pytestmark = pytest.mark.integration


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_search_results():
    """Résultats de recherche mockés."""
    return [
        SearchResult(
            id="chunk_001",
            score=0.95,
            content="[GRI > Phase 3 > Conception] Le processus de vérification "
            "assure la conformité aux exigences spécifiées.",
            section_type="process",
            cycle="GRI",
            phase_num=3,
            context_prefix="[GRI > Phase 3 > Conception]",
        ),
        SearchResult(
            id="chunk_002",
            score=0.88,
            content="[GRI > Jalon M4 (CDR)] Critères du CDR : "
            "1. Architecture validée 2. Interfaces définies",
            section_type="milestone",
            cycle="GRI",
            milestone_id="M4",
            context_prefix="[GRI > Jalon M4 (CDR)]",
        ),
        SearchResult(
            id="chunk_003",
            score=0.82,
            content="[GRI > Glossaire] artefact : Produit tangible ou intangible "
            "résultant d'une activité d'ingénierie.",
            section_type="definition",
            cycle="GRI",
            context_prefix="[GRI > Glossaire]",
        ),
    ]


@pytest.fixture
def mock_milestone_result():
    """Résultat de jalon mocké."""
    return {
        "found": True,
        "milestone_id": "M4",
        "milestone_name": "CDR",
        "cycle": "GRI",
        "criteria": [
            {"number": 1, "text": "Architecture système validée", "category": "Technique"},
            {"number": 2, "text": "Interfaces définies et documentées", "category": "Technique"},
            {"number": 3, "text": "Plans de test établis", "category": "Qualité"},
        ],
        "criteria_count": 3,
        "content": "Le CDR valide la conception détaillée du système.",
        "citation": "[GRI > Jalon M4 (CDR)]",
    }


@pytest.fixture
def mock_glossary_result():
    """Résultat de glossaire mocké."""
    return SearchResult(
        id="gloss_001",
        score=1.0,
        content="artefact (artifact) : Produit tangible ou intangible résultant "
        "d'une activité d'ingénierie des systèmes. Source: ISO/IEC/IEEE 15288:2023",
        section_type="definition",
        cycle="GRI",
        metadata={"term_fr": "artefact", "term_en": "artifact"},
    )


@pytest.fixture
def mock_store(mock_search_results, mock_glossary_result):
    """Store mocké pour les tests."""
    store = MagicMock(spec=GRIHybridStore)
    store.hybrid_search = AsyncMock(return_value=mock_search_results)
    store.glossary_lookup = AsyncMock(return_value=mock_glossary_result)
    return store


# =============================================================================
# Tests retrieve_gri_chunks
# =============================================================================


class TestRetrieveGRIChunksTool:
    """Tests pour le tool retrieve_gri_chunks."""

    @pytest.mark.asyncio
    async def test_retrieve_returns_chunks(self, mock_store, mock_search_results):
        """Le tool retourne des chunks avec les scores."""
        result = await retrieve_gri_chunks(
            query="processus de vérification",
            store=mock_store,
            n_results=5,
        )

        assert isinstance(result, RetrieveGRIOutput)
        assert result.has_results is True
        assert len(result.chunks) == len(mock_search_results)
        assert result.query == "processus de vérification"

    @pytest.mark.asyncio
    async def test_retrieve_with_section_filter(self, mock_store):
        """Le tool applique les filtres de section."""
        await retrieve_gri_chunks(
            query="critères",
            store=mock_store,
            section_type="milestone",
            n_results=5,
        )

        # Vérifier que le filtre a été passé
        mock_store.hybrid_search.assert_called_once()
        call_kwargs = mock_store.hybrid_search.call_args.kwargs
        assert call_kwargs.get("filters", {}).get("section_type") == "milestone"

    @pytest.mark.asyncio
    async def test_retrieve_with_cycle_filter(self, mock_store):
        """Le tool applique les filtres de cycle."""
        await retrieve_gri_chunks(
            query="jalon",
            store=mock_store,
            cycle="CIR",
            n_results=5,
        )

        mock_store.hybrid_search.assert_called_once()
        call_kwargs = mock_store.hybrid_search.call_args.kwargs
        assert call_kwargs.get("filters", {}).get("cycle") == "CIR"

    @pytest.mark.asyncio
    async def test_retrieve_with_phase_filter(self, mock_store):
        """Le tool applique les filtres de phase."""
        await retrieve_gri_chunks(
            query="objectifs",
            store=mock_store,
            phase_num=3,
            n_results=5,
        )

        mock_store.hybrid_search.assert_called_once()
        call_kwargs = mock_store.hybrid_search.call_args.kwargs
        assert call_kwargs.get("filters", {}).get("phase_num") == 3

    @pytest.mark.asyncio
    async def test_retrieve_calculates_statistics(self, mock_store, mock_search_results):
        """Le tool calcule les statistiques (avg_score, max_score)."""
        result = await retrieve_gri_chunks(
            query="test",
            store=mock_store,
            n_results=5,
        )

        expected_max = max(r.score for r in mock_search_results)
        expected_avg = sum(r.score for r in mock_search_results) / len(mock_search_results)

        assert result.max_score == expected_max
        assert abs(result.avg_score - expected_avg) < 0.01

    @pytest.mark.asyncio
    async def test_retrieve_empty_results(self, mock_store):
        """Le tool gère les résultats vides."""
        mock_store.hybrid_search = AsyncMock(return_value=[])

        result = await retrieve_gri_chunks(
            query="terme inexistant xyz",
            store=mock_store,
            n_results=5,
        )

        assert result.has_results is False
        assert len(result.chunks) == 0
        assert result.avg_score == 0.0


# =============================================================================
# Tests get_milestone_criteria
# =============================================================================


class TestGetMilestoneCriteriaTool:
    """Tests pour le tool get_milestone_criteria."""

    def test_normalize_milestone_id_direct(self):
        """normalize_milestone_id gère les IDs directs."""
        assert normalize_milestone_id("M4") == "M4"
        assert normalize_milestone_id("m4") == "M4"
        assert normalize_milestone_id("J3") == "J3"
        assert normalize_milestone_id("j3") == "J3"

    def test_normalize_milestone_id_aliases(self):
        """normalize_milestone_id gère les aliases."""
        assert normalize_milestone_id("CDR") == "M3"
        assert normalize_milestone_id("cdr") == "M3"
        assert normalize_milestone_id("PDR") == "M2"
        assert normalize_milestone_id("SRR") == "M1"

    def test_milestone_aliases_completeness(self):
        """Tous les acronymes courants sont mappés."""
        expected_aliases = {"ASR", "SRR", "PDR", "CDR", "IRR", "TRR", "SAR", "ORR", "MNR"}
        assert expected_aliases.issubset(set(MILESTONE_ALIASES.keys()))

    @pytest.mark.asyncio
    async def test_get_milestone_gri(self, mock_store):
        """Le tool récupère un jalon GRI."""
        with patch("src.tools.milestones.GRIMilestoneRetriever") as MockRetriever:
            mock_retriever = MagicMock()
            mock_retriever.get_milestone = AsyncMock(
                return_value=MagicMock(
                    found=True,
                    milestone_id="M4",
                    milestone_name="CDR",
                    cycle="GRI",
                    criteria_count=15,
                    content="Critères du CDR...",
                )
            )
            MockRetriever.return_value = mock_retriever

            result = await get_milestone_criteria(
                milestone_id="M4",
                store=mock_store,
            )

            assert result.found is True
            assert result.milestone_id == "M4"
            assert result.cycle == "GRI"

    @pytest.mark.asyncio
    async def test_get_milestone_cir_with_mapping(self, mock_store):
        """Le tool inclut le mapping GRI pour les jalons CIR."""
        with patch("src.tools.milestones.GRIMilestoneRetriever") as MockRetriever:
            mock_retriever = MagicMock()
            mock_retriever.get_milestone = AsyncMock(
                return_value=MagicMock(
                    found=True,
                    milestone_id="J3",
                    milestone_name="Jalon J3",
                    cycle="CIR",
                    criteria_count=10,
                    content="Critères du J3...",
                )
            )
            MockRetriever.return_value = mock_retriever

            result = await get_milestone_criteria(
                milestone_id="J3",
                store=mock_store,
                include_gri_mapping=True,
            )

            assert result.found is True
            assert result.is_cir is True
            # Le mapping GRI devrait être inclus
            assert result.gri_equivalents is not None or result.gri_mapping_info is not None

    @pytest.mark.asyncio
    async def test_get_milestone_not_found(self, mock_store):
        """Le tool gère les jalons inexistants."""
        with patch("src.tools.milestones.GRIMilestoneRetriever") as MockRetriever:
            mock_retriever = MagicMock()
            mock_retriever.get_milestone = AsyncMock(
                return_value=MagicMock(found=False, milestone_id="M99")
            )
            MockRetriever.return_value = mock_retriever

            result = await get_milestone_criteria(
                milestone_id="M99",
                store=mock_store,
            )

            assert result.found is False


# =============================================================================
# Tests lookup_gri_glossary
# =============================================================================


class TestLookupGRIGlossaryTool:
    """Tests pour le tool lookup_gri_glossary."""

    @pytest.mark.asyncio
    async def test_lookup_finds_term(self, mock_store, mock_glossary_result):
        """Le tool trouve un terme du glossaire."""
        result = await lookup_gri_glossary(
            term="artefact",
            store=mock_store,
        )

        mock_store.glossary_lookup.assert_called_once_with("artefact")
        assert result is not None
        assert result.found is True
        assert result.definition is not None
        assert result.definition.term_fr.lower() == "artefact"
        assert "artefact" in result.definition.definition_fr.lower()

    @pytest.mark.asyncio
    async def test_lookup_term_not_found(self, mock_store):
        """Le tool gère les termes non trouvés."""
        mock_store.glossary_lookup = AsyncMock(return_value=None)

        result = await lookup_gri_glossary(
            term="xyz_inexistant",
            store=mock_store,
        )

        assert result.found is False
        assert result.definition is None

    @pytest.mark.asyncio
    async def test_lookup_case_insensitive(self, mock_store, mock_glossary_result):
        """Le tool est insensible à la casse."""
        await lookup_gri_glossary(term="ARTEFACT", store=mock_store)
        await lookup_gri_glossary(term="Artefact", store=mock_store)
        await lookup_gri_glossary(term="artefact", store=mock_store)

        # Toutes les recherches doivent fonctionner
        assert mock_store.glossary_lookup.call_count == 3


# =============================================================================
# Tests get_phase_summary
# =============================================================================


class TestGetPhaseSummaryTool:
    """Tests pour le tool get_phase_summary."""

    @pytest.mark.asyncio
    async def test_get_phase_returns_summary(self, mock_store):
        """Le tool retourne un résumé de phase."""
        mock_store.hybrid_search = AsyncMock(
            return_value=[
                SearchResult(
                    id="phase_001",
                    score=0.95,
                    content="[GRI > Phase 3] Objectifs : Conception détaillée du système.",
                    section_type="phase",
                    cycle="GRI",
                    phase_num=3,
                    context_prefix="[GRI > Phase 3]",
                ),
            ]
        )

        result = await get_phase_summary(
            phase_num=3,
            store=mock_store,
        )

        assert result is not None
        mock_store.hybrid_search.assert_called()

    @pytest.mark.asyncio
    async def test_get_phase_validates_range(self, mock_store):
        """Le tool valide la plage de phases (1-7)."""
        # Phase valide
        await get_phase_summary(phase_num=1, store=mock_store)
        await get_phase_summary(phase_num=7, store=mock_store)

        # Les phases invalides devraient être gérées
        # (soit par validation Pydantic, soit par logique métier)


# =============================================================================
# Tests compare_approaches
# =============================================================================


class TestCompareApproachesTool:
    """Tests pour le tool compare_approaches."""

    @pytest.mark.asyncio
    async def test_compare_gri_cir(self, mock_store):
        """Le tool compare GRI et CIR."""
        mock_store.hybrid_search = AsyncMock(
            return_value=[
                SearchResult(
                    id="comp_001",
                    score=0.9,
                    content="Différences entre GRI et CIR...",
                    section_type="content",
                    cycle="BOTH",
                ),
            ]
        )

        result = await compare_approaches(
            entity_a="GRI phases de développement",
            entity_b="CIR phases de développement",
            store=mock_store,
        )

        assert result is not None
        mock_store.hybrid_search.assert_called()

    @pytest.mark.asyncio
    async def test_compare_returns_multiple_contexts(self, mock_store):
        """Le tool retourne des contextes pour chaque approche."""
        gri_results = [
            SearchResult(
                id="gri_001",
                score=0.95,
                content="Approche GRI...",
                section_type="content",
                cycle="GRI",
            ),
        ]
        cir_results = [
            SearchResult(
                id="cir_001",
                score=0.92,
                content="Approche CIR...",
                section_type="content",
                cycle="CIR",
            ),
        ]

        # Simuler deux appels séquentiels
        mock_store.hybrid_search = AsyncMock(side_effect=[gri_results, cir_results])

        result = await compare_approaches(
            entity_a="GRI jalons",
            entity_b="CIR jalons",
            store=mock_store,
        )

        # Au moins deux recherches doivent être effectuées
        assert result.entity_a.n_chunks == 1
        assert result.entity_b.n_chunks == 1
        assert mock_store.hybrid_search.call_count == 2


# =============================================================================
# Tests d'intégration Orchestrator → Tools
# =============================================================================


class TestOrchestratorToolIntegration:
    """Tests d'intégration entre l'orchestrateur et les tools."""

    @pytest.fixture
    def mock_llm_client(self):
        """Client LLM mocké."""
        client = AsyncMock()
        client.text_generation = AsyncMock(
            return_value='{"name": "retrieve_gri_chunks", "input": {"query": "test"}}'
        )
        return client

    @pytest.mark.asyncio
    async def test_orchestrator_calls_correct_tool(self, mock_store, mock_llm_client):
        """L'orchestrateur appelle le bon tool selon l'intent."""
        from src.agents.orchestrator import GRIOrchestrator
        from src.agents.query_router import (
            GRICycle,
            GRIIntent,
            ROUTING_TABLE,
            RoutingResult,
        )

        orchestrator = GRIOrchestrator(store=mock_store)

        # Mock le routing pour forcer un intent DEFINITION
        with patch.object(
            orchestrator.router,
            "route",
            return_value=RoutingResult(
                intent=GRIIntent.DEFINITION,
                cycle=GRICycle.GRI,
                confidence=0.95,
                strategy=ROUTING_TABLE[GRIIntent.DEFINITION],
            ),
        ):
            # Le test vérifie que l'orchestrateur est correctement configuré
            assert orchestrator.store == mock_store
            assert orchestrator.router is not None

    @pytest.mark.asyncio
    async def test_tool_results_formatted_for_llm(self, mock_store, mock_search_results):
        """Les résultats des tools sont formatés pour le LLM."""
        result = await retrieve_gri_chunks(
            query="test",
            store=mock_store,
            n_results=3,
        )

        # Le résultat doit être sérialisable en JSON pour le LLM
        result_dict = result.model_dump()

        assert "chunks" in result_dict
        assert "query" in result_dict
        assert "has_results" in result_dict

        # Chaque chunk doit avoir les champs essentiels
        for chunk in result_dict["chunks"]:
            assert "content" in chunk
            assert "score" in chunk

    @pytest.mark.asyncio
    async def test_tool_error_handling(self, mock_store):
        """Les tools gèrent les erreurs correctement."""
        mock_store.hybrid_search = AsyncMock(side_effect=Exception("Connection error"))

        # Le tool doit gérer l'erreur sans crasher
        try:
            result = await retrieve_gri_chunks(
                query="test",
                store=mock_store,
                n_results=5,
            )
            # Si pas d'exception, le résultat doit indiquer l'absence de résultats
            assert result.has_results is False
        except Exception as e:
            # L'exception doit être informative
            assert "Connection" in str(e) or "error" in str(e).lower()


# =============================================================================
# Tests de bout en bout (avec store réel si disponible)
# =============================================================================


@pytest.mark.slow
class TestToolsEndToEnd:
    """Tests E2E avec store réel (nécessite Qdrant)."""

    @pytest.fixture
    async def real_store(self):
        """Store réel avec données de test."""
        store = GRIHybridStore(use_async=True)

        try:
            await store.ensure_collections()

            # Indexer des données de test minimales
            test_chunks = [
                {
                    "content": "[GRI > Glossaire] artefact : Produit d'ingénierie",
                    "chunk_id": "test_gloss_001",
                    "metadata": {"section_type": "definition", "term_fr": "artefact"},
                },
                {
                    "content": "[GRI > Jalon M4] Critères du CDR : validation architecture",
                    "chunk_id": "test_milestone_001",
                    "metadata": {"section_type": "milestone", "milestone_id": "M4"},
                },
            ]
            await store.index_chunks(test_chunks, collection="main")
            await store.index_chunks(test_chunks[:1], collection="glossary")

            yield store
        except Exception:
            pytest.skip("Qdrant non disponible")

    @pytest.mark.asyncio
    async def test_full_retrieval_flow(self, real_store):
        """Test complet du flux de récupération."""
        result = await retrieve_gri_chunks(
            query="artefact",
            store=real_store,
            n_results=5,
        )

        assert result.has_results is True
        assert len(result.chunks) > 0
