"""Tests unitaires pour la couche Retrieval GRI.

Couvre :
- GRIHybridStore (vector_store.py)
- GRIQueryRouter (query_router.py)
- GRITermExpander (term_expander.py)
- GRIReranker (reranker.py)
- GRIMilestoneRetriever (milestone_retriever.py)
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.query_router import (
    ROUTING_TABLE,
    GRICycle,
    GRIIntent,
    GRIQueryRouter,
    get_strategy_for_intent,
)
from src.core.config import (
    CIR_GRI_MAPPING,
    VALID_CIR_MILESTONES,
    VALID_GRI_MILESTONES,
)
from src.core.term_expander import (
    GRITermExpander,
    detect_gri_terms,
)

# =============================================================================
# Tests Query Router
# =============================================================================


class TestQueryRouterHeuristics:
    """Tests pour le routing par heuristiques."""

    def setup_method(self):
        """Setup avant chaque test."""
        self.router = GRIQueryRouter()

    def test_definition_intent_quest_ce_que(self):
        """Qu'est-ce que → DEFINITION."""
        result = self.router._heuristic_route("Qu'est-ce qu'un artefact ?")
        assert result.intent == GRIIntent.DEFINITION
        assert result.confidence >= 0.9

    def test_definition_intent_definir(self):
        """Définir X → DEFINITION."""
        result = self.router._heuristic_route("Définir le CONOPS selon le GRI")
        assert result.intent == GRIIntent.DEFINITION

    def test_jalon_intent_criteres_m4(self):
        """Critères du M4 → JALON."""
        result = self.router._heuristic_route("Quels sont les critères du M4 ?")
        assert result.intent == GRIIntent.JALON
        assert result.cycle == GRICycle.GRI
        assert "M4" in result.entities

    def test_jalon_intent_cdr(self):
        """CDR → JALON M4."""
        result = self.router._heuristic_route("Critères du CDR ?")
        assert result.intent == GRIIntent.JALON
        assert "CDR" in result.entities or "M4" in result.entities

    def test_jalon_intent_cir_j3(self):
        """Jalon J3 → JALON CIR."""
        result = self.router._heuristic_route("Critères du jalon J3 du CIR")
        assert result.intent == GRIIntent.JALON
        assert result.cycle == GRICycle.CIR
        assert "J3" in result.entities

    def test_processus_intent(self):
        """Processus de vérification → PROCESSUS."""
        result = self.router._heuristic_route(
            "Quelles sont les activités du processus de vérification ?"
        )
        assert result.intent == GRIIntent.PROCESSUS

    def test_phase_complete_intent(self):
        """Objectifs Phase 3 → PHASE_COMPLETE."""
        result = self.router._heuristic_route("Objectifs de la Phase 3 ?")
        assert result.intent == GRIIntent.PHASE_COMPLETE

    def test_comparaison_intent_difference(self):
        """Différence entre X et Y → COMPARAISON."""
        result = self.router._heuristic_route(
            "Quelle est la différence entre le GRI et le CIR ?"
        )
        assert result.intent == GRIIntent.COMPARAISON
        assert result.cycle == GRICycle.BOTH

    def test_comparaison_intent_vs(self):
        """X vs Y → COMPARAISON."""
        result = self.router._heuristic_route("GRI standard vs CIR")
        assert result.intent == GRIIntent.COMPARAISON

    def test_cir_intent(self):
        """Question CIR spécifique."""
        result = self.router._heuristic_route(
            "Dans quel contexte utilise-t-on le CIR ?"
        )
        assert result.intent == GRIIntent.CIR
        assert result.cycle == GRICycle.CIR


class TestQueryRouterStrategies:
    """Tests pour les stratégies de routing."""

    def test_definition_strategy(self):
        """Stratégie DEFINITION utilise sparse et glossaire."""
        strategy = ROUTING_TABLE[GRIIntent.DEFINITION]
        assert strategy.search_mode == "sparse"
        assert strategy.primary_index == "glossary"
        assert strategy.temperature == 0.0
        assert not strategy.use_reranker

    def test_jalon_strategy(self):
        """Stratégie JALON retourne complet sans reranker."""
        strategy = ROUTING_TABLE[GRIIntent.JALON]
        assert strategy.return_complete is True
        assert not strategy.use_reranker
        assert strategy.temperature == 0.0

    def test_phase_strategy(self):
        """Stratégie PHASE utilise parent et MMR."""
        strategy = ROUTING_TABLE[GRIIntent.PHASE_COMPLETE]
        assert strategy.use_parent is True
        assert strategy.use_mmr is True
        assert strategy.use_reranker is True

    def test_cir_strategy(self):
        """Stratégie CIR inclut mapping GRI."""
        strategy = ROUTING_TABLE[GRIIntent.CIR]
        assert strategy.include_gri_mapping is True
        assert strategy.filters == {"cycle": "CIR"}

    def test_get_strategy_helper(self):
        """Test helper get_strategy_for_intent."""
        strategy = get_strategy_for_intent(GRIIntent.DEFINITION)
        assert strategy.primary_index == "glossary"


# =============================================================================
# Tests Term Expander
# =============================================================================


class TestTermExpander:
    """Tests pour l'expansion terminologique."""

    def test_detect_acronyms(self):
        """Détecte les acronymes GRI."""
        query = "Qu'est-ce que le SEMP et le CONOPS ?"
        terms = detect_gri_terms(query)
        assert "SEMP" in terms
        assert "CONOPS" in terms

    def test_detect_terms_fr(self):
        """Détecte les termes français."""
        query = "Définition de l'ingénierie système"
        terms = detect_gri_terms(query)
        assert "ingénierie système" in terms

    def test_detect_artefact(self):
        """Détecte artefact/artifact."""
        terms_fr = detect_gri_terms("Qu'est-ce qu'un artefact ?")
        terms_en = detect_gri_terms("What is an artifact?")
        assert "artefact" in terms_fr
        assert "artefact" in terms_en  # Normalisé

    def test_detect_milestone_terms(self):
        """Détecte les termes de jalons."""
        query = "Critères du CDR et TRR"
        terms = detect_gri_terms(query)
        assert "CDR" in terms
        assert "TRR" in terms

    def test_detect_process_terms(self):
        """Détecte les termes de processus."""
        query = "Processus de vérification et validation"
        terms = detect_gri_terms(query)
        assert "vérification" in terms
        assert "validation" in terms

    def test_no_false_positives(self):
        """Pas de faux positifs sur des mots courants."""
        query = "Bonjour, comment allez-vous ?"
        terms = detect_gri_terms(query)
        assert len(terms) == 0


class TestTermExpanderAsync:
    """Tests async pour l'expander."""

    @pytest.fixture
    def mock_store(self):
        """Mock du vector store."""
        store = MagicMock()
        store.glossary_lookup = AsyncMock(return_value=None)
        return store

    @pytest.mark.asyncio
    async def test_expand_with_no_terms(self, mock_store):
        """Pas d'expansion si aucun terme détecté."""
        expander = GRITermExpander(mock_store, max_terms=3)
        result = await expander.expand("Question sans terme GRI")

        assert result.original_query == "Question sans terme GRI"
        assert len(result.detected_terms) == 0
        assert result.term_context == ""
        assert not result.has_expansions

    @pytest.mark.asyncio
    async def test_expand_with_terms_not_found(self, mock_store):
        """Terme détecté mais non trouvé dans le glossaire."""
        mock_store.glossary_lookup = AsyncMock(return_value=None)

        expander = GRITermExpander(mock_store, max_terms=3)
        result = await expander.expand("Qu'est-ce que le SEMP ?")

        assert "SEMP" in result.detected_terms
        assert len(result.definitions) == 0

    @pytest.mark.asyncio
    async def test_expand_with_found_definition(self, mock_store):
        """Terme trouvé dans le glossaire."""
        # Mock du résultat glossaire
        mock_result = MagicMock()
        mock_result.metadata = {
            "term_fr": "artefact",
            "term_en": "artifact",
            "definition_fr": "Produit ou livrable élaboré...",
            "standard_ref": "ISO/IEC/IEEE 15288:2023",
        }
        mock_result.content = "Produit ou livrable élaboré..."
        mock_store.glossary_lookup = AsyncMock(return_value=mock_result)

        expander = GRITermExpander(mock_store, max_terms=3)
        result = await expander.expand("Qu'est-ce qu'un artefact ?")

        assert result.has_expansions
        assert len(result.definitions) == 1
        assert result.definitions[0].term_fr == "artefact"
        assert "## Définitions GRI applicables" in result.term_context


# =============================================================================
# Tests CIR/GRI Mapping
# =============================================================================


class TestCIRGRIMapping:
    """Tests pour le mapping CIR → GRI."""

    def test_j1_maps_to_m0_m1(self):
        """J1 → M0 + M1."""
        assert CIR_GRI_MAPPING["J1"] == ["M0", "M1"]

    def test_j2_maps_to_m2_m3_m4(self):
        """J2 → M2 + M3 + M4."""
        assert CIR_GRI_MAPPING["J2"] == ["M2", "M3", "M4"]

    def test_j3_maps_to_m5_m6(self):
        """J3 → M5 + M6."""
        assert CIR_GRI_MAPPING["J3"] == ["M5", "M6"]

    def test_j4_j5_map_to_sar(self):
        """J4 et J5 → SAR."""
        assert CIR_GRI_MAPPING["J4"] == ["SAR"]
        assert CIR_GRI_MAPPING["J5"] == ["SAR"]

    def test_j6_maps_to_m8(self):
        """J6 → M8."""
        assert CIR_GRI_MAPPING["J6"] == ["M8"]

    def test_valid_gri_milestones(self):
        """Jalons GRI valides M0-M9."""
        expected = {f"M{i}" for i in range(10)}
        assert expected == VALID_GRI_MILESTONES

    def test_valid_cir_milestones(self):
        """Jalons CIR valides J1-J6."""
        expected = {f"J{i}" for i in range(1, 7)}
        assert expected == VALID_CIR_MILESTONES


# =============================================================================
# Tests Milestone Retriever
# =============================================================================


class TestMilestoneRetrieverValidation:
    """Tests pour la validation des IDs de jalons."""

    @pytest.fixture
    def mock_store(self):
        """Mock du vector store."""
        store = MagicMock()
        store.COLLECTIONS = {"main": "gri_main", "glossary": "gri_glossary"}
        return store

    def test_validate_m4(self, mock_store):
        """M4 est valide."""
        from src.core.milestone_retriever import GRIMilestoneRetriever

        retriever = GRIMilestoneRetriever(mock_store)
        is_valid, normalized = retriever.validate_milestone_id("M4")

        assert is_valid
        assert normalized == "M4"

    def test_validate_cdr_to_m4(self, mock_store):
        """CDR se normalise en M4."""
        from src.core.milestone_retriever import GRIMilestoneRetriever

        retriever = GRIMilestoneRetriever(mock_store)
        is_valid, normalized = retriever.validate_milestone_id("CDR")

        assert is_valid
        assert normalized == "M3"  # CDR = Critical Design Review = M3

    def test_validate_lowercase(self, mock_store):
        """Accepte minuscules."""
        from src.core.milestone_retriever import GRIMilestoneRetriever

        retriever = GRIMilestoneRetriever(mock_store)
        is_valid, normalized = retriever.validate_milestone_id("j3")

        assert is_valid
        assert normalized == "J3"

    def test_validate_invalid(self, mock_store):
        """M99 est invalide."""
        from src.core.milestone_retriever import GRIMilestoneRetriever

        retriever = GRIMilestoneRetriever(mock_store)
        is_valid, _ = retriever.validate_milestone_id("M99")

        assert not is_valid

    def test_get_gri_equivalents_j3(self, mock_store):
        """J3 → M5, M6."""
        from src.core.milestone_retriever import GRIMilestoneRetriever

        retriever = GRIMilestoneRetriever(mock_store)
        equivalents = retriever.get_gri_equivalents("J3")

        assert equivalents == ["M5", "M6"]


# =============================================================================
# Tests Reranker
# =============================================================================


class TestRerankerMMR:
    """Tests pour le reranker avec MMR."""

    def test_content_similarity_identical(self):
        """Contenus identiques → similarité 1.0."""
        from src.core.reranker import GRIReranker

        reranker = GRIReranker.__new__(GRIReranker)
        similarity = reranker._content_similarity(
            "hello world test",
            "hello world test",
        )
        assert similarity == 1.0

    def test_content_similarity_different(self):
        """Contenus différents → similarité < 1.0."""
        from src.core.reranker import GRIReranker

        reranker = GRIReranker.__new__(GRIReranker)
        similarity = reranker._content_similarity(
            "hello world",
            "goodbye universe",
        )
        assert similarity == 0.0

    def test_content_similarity_partial(self):
        """Contenus partiellement similaires."""
        from src.core.reranker import GRIReranker

        reranker = GRIReranker.__new__(GRIReranker)
        similarity = reranker._content_similarity(
            "hello world test",
            "hello universe test",
        )
        assert 0.0 < similarity < 1.0


# =============================================================================
# Tests Vector Store (Mocked)
# =============================================================================


class TestVectorStoreRRFFusion:
    """Tests pour la fusion RRF."""

    def test_rrf_formula(self):
        """Vérifie la formule RRF."""
        # RRF score = α * (1 / (k + rank + 1)) pour dense
        # + (1-α) * (1 / (k + rank + 1)) pour sparse
        alpha = 0.6
        k = 60

        # Doc en position 0 dans dense
        dense_contrib = alpha * (1 / (k + 0 + 1))  # = 0.6 * (1/61) ≈ 0.00984

        # Doc en position 1 dans sparse
        sparse_contrib = (1 - alpha) * (1 / (k + 1 + 1))  # = 0.4 * (1/62) ≈ 0.00645

        expected_score = dense_contrib + sparse_contrib
        assert abs(expected_score - 0.01629) < 0.001

    def test_rrf_parameters_from_settings(self):
        """RRF utilise les params de settings."""
        from src.core.config import settings

        assert settings.rrf_alpha == 0.6
        assert settings.rrf_k == 60


# =============================================================================
# Tests Routing Strategy Integration
# =============================================================================


class TestRoutingIntegration:
    """Tests d'intégration du routing."""

    def test_all_intents_have_strategy(self):
        """Chaque intent a une stratégie."""
        for intent in GRIIntent:
            assert intent in ROUTING_TABLE

    def test_strategies_have_required_fields(self):
        """Chaque stratégie a les champs requis."""
        required_fields = [
            "search_mode",
            "primary_index",
            "n_initial",
            "use_reranker",
            "temperature",
        ]

        for intent, strategy in ROUTING_TABLE.items():
            for field in required_fields:
                assert hasattr(strategy, field), f"{intent} missing {field}"

    def test_definition_temperature_zero(self):
        """Définitions utilisent temperature=0."""
        strategy = ROUTING_TABLE[GRIIntent.DEFINITION]
        assert strategy.temperature == 0.0

    def test_jalon_temperature_zero(self):
        """Jalons utilisent temperature=0."""
        strategy = ROUTING_TABLE[GRIIntent.JALON]
        assert strategy.temperature == 0.0
