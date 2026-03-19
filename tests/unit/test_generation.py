"""Tests unitaires pour le module de génération GRI.

Ces tests vérifient :
- Les prompts par type de contenu
- Le formatage du contexte
- Le post-processing et validations
- L'extraction des citations
"""

import pytest

from src.generation.context_formatter import (
    check_context_sufficiency,
    extract_context_variables,
    format_gri_context,
    truncate_context,
)
from src.generation.postprocessor import (
    extract_citations,
    postprocess_gri_answer,
    validate_milestones,
    validate_phases,
)
from src.generation.prompts import (
    GRIResponseType,
    get_max_tokens,
    get_prompt,
    get_system_prompt,
    get_temperature,
    intent_to_response_type,
)


# === Tests des prompts ===


class TestPrompts:
    """Tests pour les templates de prompts."""

    def test_get_temperature_definition(self):
        """Temperature 0.0 pour les définitions."""
        temp = get_temperature(GRIResponseType.DEFINITION)
        assert temp == 0.0

    def test_get_temperature_milestone(self):
        """Temperature 0.0 pour les jalons."""
        temp = get_temperature(GRIResponseType.MILESTONE)
        assert temp == 0.0

    def test_get_temperature_general(self):
        """Temperature 0.1 pour les réponses générales."""
        temp = get_temperature(GRIResponseType.GENERAL)
        assert temp == 0.1

    def test_get_max_tokens_by_type(self):
        """Vérifier les max_tokens par type."""
        assert get_max_tokens(GRIResponseType.DEFINITION) == 256
        assert get_max_tokens(GRIResponseType.MILESTONE) == 1024
        assert get_max_tokens(GRIResponseType.PHASE_COMPLETE) == 2048

    def test_get_prompt_definition(self):
        """Test du prompt de définition."""
        prompt = get_prompt(
            GRIResponseType.DEFINITION,
            term="artefact",
            context="Contexte de test",
        )

        assert "artefact" in prompt
        assert "définition" in prompt.lower()
        assert "ISO" in prompt

    def test_get_prompt_milestone(self):
        """Test du prompt de jalon."""
        prompt = get_prompt(
            GRIResponseType.MILESTONE,
            milestone_id="M4",
            milestone_name="CDR",
            context="Contexte de test",
        )

        assert "M4" in prompt
        assert "CDR" in prompt
        assert "critères" in prompt.lower()

    def test_get_prompt_missing_variable(self):
        """Test qu'une erreur est levée si variable manquante."""
        with pytest.raises(ValueError) as exc_info:
            get_prompt(GRIResponseType.DEFINITION, context="test")

        assert "term" in str(exc_info.value)

    def test_get_system_prompt_includes_rules(self):
        """Test que le system prompt inclut les règles de base."""
        prompt = get_system_prompt(GRIResponseType.GENERAL)

        assert "GRI" in prompt
        assert "citations" in prompt.lower()
        assert "ISO" in prompt

    def test_get_system_prompt_definition_addon(self):
        """Test que le prompt définition a l'addon spécifique."""
        prompt = get_system_prompt(GRIResponseType.DEFINITION)

        assert "DÉFINITIONS" in prompt
        assert "mot pour mot" in prompt.lower()

    def test_get_system_prompt_milestone_addon(self):
        """Test que le prompt jalon a l'addon spécifique."""
        prompt = get_system_prompt(GRIResponseType.MILESTONE)

        assert "JALONS" in prompt
        assert "TOUS les critères" in prompt

    def test_intent_to_response_type_mapping(self):
        """Test du mapping intent -> response type."""
        assert intent_to_response_type("DEFINITION") == GRIResponseType.DEFINITION
        assert intent_to_response_type("JALON") == GRIResponseType.MILESTONE
        assert intent_to_response_type("PROCESSUS") == GRIResponseType.PROCESS
        assert intent_to_response_type("COMPARAISON") == GRIResponseType.COMPARISON
        assert intent_to_response_type("UNKNOWN") == GRIResponseType.GENERAL


# === Tests du formatage de contexte ===


class TestContextFormatter:
    """Tests pour le formatage du contexte."""

    def test_format_gri_context_empty(self):
        """Test avec chunks vides."""
        result = format_gri_context([])
        assert "Aucune source" in result

    def test_format_gri_context_single_chunk(self):
        """Test avec un seul chunk."""
        chunks = [
            {
                "content": "Contenu de test",
                "score": 0.85,
                "cycle": "GRI",
                "section_type": "definition",
            }
        ]

        result = format_gri_context(chunks)

        assert "[SOURCE 1]" in result
        assert "GRI" in result
        assert "DEFINITION" in result
        assert "0.85" in result
        assert "Contenu de test" in result

    def test_format_gri_context_multiple_chunks(self):
        """Test avec plusieurs chunks."""
        chunks = [
            {"content": "Chunk 1", "score": 0.9, "cycle": "GRI", "section_type": "phase"},
            {"content": "Chunk 2", "score": 0.7, "cycle": "CIR", "section_type": "milestone"},
        ]

        result = format_gri_context(chunks)

        assert "[SOURCE 1]" in result
        assert "[SOURCE 2]" in result
        assert "---" in result  # Séparateur

    def test_format_gri_context_with_metadata(self):
        """Test avec métadonnées complètes."""
        chunks = [
            {
                "content": "Critères du CDR",
                "score": 0.95,
                "cycle": "GRI",
                "section_type": "milestone",
                "milestone_id": "M4",
                "phase_num": 3,
                "context_prefix": "GRI > Jalon M4 (CDR)",
            }
        ]

        result = format_gri_context(chunks)

        assert "M4" in result
        assert "Phase 3" in result
        assert "GRI > Jalon M4" in result

    def test_truncate_context_short(self):
        """Test que les contextes courts ne sont pas tronqués."""
        context = "Court contexte"
        result = truncate_context(context, max_chars=1000)
        assert result == context

    def test_truncate_context_long(self):
        """Test que les contextes longs sont tronqués."""
        # Créer un contexte de plusieurs chunks
        chunks = "\n\n---\n\n".join([f"Chunk {i}" * 100 for i in range(10)])
        result = truncate_context(chunks, max_chars=500)

        assert len(result) <= 600  # Un peu de marge pour le message
        assert "omise(s)" in result

    def test_check_context_sufficiency_no_chunks(self):
        """Test avec aucun chunk."""
        result = check_context_sufficiency([], GRIResponseType.GENERAL)

        assert result["sufficient"] is False
        assert result["reason"] == "no_chunks"

    def test_check_context_sufficiency_low_scores(self):
        """Test avec scores trop faibles."""
        chunks = [
            {"content": "Test", "score": 0.2},
            {"content": "Test 2", "score": 0.3},
        ]

        result = check_context_sufficiency(chunks, GRIResponseType.GENERAL)

        assert result["sufficient"] is False
        assert result["reason"] == "low_scores"

    def test_check_context_sufficiency_ok(self):
        """Test avec contexte suffisant."""
        chunks = [
            {"content": "Test", "score": 0.8},
        ]

        result = check_context_sufficiency(chunks, GRIResponseType.GENERAL)

        assert result["sufficient"] is True

    def test_check_context_sufficiency_milestone_needs_milestone_chunk(self):
        """Test que les jalons nécessitent des chunks milestone."""
        chunks = [
            {"content": "Test", "score": 0.8, "section_type": "phase"},
        ]

        result = check_context_sufficiency(chunks, GRIResponseType.MILESTONE)

        assert result["sufficient"] is False
        assert "milestone" in result["reason"]


# === Tests du post-processing ===


class TestPostprocessor:
    """Tests pour le post-processing des réponses."""

    def test_validate_milestones_valid(self):
        """Test avec jalons valides."""
        text = "Le jalon M4 (CDR) précède M5 et M6."

        result = validate_milestones(text)

        assert "M4" in result["valid"]
        assert "M5" in result["valid"]
        assert "M6" in result["valid"]
        assert len(result["invalid"]) == 0

    def test_validate_milestones_invalid(self):
        """Test avec jalons invalides."""
        text = "Le jalon M15 n'existe pas, ni J10."

        result = validate_milestones(text)

        assert "M15" in result["invalid"] or len(result["invalid"]) > 0

    def test_validate_milestones_cir(self):
        """Test avec jalons CIR."""
        text = "Les jalons J1, J3 et J6 du CIR."

        result = validate_milestones(text)

        assert "J1" in result["valid"]
        assert "J3" in result["valid"]
        assert "J6" in result["valid"]

    def test_validate_phases_gri(self):
        """Test avec phases GRI valides."""
        text = "La Phase 3 précède la Phase 4."

        result = validate_phases(text)

        assert 3 in result["valid"]
        assert 4 in result["valid"]

    def test_validate_phases_invalid(self):
        """Test avec phases invalides."""
        text = "La Phase 10 n'existe pas dans le GRI."

        result = validate_phases(text)

        assert 10 in result["invalid"]

    def test_extract_citations_gri(self):
        """Test extraction de citations GRI."""
        text = """
        Selon le GRI [GRI > Jalon M4 (CDR) > Critère #1],
        les objectifs sont [GRI > Phase 3 > Conception].
        """

        citations = extract_citations(text)

        assert len(citations) == 2
        assert any("Jalon M4" in c for c in citations)
        assert any("Phase 3" in c for c in citations)

    def test_extract_citations_cir(self):
        """Test extraction de citations CIR."""
        text = "Le jalon [CIR > Phase 2 > J2] correspond à M2+M3."

        citations = extract_citations(text)

        assert len(citations) == 1
        assert "CIR" in citations[0]

    def test_extract_citations_none(self):
        """Test sans citations."""
        text = "Réponse sans aucune citation."

        citations = extract_citations(text)

        assert len(citations) == 0

    def test_postprocess_adds_warnings_for_invalid_milestones(self):
        """Test que les jalons invalides génèrent des warnings."""
        answer = "Le jalon M15 a 5 critères."

        result = postprocess_gri_answer(answer, GRIResponseType.MILESTONE)

        assert len(result["warnings"]) > 0
        assert "M15" in str(result["warnings"])

    def test_postprocess_counts_criteria(self):
        """Test que le nombre de critères est compté."""
        answer = """
        1. Premier critère
        2. Deuxième critère
        3. Troisième critère
        """

        result = postprocess_gri_answer(answer, GRIResponseType.MILESTONE)

        assert result["validation"]["criteria_count"] == 3


# === Tests d'extraction de variables ===


class TestContextVariables:
    """Tests pour l'extraction des variables de contexte."""

    def test_extract_term_from_definition_query(self):
        """Test extraction du terme pour une définition."""
        chunks = [{"content": "Test", "score": 0.8}]

        result = extract_context_variables(
            chunks,
            GRIResponseType.DEFINITION,
            "Qu'est-ce qu'un artefact ?",
        )

        assert "term" in result
        assert result["term"] == "artefact"

    def test_extract_milestone_id(self):
        """Test extraction de l'ID de jalon."""
        chunks = [{"content": "Test", "score": 0.8}]

        result = extract_context_variables(
            chunks,
            GRIResponseType.MILESTONE,
            "Critères du CDR M4",
        )

        assert result.get("milestone_id") == "M4"

    def test_extract_phase_num(self):
        """Test extraction du numéro de phase."""
        chunks = [{"content": "Test", "score": 0.8}]

        result = extract_context_variables(
            chunks,
            GRIResponseType.PHASE_COMPLETE,
            "Objectifs de la phase 3",
        )

        assert result.get("phase_num") == 3

    def test_extract_cycle_cir(self):
        """Test détection du cycle CIR."""
        chunks = [{"content": "Test", "score": 0.8}]

        result = extract_context_variables(
            chunks,
            GRIResponseType.PHASE_COMPLETE,
            "Phase 2 du CIR",
        )

        assert result.get("cycle") == "CIR"


# === Tests d'intégration (nécessitent HF API) ===


@pytest.mark.integration
class TestGeneratorIntegration:
    """Tests d'intégration pour le générateur.

    Ces tests sont marqués 'integration' et utilisent des mocks pour le client LLM.
    """

    @pytest.fixture
    def mock_llm_response_definition(self):
        """Réponse LLM mockée pour une définition."""
        return """Un **artefact** est un produit tangible ou intangible résultant d'une activité d'ingénierie des systèmes.

[Source: GRI > Glossaire]"""

    @pytest.fixture
    def mock_llm_response_milestone(self):
        """Réponse LLM mockée pour un jalon."""
        return """## Critères du jalon M4 (CDR)

Le CDR (Critical Design Review) valide la conception détaillée du système.

### Critères de passage :
1. Critère 1
2. Critère 2
3. Critère 3

[Source: GRI > Jalon M4 (CDR)]"""

    @pytest.mark.asyncio
    async def test_generate_definition(self, mock_llm_response_definition):
        """Test génération d'une définition avec mock."""
        from src.generation import GRIGenerator
        from unittest.mock import AsyncMock, patch, MagicMock

        chunks = [
            {
                "content": "Artefact : produit tangible du processus d'ingénierie",
                "score": 0.9,
                "section_type": "definition",
            }
        ]

        with patch("src.generation.generator.AsyncInferenceClient") as MockClient:
            mock_client = MagicMock()
            mock_client.text_generation = AsyncMock(return_value=mock_llm_response_definition)
            MockClient.return_value = mock_client

            generator = GRIGenerator()
            generator._client = mock_client

            result = await generator.generate(
                query="Définition d'artefact",
                chunks=chunks,
                response_type=GRIResponseType.DEFINITION,
            )

            # Vérifier la configuration de température pour les définitions
            assert result.temperature_used == 0.0
            assert "artefact" in result.answer.lower()

    @pytest.mark.asyncio
    async def test_generate_milestone_criteria(self, mock_llm_response_milestone):
        """Test génération de critères de jalon avec mock."""
        from src.generation import GRIGenerator
        from unittest.mock import AsyncMock, patch, MagicMock

        with patch("src.generation.generator.AsyncInferenceClient") as MockClient:
            mock_client = MagicMock()
            mock_client.text_generation = AsyncMock(return_value=mock_llm_response_milestone)
            MockClient.return_value = mock_client

            generator = GRIGenerator()
            generator._client = mock_client

            result = await generator.generate_milestone_criteria(
                milestone_id="M4",
                milestone_data={
                    "name": "CDR",
                    "cycle": "GRI",
                    "criteria": [
                        "Critère 1",
                        "Critère 2",
                        "Critère 3",
                    ],
                },
            )

            assert "M4" in result.answer or "CDR" in result.answer
            # La réponse doit contenir les critères
            assert "Critère" in result.answer
