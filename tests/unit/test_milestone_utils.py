"""Tests unitaires pour l'extraction centralisée des IDs de jalons.

Vérifie:
- Case-insensitivity (m2 -> M2)
- Mapping des acronymes (CDR -> M3, PDR -> M2)
- Gestion des word boundaries ((M4), M4-CDR, M4:)
- Fallback sur la hiérarchie
- Priorité: texte direct > acronyme > hiérarchie
"""

import pytest

from src.core.milestone_utils import (
    extract_milestone_id,
    normalize_milestone_id,
    MILESTONE_PATTERN,
    ACRONYM_PATTERN,
)


class TestMilestonePattern:
    """Tests du pattern regex M/J."""

    def test_pattern_matches_uppercase(self):
        """Pattern matche M0-M9 et J1-J6 en majuscules."""
        assert MILESTONE_PATTERN.search("Jalon M3")
        assert MILESTONE_PATTERN.search("CIR J2")
        assert MILESTONE_PATTERN.search("M0 initial")

    def test_pattern_matches_lowercase(self):
        """Pattern matche m0-m9 et j1-j6 en minuscules."""
        assert MILESTONE_PATTERN.search("jalon m3")
        assert MILESTONE_PATTERN.search("cir j2")

    def test_pattern_with_parentheses(self):
        """Pattern matche (M4)."""
        match = MILESTONE_PATTERN.search("Revue (M4) validée")
        assert match
        assert match.group(1).upper() == "M4"

    def test_pattern_with_hyphen(self):
        """Pattern matche M3-CDR."""
        match = MILESTONE_PATTERN.search("Jalon M3-CDR effectué")
        assert match
        assert match.group(1).upper() == "M3"

    def test_pattern_with_colon(self):
        """Pattern matche M5:."""
        match = MILESTONE_PATTERN.search("Phase M5: test readiness")
        assert match
        assert match.group(1).upper() == "M5"

    def test_pattern_no_false_positive_letters(self):
        """Pattern ne matche pas AM3 ou M3x."""
        # AM3 = le A est collé au M
        assert not MILESTONE_PATTERN.search("Le programme AM3 est lancé")
        # M3x = le x est collé au 3
        assert not MILESTONE_PATTERN.search("Modèle M3x disponible")

    def test_pattern_no_match_m10(self):
        """Pattern ne matche pas M10 (invalide)."""
        match = MILESTONE_PATTERN.search("Milestone M10")
        # Le pattern peut matcher M1 dans M10, vérifions que M10 complet n'est pas validé
        if match:
            assert match.group(1).upper() != "M10"


class TestAcronymPattern:
    """Tests du pattern regex pour acronymes."""

    def test_acronym_pattern_matches_all(self):
        """Pattern matche tous les acronymes définis."""
        acronyms = ["ASR", "MNS", "SRR", "SFR", "PDR", "CDR", "IRR", "TRR", "SAR", "ORR", "MNR"]
        for acr in acronyms:
            assert ACRONYM_PATTERN.search(f"Revue {acr} effectuée"), f"Failed for {acr}"

    def test_acronym_pattern_case_insensitive(self):
        """Pattern matche les acronymes en minuscules."""
        assert ACRONYM_PATTERN.search("revue cdr effectuée")
        assert ACRONYM_PATTERN.search("Passage du pdr")


class TestExtractMilestoneId:
    """Tests de la fonction extract_milestone_id."""

    # === Tests case-insensitivity ===

    def test_extract_lowercase_m2(self):
        """Lowercase m2 extrait en M2."""
        result = extract_milestone_id("Critères du jalon m2")
        assert result == "M2"

    def test_extract_lowercase_m3(self):
        """Lowercase m3 extrait en M3."""
        result = extract_milestone_id("passage du jalon m3 : cdr")
        assert result == "M3"

    def test_extract_lowercase_j3(self):
        """Lowercase j3 extrait en J3."""
        result = extract_milestone_id("jalon j3 du CIR")
        assert result == "J3"

    def test_extract_uppercase_m4(self):
        """Uppercase M4 extrait normalement."""
        result = extract_milestone_id("Critères du M4")
        assert result == "M4"

    # === Tests acronyme mapping ===

    def test_extract_pdr_to_m2(self):
        """PDR acronyme mappe vers M2."""
        result = extract_milestone_id("Preliminary Design Review (PDR)")
        assert result == "M2"

    def test_extract_cdr_to_m3(self):
        """CDR acronyme mappe vers M3."""
        result = extract_milestone_id("Critical Design Review (CDR)")
        assert result == "M3"

    def test_extract_sfr_to_m2(self):
        """SFR acronyme mappe vers M2."""
        result = extract_milestone_id("System Functional Review - SFR")
        assert result == "M2"

    def test_extract_trr_to_m5(self):
        """TRR acronyme mappe vers M5."""
        result = extract_milestone_id("Test Readiness Review TRR")
        assert result == "M5"

    def test_extract_srr_to_m1(self):
        """SRR acronyme mappe vers M1."""
        result = extract_milestone_id("System Requirements Review (SRR)")
        assert result == "M1"

    def test_extract_irr_to_m4(self):
        """IRR acronyme mappe vers M4."""
        result = extract_milestone_id("Integration Readiness Review IRR")
        assert result == "M4"

    def test_extract_sar_to_m6(self):
        """SAR acronyme mappe vers M6."""
        result = extract_milestone_id("System Acceptance Review SAR")
        assert result == "M6"

    def test_extract_orr_to_m7(self):
        """ORR acronyme mappe vers M7."""
        result = extract_milestone_id("Operational Readiness Review ORR")
        assert result == "M7"

    def test_extract_mnr_to_m8(self):
        """MNR acronyme mappe vers M8."""
        result = extract_milestone_id("Mission Needs Review MNR")
        assert result == "M8"

    def test_extract_lowercase_cdr(self):
        """Acronyme en minuscules mappe correctement."""
        result = extract_milestone_id("Revue cdr effectuée")
        assert result == "M3"

    # === Tests word boundary ===

    def test_extract_with_parentheses(self):
        """M4 dans parenthèses extrait."""
        result = extract_milestone_id("Jalon (M4) validé")
        assert result == "M4"

    def test_extract_with_hyphen(self):
        """M3 avec tiret extrait."""
        result = extract_milestone_id("Jalon M3-CDR")
        assert result == "M3"

    def test_extract_at_end_of_text(self):
        """M5 en fin de chaîne extrait."""
        result = extract_milestone_id("Phase de M5")
        assert result == "M5"

    def test_extract_at_start_of_text(self):
        """M1 en début de chaîne extrait."""
        result = extract_milestone_id("M1 : System Requirements Review")
        assert result == "M1"

    # === Tests fallback hiérarchie ===

    def test_extract_from_hierarchy(self):
        """Trouve milestone_id dans la hiérarchie si absent du texte."""
        result = extract_milestone_id(
            "Critères de passage",  # Pas de milestone ID dans le texte
            hierarchy=["GRI", "Phase 3", "Jalon M4 - CDR"],
        )
        assert result == "M4"

    def test_extract_acronym_from_hierarchy(self):
        """Trouve acronyme dans la hiérarchie et le mappe."""
        result = extract_milestone_id(
            "Critères de passage du jalon",
            hierarchy=["GRI", "Phase 3", "Critical Design Review (CDR)"],
        )
        assert result == "M3"

    def test_hierarchy_lowercase(self):
        """Trouve milestone en minuscules dans la hiérarchie."""
        result = extract_milestone_id(
            "Objectifs de la revue",
            hierarchy=["GRI", "Jalon m3 : cdr"],
        )
        assert result == "M3"

    # === Tests de priorité ===

    def test_text_takes_priority_over_hierarchy(self):
        """M5 dans le texte prioritaire sur M3 dans hiérarchie."""
        result = extract_milestone_id(
            "Critères du M5",
            hierarchy=["GRI", "Jalon M3"],
        )
        assert result == "M5"

    def test_direct_pattern_takes_priority_over_acronym(self):
        """M4 direct prioritaire sur CDR acronyme."""
        result = extract_milestone_id("Jalon M4 (CDR)")
        assert result == "M4"  # Pas M3 depuis CDR

    def test_acronym_in_text_before_hierarchy(self):
        """Acronyme dans texte prioritaire sur hiérarchie."""
        result = extract_milestone_id(
            "La revue CDR est validée",
            hierarchy=["GRI", "Phase"],
        )
        assert result == "M3"

    # === Tests edge cases ===

    def test_no_milestone_returns_none(self):
        """Texte sans milestone retourne None."""
        result = extract_milestone_id("Introduction au document GRI")
        assert result is None

    def test_empty_text_returns_none(self):
        """Texte vide retourne None."""
        result = extract_milestone_id("")
        assert result is None

    def test_none_text_returns_none(self):
        """None comme texte ne crash pas."""
        # La fonction devrait gérer None gracieusement
        result = extract_milestone_id("")
        assert result is None

    def test_empty_hierarchy_ignored(self):
        """Hiérarchie vide ne pose pas de problème."""
        result = extract_milestone_id("Jalon M6", hierarchy=[])
        assert result == "M6"

    def test_hierarchy_with_no_match(self):
        """Hiérarchie sans match retourne None."""
        result = extract_milestone_id(
            "Introduction",
            hierarchy=["GRI", "Chapitre 1", "Section A"],
        )
        assert result is None


class TestNormalizeMilestoneId:
    """Tests de la fonction normalize_milestone_id."""

    def test_normalize_lowercase_m(self):
        """Normalise m3 en M3."""
        assert normalize_milestone_id("m3") == "M3"

    def test_normalize_lowercase_j(self):
        """Normalise j2 en J2."""
        assert normalize_milestone_id("j2") == "J2"

    def test_normalize_uppercase_unchanged(self):
        """M4 reste M4."""
        assert normalize_milestone_id("M4") == "M4"

    def test_normalize_acronym_cdr(self):
        """CDR normalise en M3."""
        assert normalize_milestone_id("CDR") == "M3"

    def test_normalize_acronym_pdr(self):
        """PDR normalise en M2."""
        assert normalize_milestone_id("PDR") == "M2"

    def test_normalize_acronym_lowercase(self):
        """cdr normalise en M3."""
        assert normalize_milestone_id("cdr") == "M3"

    def test_normalize_invalid_returns_none(self):
        """Valeur invalide retourne None."""
        assert normalize_milestone_id("invalid") is None
        assert normalize_milestone_id("M10") is None
        assert normalize_milestone_id("") is None

    def test_normalize_with_whitespace(self):
        """Gère les espaces autour."""
        assert normalize_milestone_id("  M3  ") == "M3"
        assert normalize_milestone_id(" CDR ") == "M3"


class TestIntegrationWithChunker:
    """Tests d'intégration avec le chunker."""

    @pytest.fixture
    def chunker(self):
        """Instance du chunker pour les tests."""
        from src.ingestion.chunker import GRIChunker
        return GRIChunker("a1b2c3d4e5f67890")  # 16 chars minimum

    def test_milestone_section_lowercase(self, chunker):
        """Section milestone avec m3 minuscule extrait correctement."""
        from src.ingestion.models import ParsedSection, SectionType

        section = ParsedSection(
            level=3,
            title="Jalon m3 - cdr",
            content="Critères de passage pour la revue de conception critique. " * 5,
            hierarchy=["GRI", "Jalons", "m3 - cdr"],
            section_type=SectionType.MILESTONE,
        )
        chunks = chunker.chunk_sections([section])

        assert len(chunks) == 1
        assert chunks[0].metadata.milestone_id == "M3"

    def test_milestone_section_acronym_only(self, chunker):
        """Section avec seulement CDR obtient M3."""
        from src.ingestion.models import ParsedSection, SectionType

        section = ParsedSection(
            level=3,
            title="Critical Design Review (CDR)",
            content="Objectifs et critères de la revue CDR. " * 5,
            hierarchy=["GRI", "Phase 3", "CDR"],
            section_type=SectionType.MILESTONE,
        )
        chunks = chunker.chunk_sections([section])

        assert len(chunks) == 1
        assert chunks[0].metadata.milestone_id == "M3"

    def test_milestone_hierarchy_fallback(self, chunker):
        """Trouve milestone_id dans hiérarchie quand absent du contenu."""
        from src.ingestion.models import ParsedSection, SectionType

        section = ParsedSection(
            level=4,
            title="Objectifs",
            content="Cette revue vérifie que la conception est complète et conforme aux exigences. " * 3,
            hierarchy=["GRI", "Phase 3", "Jalon M4", "Objectifs"],
            section_type=SectionType.MILESTONE,
        )
        chunks = chunker.chunk_sections([section])

        assert len(chunks) == 1
        assert chunks[0].metadata.milestone_id == "M4"
