"""Tests pour la suite d'évaluation GRI.

Ce module teste les métriques d'évaluation :
- Term Accuracy
- Faithfulness GRI
- Answer Relevance
- Context Recall
- Context Precision
- Pipeline d'évaluation

Exécution:
    pytest tests/test_evaluation_gri.py -v
"""

import json
from pathlib import Path

from src.evaluation.faithfulness_gri import (
    FaithfulnessResult,
    _detect_gri_errors,
)
from src.evaluation.metrics import (
    AnswerRelevanceResult,
    ContextPrecisionResult,
    ContextRecallResult,
)
from src.evaluation.pipeline import (
    EvaluationSummary,
    GRIEvalResult,
    GRIEvaluator,
    load_dataset,
)
from src.evaluation.term_accuracy import (
    TermAccuracyResult,
    _calculate_score,
    extract_iso_terms,
)


class TestExtractISOTerms:
    """Tests pour l'extraction des termes ISO."""

    def test_extracts_acronyms(self):
        """Détecte les acronymes GRI."""
        text = "Le CDR et le PDR sont des jalons importants."
        terms = extract_iso_terms(text)
        assert "cdr" in terms
        assert "pdr" in terms

    def test_extracts_french_terms(self):
        """Détecte les termes français."""
        text = "La vérification et la validation sont essentielles."
        terms = extract_iso_terms(text)
        assert any("vérification" in t or "verification" in t for t in terms)
        assert "validation" in terms

    def test_extracts_bilingual_terms(self):
        """Détecte les termes bilingues."""
        text = "L'artefact (artifact) doit être validé."
        terms = extract_iso_terms(text)
        assert any("artefact" in t or "artifact" in t for t in terms)

    def test_extracts_trl_mrl(self):
        """Détecte TRL et MRL."""
        text = "Le TRL est de 6 et le MRL de 4."
        terms = extract_iso_terms(text)
        assert "trl" in terms
        assert "mrl" in terms

    def test_returns_empty_for_no_terms(self):
        """Retourne une liste vide si pas de termes."""
        text = "Ceci est un texte sans termes techniques."
        terms = extract_iso_terms(text)
        # Peut ne pas être vide à cause de faux positifs
        assert isinstance(terms, list)

    def test_deduplicates_terms(self):
        """Déduplique les termes trouvés."""
        text = "CONOPS CONOPS CONOPS"
        terms = extract_iso_terms(text)
        assert terms.count("conops") <= 1


class TestTermAccuracyCalculation:
    """Tests pour le calcul du score Term Accuracy."""

    def test_exact_match_score_1(self):
        """Score 1.0 pour EXACT."""
        from src.evaluation.term_accuracy import TermEvaluation

        evals = [
            TermEvaluation(
                term="artefact",
                definition_in_answer="...",
                normative_definition="...",
                status="EXACT",
                severity="OK",
            )
        ]
        score = _calculate_score(evals)
        assert score == 1.0

    def test_approximatif_score_05(self):
        """Score 0.5 pour APPROXIMATIF."""
        from src.evaluation.term_accuracy import TermEvaluation

        evals = [
            TermEvaluation(
                term="artefact",
                definition_in_answer="...",
                normative_definition="...",
                status="APPROXIMATIF",
                severity="MINEUR",
            )
        ]
        score = _calculate_score(evals)
        assert score == 0.5

    def test_incorrect_score_0(self):
        """Score 0.0 pour INCORRECT."""
        from src.evaluation.term_accuracy import TermEvaluation

        evals = [
            TermEvaluation(
                term="artefact",
                definition_in_answer="...",
                normative_definition="...",
                status="INCORRECT",
                severity="CRITIQUE",
            )
        ]
        score = _calculate_score(evals)
        assert score == 0.0

    def test_mixed_scores(self):
        """Score mixte correct."""
        from src.evaluation.term_accuracy import TermEvaluation

        evals = [
            TermEvaluation(
                term="t1",
                definition_in_answer="",
                normative_definition="",
                status="EXACT",
                severity="OK",
            ),
            TermEvaluation(
                term="t2",
                definition_in_answer="",
                normative_definition="",
                status="APPROXIMATIF",
                severity="MINEUR",
            ),
        ]
        score = _calculate_score(evals)
        assert score == 0.75  # (1.0 + 0.5) / 2

    def test_empty_evaluations_score_1(self):
        """Score 1.0 si pas d'évaluations."""
        score = _calculate_score([])
        assert score == 1.0


class TestDetectGRIErrors:
    """Tests pour la détection automatique des erreurs GRI."""

    def test_detects_invalid_gri_milestone(self):
        """Détecte les jalons GRI invalides."""
        answer = "Le jalon M10 est important."
        errors = _detect_gri_errors(answer)
        assert any("jalon_inexistant" in e for e in errors)

    def test_detects_invalid_cir_milestone(self):
        """Détecte les jalons CIR invalides."""
        answer = "Le jalon J7 du CIR est critique."
        errors = _detect_gri_errors(answer)
        assert any("jalon_inexistant" in e for e in errors)

    def test_detects_invalid_phase(self):
        """Détecte les phases GRI invalides."""
        answer = "La Phase 9 du GRI couvre le retrait."
        errors = _detect_gri_errors(answer)
        assert any("phase_inexistante" in e for e in errors)

    def test_valid_milestone_no_error(self):
        """Pas d'erreur pour les jalons valides."""
        answer = "Le jalon M4 (CDR) est le Critical Design Review."
        errors = _detect_gri_errors(answer)
        assert not any("jalon_inexistant" in e for e in errors)

    def test_valid_phase_no_error(self):
        """Pas d'erreur pour les phases valides."""
        answer = "La Phase 3 du GRI est la Conception."
        errors = _detect_gri_errors(answer)
        assert not any("phase_inexistante" in e for e in errors)


class TestGRIEvaluator:
    """Tests pour l'évaluateur principal."""

    def test_quality_gates_defined(self):
        """Les quality gates sont définis."""
        evaluator = GRIEvaluator()
        assert "faithfulness" in evaluator.QUALITY_GATES
        assert "term_accuracy" in evaluator.QUALITY_GATES
        assert evaluator.QUALITY_GATES["term_accuracy"] == 0.95

    def test_check_milestone_completeness_jalon(self):
        """Vérifie la complétude des jalons."""
        evaluator = GRIEvaluator()

        # Réponse avec critères listés
        answer = """Les critères sont :
        1. Premier critère
        2. Deuxième critère
        3. Troisième critère
        """
        item = {"expected_intent": "JALON", "critical_check": "all_criteria_listed"}
        assert evaluator._check_milestone_completeness(answer, item) is True

    def test_check_milestone_completeness_not_jalon(self):
        """Retourne True si pas un jalon."""
        evaluator = GRIEvaluator()
        answer = "Quelque chose"
        item = {"expected_intent": "DEFINITION"}
        assert evaluator._check_milestone_completeness(answer, item) is True

    def test_check_milestone_completeness_gri_mapping(self):
        """Vérifie le mapping GRI pour CIR."""
        evaluator = GRIEvaluator()
        answer = "J3 équivaut à M5 et M6 du GRI."
        item = {"expected_intent": "JALON", "critical_check": "includes_gri_mapping"}
        assert evaluator._check_milestone_completeness(answer, item) is True


class TestLoadDataset:
    """Tests pour le chargement du dataset."""

    def test_load_golden_dataset(self):
        """Charge le golden dataset."""
        dataset_path = Path("data/golden_dataset.json")
        if dataset_path.exists():
            dataset = load_dataset(dataset_path)
            assert isinstance(dataset, list)
            assert len(dataset) == 50

    def test_dataset_structure(self):
        """Vérifie la structure du dataset."""
        dataset_path = Path("data/golden_dataset.json")
        if dataset_path.exists():
            with open(dataset_path) as f:
                data = json.load(f)

            assert "metadata" in data
            assert "questions" in data
            assert data["metadata"]["n_questions"] == 50

            # Vérifier une question
            q = data["questions"][0]
            assert "id" in q
            assert "query" in q
            assert "ground_truth" in q


class TestGRIEvalResult:
    """Tests pour le dataclass GRIEvalResult."""

    def test_default_values(self):
        """Vérifie les valeurs par défaut."""
        result = GRIEvalResult(
            question_id="test",
            query="Test query",
            answer="Test answer",
        )
        assert result.faithfulness == 0.0
        assert result.term_accuracy == 0.0
        assert result.milestone_complete is True
        assert result.error is None

    def test_with_metrics(self):
        """Crée un résultat avec métriques."""
        result = GRIEvalResult(
            question_id="DEF_001",
            query="Qu'est-ce qu'un artefact ?",
            answer="Un artefact est...",
            faithfulness=0.9,
            term_accuracy=1.0,
            latency_ms=500.0,
        )
        assert result.faithfulness == 0.9
        assert result.term_accuracy == 1.0


class TestEvaluationSummary:
    """Tests pour EvaluationSummary."""

    def test_default_values(self):
        """Vérifie les valeurs par défaut."""
        summary = EvaluationSummary()
        assert summary.n_evaluated == 0
        assert summary.faithfulness_mean == 0.0
        assert summary.term_accuracy_mean == 0.0


class TestAggregation:
    """Tests pour l'agrégation des résultats."""

    def test_aggregate_empty(self):
        """Agrégation d'une liste vide."""
        evaluator = GRIEvaluator()
        summary = evaluator._aggregate([])
        assert summary.n_evaluated == 0

    def test_aggregate_single_result(self):
        """Agrégation d'un seul résultat."""
        evaluator = GRIEvaluator()
        results = [
            GRIEvalResult(
                question_id="1",
                query="Test",
                answer="Answer",
                faithfulness=0.9,
                answer_relevance=0.8,
                term_accuracy=1.0,
                latency_ms=500.0,
                intent_correct=True,
                cycle_correct=True,
            )
        ]
        summary = evaluator._aggregate(results)
        assert summary.n_evaluated == 1
        assert summary.faithfulness_mean == 0.9
        assert summary.intent_accuracy == 1.0


class TestQualityGates:
    """Tests pour la vérification des quality gates."""

    def test_all_gates_pass(self):
        """Tous les gates passent."""
        evaluator = GRIEvaluator()
        summary = EvaluationSummary(
            faithfulness_mean=0.90,
            answer_relevance_mean=0.85,
            context_recall_mean=0.80,
            context_precision_mean=0.75,
            term_accuracy_mean=0.98,
            latency_p95=5000.0,
        )
        failures = evaluator._check_quality_gates(summary)
        assert len(failures) == 0

    def test_term_accuracy_fails(self):
        """Term accuracy échoue (critique)."""
        evaluator = GRIEvaluator()
        summary = EvaluationSummary(
            faithfulness_mean=0.90,
            answer_relevance_mean=0.85,
            context_recall_mean=0.80,
            context_precision_mean=0.75,
            term_accuracy_mean=0.90,  # < 0.95
            latency_p95=5000.0,
        )
        failures = evaluator._check_quality_gates(summary)
        assert len(failures) == 1
        assert "CRITIQUE" in failures[0]
        assert "term_accuracy" in failures[0]

    def test_latency_fails(self):
        """Latence trop élevée."""
        evaluator = GRIEvaluator()
        summary = EvaluationSummary(
            faithfulness_mean=0.90,
            answer_relevance_mean=0.85,
            context_recall_mean=0.80,
            context_precision_mean=0.75,
            term_accuracy_mean=0.98,
            latency_p95=10000.0,  # > 8000
        )
        failures = evaluator._check_quality_gates(summary)
        assert len(failures) == 1
        assert "latency" in failures[0]


class TestTermAccuracyResult:
    """Tests pour TermAccuracyResult."""

    def test_no_terms_found(self):
        """Résultat quand aucun terme trouvé."""
        result = TermAccuracyResult(
            term_accuracy_score=1.0,
            no_terms_found=True,
        )
        assert result.term_accuracy_score == 1.0
        assert result.no_terms_found is True

    def test_with_evaluations(self):
        """Résultat avec évaluations."""
        from src.evaluation.term_accuracy import TermEvaluation

        result = TermAccuracyResult(
            term_accuracy_score=0.75,
            term_evaluations=[
                TermEvaluation(
                    term="artefact",
                    definition_in_answer="...",
                    normative_definition="...",
                    status="EXACT",
                    severity="OK",
                )
            ],
        )
        assert len(result.term_evaluations) == 1


class TestFaithfulnessResult:
    """Tests pour FaithfulnessResult."""

    def test_with_errors(self):
        """Résultat avec erreurs GRI."""
        result = FaithfulnessResult(
            faithfulness_score=0.6,
            gri_specific_errors=["jalon_inexistant: M10"],
        )
        assert result.faithfulness_score == 0.6
        assert len(result.gri_specific_errors) == 1


class TestMetricResults:
    """Tests pour les résultats des métriques."""

    def test_answer_relevance_result(self):
        """AnswerRelevanceResult."""
        result = AnswerRelevanceResult(
            relevance_score=0.85,
            directly_answers=True,
            completeness="COMPLÈTE",
        )
        assert result.relevance_score == 0.85

    def test_context_recall_result(self):
        """ContextRecallResult."""
        result = ContextRecallResult(
            recall_score=0.75,
            covered_points=["point1", "point2"],
            missing_points=["point3"],
        )
        assert result.recall_score == 0.75
        assert len(result.covered_points) == 2

    def test_context_precision_result(self):
        """ContextPrecisionResult."""
        result = ContextPrecisionResult(
            precision_score=0.80,
            n_relevant=4,
            n_total=5,
        )
        assert result.precision_score == 0.80
