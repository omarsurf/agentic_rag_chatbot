"""Pipeline d'évaluation complet pour le GRI RAG.

Ce module orchestre l'évaluation complète du système RAG
sur le golden dataset avec toutes les métriques.

Usage:
    # Évaluation complète
    python -m src.evaluation.pipeline --dataset data/golden_dataset.json

    # Smoke test (10 questions)
    python -m src.evaluation.pipeline --smoke-test

    # API
    from src.evaluation.pipeline import GRIEvaluator
    evaluator = GRIEvaluator(rag_system)
    results = await evaluator.evaluate_dataset(dataset)
"""

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import structlog
from huggingface_hub import AsyncInferenceClient

from src.core.config import settings
from src.evaluation.faithfulness_gri import FaithfulnessResult
from src.evaluation.metrics import (
    AnswerRelevanceResult,
    ContextPrecisionResult,
    ContextRecallResult,
)
from src.evaluation.term_accuracy import TermAccuracyResult

log = structlog.get_logger()


@dataclass
class GRIEvalResult:
    """Résultat d'évaluation pour une question."""

    question_id: str
    query: str
    answer: str
    expected_intent: str | None = None
    actual_intent: str | None = None
    expected_cycle: str | None = None
    actual_cycle: str | None = None

    # Métriques
    faithfulness: float = 0.0
    faithfulness_gri_errors: list[str] = field(default_factory=list)
    answer_relevance: float = 0.0
    context_recall: float = 0.0
    context_precision: float = 0.0
    term_accuracy: float = 0.0
    term_critical_errors: list[str] = field(default_factory=list)

    # Performance
    latency_ms: float = 0.0

    # Checks
    intent_correct: bool = False
    cycle_correct: bool = False
    milestone_complete: bool = True

    # Metadata
    n_tool_calls: int = 0
    iterations: int = 0
    error: str | None = None


@dataclass
class EvaluationSummary:
    """Résumé de l'évaluation."""

    n_evaluated: int = 0
    n_errors: int = 0

    # Moyennes des métriques
    faithfulness_mean: float = 0.0
    faithfulness_p25: float = 0.0
    faithfulness_p75: float = 0.0

    answer_relevance_mean: float = 0.0
    answer_relevance_p25: float = 0.0
    answer_relevance_p75: float = 0.0

    context_recall_mean: float = 0.0
    context_recall_p25: float = 0.0
    context_recall_p75: float = 0.0

    context_precision_mean: float = 0.0
    context_precision_p25: float = 0.0
    context_precision_p75: float = 0.0

    term_accuracy_mean: float = 0.0
    term_accuracy_min: float = 0.0

    # Latence
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_max: float = 0.0

    # Accuracy
    intent_accuracy: float = 0.0
    cycle_accuracy: float = 0.0
    milestone_completeness: float = 0.0


class GRIEvaluator:
    """Évaluateur complet pour le système RAG GRI.

    Évalue le système sur un golden dataset avec les métriques :
    - Faithfulness (adaptée GRI)
    - Answer Relevance
    - Context Recall
    - Context Precision
    - Term Accuracy (critique)

    Attributes:
        QUALITY_GATES: Seuils de qualité à respecter
    """

    QUALITY_GATES = {
        "faithfulness": 0.85,
        "answer_relevance": 0.80,
        "context_recall": 0.75,
        "context_precision": 0.70,
        "term_accuracy": 0.95,  # CRITIQUE
        "latency_p95_ms": 8000,
    }

    def __init__(
        self,
        rag_system: Any | None = None,
        store: Any | None = None,
    ) -> None:
        """Initialise l'évaluateur.

        Args:
            rag_system: Système RAG à évaluer (orchestrateur)
            store: Vector store pour les lookups
        """
        self.rag_system = rag_system
        self.store = store
        self._client: AsyncInferenceClient | None = None

    @property
    def client(self) -> AsyncInferenceClient:
        """Lazy loading du client HF."""
        if self._client is None:
            self._client = AsyncInferenceClient(token=settings.hf_api_key)
        return self._client

    async def evaluate_dataset(
        self,
        dataset: list[dict],
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Évalue le système sur un dataset complet.

        Args:
            dataset: Liste de questions avec ground truth
            verbose: Afficher les détails

        Returns:
            Rapport d'évaluation complet
        """
        log.info("evaluation.start", n_questions=len(dataset))
        start_time = time.time()

        results: list[GRIEvalResult] = []

        for i, item in enumerate(dataset):
            if verbose:
                log.info(
                    "evaluation.progress",
                    current=i + 1,
                    total=len(dataset),
                    question_id=item.get("id"),
                )

            try:
                result = await self._evaluate_one(item)
                results.append(result)
            except Exception as e:
                log.error(
                    "evaluation.question_error",
                    question_id=item.get("id"),
                    error=str(e),
                )
                results.append(
                    GRIEvalResult(
                        question_id=item.get("id", "unknown"),
                        query=item.get("query", ""),
                        answer="",
                        error=str(e),
                    )
                )

        # Agréger les résultats
        summary = self._aggregate(results)
        failures = self._check_quality_gates(summary)

        total_time = time.time() - start_time

        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "n_evaluated": len(results),
                "n_errors": sum(1 for r in results if r.error),
                "total_time_seconds": total_time,
                "avg_time_per_question_seconds": total_time / len(results) if results else 0,
            },
            "summary": asdict(summary),
            "quality_gates": {
                "thresholds": self.QUALITY_GATES,
                "passed": len(failures) == 0,
                "failures": failures,
            },
            "results": [asdict(r) for r in results],
        }

        log.info(
            "evaluation.done",
            n_evaluated=len(results),
            quality_gates_passed=len(failures) == 0,
            n_failures=len(failures),
            total_time=f"{total_time:.1f}s",
        )

        return report

    async def _evaluate_one(self, item: dict) -> GRIEvalResult:
        """Évalue une seule question.

        Args:
            item: Question du dataset

        Returns:
            GRIEvalResult
        """
        query = item.get("query", "")
        ground_truth = item.get("ground_truth", "")
        expected_intent = item.get("expected_intent")
        expected_cycle = item.get("expected_cycle")

        # Exécuter la query si le système RAG est disponible
        answer = ""
        actual_intent = None
        actual_cycle = None
        latency_ms = 0.0
        n_tool_calls = 0
        iterations = 0
        chunks_used = []

        if self.rag_system is not None:
            start = time.time()
            try:
                response = await self.rag_system.run(query)
                latency_ms = (time.time() - start) * 1000

                answer = (
                    response.answer
                    if hasattr(response, "answer")
                    else str(response.get("answer", ""))
                )
                actual_intent = (
                    response.intent if hasattr(response, "intent") else response.get("intent")
                )
                actual_cycle = (
                    response.cycle if hasattr(response, "cycle") else response.get("cycle")
                )
                n_tool_calls = len(response.tool_calls) if hasattr(response, "tool_calls") else 0
                iterations = response.iterations if hasattr(response, "iterations") else 0

                # Récupérer les chunks utilisés si disponibles
                if hasattr(response, "chunks_used"):
                    chunks_used = [c.get("content", "") for c in response.chunks_used]
            except Exception as e:
                log.error("evaluation.rag_error", error=str(e))
                answer = f"Erreur: {str(e)}"
        else:
            # Mode sans RAG - utiliser une réponse simulée
            answer = item.get("simulated_answer", "")

        # Calculer les métriques
        from src.evaluation.faithfulness_gri import compute_faithfulness_gri
        from src.evaluation.metrics import (
            compute_answer_relevance,
            compute_context_precision,
            compute_context_recall,
        )
        from src.evaluation.term_accuracy import compute_term_accuracy

        # Exécuter les métriques en parallèle
        faithfulness_result: FaithfulnessResult | BaseException | None = None
        relevance_result: AnswerRelevanceResult | BaseException | None = None
        recall_result: ContextRecallResult | BaseException | None = None
        precision_result: ContextPrecisionResult | BaseException | None = None
        term_result: TermAccuracyResult | BaseException | None = None

        if answer and chunks_used:
            faithfulness_result, relevance_result = cast(
                tuple[FaithfulnessResult | BaseException, AnswerRelevanceResult | BaseException],
                await asyncio.gather(
                    compute_faithfulness_gri(answer, chunks_used, self.client),
                    compute_answer_relevance(query, answer, self.client),
                    return_exceptions=True,
                ),
            )

            if ground_truth:
                recall_result = await compute_context_recall(
                    query, ground_truth, chunks_used, self.client
                )

            precision_result = await compute_context_precision(query, chunks_used, self.client)

        if answer and self.store is not None:
            term_result = await compute_term_accuracy(answer, self.store, self.client)

        # Extraire les scores
        faithfulness = 0.0
        faithfulness_errors: list[str] = []
        if faithfulness_result is not None and not isinstance(faithfulness_result, BaseException):
            faithfulness = faithfulness_result.faithfulness_score
            faithfulness_errors = faithfulness_result.gri_specific_errors

        relevance = 0.0
        if relevance_result is not None and not isinstance(relevance_result, BaseException):
            relevance = relevance_result.relevance_score

        recall = 0.0
        if recall_result is not None and not isinstance(recall_result, BaseException):
            recall = recall_result.recall_score

        precision = 0.0
        if precision_result is not None and not isinstance(precision_result, BaseException):
            precision = precision_result.precision_score

        term_acc = 1.0  # Défaut si pas de termes
        term_errors: list[str] = []
        if term_result is not None and not isinstance(term_result, BaseException):
            term_acc = term_result.term_accuracy_score
            term_errors = term_result.critical_errors

        # Vérifications
        intent_correct = expected_intent is None or actual_intent == expected_intent
        cycle_correct = expected_cycle is None or actual_cycle == expected_cycle
        milestone_complete = self._check_milestone_completeness(answer, item)

        return GRIEvalResult(
            question_id=item.get("id", "unknown"),
            query=query,
            answer=answer,
            expected_intent=expected_intent,
            actual_intent=actual_intent,
            expected_cycle=expected_cycle,
            actual_cycle=actual_cycle,
            faithfulness=faithfulness,
            faithfulness_gri_errors=faithfulness_errors,
            answer_relevance=relevance,
            context_recall=recall,
            context_precision=precision,
            term_accuracy=term_acc,
            term_critical_errors=term_errors,
            latency_ms=latency_ms,
            intent_correct=intent_correct,
            cycle_correct=cycle_correct,
            milestone_complete=milestone_complete,
            n_tool_calls=n_tool_calls,
            iterations=iterations,
        )

    def _check_milestone_completeness(self, answer: str, item: dict) -> bool:
        """Vérifie si les critères de jalon sont complets.

        Args:
            answer: Réponse générée
            item: Question du dataset

        Returns:
            True si complet ou N/A
        """
        if item.get("expected_intent") != "JALON":
            return True

        if item.get("critical_check") == "all_criteria_listed":
            import re

            criteria = re.findall(r"^\s*[-•\d]+[.)]", answer, re.MULTILINE)
            return len(criteria) >= 3

        if item.get("critical_check") == "includes_gri_mapping":
            return "M" in answer.upper() and "J" in answer.upper()

        return True

    def _aggregate(self, results: list[GRIEvalResult]) -> EvaluationSummary:
        """Agrège les résultats en un résumé.

        Args:
            results: Liste des résultats

        Returns:
            EvaluationSummary
        """
        if not results:
            return EvaluationSummary()

        valid_results = [r for r in results if not r.error]

        if not valid_results:
            return EvaluationSummary(
                n_evaluated=len(results),
                n_errors=len(results),
            )

        def safe_stats(values: list[float]) -> tuple[float, float, float]:
            if not values:
                return 0.0, 0.0, 0.0
            return (
                float(np.mean(values)),
                float(np.percentile(values, 25)),
                float(np.percentile(values, 75)),
            )

        faith = [r.faithfulness for r in valid_results if r.faithfulness > 0]
        relev = [r.answer_relevance for r in valid_results if r.answer_relevance > 0]
        recall = [r.context_recall for r in valid_results if r.context_recall > 0]
        prec = [r.context_precision for r in valid_results if r.context_precision > 0]
        term = [r.term_accuracy for r in valid_results]
        latencies = [r.latency_ms for r in valid_results if r.latency_ms > 0]

        faith_mean, faith_p25, faith_p75 = safe_stats(faith)
        relev_mean, relev_p25, relev_p75 = safe_stats(relev)
        recall_mean, recall_p25, recall_p75 = safe_stats(recall)
        prec_mean, prec_p25, prec_p75 = safe_stats(prec)

        return EvaluationSummary(
            n_evaluated=len(results),
            n_errors=len(results) - len(valid_results),
            faithfulness_mean=faith_mean,
            faithfulness_p25=faith_p25,
            faithfulness_p75=faith_p75,
            answer_relevance_mean=relev_mean,
            answer_relevance_p25=relev_p25,
            answer_relevance_p75=relev_p75,
            context_recall_mean=recall_mean,
            context_recall_p25=recall_p25,
            context_recall_p75=recall_p75,
            context_precision_mean=prec_mean,
            context_precision_p25=prec_p25,
            context_precision_p75=prec_p75,
            term_accuracy_mean=float(np.mean(term)) if term else 0.0,
            term_accuracy_min=float(np.min(term)) if term else 0.0,
            latency_p50=float(np.percentile(latencies, 50)) if latencies else 0.0,
            latency_p95=float(np.percentile(latencies, 95)) if latencies else 0.0,
            latency_max=float(np.max(latencies)) if latencies else 0.0,
            intent_accuracy=sum(r.intent_correct for r in valid_results) / len(valid_results),
            cycle_accuracy=sum(r.cycle_correct for r in valid_results) / len(valid_results),
            milestone_completeness=sum(r.milestone_complete for r in valid_results)
            / len(valid_results),
        )

    def _check_quality_gates(self, summary: EvaluationSummary) -> list[str]:
        """Vérifie les quality gates.

        Args:
            summary: Résumé de l'évaluation

        Returns:
            Liste des échecs
        """
        failures = []

        checks = [
            ("faithfulness", summary.faithfulness_mean),
            ("answer_relevance", summary.answer_relevance_mean),
            ("context_recall", summary.context_recall_mean),
            ("context_precision", summary.context_precision_mean),
            ("term_accuracy", summary.term_accuracy_mean),
        ]

        for metric, value in checks:
            threshold = self.QUALITY_GATES.get(metric, 0)
            if value < threshold:
                label = "🔴 CRITIQUE" if metric == "term_accuracy" else "⚠️"
                failures.append(f"{label} {metric}: {value:.3f} < {threshold}")

        # Latence
        if summary.latency_p95 > self.QUALITY_GATES["latency_p95_ms"]:
            failures.append(
                f"⚠️ latency_p95: {summary.latency_p95:.0f}ms > "
                f"{self.QUALITY_GATES['latency_p95_ms']}ms"
            )

        return failures


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Charge un dataset JSON.

    Args:
        path: Chemin vers le fichier

    Returns:
        Liste des questions
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return cast(list[dict[str, Any]], data.get("questions", []))
    return cast(list[dict[str, Any]], data)


def save_report(report: dict, path: str | Path) -> None:
    """Sauvegarde le rapport d'évaluation.

    Args:
        report: Rapport à sauvegarder
        path: Chemin de sortie
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log.info("evaluation.report_saved", path=str(path))


async def main_async(args: argparse.Namespace) -> None:
    """Point d'entrée async."""
    # Charger le dataset
    dataset = load_dataset(args.dataset)

    if args.smoke_test:
        n = args.n or 10
        dataset = dataset[:n]
        log.info("evaluation.smoke_test", n_questions=len(dataset))

    # Créer l'évaluateur
    rag_system = None
    store = None

    if not args.no_rag:
        try:
            from src.agents.orchestrator import GRIOrchestrator
            from src.core.vector_store import GRIHybridStore

            store = GRIHybridStore()
            rag_system = GRIOrchestrator(store=store)
            log.info("evaluation.rag_loaded")
        except Exception as e:
            log.warning("evaluation.rag_load_failed", error=str(e))

    evaluator = GRIEvaluator(rag_system=rag_system, store=store)

    # Évaluer
    report = await evaluator.evaluate_dataset(dataset, verbose=args.verbose)

    # Sauvegarder
    if args.output:
        save_report(report, args.output)
    else:
        # Afficher le résumé
        print("\n" + "=" * 60)
        print("RAPPORT D'ÉVALUATION GRI RAG")
        print("=" * 60)

        summary = report["summary"]
        print(f"\nQuestions évaluées: {summary['n_evaluated']}")
        print(f"Erreurs: {summary['n_errors']}")

        print("\n--- MÉTRIQUES ---")
        print(f"Faithfulness:      {summary['faithfulness_mean']:.3f}")
        print(f"Answer Relevance:  {summary['answer_relevance_mean']:.3f}")
        print(f"Context Recall:    {summary['context_recall_mean']:.3f}")
        print(f"Context Precision: {summary['context_precision_mean']:.3f}")
        print(f"Term Accuracy:     {summary['term_accuracy_mean']:.3f}")

        print("\n--- LATENCE ---")
        print(f"P50: {summary['latency_p50']:.0f}ms")
        print(f"P95: {summary['latency_p95']:.0f}ms")

        print("\n--- QUALITY GATES ---")
        qg = report["quality_gates"]
        if qg["passed"]:
            print("✅ Tous les quality gates passent")
        else:
            print("❌ Quality gates échoués:")
            for failure in qg["failures"]:
                print(f"   {failure}")

        print("=" * 60)


def main() -> None:
    """Point d'entrée CLI."""
    parser = argparse.ArgumentParser(description="Évaluation du système RAG GRI")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/golden_dataset.json",
        help="Chemin vers le golden dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Chemin de sortie pour le rapport JSON",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Mode smoke test (10 questions)",
    )
    parser.add_argument(
        "-n",
        type=int,
        help="Nombre de questions (avec --smoke-test)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Afficher les détails",
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Évaluer sans système RAG (métriques seulement)",
    )

    args = parser.parse_args()

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
