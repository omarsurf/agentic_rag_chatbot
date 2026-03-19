"""Suite d'évaluation GRI - RAGAS + Term Accuracy.

Ce module fournit les outils d'évaluation pour le système RAG GRI :
- Faithfulness GRI (adaptée au domaine)
- Term Accuracy (métrique custom critique)
- Answer Relevance
- Context Recall
- Context Precision
- Pipeline d'évaluation complet

Usage:
    from src.evaluation import GRIEvaluator, compute_term_accuracy

    evaluator = GRIEvaluator(rag_system)
    report = await evaluator.evaluate_dataset(dataset)

CLI:
    python -m src.evaluation.pipeline --dataset data/golden_dataset.json
"""

from src.evaluation.faithfulness_gri import (
    FaithfulnessResult,
    compute_faithfulness_gri,
    faithfulness_gri,
)
from src.evaluation.metrics import (
    AnswerRelevanceResult,
    ContextPrecisionResult,
    ContextRecallResult,
    answer_relevance,
    compute_answer_relevance,
    compute_context_precision,
    compute_context_recall,
    context_precision,
    context_recall,
)
from src.evaluation.pipeline import (
    EvaluationSummary,
    GRIEvalResult,
    GRIEvaluator,
    load_dataset,
    save_report,
)
from src.evaluation.term_accuracy import (
    TermAccuracyResult,
    compute_term_accuracy,
    extract_iso_terms,
    term_accuracy,
)

__all__ = [
    # Pipeline
    "GRIEvaluator",
    "GRIEvalResult",
    "EvaluationSummary",
    "load_dataset",
    "save_report",
    # Faithfulness GRI
    "compute_faithfulness_gri",
    "faithfulness_gri",
    "FaithfulnessResult",
    # Term Accuracy
    "compute_term_accuracy",
    "term_accuracy",
    "extract_iso_terms",
    "TermAccuracyResult",
    # Answer Relevance
    "compute_answer_relevance",
    "answer_relevance",
    "AnswerRelevanceResult",
    # Context Recall
    "compute_context_recall",
    "context_recall",
    "ContextRecallResult",
    # Context Precision
    "compute_context_precision",
    "context_precision",
    "ContextPrecisionResult",
]
