"""Métriques RAGAS standards pour le GRI RAG.

Ce module implémente les métriques RAGAS classiques :
- Answer Relevance
- Context Recall
- Context Precision

Usage:
    from src.evaluation.metrics import (
        compute_answer_relevance,
        compute_context_recall,
        compute_context_precision,
    )
"""

import json
import re
from typing import Any

import structlog
from huggingface_hub import AsyncInferenceClient
from pydantic import BaseModel, Field

from src.core.config import settings

log = structlog.get_logger()


# === Answer Relevance ===

ANSWER_RELEVANCE_PROMPT = """Tu es un évaluateur expert en pertinence des réponses RAG.

Évalue si cette réponse est pertinente par rapport à la question posée.

## QUESTION

{question}

## RÉPONSE

{answer}

## INSTRUCTIONS

Évalue selon ces critères :
1. La réponse répond-elle directement à la question ?
2. La réponse est-elle complète ?
3. La réponse contient-elle des informations hors sujet ?
4. La réponse est-elle claire et bien structurée ?

## FORMAT DE SORTIE (JSON uniquement)

{{
  "relevance_score": 0.0,
  "directly_answers": true,
  "completeness": "COMPLÈTE|PARTIELLE|INSUFFISANTE",
  "off_topic_content": false,
  "explanation": "explication de l'évaluation"
}}

Score de 0.0 à 1.0 :
- 1.0 : Répond parfaitement et complètement
- 0.8 : Répond bien mais quelques lacunes
- 0.5 : Répond partiellement
- 0.2 : Répond à côté de la question
- 0.0 : Ne répond pas du tout

Réponds UNIQUEMENT avec le JSON."""


class AnswerRelevanceResult(BaseModel):
    """Résultat de l'évaluation Answer Relevance."""

    relevance_score: float = Field(ge=0.0, le=1.0)
    directly_answers: bool = True
    completeness: str = "COMPLÈTE"
    off_topic_content: bool = False
    explanation: str = ""
    error: str | None = None


async def compute_answer_relevance(
    question: str,
    answer: str,
    client: AsyncInferenceClient | None = None,
    model: str | None = None,
) -> AnswerRelevanceResult:
    """Calcule le score Answer Relevance.

    Évalue si la réponse est pertinente par rapport à la question.

    Args:
        question: Question posée
        answer: Réponse générée
        client: Client HF (optionnel)
        model: Modèle d'évaluation (optionnel)

    Returns:
        AnswerRelevanceResult
    """
    log.info("answer_relevance.start")

    if not answer.strip():
        return AnswerRelevanceResult(
            relevance_score=0.0,
            error="Empty answer",
        )

    if client is None:
        client = AsyncInferenceClient(token=settings.hf_api_key)

    model = model or settings.hf_eval_model

    prompt_content = ANSWER_RELEVANCE_PROMPT.format(
        question=question,
        answer=answer,
    )
    full_prompt = f"<s>[INST] {prompt_content} [/INST]"

    try:
        response = await client.text_generation(
            prompt=full_prompt,
            model=model,
            max_new_tokens=512,
            temperature=0.1,
            return_full_text=False,
        )

        # Parser le JSON
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            data = json.loads(json_match.group())
            return AnswerRelevanceResult(
                relevance_score=data.get("relevance_score", 0.5),
                directly_answers=data.get("directly_answers", True),
                completeness=data.get("completeness", "PARTIELLE"),
                off_topic_content=data.get("off_topic_content", False),
                explanation=data.get("explanation", ""),
            )

        return AnswerRelevanceResult(
            relevance_score=0.5,
            error="No JSON found in response",
        )

    except Exception as e:
        log.error("answer_relevance.error", error=str(e))
        return AnswerRelevanceResult(
            relevance_score=0.5,
            error=str(e),
        )


# === Context Recall ===

CONTEXT_RECALL_PROMPT = """Tu es un évaluateur expert en recall de contexte RAG.

Évalue si les chunks de contexte récupérés couvrent les informations nécessaires pour répondre correctement à la question.

## QUESTION

{question}

## RÉPONSE ATTENDUE (Ground Truth)

{ground_truth}

## CONTEXTE RÉCUPÉRÉ

{context}

## INSTRUCTIONS

Évalue selon ces critères :
1. Quelles informations de la réponse attendue sont présentes dans le contexte ?
2. Quelles informations manquent ?
3. Le contexte permet-il de construire la réponse attendue ?

## FORMAT DE SORTIE (JSON uniquement)

{{
  "recall_score": 0.0,
  "covered_points": ["point 1", "point 2"],
  "missing_points": ["point manquant"],
  "coverage_ratio": "X/Y points couverts",
  "explanation": "explication"
}}

Score = nb_points_couverts / nb_points_total

Réponds UNIQUEMENT avec le JSON."""


class ContextRecallResult(BaseModel):
    """Résultat de l'évaluation Context Recall."""

    recall_score: float = Field(ge=0.0, le=1.0)
    covered_points: list[str] = Field(default_factory=list)
    missing_points: list[str] = Field(default_factory=list)
    coverage_ratio: str = ""
    explanation: str = ""
    error: str | None = None


async def compute_context_recall(
    question: str,
    ground_truth: str,
    context_chunks: list[str],
    client: AsyncInferenceClient | None = None,
    model: str | None = None,
) -> ContextRecallResult:
    """Calcule le score Context Recall.

    Évalue si le contexte récupéré couvre les informations
    nécessaires pour la réponse attendue.

    Args:
        question: Question posée
        ground_truth: Réponse attendue
        context_chunks: Chunks de contexte récupérés
        client: Client HF (optionnel)
        model: Modèle d'évaluation (optionnel)

    Returns:
        ContextRecallResult
    """
    log.info("context_recall.start", n_chunks=len(context_chunks))

    if not context_chunks:
        return ContextRecallResult(
            recall_score=0.0,
            error="No context chunks",
        )

    if not ground_truth:
        return ContextRecallResult(
            recall_score=1.0,  # Pas de ground truth = pas de recall à mesurer
            error="No ground truth provided",
        )

    if client is None:
        client = AsyncInferenceClient(token=settings.hf_api_key)

    model = model or settings.hf_eval_model

    context = "\n\n---\n\n".join(context_chunks)

    prompt_content = CONTEXT_RECALL_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        context=context,
    )
    full_prompt = f"<s>[INST] {prompt_content} [/INST]"

    try:
        response = await client.text_generation(
            prompt=full_prompt,
            model=model,
            max_new_tokens=768,
            temperature=0.1,
            return_full_text=False,
        )

        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            data = json.loads(json_match.group())
            return ContextRecallResult(
                recall_score=data.get("recall_score", 0.5),
                covered_points=data.get("covered_points", []),
                missing_points=data.get("missing_points", []),
                coverage_ratio=data.get("coverage_ratio", ""),
                explanation=data.get("explanation", ""),
            )

        return ContextRecallResult(
            recall_score=0.5,
            error="No JSON found in response",
        )

    except Exception as e:
        log.error("context_recall.error", error=str(e))
        return ContextRecallResult(
            recall_score=0.5,
            error=str(e),
        )


# === Context Precision ===

CONTEXT_PRECISION_PROMPT = """Tu es un évaluateur expert en précision de contexte RAG.

Évalue si les chunks de contexte récupérés sont pertinents pour répondre à la question.

## QUESTION

{question}

## CONTEXTE RÉCUPÉRÉ (classé par score)

{context}

## INSTRUCTIONS

Pour chaque chunk de contexte :
1. Est-il pertinent pour répondre à la question ?
2. Apporte-t-il des informations utiles ?

## FORMAT DE SORTIE (JSON uniquement)

{{
  "precision_score": 0.0,
  "chunk_evaluations": [
    {{"chunk_index": 0, "relevant": true, "reason": "..."}},
    {{"chunk_index": 1, "relevant": false, "reason": "..."}}
  ],
  "n_relevant": 0,
  "n_total": 0
}}

Score = n_relevant / n_total

Réponds UNIQUEMENT avec le JSON."""


class ChunkEvaluation(BaseModel):
    """Évaluation d'un chunk."""

    chunk_index: int
    relevant: bool
    reason: str = ""


class ContextPrecisionResult(BaseModel):
    """Résultat de l'évaluation Context Precision."""

    precision_score: float = Field(ge=0.0, le=1.0)
    chunk_evaluations: list[ChunkEvaluation] = Field(default_factory=list)
    n_relevant: int = 0
    n_total: int = 0
    error: str | None = None


async def compute_context_precision(
    question: str,
    context_chunks: list[str],
    client: AsyncInferenceClient | None = None,
    model: str | None = None,
) -> ContextPrecisionResult:
    """Calcule le score Context Precision.

    Évalue quelle proportion des chunks récupérés est pertinente.

    Args:
        question: Question posée
        context_chunks: Chunks de contexte récupérés
        client: Client HF (optionnel)
        model: Modèle d'évaluation (optionnel)

    Returns:
        ContextPrecisionResult
    """
    log.info("context_precision.start", n_chunks=len(context_chunks))

    if not context_chunks:
        return ContextPrecisionResult(
            precision_score=0.0,
            error="No context chunks",
        )

    if client is None:
        client = AsyncInferenceClient(token=settings.hf_api_key)

    model = model or settings.hf_eval_model

    # Formater les chunks avec indices
    context_formatted = []
    for i, chunk in enumerate(context_chunks):
        context_formatted.append(f"[Chunk {i}]\n{chunk[:500]}...")

    context = "\n\n".join(context_formatted)

    prompt_content = CONTEXT_PRECISION_PROMPT.format(
        question=question,
        context=context,
    )
    full_prompt = f"<s>[INST] {prompt_content} [/INST]"

    try:
        response = await client.text_generation(
            prompt=full_prompt,
            model=model,
            max_new_tokens=768,
            temperature=0.1,
            return_full_text=False,
        )

        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            data = json.loads(json_match.group())

            chunk_evals = [
                ChunkEvaluation(**ce) for ce in data.get("chunk_evaluations", [])
            ]

            return ContextPrecisionResult(
                precision_score=data.get("precision_score", 0.5),
                chunk_evaluations=chunk_evals,
                n_relevant=data.get("n_relevant", 0),
                n_total=data.get("n_total", len(context_chunks)),
            )

        return ContextPrecisionResult(
            precision_score=0.5,
            n_total=len(context_chunks),
            error="No JSON found in response",
        )

    except Exception as e:
        log.error("context_precision.error", error=str(e))
        return ContextPrecisionResult(
            precision_score=0.5,
            n_total=len(context_chunks),
            error=str(e),
        )


# === Aliases pour compatibilité ===


async def answer_relevance(question: str, answer: str) -> float:
    """Version simplifiée de compute_answer_relevance."""
    result = await compute_answer_relevance(question, answer)
    return result.relevance_score


async def context_recall(
    question: str,
    ground_truth: str,
    context_chunks: list[str],
) -> float:
    """Version simplifiée de compute_context_recall."""
    result = await compute_context_recall(question, ground_truth, context_chunks)
    return result.recall_score


async def context_precision(question: str, context_chunks: list[str]) -> float:
    """Version simplifiée de compute_context_precision."""
    result = await compute_context_precision(question, context_chunks)
    return result.precision_score
