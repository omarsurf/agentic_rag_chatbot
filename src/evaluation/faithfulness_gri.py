"""Faithfulness GRI - Métrique adaptée au domaine GRI.

Cette métrique évalue si les affirmations de la réponse sont
supportées par les sources GRI, avec une attention particulière
aux erreurs GRI-spécifiques.

Seuil : ≥ 0.85

Usage:
    from src.evaluation.faithfulness_gri import compute_faithfulness_gri

    result = await compute_faithfulness_gri(answer, context_chunks)
    print(result["faithfulness_score"])
"""

import json
import re

import structlog
from huggingface_hub import AsyncInferenceClient
from pydantic import BaseModel, Field

from src.core.config import settings

log = structlog.get_logger()


# Types d'erreurs GRI spécifiques
GRI_ERROR_TYPES = [
    "jalon_inexistant",  # Citer un jalon qui n'existe pas (M10, J7...)
    "mauvais_mapping",  # Mauvais mapping CIR↔GRI
    "phase_inexistante",  # Phase inexistante (Phase 8, 9...)
    "critere_invente",  # Critère de jalon inventé
    "duree_inventee",  # Durée non mentionnée dans les sources
    "processus_inexistant",  # Processus IS 15288 inexistant
    "definition_incorrecte",  # Définition ISO incorrecte
]


FAITHFULNESS_GRI_PROMPT = """Tu es un évaluateur expert en fidélité des réponses RAG pour le domaine GRI.

Évalue si cette réponse sur le GRI est factuellement supportée par les sources GRI fournies.

## SOURCES GRI

{context}

## RÉPONSE À ÉVALUER

{answer}

## INSTRUCTIONS

Pour chaque affirmation factuelle de la réponse :
1. Identifier l'affirmation
2. Chercher le support dans les sources GRI fournies
3. Classifier selon ces critères :
   - SUPPORTÉE : L'affirmation est explicitement dans les sources
   - INFÉRÉE : L'affirmation peut être déduite des sources (acceptable)
   - INVENTÉE : L'affirmation n'est pas dans les sources (ERREUR)
   - HORS_PÉRIMÈTRE : L'affirmation est vraie mais hors du périmètre GRI

## ERREURS GRI SPÉCIFIQUES À DÉTECTER

- jalon_inexistant : Citer un jalon qui n'existe pas (ex: M10, J7)
- mauvais_mapping : Mauvais mapping CIR↔GRI (ex: dire J3 = M4 au lieu de M5+M6)
- phase_inexistante : Phase inexistante (ex: Phase 8 du GRI)
- critere_invente : Critère de jalon non présent dans les sources
- duree_inventee : Durée inventée non mentionnée
- processus_inexistant : Processus IS 15288 inexistant
- definition_incorrecte : Définition ISO incorrecte

## FORMAT DE SORTIE (JSON uniquement)

{{
  "claims": [
    {{
      "claim": "l'affirmation exacte",
      "status": "SUPPORTÉE|INFÉRÉE|INVENTÉE|HORS_PÉRIMÈTRE",
      "evidence": "extrait de la source qui supporte ou contredit",
      "gri_error_type": "type d'erreur ou null"
    }}
  ],
  "faithfulness_score": 0.0,
  "gri_specific_errors": [],
  "summary": "résumé de l'évaluation"
}}

## SCORING

- SUPPORTÉE → 1.0
- INFÉRÉE → 0.8
- HORS_PÉRIMÈTRE → 0.5 (pénalité légère)
- INVENTÉE → 0.0

Score = (Σ scores) / nb_claims

Réponds UNIQUEMENT avec le JSON, sans texte avant ou après."""


class ClaimEvaluation(BaseModel):
    """Évaluation d'une affirmation."""

    claim: str
    status: str = Field(pattern="^(SUPPORTÉE|INFÉRÉE|INVENTÉE|HORS_PÉRIMÈTRE)$")
    evidence: str = ""
    gri_error_type: str | None = None


class FaithfulnessResult(BaseModel):
    """Résultat de l'évaluation Faithfulness GRI."""

    faithfulness_score: float = Field(ge=0.0, le=1.0)
    claims: list[ClaimEvaluation] = Field(default_factory=list)
    gri_specific_errors: list[str] = Field(default_factory=list)
    summary: str = ""
    raw_response: str | None = None
    error: str | None = None


async def compute_faithfulness_gri(
    answer: str,
    context_chunks: list[str],
    client: AsyncInferenceClient | None = None,
    model: str | None = None,
) -> FaithfulnessResult:
    """Calcule le score Faithfulness adapté au GRI.

    Cette métrique vérifie que les affirmations de la réponse
    sont supportées par les sources GRI, avec détection des
    erreurs GRI-spécifiques.

    Args:
        answer: Réponse à évaluer
        context_chunks: Liste des chunks de contexte utilisés
        client: Client HF (optionnel)
        model: Modèle d'évaluation (optionnel)

    Returns:
        FaithfulnessResult avec le score et les détails
    """
    log.info(
        "faithfulness_gri.start",
        answer_length=len(answer),
        n_chunks=len(context_chunks),
    )

    if not context_chunks:
        log.warning("faithfulness_gri.no_context")
        return FaithfulnessResult(
            faithfulness_score=0.0,
            error="No context chunks provided",
        )

    if not answer.strip():
        log.warning("faithfulness_gri.empty_answer")
        return FaithfulnessResult(
            faithfulness_score=0.0,
            error="Empty answer",
        )

    # Créer le client si nécessaire
    if client is None:
        client = AsyncInferenceClient(token=settings.hf_api_key)

    model = model or settings.hf_eval_model

    # Formater le contexte
    context = "\n\n---\n\n".join(context_chunks)

    # Construire le prompt
    prompt_content = FAITHFULNESS_GRI_PROMPT.format(
        context=context,
        answer=answer,
    )
    full_prompt = f"<s>[INST] {prompt_content} [/INST]"

    # Appel LLM
    try:
        response = await client.text_generation(
            prompt=full_prompt,
            model=model,
            max_new_tokens=1536,
            temperature=0.1,
            return_full_text=False,
        )

        log.info("faithfulness_gri.llm_response", response_length=len(response))

        # Parser le JSON
        result = _parse_llm_response(response)
        result.raw_response = response

        # Détecter les erreurs GRI automatiquement
        auto_errors = _detect_gri_errors(answer)
        result.gri_specific_errors.extend(auto_errors)

        log.info(
            "faithfulness_gri.done",
            score=result.faithfulness_score,
            n_claims=len(result.claims),
            n_errors=len(result.gri_specific_errors),
        )

        return result

    except Exception as e:
        log.error("faithfulness_gri.error", error=str(e))
        return FaithfulnessResult(
            faithfulness_score=0.0,
            error=str(e),
        )


def _parse_llm_response(response: str) -> FaithfulnessResult:
    """Parse la réponse JSON du LLM.

    Args:
        response: Réponse brute du LLM

    Returns:
        FaithfulnessResult
    """
    cleaned = response.strip()

    # Chercher le JSON
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    if not json_match:
        return FaithfulnessResult(
            faithfulness_score=0.5,
            error="No JSON found in response",
        )

    try:
        data = json.loads(json_match.group())

        claims = []
        for claim_data in data.get("claims", []):
            claims.append(
                ClaimEvaluation(
                    claim=claim_data.get("claim", ""),
                    status=claim_data.get("status", "INVENTÉE"),
                    evidence=claim_data.get("evidence", ""),
                    gri_error_type=claim_data.get("gri_error_type"),
                )
            )

        # Recalculer le score
        score = _calculate_score(claims)

        # Collecter les erreurs GRI
        gri_errors = []
        for claim in claims:
            if claim.gri_error_type:
                gri_errors.append(f"{claim.gri_error_type}: {claim.claim}")

        return FaithfulnessResult(
            faithfulness_score=score,
            claims=claims,
            gri_specific_errors=gri_errors,
            summary=data.get("summary", ""),
        )

    except json.JSONDecodeError as e:
        return FaithfulnessResult(
            faithfulness_score=0.5,
            error=f"JSON parse error: {str(e)}",
        )


def _calculate_score(claims: list[ClaimEvaluation]) -> float:
    """Recalcule le score à partir des évaluations.

    Args:
        claims: Liste des évaluations

    Returns:
        Score entre 0.0 et 1.0
    """
    if not claims:
        return 1.0

    scores = []
    for claim in claims:
        if claim.status == "SUPPORTÉE":
            scores.append(1.0)
        elif claim.status == "INFÉRÉE":
            scores.append(0.8)
        elif claim.status == "HORS_PÉRIMÈTRE":
            scores.append(0.5)
        elif claim.status == "INVENTÉE":
            scores.append(0.0)

    return sum(scores) / len(scores) if scores else 1.0


def _detect_gri_errors(answer: str) -> list[str]:
    """Détecte automatiquement certaines erreurs GRI.

    Args:
        answer: Réponse à analyser

    Returns:
        Liste d'erreurs détectées
    """
    errors = []

    # Jalons GRI invalides
    invalid_milestones = re.findall(r"\bM1[0-9]\b|\bM[2-9][0-9]\b", answer)
    if invalid_milestones:
        errors.append(f"jalon_inexistant: {', '.join(invalid_milestones)}")

    # Jalons CIR invalides
    invalid_cir = re.findall(r"\bJ[7-9]\b|\bJ[1-9][0-9]\b", answer)
    if invalid_cir:
        errors.append(f"jalon_inexistant: {', '.join(invalid_cir)}")

    # Phases GRI invalides (> 7)
    invalid_phases = re.findall(r"\bPhase\s+([89]|[1-9][0-9])\b", answer, re.IGNORECASE)
    if invalid_phases:
        errors.append(f"phase_inexistante: Phase {', '.join(invalid_phases)}")

    # Phases CIR invalides (> 4)
    if "CIR" in answer.upper():
        cir_phases = re.findall(r"\bPhase\s+([5-9])\b.*CIR", answer, re.IGNORECASE)
        if cir_phases:
            errors.append(f"phase_inexistante (CIR): Phase {', '.join(cir_phases)}")

    return errors


# Alias pour compatibilité
async def faithfulness_gri(
    answer: str,
    context_chunks: list[str],
) -> float:
    """Calcule le score Faithfulness GRI (version simplifiée).

    Args:
        answer: Réponse à évaluer
        context_chunks: Chunks de contexte

    Returns:
        Score entre 0.0 et 1.0
    """
    result = await compute_faithfulness_gri(answer, context_chunks)
    return result.faithfulness_score
