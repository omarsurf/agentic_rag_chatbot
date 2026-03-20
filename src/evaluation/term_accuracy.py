"""Term Accuracy - Métrique custom pour le GRI.

Cette métrique est CRITIQUE pour le projet GRI :
une définition ISO paraphrasée est une erreur réglementaire.

Seuil : ≥ 0.95

Usage:
    from src.evaluation.term_accuracy import compute_term_accuracy

    result = await compute_term_accuracy(answer, glossary_index)
    print(result["term_accuracy_score"])
"""

import json
import re

import structlog
from huggingface_hub import AsyncInferenceClient
from pydantic import BaseModel, Field

from src.core.config import settings
from src.core.vector_store import GRIHybridStore

log = structlog.get_logger()


# Patterns pour détecter les termes GRI/ISO
GRI_TERMS_PATTERNS = [
    r"\b(artefact|artifact)\b",
    r"\b(CONOPS|SEMP|SRR|PDR|CDR|IRR|TRR|SAR|ORR|MNR)\b",
    r"\b(TRL|MRL|IRL)\b",
    r"\b(cycle de vie|lifecycle)\b",
    r"\b(jalon[s]? d[eé]cisionnel[s]?)\b",
    r"\b(v[eé]rification|validation|int[eé]gration)\b",
    r"\b(ing[eé]nierie syst[eè]me|systems? engineering)\b",
    r"\b(parties? prenantes?|stakeholders?)\b",
    r"\b(tra[cç]abilit[eé]|traceability)\b",
    r"\b(exigences? syst[eè]me|system requirements?)\b",
    r"\b(architecture syst[eè]me|system architecture)\b",
    r"\b(conception d[eé]taill[eé]e|detailed design)\b",
    r"\b(qualification|homologation)\b",
    r"\b(transition|d[eé]ploiement)\b",
    r"\b(maintien en condition op[eé]rationnelle|MCO)\b",
]


TERM_ACCURACY_PROMPT = """Tu es un évaluateur expert en terminologie ISO/GRI pour l'ingénierie système.

Vérifie si les définitions de termes ISO/GRI dans cette réponse correspondent exactement aux définitions normatives du glossaire GRI.

## GLOSSAIRE DE RÉFÉRENCE (définitions normatives)

{glossary_context}

## RÉPONSE À ÉVALUER

{answer}

## INSTRUCTIONS

Pour chaque terme ISO/GRI mentionné dans la réponse :
1. Identifier le terme et sa définition/utilisation dans la réponse
2. Comparer avec la définition normative du glossaire
3. Évaluer selon ces critères :
   - EXACT : Le sens est identique à la définition normative
   - APPROXIMATIF : Le sens est préservé mais la formulation diffère légèrement
   - INCORRECT : Le sens est différent de la norme (ERREUR CRITIQUE)
   - NON_TROUVÉ : Le terme n'est pas dans le glossaire fourni

## FORMAT DE SORTIE (JSON uniquement)

{{
  "term_evaluations": [
    {{
      "term": "nom du terme",
      "definition_in_answer": "comment le terme est défini/utilisé dans la réponse",
      "normative_definition": "définition du glossaire",
      "status": "EXACT|APPROXIMATIF|INCORRECT|NON_TROUVÉ",
      "severity": "CRITIQUE|MINEUR|OK",
      "explanation": "explication de l'évaluation"
    }}
  ],
  "term_accuracy_score": 0.0,
  "critical_errors": []
}}

## SCORING

- EXACT → 1.0 par terme
- APPROXIMATIF → 0.5 par terme (acceptable si le sens est préservé)
- INCORRECT → 0.0 par terme (sens différent = erreur critique)
- NON_TROUVÉ → ignoré dans le calcul

Score final = (Σ scores) / (nb termes évalués hors NON_TROUVÉ)
Si aucun terme trouvé : score = 1.0

Réponds UNIQUEMENT avec le JSON, sans texte avant ou après."""


class TermEvaluation(BaseModel):
    """Évaluation d'un terme ISO/GRI."""

    term: str
    definition_in_answer: str
    normative_definition: str
    status: str = Field(pattern="^(EXACT|APPROXIMATIF|INCORRECT|NON_TROUVÉ)$")
    severity: str = Field(pattern="^(CRITIQUE|MINEUR|OK)$")
    explanation: str = ""


class TermAccuracyResult(BaseModel):
    """Résultat de l'évaluation Term Accuracy."""

    term_accuracy_score: float = Field(ge=0.0, le=1.0)
    term_evaluations: list[TermEvaluation] = Field(default_factory=list)
    critical_errors: list[str] = Field(default_factory=list)
    no_terms_found: bool = False
    no_normative_terms: bool = False
    raw_response: str | None = None
    error: str | None = None


def extract_iso_terms(text: str) -> list[str]:
    """Détecte les termes GRI/ISO dans le texte.

    Args:
        text: Texte à analyser

    Returns:
        Liste de termes uniques détectés
    """
    found = []
    for pattern in GRI_TERMS_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            term = match if isinstance(match, str) else match[0]
            found.append(term.lower())

    return list(set(found))


async def compute_term_accuracy(
    answer: str,
    store: GRIHybridStore,
    client: AsyncInferenceClient | None = None,
    model: str | None = None,
) -> TermAccuracyResult:
    """Calcule le score Term Accuracy pour une réponse.

    Cette métrique vérifie que les définitions ISO/GRI dans la réponse
    correspondent exactement aux définitions normatives du glossaire.

    Args:
        answer: Réponse à évaluer
        store: Vector store avec l'index glossaire
        client: Client HF (optionnel)
        model: Modèle d'évaluation (optionnel)

    Returns:
        TermAccuracyResult avec le score et les détails
    """
    log.info("term_accuracy.start", answer_length=len(answer))

    # Extraire les termes ISO potentiels
    iso_terms = extract_iso_terms(answer)

    if not iso_terms:
        log.info("term_accuracy.no_terms_found")
        return TermAccuracyResult(
            term_accuracy_score=1.0,
            no_terms_found=True,
        )

    log.info("term_accuracy.terms_detected", n_terms=len(iso_terms), terms=iso_terms[:5])

    # Récupérer les définitions normatives
    glossary_context = []
    for term in iso_terms[:10]:  # Max 10 termes par évaluation
        try:
            defn = await store.glossary_lookup(term)
            if defn:
                term_fr = defn.metadata.get("term_fr", term)
                term_en = defn.metadata.get("term_en", "")
                definition = defn.content
                standard_ref = defn.metadata.get("standard_ref", "GRI")

                glossary_context.append(
                    f"**{term_fr}** ({term_en}): {definition}\n  Source: {standard_ref}"
                )
        except Exception as e:
            log.warning("term_accuracy.lookup_failed", term=term, error=str(e))

    if not glossary_context:
        log.info("term_accuracy.no_normative_terms")
        return TermAccuracyResult(
            term_accuracy_score=1.0,
            no_normative_terms=True,
        )

    # Créer le client si nécessaire
    if client is None:
        client = AsyncInferenceClient(token=settings.hf_api_key)

    model = model or settings.hf_eval_model

    # Construire le prompt
    prompt_content = TERM_ACCURACY_PROMPT.format(
        glossary_context="\n\n".join(glossary_context),
        answer=answer,
    )
    full_prompt = f"<s>[INST] {prompt_content} [/INST]"

    # Appel LLM
    try:
        response = await client.text_generation(
            prompt=full_prompt,
            model=model,
            max_new_tokens=1024,
            temperature=0.1,
            return_full_text=False,
        )

        log.info("term_accuracy.llm_response", response_length=len(response))

        # Parser le JSON
        result = _parse_llm_response(response)
        result.raw_response = response

        log.info(
            "term_accuracy.done",
            score=result.term_accuracy_score,
            n_evaluations=len(result.term_evaluations),
            n_critical=len(result.critical_errors),
        )

        return result

    except Exception as e:
        log.error("term_accuracy.error", error=str(e))
        return TermAccuracyResult(
            term_accuracy_score=0.0,
            error=str(e),
        )


def _parse_llm_response(response: str) -> TermAccuracyResult:
    """Parse la réponse JSON du LLM.

    Args:
        response: Réponse brute du LLM

    Returns:
        TermAccuracyResult
    """
    # Nettoyer la réponse
    cleaned = response.strip()

    # Chercher le JSON dans la réponse
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    if not json_match:
        return TermAccuracyResult(
            term_accuracy_score=0.5,  # Score par défaut si parsing échoue
            error="No JSON found in response",
        )

    try:
        data = json.loads(json_match.group())

        evaluations = []
        for eval_data in data.get("term_evaluations", []):
            evaluations.append(
                TermEvaluation(
                    term=eval_data.get("term", ""),
                    definition_in_answer=eval_data.get("definition_in_answer", ""),
                    normative_definition=eval_data.get("normative_definition", ""),
                    status=eval_data.get("status", "NON_TROUVÉ"),
                    severity=eval_data.get("severity", "OK"),
                    explanation=eval_data.get("explanation", ""),
                )
            )

        # Recalculer le score pour être sûr
        score = _calculate_score(evaluations)

        return TermAccuracyResult(
            term_accuracy_score=score,
            term_evaluations=evaluations,
            critical_errors=data.get("critical_errors", []),
        )

    except json.JSONDecodeError as e:
        return TermAccuracyResult(
            term_accuracy_score=0.5,
            error=f"JSON parse error: {str(e)}",
        )


def _calculate_score(evaluations: list[TermEvaluation]) -> float:
    """Recalcule le score à partir des évaluations.

    Args:
        evaluations: Liste des évaluations de termes

    Returns:
        Score entre 0.0 et 1.0
    """
    if not evaluations:
        return 1.0

    scores = []
    for eval in evaluations:
        if eval.status == "EXACT":
            scores.append(1.0)
        elif eval.status == "APPROXIMATIF":
            scores.append(0.5)
        elif eval.status == "INCORRECT":
            scores.append(0.0)
        # NON_TROUVÉ est ignoré

    if not scores:
        return 1.0

    return sum(scores) / len(scores)


# Alias pour compatibilité
async def term_accuracy(
    answer: str,
    store: GRIHybridStore,
) -> float:
    """Calcule le score Term Accuracy (version simplifiée).

    Args:
        answer: Réponse à évaluer
        store: Vector store

    Returns:
        Score entre 0.0 et 1.0
    """
    result = await compute_term_accuracy(answer, store)
    return result.term_accuracy_score
