"""GRIGenerator - Génération de réponses grounded pour le GRI/FAR.

Ce module implémente le générateur principal avec :
- Sélection automatique du prompt par type de contenu
- Températures adaptées (0.0 pour normatif, 0.1 sinon)
- Post-processing avec validation GRI
- Gestion du contexte insuffisant

Usage:
    from src.generation import GRIGenerator, GRIResponseType

    generator = GRIGenerator()
    result = await generator.generate(
        query="Définition d'artefact",
        chunks=[...],
        response_type=GRIResponseType.DEFINITION,
    )
"""

import os
import time
from typing import Any

import structlog
from huggingface_hub import AsyncInferenceClient
from pydantic import BaseModel, Field

from src.generation.context_formatter import (
    check_context_sufficiency,
    extract_context_variables,
    format_comparison_context,
    format_gri_context,
    truncate_context,
)
from src.generation.postprocessor import (
    add_source_footer,
    clean_response,
    extract_citations,
    postprocess_gri_answer,
)
from src.generation.prompts import (
    GRIResponseType,
    get_max_tokens,
    get_prompt,
    get_system_prompt,
    get_temperature,
    intent_to_response_type,
)

log = structlog.get_logger()

# Modèle par défaut
DEFAULT_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"


class GenerationResult(BaseModel):
    """Résultat de génération GRI."""

    answer: str
    response_type: GRIResponseType
    temperature_used: float
    citations: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    has_normative_content: bool = False
    context_sufficient: bool = True
    latency_ms: float = 0.0
    model: str = ""
    validation: dict[str, Any] = Field(default_factory=dict)


class GRIGenerator:
    """Générateur de réponses grounded pour le GRI.

    Attributs:
        model: Modèle HuggingFace à utiliser
        client: Client HF Inference API
    """

    def __init__(
        self,
        model: str | None = None,
        hf_token: str | None = None,
    ):
        """Initialise le générateur.

        Args:
            model: Modèle HF (défaut: Mixtral-8x7B-Instruct)
            hf_token: Token HuggingFace (défaut: HF_API_KEY env)
        """
        self.model = model or os.getenv("HF_GENERATION_MODEL", DEFAULT_MODEL)
        token = hf_token or os.getenv("HF_API_KEY")

        self.client = AsyncInferenceClient(token=token)

        log.info(
            "generator.init",
            model=self.model,
        )

    async def generate(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        response_type: GRIResponseType | None = None,
        intent: str | None = None,
        context_vars: dict[str, Any] | None = None,
    ) -> GenerationResult:
        """Génère une réponse grounded.

        Args:
            query: Question utilisateur
            chunks: Chunks de contexte récupérés
            response_type: Type de réponse (optionnel, déduit de intent)
            intent: Intent du query router (optionnel)
            context_vars: Variables additionnelles pour le prompt

        Returns:
            GenerationResult avec la réponse et métadonnées
        """
        start_time = time.time()

        # Déterminer le type de réponse
        if response_type is None:
            if intent:
                response_type = intent_to_response_type(intent)
            else:
                response_type = GRIResponseType.GENERAL

        log.info(
            "generator.generate.start",
            query=query[:50],
            response_type=response_type.value,
            n_chunks=len(chunks),
        )

        # Vérifier la suffisance du contexte
        sufficiency = check_context_sufficiency(chunks, response_type)
        if not sufficiency["sufficient"]:
            log.warning(
                "generator.context_insufficient",
                reason=sufficiency["reason"],
            )
            return GenerationResult(
                answer=sufficiency["message"],
                response_type=response_type,
                temperature_used=0.0,
                citations=[],
                warnings=[f"Contexte insuffisant: {sufficiency['reason']}"],
                has_normative_content=False,
                context_sufficient=False,
                latency_ms=(time.time() - start_time) * 1000,
                model=self.model,
            )

        # Extraire les variables de contexte
        variables = extract_context_variables(chunks, response_type, query)
        if context_vars:
            variables.update(context_vars)

        # Gérer le cas de comparaison
        if response_type == GRIResponseType.COMPARISON:
            answer_result = await self._generate_comparison(
                query, chunks, variables
            )
        else:
            answer_result = await self._generate_standard(
                query, chunks, response_type, variables
            )

        latency_ms = (time.time() - start_time) * 1000

        log.info(
            "generator.generate.done",
            response_type=response_type.value,
            latency_ms=f"{latency_ms:.0f}",
            n_citations=len(answer_result["citations"]),
        )

        return GenerationResult(
            answer=answer_result["answer"],
            response_type=response_type,
            temperature_used=answer_result["temperature"],
            citations=answer_result["citations"],
            warnings=answer_result.get("warnings", []),
            has_normative_content=answer_result.get("has_normative", False),
            context_sufficient=True,
            latency_ms=latency_ms,
            model=self.model,
            validation=answer_result.get("validation", {}),
        )

    async def _generate_standard(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        response_type: GRIResponseType,
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        """Génère une réponse standard (non-comparaison).

        Args:
            query: Question
            chunks: Chunks de contexte
            response_type: Type de réponse
            variables: Variables pour le prompt

        Returns:
            Dict avec answer, citations, temperature, etc.
        """
        # Formater le contexte
        context = format_gri_context(chunks)
        context = truncate_context(context)
        variables["context"] = context

        # Construire les prompts
        system_prompt = get_system_prompt(response_type)
        temperature = get_temperature(response_type)
        max_tokens = get_max_tokens(response_type)

        try:
            user_prompt = get_prompt(response_type, **variables)
        except ValueError as e:
            log.error("generator.prompt_error", error=str(e))
            # Fallback vers GENERAL
            user_prompt = get_prompt(
                GRIResponseType.GENERAL,
                query=query,
                context=context,
            )
            response_type = GRIResponseType.GENERAL

        # Appel LLM
        full_prompt = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"

        try:
            response = await self.client.text_generation(
                prompt=full_prompt,
                model=self.model,
                max_new_tokens=max_tokens,
                temperature=temperature,
                return_full_text=False,
            )
        except Exception as e:
            log.error("generator.llm_error", error=str(e))
            return {
                "answer": f"Erreur lors de la génération : {str(e)}",
                "citations": [],
                "temperature": temperature,
                "warnings": ["Erreur LLM"],
                "has_normative": False,
            }

        # Post-processing
        response = clean_response(response)
        processed = postprocess_gri_answer(response, response_type, chunks)

        # Ajouter le footer de sources si nécessaire
        answer = add_source_footer(
            processed["answer"],
            processed["citations"],
            response_type,
        )

        return {
            "answer": answer,
            "citations": processed["citations"],
            "temperature": temperature,
            "warnings": processed["warnings"],
            "has_normative": processed["has_normative_content"],
            "validation": processed["validation"],
        }

    async def _generate_comparison(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        variables: dict[str, Any],
    ) -> dict[str, Any]:
        """Génère une réponse de comparaison.

        Pour une comparaison, on sépare les chunks par entité
        et on génère un tableau comparatif.

        Args:
            query: Question
            chunks: Tous les chunks
            variables: Variables extraites

        Returns:
            Dict avec answer, citations, etc.
        """
        response_type = GRIResponseType.COMPARISON
        temperature = get_temperature(response_type)
        max_tokens = get_max_tokens(response_type)

        # Séparer les chunks par entité si possible
        entity_a = variables.get("entity_a", "Entité A")
        entity_b = variables.get("entity_b", "Entité B")

        chunks_a = []
        chunks_b = []
        for chunk in chunks:
            content_lower = chunk.get("content", "").lower()
            cycle = chunk.get("cycle", "").lower()

            # Heuristique simple pour séparer
            if entity_a.lower() in content_lower or entity_a.lower() in cycle:
                chunks_a.append(chunk)
            elif entity_b.lower() in content_lower or entity_b.lower() in cycle:
                chunks_b.append(chunk)
            else:
                # Ajouter aux deux si incertain
                chunks_a.append(chunk)
                chunks_b.append(chunk)

        # Formater les contextes
        context_a, context_b = format_comparison_context(
            chunks_a, chunks_b, entity_a, entity_b
        )

        variables["context_a"] = truncate_context(context_a, max_chars=4000)
        variables["context_b"] = truncate_context(context_b, max_chars=4000)

        # Construire les prompts
        system_prompt = get_system_prompt(response_type)

        try:
            user_prompt = get_prompt(response_type, **variables)
        except ValueError as e:
            log.error("generator.comparison_prompt_error", error=str(e))
            # Fallback
            user_prompt = f"Compare {entity_a} et {entity_b} selon le GRI.\n\nContexte:\n{variables.get('context', '')}"

        # Appel LLM
        full_prompt = f"<s>[INST] {system_prompt}\n\n{user_prompt} [/INST]"

        try:
            response = await self.client.text_generation(
                prompt=full_prompt,
                model=self.model,
                max_new_tokens=max_tokens,
                temperature=temperature,
                return_full_text=False,
            )
        except Exception as e:
            log.error("generator.comparison_llm_error", error=str(e))
            return {
                "answer": f"Erreur lors de la génération : {str(e)}",
                "citations": [],
                "temperature": temperature,
                "warnings": ["Erreur LLM"],
                "has_normative": False,
            }

        # Post-processing
        response = clean_response(response)
        processed = postprocess_gri_answer(response, response_type, chunks)

        answer = add_source_footer(
            processed["answer"],
            processed["citations"],
            response_type,
        )

        return {
            "answer": answer,
            "citations": processed["citations"],
            "temperature": temperature,
            "warnings": processed["warnings"],
            "has_normative": False,
            "validation": processed["validation"],
        }

    async def generate_definition(
        self,
        term: str,
        glossary_entry: dict[str, Any] | None = None,
    ) -> GenerationResult:
        """Génère une définition ISO exacte.

        Méthode spécialisée pour les définitions.
        Si glossary_entry est fourni, utilise directement les données.

        Args:
            term: Terme à définir
            glossary_entry: Entrée du glossaire (optionnel)

        Returns:
            GenerationResult
        """
        if glossary_entry:
            # Utiliser directement l'entrée du glossaire
            definition_fr = glossary_entry.get("definition_fr", "")
            definition_en = glossary_entry.get("definition_en", "")
            term_en = glossary_entry.get("term_en", "")
            source = glossary_entry.get("source", "ISO/IEC/IEEE 15288:2023")

            answer = f"**{term}** ({term_en}) : {definition_fr}"
            if definition_en:
                answer += f"\n\n*English: {definition_en}*"
            answer += f"\n\n**Source : [{source}]**"

            return GenerationResult(
                answer=answer,
                response_type=GRIResponseType.DEFINITION,
                temperature_used=0.0,
                citations=[f"[GRI > Terminologie > {term}]"],
                has_normative_content=True,
                context_sufficient=True,
                latency_ms=0.0,
                model="direct_lookup",
            )

        # Sinon, générer via le LLM
        chunks = [{"content": f"Recherche de définition pour : {term}", "score": 0.5}]
        return await self.generate(
            query=f"Définition de {term}",
            chunks=chunks,
            response_type=GRIResponseType.DEFINITION,
            context_vars={"term": term},
        )

    async def generate_milestone_criteria(
        self,
        milestone_id: str,
        milestone_data: dict[str, Any],
    ) -> GenerationResult:
        """Génère la liste des critères d'un jalon.

        Méthode spécialisée pour les jalons avec données complètes.

        Args:
            milestone_id: ID du jalon (M4, J3, etc.)
            milestone_data: Données complètes du jalon

        Returns:
            GenerationResult
        """
        start_time = time.time()

        milestone_name = milestone_data.get("name", milestone_id)
        criteria = milestone_data.get("criteria", [])
        cycle = milestone_data.get("cycle", "GRI")

        # Construire la réponse directement depuis les données
        lines = [f"## Critères du jalon {milestone_id} — {milestone_name}"]
        lines.append("")

        for i, criterion in enumerate(criteria, 1):
            if isinstance(criterion, dict):
                text = criterion.get("text", criterion.get("content", str(criterion)))
            else:
                text = str(criterion)
            lines.append(f"{i}. {text}")

        lines.append("")

        # Mapping CIR si applicable
        if milestone_id.startswith("J"):
            gri_mapping = milestone_data.get("gri_equivalents", [])
            if gri_mapping:
                lines.append(f"**Équivalents GRI :** {', '.join(gri_mapping)}")
                lines.append("")

        lines.append(f"**Source : [{cycle} > Jalon {milestone_id} ({milestone_name})]**")

        answer = "\n".join(lines)
        latency_ms = (time.time() - start_time) * 1000

        return GenerationResult(
            answer=answer,
            response_type=GRIResponseType.MILESTONE,
            temperature_used=0.0,
            citations=[f"[{cycle} > Jalon {milestone_id} ({milestone_name})]"],
            has_normative_content=True,
            context_sufficient=True,
            latency_ms=latency_ms,
            model="direct_data",
            validation={"criteria_count": len(criteria)},
        )


# Fonction utilitaire pour usage simple
async def generate_gri_answer(
    query: str,
    chunks: list[dict[str, Any]],
    response_type: GRIResponseType | None = None,
    intent: str | None = None,
) -> GenerationResult:
    """Fonction utilitaire pour génération simple.

    Args:
        query: Question
        chunks: Chunks de contexte
        response_type: Type de réponse
        intent: Intent du router

    Returns:
        GenerationResult
    """
    generator = GRIGenerator()
    return await generator.generate(
        query=query,
        chunks=chunks,
        response_type=response_type,
        intent=intent,
    )
