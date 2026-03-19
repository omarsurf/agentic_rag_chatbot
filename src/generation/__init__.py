"""Generation - Module de génération grounded pour le GRI/FAR.

Ce module contient les composants pour la génération de réponses :

- prompts: Templates de prompts par type de contenu (DEFINITION, MILESTONE, etc.)
- context_formatter: Formatage du contexte pour le LLM
- postprocessor: Validation et post-traitement des réponses
- generator: GRIGenerator - générateur principal

Usage:
    from src.generation import GRIGenerator, GRIResponseType

    generator = GRIGenerator()
    result = await generator.generate(
        query="Définition d'artefact",
        chunks=[...],
        response_type=GRIResponseType.DEFINITION,
    )
"""

from src.generation.context_formatter import (
    check_context_sufficiency,
    extract_context_variables,
    format_comparison_context,
    format_gri_context,
    truncate_context,
)
from src.generation.generator import (
    GenerationResult,
    GRIGenerator,
    generate_gri_answer,
)
from src.generation.postprocessor import (
    add_source_footer,
    clean_response,
    extract_citations,
    postprocess_gri_answer,
    validate_citations_against_context,
    validate_milestones,
    validate_phases,
)
from src.generation.prompts import (
    GRIResponseType,
    MAX_TOKENS_MAP,
    TEMPERATURE_MAP,
    get_max_tokens,
    get_prompt,
    get_system_prompt,
    get_temperature,
    intent_to_response_type,
)

__all__ = [
    # Generator
    "GRIGenerator",
    "GenerationResult",
    "generate_gri_answer",
    # Response Types
    "GRIResponseType",
    # Prompts
    "get_prompt",
    "get_system_prompt",
    "get_temperature",
    "get_max_tokens",
    "intent_to_response_type",
    "TEMPERATURE_MAP",
    "MAX_TOKENS_MAP",
    # Context Formatter
    "format_gri_context",
    "format_comparison_context",
    "check_context_sufficiency",
    "extract_context_variables",
    "truncate_context",
    # Postprocessor
    "postprocess_gri_answer",
    "extract_citations",
    "validate_milestones",
    "validate_phases",
    "validate_citations_against_context",
    "clean_response",
    "add_source_footer",
]
