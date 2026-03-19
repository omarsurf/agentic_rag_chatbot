"""Tools GRI - 5 outils spécialisés pour l'agent.

Ce module contient les 5 tools utilisés par l'orchestrateur ReAct :

1. retrieve_gri_chunks  - Recherche hybride avec filtres section_type
2. lookup_gri_glossary  - Définitions exactes ISO (FR+EN)
3. get_milestone_criteria - Checklist complète d'un jalon M0-M9 ou J1-J6
4. compare_approaches   - Multi-retrieve parallèle pour comparaisons
5. get_phase_summary    - Parent doc retrieval - objectifs + livrables

Usage:
    from src.tools import execute_tool, TOOLS

    # Exécuter un tool
    result = await execute_tool(
        "retrieve_gri_chunks",
        {"query": "processus de vérification"},
        store,
    )

    # Liste des tools disponibles
    for tool in TOOLS:
        print(tool["name"])
"""

from src.tools.compare import (
    CompareApproachesInput,
    CompareApproachesOutput,
    compare_approaches,
)
from src.tools.definitions import (
    COMPARE_APPROACHES,
    GET_MILESTONE_CRITERIA,
    GET_PHASE_SUMMARY,
    LOOKUP_GRI_GLOSSARY,
    RETRIEVE_GRI_CHUNKS,
    TOOLS,
    format_tools_for_prompt,
    get_tool_by_name,
    get_tool_names,
)
from src.tools.executor import (
    ToolExecutor,
    ToolResult,
    execute_tool,
    execute_tools_parallel,
    format_tool_result_for_llm,
    get_executor,
)
from src.tools.glossary import (
    GlossaryDefinition,
    LookupGlossaryInput,
    LookupGlossaryOutput,
    format_definition_for_response,
    lookup_gri_glossary,
)
from src.tools.milestones import (
    GetMilestoneInput,
    GetMilestoneOutput,
    MilestoneCriterion,
    format_milestone_for_response,
    get_milestone_criteria,
    normalize_milestone_id,
)
from src.tools.phases import (
    GetPhaseSummaryInput,
    GetPhaseSummaryOutput,
    PhaseDeliverable,
    PhaseObjective,
    format_phase_for_response,
    get_phase_summary,
)
from src.tools.retrieve_gri import (
    RetrieveChunk,
    RetrieveGRIInput,
    RetrieveGRIOutput,
    retrieve_gri_chunks,
)

__all__ = [
    # Definitions
    "TOOLS",
    "RETRIEVE_GRI_CHUNKS",
    "LOOKUP_GRI_GLOSSARY",
    "GET_MILESTONE_CRITERIA",
    "COMPARE_APPROACHES",
    "GET_PHASE_SUMMARY",
    "get_tool_by_name",
    "get_tool_names",
    "format_tools_for_prompt",
    # Executor
    "ToolExecutor",
    "ToolResult",
    "execute_tool",
    "execute_tools_parallel",
    "format_tool_result_for_llm",
    "get_executor",
    # retrieve_gri_chunks
    "retrieve_gri_chunks",
    "RetrieveGRIInput",
    "RetrieveGRIOutput",
    "RetrieveChunk",
    # lookup_gri_glossary
    "lookup_gri_glossary",
    "LookupGlossaryInput",
    "LookupGlossaryOutput",
    "GlossaryDefinition",
    "format_definition_for_response",
    # get_milestone_criteria
    "get_milestone_criteria",
    "GetMilestoneInput",
    "GetMilestoneOutput",
    "MilestoneCriterion",
    "format_milestone_for_response",
    "normalize_milestone_id",
    # compare_approaches
    "compare_approaches",
    "CompareApproachesInput",
    "CompareApproachesOutput",
    # get_phase_summary
    "get_phase_summary",
    "GetPhaseSummaryInput",
    "GetPhaseSummaryOutput",
    "PhaseObjective",
    "PhaseDeliverable",
    "format_phase_for_response",
]
