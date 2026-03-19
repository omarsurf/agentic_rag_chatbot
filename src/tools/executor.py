"""Tool Executor - Dispatcher central pour les 5 tools GRI.

Ce module route les appels de tools vers les bonnes implémentations
et gère les erreurs de manière uniforme.

Usage:
    from src.tools.executor import execute_tool, ToolExecutor

    result = await execute_tool(
        tool_name="retrieve_gri_chunks",
        input_data={"query": "processus de vérification"},
        store=store,
    )
"""

from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, Field

from src.tools import compare, glossary, milestones, phases, retrieve_gri
from src.tools.definitions import get_tool_names

if TYPE_CHECKING:
    from src.core.vector_store import GRIHybridStore

log = structlog.get_logger()


class ToolResult(BaseModel):
    """Résultat d'exécution d'un tool."""

    tool_name: str
    success: bool = True
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    execution_time_ms: float = 0.0


class ToolExecutor:
    """Executor central pour les tools GRI.

    Gère le routing, l'exécution et la gestion d'erreurs
    pour tous les tools de l'agent.

    Attributes:
        store: Vector store GRI
    """

    # Mapping tool_name -> module.execute
    TOOL_HANDLERS = {
        "retrieve_gri_chunks": retrieve_gri.execute,
        "lookup_gri_glossary": glossary.execute,
        "get_milestone_criteria": milestones.execute,
        "compare_approaches": compare.execute,
        "get_phase_summary": phases.execute,
    }

    def __init__(self, store: "GRIHybridStore") -> None:
        """Initialise l'executor.

        Args:
            store: Vector store GRI
        """
        self.store = store
        log.info("tool_executor.init", available_tools=list(self.TOOL_HANDLERS.keys()))

    async def execute(
        self,
        tool_name: str,
        input_data: dict[str, Any],
    ) -> ToolResult:
        """Exécute un tool par son nom.

        Args:
            tool_name: Nom du tool
            input_data: Paramètres d'entrée

        Returns:
            ToolResult avec le résultat ou l'erreur
        """
        import time

        start = time.time()

        log.info(
            "tool_executor.execute_start",
            tool=tool_name,
            input_keys=list(input_data.keys()),
        )

        # Vérifier que le tool existe
        if tool_name not in self.TOOL_HANDLERS:
            available = get_tool_names()
            error_msg = (
                f"Tool '{tool_name}' non reconnu. "
                f"Tools disponibles : {', '.join(available)}"
            )
            log.error("tool_executor.unknown_tool", tool=tool_name)
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=error_msg,
            )

        # Exécuter le tool
        try:
            handler = self.TOOL_HANDLERS[tool_name]
            result = await handler(input_data, self.store)

            execution_time = (time.time() - start) * 1000

            log.info(
                "tool_executor.execute_done",
                tool=tool_name,
                execution_time_ms=f"{execution_time:.1f}",
            )

            return ToolResult(
                tool_name=tool_name,
                success=True,
                result=result,
                execution_time_ms=execution_time,
            )

        except Exception as e:
            execution_time = (time.time() - start) * 1000

            log.error(
                "tool_executor.execute_error",
                tool=tool_name,
                error=str(e),
                execution_time_ms=f"{execution_time:.1f}",
            )

            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=str(e),
                execution_time_ms=execution_time,
            )

    def get_available_tools(self) -> list[str]:
        """Retourne la liste des tools disponibles."""
        return list(self.TOOL_HANDLERS.keys())

    def is_valid_tool(self, tool_name: str) -> bool:
        """Vérifie si un tool existe."""
        return tool_name in self.TOOL_HANDLERS


# Singleton pour usage global
_executor: ToolExecutor | None = None


def get_executor(store: "GRIHybridStore") -> ToolExecutor:
    """Retourne le singleton de l'executor."""
    global _executor
    if _executor is None:
        _executor = ToolExecutor(store)
    return _executor


async def execute_tool(
    tool_name: str,
    input_data: dict[str, Any],
    store: "GRIHybridStore",
) -> dict[str, Any]:
    """Fonction helper pour exécuter un tool.

    Args:
        tool_name: Nom du tool
        input_data: Paramètres d'entrée
        store: Vector store GRI

    Returns:
        Résultat du tool sous forme de dict
    """
    executor = get_executor(store)
    result = await executor.execute(tool_name, input_data)

    if not result.success:
        return {
            "error": result.error,
            "tool": tool_name,
            "success": False,
        }

    return result.result


async def execute_tools_parallel(
    tool_calls: list[tuple[str, dict[str, Any]]],
    store: "GRIHybridStore",
) -> list[ToolResult]:
    """Exécute plusieurs tools en parallèle.

    Args:
        tool_calls: Liste de (tool_name, input_data)
        store: Vector store GRI

    Returns:
        Liste de ToolResult dans le même ordre
    """
    import asyncio

    executor = get_executor(store)

    tasks = [
        executor.execute(tool_name, input_data)
        for tool_name, input_data in tool_calls
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Convertir les exceptions en ToolResult
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            tool_name = tool_calls[i][0]
            processed_results.append(
                ToolResult(
                    tool_name=tool_name,
                    success=False,
                    error=str(result),
                )
            )
        else:
            processed_results.append(result)

    return processed_results


def format_tool_result_for_llm(result: ToolResult) -> str:
    """Formate un résultat de tool pour le LLM.

    Args:
        result: Résultat du tool

    Returns:
        Texte formaté
    """
    if not result.success:
        return f"[ERREUR] Tool {result.tool_name}: {result.error}"

    # Formater selon le type de tool
    tool_formatters = {
        "retrieve_gri_chunks": _format_retrieve_result,
        "lookup_gri_glossary": _format_glossary_result,
        "get_milestone_criteria": _format_milestone_result,
        "compare_approaches": _format_compare_result,
        "get_phase_summary": _format_phase_result,
    }

    formatter = tool_formatters.get(result.tool_name, _format_generic_result)
    return formatter(result.result)


def _format_retrieve_result(result: dict) -> str:
    """Formate un résultat retrieve_gri_chunks."""
    lines = [f"Résultats de recherche ({result.get('n_results', 0)} chunks):"]
    lines.append(f"Score max: {result.get('max_score', 0):.2f}")
    lines.append("")

    for i, chunk in enumerate(result.get("chunks", []), 1):
        lines.append(f"[{i}] Score: {chunk.get('score', 0):.2f}")
        if chunk.get("context_prefix"):
            lines.append(f"    {chunk['context_prefix']}")
        content = chunk.get("content", "")[:300]
        if len(chunk.get("content", "")) > 300:
            content += "..."
        lines.append(f"    {content}")
        lines.append("")

    return "\n".join(lines)


def _format_glossary_result(result: dict) -> str:
    """Formate un résultat lookup_gri_glossary."""
    if not result.get("found"):
        alternatives = result.get("alternatives", [])
        if alternatives:
            return (
                f"Terme '{result.get('term_searched')}' non trouvé. "
                f"Suggestions: {', '.join(alternatives)}"
            )
        return f"Terme '{result.get('term_searched')}' non trouvé dans le glossaire."

    defn = result.get("definition", {})
    lines = []

    term_fr = defn.get("term_fr", "")
    term_en = defn.get("term_en", "")
    if term_en:
        lines.append(f"**{term_fr}** ({term_en})")
    else:
        lines.append(f"**{term_fr}**")

    lines.append("")
    lines.append(defn.get("definition_fr", ""))

    if defn.get("definition_en"):
        lines.append("")
        lines.append(f"*EN: {defn['definition_en']}*")

    if defn.get("standard_ref"):
        lines.append("")
        lines.append(f"[{defn['standard_ref']}]")

    if result.get("citation"):
        lines.append("")
        lines.append(result["citation"])

    return "\n".join(lines)


def _format_milestone_result(result: dict) -> str:
    """Formate un résultat get_milestone_criteria."""
    if not result.get("found"):
        return result.get("content", "Jalon non trouvé.")

    lines = []

    milestone_id = result.get("milestone_id", "")
    milestone_name = result.get("milestone_name", "")
    lines.append(f"## Jalon {milestone_id} — {milestone_name}")
    lines.append(f"Cycle: {result.get('cycle', 'GRI')}")

    if result.get("gri_mapping_info"):
        lines.append(result["gri_mapping_info"])

    lines.append("")

    # Critères
    criteria = result.get("criteria", [])
    if criteria:
        lines.append("### Critères de passage")
        for c in criteria:
            lines.append(f"{c.get('number', '-')}. {c.get('text', '')}")
    else:
        lines.append("### Contenu")
        lines.append(result.get("content", ""))

    if result.get("citation"):
        lines.append("")
        lines.append(result["citation"])

    return "\n".join(lines)


def _format_compare_result(result: dict) -> str:
    """Formate un résultat compare_approaches."""
    if not result.get("has_sufficient_data"):
        return "Données insuffisantes pour la comparaison."

    return result.get("combined_context", "")


def _format_phase_result(result: dict) -> str:
    """Formate un résultat get_phase_summary."""
    if not result.get("found"):
        return result.get("content", "Phase non trouvée.")

    lines = []
    lines.append(f"## Phase {result.get('phase_num')} — {result.get('phase_name')}")
    lines.append(f"Cycle: {result.get('cycle', 'GRI')}")

    milestones = result.get("milestones", [])
    if milestones:
        lines.append(f"Jalons: {', '.join(milestones)}")

    lines.append("")

    # Objectifs
    objectives = result.get("objectives", [])
    if objectives:
        lines.append("### Objectifs")
        for o in objectives:
            prefix = "Général" if o.get("type") == "general" else "Spécifique"
            lines.append(f"- [{prefix}] {o.get('text', '')}")
        lines.append("")

    # Activités
    activities = result.get("activities", [])
    if activities:
        lines.append("### Activités")
        for a in activities:
            lines.append(f"- {a}")
        lines.append("")

    # Livrables
    deliverables = result.get("deliverables", [])
    if deliverables:
        lines.append("### Livrables")
        for d in deliverables:
            lines.append(f"- {d.get('name', '')}")

    return "\n".join(lines)


def _format_generic_result(result: dict) -> str:
    """Formate un résultat générique."""
    import json

    return json.dumps(result, ensure_ascii=False, indent=2)
