"""GRI Orchestrator - Boucle ReAct avec 5 tools spécialisés.

L'orchestrateur est le cerveau de l'agent GRI. Il :
1. Route la query vers le bon intent
2. Enrichit avec le contexte terminologique
3. Exécute une boucle ReAct (Reasoning + Acting)
4. Valide et synthétise la réponse avec citations

Usage:
    from src.agents.orchestrator import GRIOrchestrator

    orchestrator = GRIOrchestrator(store, memory)
    result = await orchestrator.run("Quels sont les critères du CDR ?")
"""

import asyncio
import json
import re
import time
from typing import Any

import structlog
from huggingface_hub import AsyncInferenceClient
from pydantic import BaseModel, Field

from src.agents.query_router import GRIQueryRouter
from src.core.config import settings
from src.core.memory import GRIMemory
from src.core.term_expander import GRITermExpander
from src.core.vector_store import GRIHybridStore
from src.generation import GRIGenerator
from src.tools import TOOLS, format_tool_result_for_llm
from src.tools.executor import ToolResult

log = structlog.get_logger()


# System prompt de l'orchestrateur
ORCHESTRATOR_SYSTEM = """Tu es un expert en ingénierie système et gestion de l'innovation selon le GRI des FAR.
Sources autorisées : GRI/FAR (ISO/IEC/IEEE 15288:2023) et INCOSE Systems Engineering Handbook 5e éd.

## PROCESSUS DE RAISONNEMENT OBLIGATOIRE

### ÉTAPE 1 : IDENTIFICATION DU CYCLE
- La question concerne-t-elle le GRI standard (7 phases, jalons M0-M9) ?
- Ou le CIR - Cycle d'Innovation Rapide (4 phases, jalons J1-J6) ?
- Ou les deux (question comparative) ?

### ÉTAPE 2 : CLASSIFICATION DE L'INTENT
Identifie parmi : DEFINITION / PROCESSUS / JALON / PHASE_COMPLETE / COMPARAISON / CIR

### ÉTAPE 3 : PLANIFICATION DES OUTILS
- DEFINITION -> lookup_gri_glossary (toujours en premier pour les termes)
- PROCESSUS -> retrieve_gri_chunks avec filter section_type='process'
- JALON -> get_milestone_criteria (retourne la checklist complète)
- PHASE_COMPLETE -> get_phase_summary (retourne objectifs + livrables + jalons)
- COMPARAISON -> compare_approaches (multi-retrieve sur les deux entités)
- CIR -> retrieve_gri_chunks avec cycle='CIR' + get_milestone_criteria pour le mapping

### ÉTAPE 4 : VALIDATION DES RÉSULTATS
Après chaque tool call, évalue :
- Les informations récupérées répondent-elles à la question ?
- Y a-t-il des contradictions entre sources ?
- Manque-t-il une information critique (ex: critères de jalon incomplets) ?
Si insuffisant -> reformuler la query et recommencer (max 2 reformulations par tool)

### ÉTAPE 5 : SYNTHÈSE GROUNDED
- Utilise UNIQUEMENT les informations des chunks récupérés
- Citation obligatoire pour chaque affirmation : [GRI > Section > ...] ou [CIR > Phase N > ...]
- Pour les définitions ISO : reproduire exactement le libellé du GRI (temperature=0 mentale)
- Pour les jalons : lister TOUS les critères, jamais en résumer certains

## RÈGLES ABSOLUES
1. Ne jamais inventer de critères, de jalons, ou de livrables non présents dans les sources
2. Les définitions ISO/IEC/IEEE 15288:2023 sont normatives : ne pas paraphraser
3. Si GRI et CIR sont mentionnés dans la même question -> distinguer clairement les deux
4. Maximum {max_iter} itérations de retrieval par question
5. Si un terme GRI est utilisé sans définition connue -> toujours appeler lookup_gri_glossary d'abord

## FORMAT DE CITATION
[GRI > Terminologie > Terme]
[GRI > Principe N°X > Titre]
[GRI > Phase N > Titre Phase > Sous-section]
[GRI > Jalon MN (Nom) > Critère #X]
[CIR > Phase N > Jalon JN > Critère]
[GRI > Processus IS 15288 > Nom Processus > Activité/Input/Output]

## OUTILS DISPONIBLES

Pour appeler un outil, réponds avec un JSON dans ce format EXACT :
```json
{{"tool_calls": [{{"name": "tool_name", "input": {{"param1": "value1"}}}}]}}
```

{tools_description}

## INSTRUCTIONS FINALES
- Si tu as besoin d'informations, appelle les outils appropriés
- Quand tu as suffisamment d'informations, génère ta réponse finale SANS le format JSON
- Ta réponse finale doit inclure des citations au format [GRI > ...]
"""


class ToolCall(BaseModel):
    """Un appel de tool parsé."""

    name: str
    input: dict[str, Any] = Field(default_factory=dict)


class OrchestratorResult(BaseModel):
    """Résultat de l'orchestrateur."""

    answer: str
    intent: str
    cycle: str
    citations: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    iterations: int = 0
    latency_ms: float = 0.0
    warning: str | None = None
    collected_chunks: list[dict[str, Any]] = Field(default_factory=list)


class GRIOrchestrator:
    """Orchestrateur ReAct pour le GRI.

    Gère la boucle de raisonnement multi-étapes avec les 5 tools
    spécialisés pour répondre aux questions sur le GRI.

    Attributes:
        MAX_ITER: Nombre maximum d'itérations
        store: Vector store GRI
        memory: Mémoire conversationnelle
        router: Query router
        term_expander: Enrichissement terminologique
    """

    MAX_ITER = 5

    def __init__(
        self,
        store: GRIHybridStore,
        memory: GRIMemory | None = None,
        max_iter: int | None = None,
        model: str | None = None,
    ) -> None:
        """Initialise l'orchestrateur.

        Args:
            store: Vector store GRI
            memory: Mémoire conversationnelle (optionnel)
            max_iter: Nombre max d'itérations (défaut: MAX_ITER)
            model: Modèle HF pour l'orchestration
        """
        self.store = store
        self.memory = memory or GRIMemory()
        self.max_iter = max_iter or self.MAX_ITER
        self.model = model or settings.hf_orchestrator_model

        # Composants
        self.router = GRIQueryRouter()
        self.term_expander = GRITermExpander(store)
        self.generator = GRIGenerator(model=self.model)

        # Client HF (lazy)
        self._client: AsyncInferenceClient | None = None

        log.info(
            "orchestrator.init",
            model=self.model,
            max_iter=self.max_iter,
        )

    @property
    def client(self) -> AsyncInferenceClient:
        """Lazy loading du client HF."""
        if self._client is None:
            self._client = AsyncInferenceClient(token=settings.hf_api_key)
        return self._client

    async def run(self, query: str) -> OrchestratorResult:
        """Exécute la boucle ReAct pour répondre à une question.

        Args:
            query: Question utilisateur

        Returns:
            OrchestratorResult avec la réponse et les stats
        """
        start_time = time.time()

        log.info("orchestrator.run.start", query=query[:100])

        # Étape 1 : Routing
        routing = await self.router.route(query)
        log.info(
            "orchestrator.routing",
            intent=routing.intent.value,
            cycle=routing.cycle.value,
            confidence=routing.confidence,
        )

        # Étape 2 : Enrichissement terminologique
        expansion = await self.term_expander.expand(query)
        term_context = expansion.term_context

        # Étape 3 : Construire le contexte
        system_prompt = self._build_system_prompt(term_context)
        conversation_context = self.memory.get_context()

        # Ajouter le contexte conversationnel au prompt utilisateur si présent
        user_prompt = query
        if conversation_context:
            user_prompt = f"{conversation_context}\n\n---\nNouvelle question : {query}"

        # Étape 4 : Boucle ReAct
        messages: list[dict[str, str]] = []
        tool_calls_history: list[dict[str, Any]] = []
        collected_chunks: list[dict[str, Any]] = []  # Chunks collectés pour GRIGenerator
        iterations = 0
        final_answer = ""

        for i in range(self.max_iter):
            iterations = i + 1

            log.info("orchestrator.iteration", iteration=iterations)

            # Appel LLM
            full_prompt = self._build_llm_prompt(system_prompt, user_prompt, messages)

            try:
                response = await self._call_llm(full_prompt)
            except Exception as e:
                error_msg = str(e)
                # Améliorer le message d'erreur pour les problèmes HF courants
                if "StopIteration" in error_msg:
                    error_msg = (
                        "Erreur de configuration HuggingFace. "
                        "Vérifiez que HF_API_KEY est configuré et que le modèle "
                        f"'{self.model}' est accessible."
                    )
                elif "403" in error_msg or "Forbidden" in error_msg:
                    error_msg = (
                        "Accès refusé à l'API HuggingFace. "
                        "Vérifiez les permissions de votre token HF_API_KEY."
                    )
                log.error("orchestrator.llm_error", error=error_msg)
                final_answer = f"Erreur lors de la génération : {error_msg}"
                break

            # Parser la réponse
            tool_calls = self._parse_tool_calls(response)

            if not tool_calls:
                # Pas d'appel de tool = réponse finale
                # Utiliser GRIGenerator si on a collecté des chunks
                if collected_chunks:
                    gen_result = await self.generator.generate(
                        query=query,
                        chunks=collected_chunks,
                        intent=routing.intent.value,
                    )
                    final_answer = gen_result.answer
                    log.info(
                        "orchestrator.final_answer.generated",
                        length=len(final_answer),
                        response_type=gen_result.response_type.value,
                        n_citations=len(gen_result.citations),
                    )
                else:
                    # Fallback sur la réponse directe du LLM si pas de chunks
                    final_answer = response.strip()
                    log.info("orchestrator.final_answer.direct", length=len(final_answer))
                break

            # Exécuter les tools
            log.info(
                "orchestrator.executing_tools",
                tools=[tc.name for tc in tool_calls],
            )

            tool_results = await self._execute_tools(tool_calls)

            # Ajouter à l'historique et collecter les chunks
            for tc, result in zip(tool_calls, tool_results, strict=False):
                tool_calls_history.append(
                    {
                        "tool": tc.name,
                        "input": tc.input,
                        "iteration": iterations,
                        "success": result.success,
                    }
                )
                # Collecter les chunks/contexte pour GRIGenerator
                if result.success:
                    extracted = self._extract_context_from_tool_result(tc.name, result.result)
                    collected_chunks.extend(extracted)

            # Ajouter au contexte de conversation
            messages.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )

            # Formater les résultats des tools
            tool_results_text = self._format_tool_results(tool_calls, tool_results)
            messages.append(
                {
                    "role": "user",
                    "content": f"Résultats des outils :\n\n{tool_results_text}",
                }
            )

        # Si on atteint max_iter sans réponse finale
        if not final_answer:
            final_answer = (
                "Je n'ai pas pu trouver une réponse complète dans les sources GRI disponibles. "
                "Essayez de reformuler votre question ou de la préciser."
            )
            log.warning("orchestrator.max_iter_reached")

        # Extraire les citations
        citations = self._extract_citations(final_answer)

        # Sauvegarder dans la mémoire
        self.memory.add_turn(
            query=query,
            answer=final_answer,
            intent=routing.intent.value,
            cycle=routing.cycle.value,
            tool_calls=[tc["tool"] for tc in tool_calls_history],
            citations=citations,
        )

        latency = (time.time() - start_time) * 1000

        log.info(
            "orchestrator.run.done",
            iterations=iterations,
            n_tool_calls=len(tool_calls_history),
            latency_ms=f"{latency:.0f}",
        )

        return OrchestratorResult(
            answer=final_answer,
            intent=routing.intent.value,
            cycle=routing.cycle.value,
            citations=citations,
            tool_calls=tool_calls_history,
            iterations=iterations,
            latency_ms=latency,
            warning=(
                "max_iterations_reached"
                if iterations >= self.max_iter and not final_answer
                else None
            ),
            collected_chunks=collected_chunks,
        )

    def _build_system_prompt(self, term_context: str) -> str:
        """Construit le system prompt avec le contexte terminologique.

        Args:
            term_context: Contexte terminologique enrichi

        Returns:
            System prompt complet
        """
        # Description des tools
        tools_desc = self._format_tools_description()

        prompt = ORCHESTRATOR_SYSTEM.format(
            max_iter=self.max_iter,
            tools_description=tools_desc,
        )

        if term_context:
            prompt += f"\n\n{term_context}"

        return prompt

    def _format_tools_description(self) -> str:
        """Formate la description des tools pour le prompt."""
        lines = []

        for tool in TOOLS:
            lines.append(f"### {tool['name']}")
            lines.append(tool["description"][:500])  # Tronquer si trop long
            lines.append("")

            # Paramètres
            schema = tool["input_schema"]
            props = schema.get("properties", {})
            required = schema.get("required", [])

            lines.append("Paramètres :")
            for param_name, param_def in props.items():
                req = "(requis)" if param_name in required else ""
                lines.append(f"- {param_name} {req}: {param_def.get('description', '')}")

            lines.append("")

        return "\n".join(lines)

    async def _call_llm(self, prompt: str) -> str:
        """Appelle le LLM via chat_completion (format OpenAI-compatible).

        Args:
            prompt: Prompt formaté (sera converti en message chat)

        Returns:
            Réponse du LLM

        Raises:
            Exception: Si l'appel échoue
        """
        # Utiliser chat_completion qui est plus stable avec huggingface_hub récent
        messages = [{"role": "user", "content": prompt}]

        try:
            response = await self.client.chat_completion(
                messages=messages,
                model=self.model,
                max_tokens=2048,
                temperature=0.1,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            log.error("orchestrator.llm_call_failed", error=str(e))
            raise

    def _build_llm_prompt(
        self,
        system: str,
        user_query: str,
        messages: list[dict[str, str]],
    ) -> str:
        """Construit le prompt complet pour le LLM.

        Args:
            system: System prompt
            user_query: Question utilisateur
            messages: Historique des messages

        Returns:
            Prompt formaté pour le modèle HF
        """
        # Format Mistral/Mixtral
        parts = [f"<s>[INST] {system}\n\nQuestion : {user_query} [/INST]"]

        for msg in messages:
            if msg["role"] == "assistant":
                parts.append(msg["content"])
            elif msg["role"] == "user":
                parts.append(f"[INST] {msg['content']} [/INST]")

        return "\n".join(parts)

    def _parse_tool_calls(self, response: str) -> list[ToolCall]:
        """Parse les appels de tools depuis la réponse LLM.

        Plusieurs stratégies de parsing sont tentées dans l'ordre :
        1. Format {"tool_calls": [...]} standard
        2. Format avec blocs ```json ... ```
        3. Format YAML-like (name: ..., input: ...)
        4. Extraction de JSON individuel {"name": ..., "input": ...}

        Args:
            response: Réponse du LLM

        Returns:
            Liste de ToolCall validés
        """
        # Stratégie 1 : Format standard avec tool_calls array
        result = self._parse_standard_format(response)
        if result:
            return self._validate_tool_calls(result)

        # Stratégie 2 : Blocs ```json ... ```
        result = self._parse_json_blocks(response)
        if result:
            return self._validate_tool_calls(result)

        # Stratégie 3 : JSON individuel inline
        result = self._parse_individual_json(response)
        if result:
            return self._validate_tool_calls(result)

        return []

    def _parse_standard_format(self, response: str) -> list[ToolCall]:
        """Parse le format {"tool_calls": [...]}."""
        # Nettoyer les backticks markdown
        cleaned = re.sub(r"```json\s*", "", response)
        cleaned = re.sub(r"\s*```", "", cleaned)

        # Chercher "tool_calls"
        start_idx = cleaned.find('"tool_calls"')
        if start_idx == -1:
            return []

        # Trouver l'accolade ouvrante
        brace_idx = cleaned.rfind("{", 0, start_idx)
        if brace_idx == -1:
            return []

        # Extraire le bloc JSON avec matching des accolades
        json_str = self._extract_json_block(cleaned, brace_idx)
        if not json_str:
            return []

        try:
            data = json.loads(json_str)
            return [
                ToolCall(name=tc.get("name", ""), input=tc.get("input", {}))
                for tc in data.get("tool_calls", [])
            ]
        except (json.JSONDecodeError, KeyError):
            return []

    def _parse_json_blocks(self, response: str) -> list[ToolCall]:
        """Parse les blocs ```json ... ```."""
        tool_calls = []

        # Trouver tous les blocs JSON
        pattern = r"```(?:json)?\s*(\{[^`]+\})\s*```"
        matches = re.findall(pattern, response, re.DOTALL)

        for match in matches:
            try:
                data = json.loads(match)

                # Format {"tool_calls": [...]}
                if "tool_calls" in data:
                    for tc in data["tool_calls"]:
                        tool_calls.append(
                            ToolCall(name=tc.get("name", ""), input=tc.get("input", {}))
                        )
                # Format direct {"name": ..., "input": ...}
                elif "name" in data and "input" in data:
                    tool_calls.append(ToolCall(name=data["name"], input=data["input"]))
            except json.JSONDecodeError:
                continue

        return tool_calls

    def _parse_individual_json(self, response: str) -> list[ToolCall]:
        """Parse les objets JSON individuels {"name": ..., "input": ...}."""
        tool_calls = []

        # Pattern pour trouver des objets JSON avec name et input
        pattern = r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"input"\s*:\s*(\{[^}]+\})\s*\}'
        matches = re.findall(pattern, response)

        for name, input_str in matches:
            try:
                input_data = json.loads(input_str)
                tool_calls.append(ToolCall(name=name, input=input_data))
            except json.JSONDecodeError:
                continue

        return tool_calls

    def _extract_json_block(self, text: str, start_idx: int) -> str | None:
        """Extrait un bloc JSON en matchant les accolades."""
        depth = 0
        end_idx = start_idx

        for i, char in enumerate(text[start_idx:], start_idx):
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break

        if depth != 0:
            return None

        return text[start_idx:end_idx]

    def _validate_tool_calls(self, tool_calls: list[ToolCall]) -> list[ToolCall]:
        """Valide les tool calls selon les définitions de tools disponibles."""
        from src.tools import TOOLS

        valid_tools = {t["name"] for t in TOOLS}
        validated = []

        for tc in tool_calls:
            if not tc.name:
                log.warning("orchestrator.tool_call_missing_name")
                continue

            if tc.name not in valid_tools:
                log.warning(
                    "orchestrator.tool_call_unknown_tool",
                    tool_name=tc.name,
                    available_tools=list(valid_tools),
                )
                continue

            # Valider les paramètres requis
            tool_def = next((t for t in TOOLS if t["name"] == tc.name), None)
            if tool_def:
                required = tool_def.get("input_schema", {}).get("required", [])
                missing = [r for r in required if r not in tc.input]
                if missing:
                    log.warning(
                        "orchestrator.tool_call_missing_params",
                        tool_name=tc.name,
                        missing_params=missing,
                    )
                    # On inclut quand même le tool call, le tool gèrera l'erreur
                    pass

            validated.append(tc)

        return validated

    async def _execute_tools(
        self,
        tool_calls: list[ToolCall],
    ) -> list[ToolResult]:
        """Exécute les tools en parallèle.

        Args:
            tool_calls: Liste des tools à exécuter

        Returns:
            Liste de ToolResult
        """
        from src.tools.executor import ToolResult as TR

        tasks = [self._execute_single_tool(tc) for tc in tool_calls]
        results: list[ToolResult | BaseException] = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        # Convertir les exceptions
        processed: list[ToolResult] = []
        for i, result in enumerate(results):
            if isinstance(result, BaseException):
                processed.append(
                    TR(
                        tool_name=tool_calls[i].name,
                        success=False,
                        error=str(result),
                    )
                )
            else:
                processed.append(result)

        return processed

    async def _execute_single_tool(self, tool_call: ToolCall) -> ToolResult:
        """Exécute un seul tool.

        Args:
            tool_call: Tool à exécuter

        Returns:
            ToolResult
        """
        from src.tools.executor import ToolExecutor

        executor = ToolExecutor(self.store)
        return await executor.execute(tool_call.name, tool_call.input)

    def _format_tool_results(
        self,
        tool_calls: list[ToolCall],
        results: list[ToolResult],
    ) -> str:
        """Formate les résultats des tools pour le LLM.

        Args:
            tool_calls: Tools appelés
            results: Résultats correspondants

        Returns:
            Texte formaté
        """
        lines = []

        for tc, result in zip(tool_calls, results, strict=False):
            lines.append(f"## Résultat de {tc.name}")
            lines.append("")
            lines.append(format_tool_result_for_llm(result))
            lines.append("")
            lines.append("---")
            lines.append("")

        return "\n".join(lines)

    def _extract_context_from_tool_result(
        self,
        tool_name: str,
        result: Any,
    ) -> list[dict[str, Any]]:
        """Extrait le contexte utilisable par GRIGenerator depuis un résultat de tool.

        Args:
            tool_name: Nom du tool
            result: Résultat du tool (Pydantic model ou dict)

        Returns:
            Liste de chunks formatés pour GRIGenerator
        """
        chunks: list[dict[str, Any]] = []

        # Convertir Pydantic en dict si nécessaire
        if hasattr(result, "model_dump"):
            data = result.model_dump()
        elif isinstance(result, dict):
            data = result
        else:
            return chunks

        # retrieve_gri_chunks : extraire les chunks directement
        if tool_name == "retrieve_gri_chunks" and "chunks" in data:
            for chunk in data["chunks"]:
                if isinstance(chunk, dict):
                    chunks.append(chunk)
                elif hasattr(chunk, "model_dump"):
                    chunks.append(chunk.model_dump())

        # lookup_gri_glossary : convertir la définition en chunk
        elif tool_name == "lookup_gri_glossary" and data.get("found"):
            definition = data.get("definition", {})
            if definition:
                chunks.append(
                    {
                        "content": (
                            f"**{definition.get('term_fr', '')}** "
                            f"({definition.get('term_en', '')}): "
                            f"{definition.get('definition_fr', '')}"
                        ),
                        "section_type": "definition",
                        "context_prefix": definition.get("context_prefix"),
                        "score": 1.0,
                    }
                )

        # get_milestone_criteria : convertir le contenu en chunk
        elif tool_name == "get_milestone_criteria" and data.get("found"):
            content_parts = []
            if data.get("milestone_name"):
                content_parts.append(f"**{data['milestone_id']} - {data['milestone_name']}**")
            if data.get("content"):
                content_parts.append(data["content"])
            if data.get("criteria"):
                for criterion in data["criteria"]:
                    if isinstance(criterion, dict):
                        content_parts.append(f"- {criterion.get('text', '')}")

            if content_parts:
                chunks.append(
                    {
                        "content": "\n".join(content_parts),
                        "section_type": "milestone",
                        "milestone_id": data.get("milestone_id"),
                        "cycle": data.get("cycle", "GRI"),
                        "score": 1.0,
                    }
                )

        # get_phase_summary : convertir en chunk
        elif tool_name == "get_phase_summary" and data.get("found"):
            chunks.append(
                {
                    "content": data.get("content", ""),
                    "section_type": "phase",
                    "phase_num": data.get("phase_num"),
                    "score": 1.0,
                }
            )

        # compare_approaches : convertir en chunk
        elif tool_name == "compare_approaches" and data.get("comparison_text"):
            chunks.append(
                {
                    "content": data["comparison_text"],
                    "section_type": "comparison",
                    "score": 1.0,
                }
            )

        return chunks

    def _extract_citations(self, text: str) -> list[str]:
        """Extrait les citations du format [GRI/CIR > ...].

        Args:
            text: Texte à analyser

        Returns:
            Liste de citations uniques
        """
        pattern = r"\[(?:GRI|CIR)[^\]]+\]"
        matches = re.findall(pattern, text)
        return list(set(matches))


# Fonction helper pour usage simple
async def run_query(
    query: str,
    store: GRIHybridStore,
    memory: GRIMemory | None = None,
) -> OrchestratorResult:
    """Exécute une query avec l'orchestrateur.

    Args:
        query: Question utilisateur
        store: Vector store GRI
        memory: Mémoire conversationnelle (optionnel)

    Returns:
        OrchestratorResult
    """
    orchestrator = GRIOrchestrator(store, memory)
    return await orchestrator.run(query)


def main() -> None:
    """CLI entry point pour l'agent GRI interactif."""
    import argparse

    parser = argparse.ArgumentParser(
        description="GRI Agent - Assistant IA pour le Guide de Référence d'Ingénierie"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Mode interactif (boucle de questions)"
    )
    parser.add_argument("--query", "-q", type=str, help="Question unique à poser")
    args = parser.parse_args()

    # Initialiser les composants
    store = GRIHybridStore()
    memory = GRIMemory()
    orchestrator = GRIOrchestrator(store, memory)

    if args.interactive:
        print("GRI Agent - Mode Interactif")
        print("Tapez 'exit' ou 'quit' pour quitter\n")
        while True:
            try:
                query = input("Question: ").strip()
                if query.lower() in ("exit", "quit", "q"):
                    print("Au revoir!")
                    break
                if not query:
                    continue
                result = asyncio.run(orchestrator.run(query))
                print(f"\nRéponse:\n{result.answer}\n")
                if result.citations:
                    print(f"Sources: {', '.join(result.citations)}\n")
            except KeyboardInterrupt:
                print("\nAu revoir!")
                break
            except Exception as e:
                print(f"Erreur: {e}\n")
    elif args.query:
        result = asyncio.run(orchestrator.run(args.query))
        print(result.answer)
        if result.citations:
            print(f"\nSources: {', '.join(result.citations)}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
