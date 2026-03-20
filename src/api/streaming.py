"""SSE Streaming pour les réponses RAG GRI.

Ce module implémente le streaming Server-Sent Events (SSE) pour
fournir des mises à jour en temps réel pendant le traitement
d'une question.

Events émis:
- routing: Intent et cycle détectés
- tool_call: Appel de tool en cours
- tool_result: Résultat d'un tool
- chunk: Partie de la réponse
- done: Réponse complète
- error: En cas d'erreur

Usage:
    from src.api.streaming import stream_query_response

    async for event in stream_query_response(query, store, memory):
        yield event
"""

import asyncio
import json
import re
import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any, cast

import structlog
from huggingface_hub import AsyncInferenceClient

from src.agents.query_router import GRIQueryRouter
from src.core.config import settings
from src.core.memory import GRIMemory
from src.core.session_store import SessionStore
from src.core.term_expander import GRITermExpander
from src.core.vector_store import GRIHybridStore
from src.generation.generator import GRIGenerator
from src.tools import TOOLS, format_tool_result_for_llm
from src.tools.executor import ToolExecutor, ToolResult

log = structlog.get_logger()


def format_sse_event(event_type: str, data: dict[str, Any]) -> str:
    """Formate un événement SSE.

    Args:
        event_type: Type d'événement
        data: Données de l'événement

    Returns:
        Chaîne formatée pour SSE
    """
    return json.dumps({"event": event_type, **data})


async def stream_query_response(
    query: str,
    store: GRIHybridStore,
    memory: GRIMemory,
    max_iter: int = 5,
    session_store: SessionStore | None = None,
    include_sources: bool = False,
    max_chunks: int = 5,
) -> AsyncGenerator[dict[str, str], None]:
    """Génère un flux SSE pour une question.

    Cette fonction exécute la boucle ReAct de manière incrémentale,
    émettant des événements SSE à chaque étape importante.

    Args:
        query: Question utilisateur
        store: Vector store GRI
        memory: Mémoire conversationnelle
        max_iter: Nombre maximum d'itérations
        include_sources: Si True, inclut les sources dans l'événement done
        max_chunks: Nombre maximum de chunks sources à retourner

    Yields:
        dict avec "event" et "data" pour EventSourceResponse
    """
    start_time = time.time()
    query_id = str(uuid.uuid4())

    log.info("streaming.start", query_id=query_id, query=query[:100])

    try:
        # === Étape 1: Routing ===
        router = GRIQueryRouter()
        routing = await router.route(query)

        yield {
            "event": "routing",
            "data": format_sse_event(
                "routing",
                {
                    "intent": routing.intent.value,
                    "cycle": routing.cycle.value,
                    "confidence": routing.confidence,
                },
            ),
        }

        # === Étape 2: Term Expansion ===
        term_expander = GRITermExpander(store)
        expansion = await term_expander.expand(query)
        term_context = expansion.term_context

        # === Étape 3: Construire le contexte ===
        system_prompt = _build_system_prompt(term_context, max_iter)
        conversation_context = memory.get_context()

        user_prompt = query
        if conversation_context:
            user_prompt = f"{conversation_context}\n\n---\nNouvelle question : {query}"

        # === Étape 4: Boucle ReAct avec streaming ===
        client = AsyncInferenceClient(token=settings.hf_api_key)
        model = settings.hf_orchestrator_model
        messages: list[dict[str, str]] = []
        tool_calls_history: list[dict[str, Any]] = []
        tool_executor = ToolExecutor(store)
        generator = GRIGenerator()  # Pour la génération finale
        collected_chunks: list[dict[str, Any]] = []  # Chunks pour GRIGenerator
        iterations = 0
        final_answer = ""

        for i in range(max_iter):
            iterations = i + 1

            # Construire le prompt
            full_prompt = _build_llm_prompt(system_prompt, user_prompt, messages)

            # Appel LLM avec gestion du bug StopIteration de huggingface_hub
            try:
                response = await _call_llm_with_fallback(
                    client=client,
                    prompt=full_prompt,
                    model=model,
                )
            except Exception as e:
                error_msg = str(e)
                if "StopIteration" in error_msg:
                    error_msg = (
                        "Erreur de configuration HuggingFace. "
                        f"Vérifiez que HF_API_KEY est configuré et que le modèle '{model}' est accessible."
                    )
                log.error("streaming.llm_error", error=error_msg)
                yield {
                    "event": "error",
                    "data": format_sse_event(
                        "error",
                        {
                            "error": "LLMError",
                            "message": error_msg,
                        },
                    ),
                }
                return

            # Parser les tool calls
            tool_calls = _parse_tool_calls(response)

            if not tool_calls:
                # Pas de tool call = réponse finale
                # Utiliser GRIGenerator si on a collecté des chunks (parité avec /query)
                if collected_chunks:
                    yield {
                        "event": "generating",
                        "data": format_sse_event(
                            "generating",
                            {
                                "status": "Génération de la réponse grounded...",
                                "n_chunks": len(collected_chunks),
                            },
                        ),
                    }

                    gen_result = await generator.generate(
                        query=query,
                        chunks=collected_chunks,
                        intent=routing.intent.value,
                    )
                    final_answer = gen_result.answer
                    log.info(
                        "streaming.generator_used",
                        response_type=gen_result.response_type.value,
                        n_citations=len(gen_result.citations),
                    )
                else:
                    # Fallback sur la réponse directe du LLM si pas de chunks
                    final_answer = response.strip()

                # Streamer la réponse par chunks
                text_chunks = _split_into_chunks(final_answer, chunk_size=100)
                for text_chunk in text_chunks:
                    yield {
                        "event": "chunk",
                        "data": format_sse_event("chunk", {"text": text_chunk}),
                    }
                    await asyncio.sleep(0.01)  # Petit délai pour effet streaming

                break

            # Exécuter les tools et collecter les résultats
            iteration_results = []
            for tc in tool_calls:
                # Émettre l'événement tool_call
                yield {
                    "event": "tool_call",
                    "data": format_sse_event(
                        "tool_call",
                        {
                            "tool": tc["name"],
                            "input": tc["input"],
                            "iteration": iterations,
                        },
                    ),
                }

                # Exécuter le tool UNE SEULE FOIS
                result = await tool_executor.execute(tc["name"], tc["input"])
                iteration_results.append(result)

                # Émettre l'événement tool_result
                yield {
                    "event": "tool_result",
                    "data": format_sse_event(
                        "tool_result",
                        {
                            "tool": tc["name"],
                            "success": result.success,
                            "n_results": (
                                len(result.result.get("chunks", []))
                                if isinstance(result.result, dict)
                                else 1
                            ),
                        },
                    ),
                }

                tool_calls_history.append(
                    {
                        "tool": tc["name"],
                        "input": tc["input"],
                        "iteration": iterations,
                        "success": result.success,
                    }
                )

                # Collecter les chunks pour GRIGenerator
                if result.success:
                    extracted = _extract_context_from_tool_result(tc["name"], result)
                    collected_chunks.extend(extracted)

            # Ajouter à l'historique de conversation
            messages.append({"role": "assistant", "content": response})

            # Formater les résultats des tools (réutiliser les résultats déjà exécutés)
            tool_results_text = _format_tool_results_from_cache(tool_calls, iteration_results)
            messages.append(
                {
                    "role": "user",
                    "content": f"Résultats des outils :\n\n{tool_results_text}",
                }
            )

        # Si max_iter atteint sans réponse
        if not final_answer:
            final_answer = (
                "Je n'ai pas pu trouver une réponse complète. "
                "Essayez de reformuler votre question."
            )
            yield {
                "event": "chunk",
                "data": format_sse_event("chunk", {"text": final_answer}),
            }

        # === Étape 5: Finalisation ===
        latency_ms = (time.time() - start_time) * 1000
        citations = _extract_citations(final_answer)

        # Sauvegarder dans la mémoire
        memory.add_turn(
            query=query,
            answer=final_answer,
            intent=routing.intent.value,
            cycle=routing.cycle.value,
            tool_calls=[tc["tool"] for tc in tool_calls_history],
            citations=citations,
        )

        if session_store is not None and memory.session_id:
            saved = await session_store.save_session(memory.session_id, memory)
            if not saved:
                log.warning(
                    "streaming.session_save_failed",
                    session_id=memory.session_id,
                )

        # Build sources if requested (parity with non-streaming /query)
        sources_data = None
        if include_sources and collected_chunks:
            sources_data = [
                {
                    "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
                    "content": chunk.get("content", "")[:500],
                    "section_type": chunk.get("section_type", "unknown"),
                    "score": chunk.get("score", 0.0),
                }
                for idx, chunk in enumerate(collected_chunks[:max_chunks])
            ]

        # Émettre l'événement done avec métadonnées complètes (parité avec /query)
        done_data = {
            "query_id": query_id,
            "answer": final_answer,
            "intent": routing.intent.value,
            "cycle": routing.cycle.value,
            "citations": citations,
            "tool_calls": tool_calls_history,
            "latency_ms": latency_ms,
            "iterations": iterations,
        }
        if sources_data is not None:
            done_data["sources"] = sources_data

        yield {
            "event": "done",
            "data": format_sse_event("done", done_data),
        }

        log.info(
            "streaming.done",
            query_id=query_id,
            iterations=iterations,
            latency_ms=f"{latency_ms:.0f}",
        )

    except Exception as e:
        log.error("streaming.error", error=str(e), exc_info=True)
        yield {
            "event": "error",
            "data": format_sse_event(
                "error",
                {
                    "error": e.__class__.__name__,
                    "message": str(e),
                },
            ),
        }


def _build_system_prompt(term_context: str, max_iter: int) -> str:
    """Construit le system prompt."""
    from src.agents.orchestrator import ORCHESTRATOR_SYSTEM

    # Description des tools
    tools_desc = _format_tools_description()

    prompt = ORCHESTRATOR_SYSTEM.format(
        max_iter=max_iter,
        tools_description=tools_desc,
    )

    if term_context:
        prompt += f"\n\n{term_context}"

    return prompt


def _format_tools_description() -> str:
    """Formate la description des tools."""
    lines = []

    for tool in TOOLS:
        lines.append(f"### {tool['name']}")
        lines.append(tool["description"][:500])
        lines.append("")

        schema = tool["input_schema"]
        props = schema.get("properties", {})
        required = schema.get("required", [])

        lines.append("Paramètres :")
        for param_name, param_def in props.items():
            req = "(requis)" if param_name in required else ""
            lines.append(f"- {param_name} {req}: {param_def.get('description', '')}")

        lines.append("")

    return "\n".join(lines)


async def _call_llm_with_fallback(
    client: AsyncInferenceClient,
    prompt: str,
    model: str,
) -> str:
    """Appelle le LLM via chat_completion (format OpenAI-compatible).

    Args:
        client: AsyncInferenceClient
        prompt: Prompt formaté
        model: Modèle HF

    Returns:
        Réponse du LLM
    """
    messages = [{"role": "user", "content": prompt}]

    try:
        response = await client.chat_completion(
            messages=messages,
            model=model,
            max_tokens=2048,
            temperature=0.1,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        log.error("streaming.llm_call_failed", error=str(e))
        raise


def _build_llm_prompt(
    system: str,
    user_query: str,
    messages: list[dict[str, str]],
) -> str:
    """Construit le prompt complet pour le LLM."""
    parts = [f"<s>[INST] {system}\n\nQuestion : {user_query} [/INST]"]

    for msg in messages:
        if msg["role"] == "assistant":
            parts.append(msg["content"])
        elif msg["role"] == "user":
            parts.append(f"[INST] {msg['content']} [/INST]")

    return "\n".join(parts)


def _parse_tool_calls(response: str) -> list[dict[str, Any]]:
    """Parse les appels de tools depuis la réponse."""
    cleaned = re.sub(r"```json\s*", "", response)
    cleaned = re.sub(r"\s*```", "", cleaned)

    start_idx = cleaned.find('"tool_calls"')
    if start_idx == -1:
        return []

    brace_idx = cleaned.rfind("{", 0, start_idx)
    if brace_idx == -1:
        return []

    depth = 0
    end_idx = brace_idx
    for i, char in enumerate(cleaned[brace_idx:], brace_idx):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break

    if depth != 0:
        return []

    try:
        import json

        json_str = cleaned[brace_idx:end_idx]
        data = cast(dict[str, Any], json.loads(json_str))
        return cast(list[dict[str, Any]], data.get("tool_calls", []))
    except (json.JSONDecodeError, KeyError):
        return []


def _format_tool_results_from_cache(
    tool_calls: list[dict[str, Any]],
    results: list[ToolResult],
) -> str:
    """Formate les résultats des tools pour le LLM.

    Args:
        tool_calls: Liste des appels de tools
        results: Liste des résultats (déjà exécutés) dans le même ordre

    Returns:
        Texte formaté pour le LLM
    """
    lines = []

    for tc, result in zip(tool_calls, results, strict=False):
        lines.append(f"## Résultat de {tc['name']}")
        lines.append("")
        lines.append(format_tool_result_for_llm(result))
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def _split_into_chunks(text: str, chunk_size: int = 100) -> list[str]:
    """Divise le texte en chunks pour le streaming."""
    chunks: list[str] = []
    words = text.split()
    current_chunk: list[str] = []
    current_length = 0

    for word in words:
        if current_length + len(word) + 1 > chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk) + " ")
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks if chunks else [text]


def _extract_citations(text: str) -> list[str]:
    """Extrait les citations du format [GRI/CIR > ...]."""
    pattern = r"\[(?:GRI|CIR)[^\]]+\]"
    matches = re.findall(pattern, text)
    return list(set(matches))


def _extract_context_from_tool_result(
    tool_name: str,
    result: Any,
) -> list[dict[str, Any]]:
    """Extrait le contexte utilisable par GRIGenerator depuis un résultat de tool.

    Args:
        tool_name: Nom du tool
        result: Résultat du tool (ToolResult)

    Returns:
        Liste de chunks formatés pour GRIGenerator
    """
    chunks: list[dict[str, Any]] = []

    # Récupérer le dict de résultat
    if hasattr(result, "result"):
        data = result.result
    elif isinstance(result, dict):
        data = result
    else:
        return chunks

    # Convertir Pydantic en dict si nécessaire
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    elif not isinstance(data, dict):
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
                    content_parts.append(
                        f"- {criterion.get('id', '')}: {criterion.get('text', '')}"
                    )
                else:
                    content_parts.append(f"- {criterion}")

        if content_parts:
            chunks.append(
                {
                    "content": "\n".join(content_parts),
                    "section_type": "milestone",
                    "milestone_id": data.get("milestone_id"),
                    "cycle": data.get("cycle"),
                    "score": 1.0,
                }
            )

    # get_phase_description : convertir en chunk
    elif tool_name == "get_phase_description" and data.get("found"):
        content_parts = []
        if data.get("phase_name"):
            content_parts.append(f"**Phase {data['phase_id']} - {data['phase_name']}**")
        if data.get("description"):
            content_parts.append(data["description"])
        if data.get("activities"):
            content_parts.append("\nActivités:")
            for activity in data["activities"]:
                content_parts.append(f"- {activity}")

        if content_parts:
            chunks.append(
                {
                    "content": "\n".join(content_parts),
                    "section_type": "phase",
                    "phase_id": data.get("phase_id"),
                    "cycle": data.get("cycle"),
                    "score": 1.0,
                }
            )

    # compare_gri_cir : convertir en chunks de comparaison
    elif tool_name == "compare_gri_cir" and data.get("found"):
        if data.get("gri_content"):
            chunks.append(
                {
                    "content": data["gri_content"],
                    "section_type": "comparison_gri",
                    "score": 1.0,
                }
            )
        if data.get("cir_content"):
            chunks.append(
                {
                    "content": data["cir_content"],
                    "section_type": "comparison_cir",
                    "score": 1.0,
                }
            )

    return chunks
