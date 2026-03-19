"""Agents GRI - Orchestrateur ReAct et Query Router.

Ce module contient les agents pour le système RAG GRI :

- query_router: Classification des queries en 6 intents GRI
- orchestrator: Boucle ReAct avec 5 tools spécialisés

Usage:
    from src.agents import GRIOrchestrator, GRIQueryRouter

    # Router seul
    router = GRIQueryRouter()
    routing = await router.route("Quels sont les critères du CDR ?")

    # Orchestrateur complet
    orchestrator = GRIOrchestrator(store, memory)
    result = await orchestrator.run("Quels sont les critères du CDR ?")
"""

from src.agents.orchestrator import (
    GRIOrchestrator,
    OrchestratorResult,
    ToolCall,
    run_query,
)
from src.agents.query_router import (
    GRICycle,
    GRIIntent,
    GRIQueryRouter,
    ROUTING_TABLE,
    RoutingResult,
    RoutingStrategy,
    get_strategy_for_intent,
    route_query,
)

__all__ = [
    # Orchestrator
    "GRIOrchestrator",
    "OrchestratorResult",
    "ToolCall",
    "run_query",
    # Query Router
    "GRIQueryRouter",
    "GRIIntent",
    "GRICycle",
    "RoutingResult",
    "RoutingStrategy",
    "ROUTING_TABLE",
    "route_query",
    "get_strategy_for_intent",
]
