"""Query Router GRI - Classification des questions en 6 intents.

Le router est la pièce la plus critique du retrieval. Sans routing,
la qualité s'effondre car les mauvaises stratégies sont appliquées.

6 Intents GRI :
- DEFINITION    : demande de définition d'un terme ISO ou GRI
- PROCESSUS     : question sur un processus IS 15288
- JALON         : critères de passage d'un jalon (M0-M9 ou J1-J6)
- PHASE_COMPLETE: résumé ou objectifs d'une phase complète
- COMPARAISON   : comparaison entre deux éléments du GRI
- CIR           : question spécifique au Cycle d'Innovation Rapide

Usage:
    from src.agents.query_router import GRIQueryRouter, route_query

    router = GRIQueryRouter()
    result = await router.route("Quels sont les critères du CDR ?")
    # result.intent == "JALON", result.cycle == "GRI"
"""

import json
import re
from enum import StrEnum
from typing import Any, Literal

import structlog
from huggingface_hub import AsyncInferenceClient
from pydantic import BaseModel, Field

from src.core.config import settings

log = structlog.get_logger()


class GRIIntent(StrEnum):
    """Les 6 intents GRI pour le routing des queries."""

    DEFINITION = "DEFINITION"
    PROCESSUS = "PROCESSUS"
    JALON = "JALON"
    PHASE_COMPLETE = "PHASE_COMPLETE"
    COMPARAISON = "COMPARAISON"
    CIR = "CIR"


class GRICycle(StrEnum):
    """Cycle GRI standard ou CIR."""

    GRI = "GRI"
    CIR = "CIR"
    BOTH = "BOTH"


class RoutingStrategy(BaseModel):
    """Stratégie de retrieval pour un intent donné."""

    search_mode: Literal["dense", "sparse", "hybrid"] = "hybrid"
    primary_index: Literal["main", "glossary"] = "main"
    fallback_index: Literal["main", "glossary"] | None = None
    filters: dict[str, Any] | None = None
    fallback_filter: dict[str, Any] | None = None
    n_initial: int = 20
    n_final: int = 5
    use_reranker: bool = True
    use_parent: bool = False
    use_mmr: bool = False
    return_complete: bool = False
    multi_query: bool = False
    include_gri_mapping: bool = False
    temperature: float = 0.1
    alpha_rrf: float = 0.6


class RoutingResult(BaseModel):
    """Résultat du routing d'une query."""

    intent: GRIIntent
    cycle: GRICycle
    entities: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    strategy: RoutingStrategy
    raw_response: str | None = None


# Table de routing : stratégie par intent
ROUTING_TABLE: dict[GRIIntent, RoutingStrategy] = {
    GRIIntent.DEFINITION: RoutingStrategy(
        search_mode="sparse",  # BM25 suffit pour les termes exacts
        primary_index="glossary",  # Index glossaire dédié
        fallback_index="main",
        filters={"section_type": "definition"},
        n_initial=5,
        n_final=2,
        use_reranker=False,  # Pas besoin — exact match
        temperature=0.0,  # Définitions ISO : fidélité absolue
        alpha_rrf=0.3,  # Favoriser sparse pour exact match
    ),
    GRIIntent.PROCESSUS: RoutingStrategy(
        search_mode="hybrid",
        primary_index="main",
        filters={"section_type": "process"},
        n_initial=20,
        n_final=5,
        use_reranker=True,  # Critique pour les processus IS 15288
        use_parent=True,  # Remonter au processus parent si besoin
        temperature=0.1,
        alpha_rrf=0.6,
    ),
    GRIIntent.JALON: RoutingStrategy(
        search_mode="hybrid",
        primary_index="main",
        filters={"section_type": "milestone"},
        n_initial=5,
        n_final=1,  # Un seul jalon complet
        use_reranker=False,
        return_complete=True,  # TOUJOURS retourner le jalon entier
        temperature=0.0,
        alpha_rrf=0.5,
    ),
    GRIIntent.PHASE_COMPLETE: RoutingStrategy(
        search_mode="dense",
        primary_index="main",
        filters={"section_type": "phase"},
        n_initial=15,
        n_final=8,
        use_parent=True,  # Parent Document Retriever obligatoire
        use_reranker=True,
        use_mmr=True,  # Diversité sous-sections
        temperature=0.1,
        alpha_rrf=0.7,
    ),
    GRIIntent.COMPARAISON: RoutingStrategy(
        search_mode="hybrid",
        primary_index="main",
        filters=None,  # Multi-filter selon entities
        n_initial=30,  # Large pour couvrir les deux sujets
        n_final=10,
        use_reranker=True,
        use_mmr=True,
        multi_query=True,  # 1 query par entité à comparer
        temperature=0.1,
        alpha_rrf=0.6,
    ),
    GRIIntent.CIR: RoutingStrategy(
        search_mode="hybrid",
        primary_index="main",
        filters={"cycle": "CIR"},
        fallback_filter={"cycle": "GRI"},  # Fallback sur GRI pour mapping
        n_initial=20,
        n_final=5,
        use_reranker=True,
        include_gri_mapping=True,  # Ajouter le mapping J→M dans le contexte
        temperature=0.1,
        alpha_rrf=0.5,
    ),
}

# Prompt pour le LLM de routing
ROUTER_PROMPT = """Classifie cette question sur le GRI dans exactement un des 6 intents.

INTENTS :
- DEFINITION    : demande de définition d'un terme ISO ou GRI (ex: "Qu'est-ce qu'un artefact ?")
- PROCESSUS     : question sur un processus IS 15288 (ex: "Activités du processus de vérification ?")
- JALON         : critères de passage d'un jalon M0-M9 ou J1-J6 (ex: "Critères du CDR ?")
- PHASE_COMPLETE: résumé ou objectifs d'une phase complète 1-7 GRI ou 1-4 CIR (ex: "Objectifs de la Phase 3 ?")
- COMPARAISON   : comparaison entre deux éléments du GRI (ex: "Différence GRI vs CIR ?")
- CIR           : question spécifique au Cycle d'Innovation Rapide (ex: "Contexte d'application du CIR ?")

CYCLES :
- GRI  : cycle standard à 7 phases, jalons M0 à M9
- CIR  : Cycle d'Innovation Rapide à 4 phases, jalons J1 à J6
- BOTH : question qui concerne les deux cycles

Retourne uniquement un JSON valide (sans markdown) :
{{"intent": "...", "cycle": "GRI"|"CIR"|"BOTH", "entities": ["..."], "confidence": 0.0-1.0}}

Question : {query}"""


class GRIQueryRouter:
    """Router de queries GRI basé sur un LLM.

    Classifie les questions en 6 intents et détermine le cycle (GRI/CIR).
    Utilise des heuristiques en fallback si le LLM échoue.

    Attributes:
        model: Modèle HF pour le routing
        client: Client HF Inference API
    """

    def __init__(
        self,
        model: str | None = None,
        hf_api_key: str | None = None,
    ) -> None:
        """Initialise le router.

        Args:
            model: Modèle HF pour le routing (défaut: settings)
            hf_api_key: Clé API HF (défaut: settings)
        """
        self.model = model or settings.hf_router_model
        self._api_key = hf_api_key or settings.hf_api_key
        self._client: AsyncInferenceClient | None = None

        log.info("query_router.init", model=self.model)

    @property
    def client(self) -> AsyncInferenceClient:
        """Lazy loading du client HF."""
        if self._client is None:
            self._client = AsyncInferenceClient(token=self._api_key)
        return self._client

    async def route(self, query: str) -> RoutingResult:
        """Route une query vers le bon intent et cycle.

        Args:
            query: Question utilisateur

        Returns:
            RoutingResult avec intent, cycle, stratégie
        """
        log.info("query_router.routing", query=query[:80])

        # Essayer d'abord les heuristiques (rapide et gratuit)
        heuristic_result = self._heuristic_route(query)
        if heuristic_result.confidence >= 0.9:
            log.info(
                "query_router.heuristic_match",
                intent=heuristic_result.intent,
                confidence=heuristic_result.confidence,
            )
            return heuristic_result

        # Sinon, utiliser le LLM
        try:
            llm_result = await self._llm_route(query)
            log.info(
                "query_router.llm_match",
                intent=llm_result.intent,
                confidence=llm_result.confidence,
            )
            return llm_result
        except Exception as e:
            log.warning("query_router.llm_failed", error=str(e))
            # Fallback sur heuristiques
            return heuristic_result

    def _heuristic_route(self, query: str) -> RoutingResult:
        """Routing par heuristiques (patterns regex).

        Args:
            query: Question utilisateur

        Returns:
            RoutingResult basé sur les patterns
        """
        query_lower = query.lower()
        entities: list[str] = []

        # Patterns pour DEFINITION
        definition_patterns = [
            r"qu'est[- ]ce qu[e']",
            r"définir?\b",
            r"définition\b",
            r"signification\b",
            r"que signifie",
            r"c'est quoi\b",
        ]
        if any(re.search(p, query_lower) for p in definition_patterns):
            # Extraire le terme potentiel
            term_match = re.search(
                r"(?:qu'est-ce qu'(?:un|une)|définition de|définir|signification de)\s+([a-zéèàùâêîôûç\s]+)",
                query_lower,
            )
            if term_match:
                entities.append(term_match.group(1).strip())
            return RoutingResult(
                intent=GRIIntent.DEFINITION,
                cycle=GRICycle.GRI,
                entities=entities,
                confidence=0.9,
                strategy=ROUTING_TABLE[GRIIntent.DEFINITION],
            )

        # Patterns pour JALON
        jalon_patterns = [
            r"critères?\s+(?:du|de|d')\s*(?:passage\s+)?(?:du\s+)?(?:jalon\s+)?([mj]\d)",
            r"jalon\s+([mj]\d)",
            r"\b(cdr|pdr|srr|irr|trr|sar|orr|mnr)\b",
            r"\b([mj][0-9])\b",
        ]
        for pattern in jalon_patterns:
            match = re.search(pattern, query_lower)
            if match:
                milestone = match.group(1).upper()
                entities.append(milestone)
                # Déterminer le cycle
                cycle = GRICycle.CIR if milestone.startswith("J") else GRICycle.GRI
                return RoutingResult(
                    intent=GRIIntent.JALON,
                    cycle=cycle,
                    entities=entities,
                    confidence=0.95,
                    strategy=ROUTING_TABLE[GRIIntent.JALON],
                )

        # Patterns pour PROCESSUS
        processus_patterns = [
            r"processus\s+(?:de\s+)?([a-zéèàùâêîôûç\s]+)",
            r"activités?\s+(?:du|de)\s+processus",
            r"inputs?\s+(?:du|de)",
            r"outputs?\s+(?:du|de)",
            r"is\s*15288",
        ]
        if any(re.search(p, query_lower) for p in processus_patterns):
            return RoutingResult(
                intent=GRIIntent.PROCESSUS,
                cycle=GRICycle.GRI,
                entities=entities,
                confidence=0.85,
                strategy=ROUTING_TABLE[GRIIntent.PROCESSUS],
            )

        # Patterns pour PHASE_COMPLETE
        phase_patterns = [
            r"phase\s+(\d+)",
            r"objectifs?\s+(?:de\s+)?(?:la\s+)?phase",
            r"livrables?\s+(?:de\s+)?(?:la\s+)?phase",
            r"résumé?\s+(?:de\s+)?(?:la\s+)?phase",
        ]
        for pattern in phase_patterns:
            match = re.search(pattern, query_lower)
            if match:
                if match.groups():
                    phase_num = int(match.group(1))
                    entities.append(f"Phase {phase_num}")
                    cycle = (
                        GRICycle.CIR if phase_num <= 4 and "cir" in query_lower else GRICycle.GRI
                    )
                else:
                    cycle = GRICycle.GRI
                return RoutingResult(
                    intent=GRIIntent.PHASE_COMPLETE,
                    cycle=cycle,
                    entities=entities,
                    confidence=0.85,
                    strategy=ROUTING_TABLE[GRIIntent.PHASE_COMPLETE],
                )

        # Patterns pour COMPARAISON
        comparaison_patterns = [
            r"différence\s+entre",
            r"comparer?\b",
            r"\bvs\.?\b",
            r"versus\b",
            r"par\s+rapport\s+à",
        ]
        if any(re.search(p, query_lower) for p in comparaison_patterns):
            # Déterminer le cycle
            has_cir = "cir" in query_lower
            has_gri = "gri" in query_lower
            cycle = (
                GRICycle.BOTH
                if (has_cir and has_gri)
                else (GRICycle.CIR if has_cir else GRICycle.GRI)
            )
            return RoutingResult(
                intent=GRIIntent.COMPARAISON,
                cycle=cycle,
                entities=entities,
                confidence=0.85,
                strategy=ROUTING_TABLE[GRIIntent.COMPARAISON],
            )

        # Patterns pour CIR
        cir_patterns = [
            r"\bcir\b",
            r"cycle\s+d['']?innovation\s+rapide",
            r"innovation\s+rapide",
            r"\bj[1-6]\b",
        ]
        if any(re.search(p, query_lower) for p in cir_patterns):
            return RoutingResult(
                intent=GRIIntent.CIR,
                cycle=GRICycle.CIR,
                entities=entities,
                confidence=0.9,
                strategy=ROUTING_TABLE[GRIIntent.CIR],
            )

        # Défaut : PROCESSUS avec faible confiance
        return RoutingResult(
            intent=GRIIntent.PROCESSUS,
            cycle=GRICycle.GRI,
            entities=entities,
            confidence=0.5,
            strategy=ROUTING_TABLE[GRIIntent.PROCESSUS],
        )

    async def _llm_route(self, query: str) -> RoutingResult:
        """Routing par LLM.

        Args:
            query: Question utilisateur

        Returns:
            RoutingResult basé sur la réponse LLM
        """
        prompt = ROUTER_PROMPT.format(query=query)
        full_prompt = f"<s>[INST] {prompt} [/INST]"

        response = await self.client.text_generation(
            prompt=full_prompt,
            model=self.model,
            max_new_tokens=256,
            temperature=0.1,
            return_full_text=False,
        )

        # Parser la réponse JSON
        try:
            # Nettoyer la réponse (enlever markdown si présent)
            clean_response = response.strip()
            if clean_response.startswith("```"):
                clean_response = re.sub(r"```(?:json)?", "", clean_response).strip()

            data = json.loads(clean_response)

            intent = GRIIntent(data.get("intent", "PROCESSUS"))
            cycle = GRICycle(data.get("cycle", "GRI"))
            entities = data.get("entities", [])
            confidence = float(data.get("confidence", 0.8))

            return RoutingResult(
                intent=intent,
                cycle=cycle,
                entities=entities,
                confidence=confidence,
                strategy=ROUTING_TABLE[intent],
                raw_response=response,
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            log.warning("query_router.parse_failed", error=str(e), response=response)
            # Fallback heuristique
            return self._heuristic_route(query)


# Fonction helper pour usage simple
async def route_query(query: str) -> RoutingResult:
    """Route une query (fonction helper).

    Args:
        query: Question utilisateur

    Returns:
        RoutingResult
    """
    router = GRIQueryRouter()
    return await router.route(query)


def get_strategy_for_intent(intent: GRIIntent) -> RoutingStrategy:
    """Retourne la stratégie pour un intent donné.

    Args:
        intent: Intent GRI

    Returns:
        RoutingStrategy associée
    """
    return ROUTING_TABLE.get(intent, ROUTING_TABLE[GRIIntent.PROCESSUS])
