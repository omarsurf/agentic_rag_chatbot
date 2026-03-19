"""Core - Vector Store, Reranker, Memory, Term Expander.

Ce module contient les composants de base pour la couche Retrieval GRI :

- vector_store: GRIHybridStore avec Qdrant + BM25 et RRF Fusion
- reranker: Cross-encoder pour reranking de précision
- term_expander: Enrichissement pré-retrieval avec définitions GRI
- milestone_retriever: Retrieval spécialisée pour les jalons GRI/CIR
- memory: Mémoire conversationnelle pour l'orchestrateur
- config: Configuration centralisée avec Pydantic Settings
- logging: Logging structuré avec structlog
"""

from src.core.config import (
    CIR_GRI_MAPPING,
    CIR_PHASES,
    GRI_PHASES,
    VALID_CIR_MILESTONES,
    VALID_GRI_MILESTONES,
    VALID_MILESTONES,
    Settings,
    get_settings,
    settings,
)
from src.core.memory import (
    ConversationTurn,
    GRIMemory,
    MemoryStats,
    get_memory,
    reset_memory,
)
from src.core.milestone_retriever import (
    GRIMilestoneRetriever,
    MilestoneChunk,
    MilestoneResult,
    get_jalon_complet,
    get_milestone_retriever,
)
from src.core.reranker import (
    GRIReranker,
    RerankedResult,
    get_reranker,
    rerank_results,
)
from src.core.term_expander import (
    ExpansionResult,
    GRITermExpander,
    TermDefinition,
    detect_gri_terms,
    expand_query_with_terms,
)
from src.core.vector_store import (
    GRIHybridStore,
    SearchResult,
    get_vector_store,
)

__all__ = [
    # Config
    "Settings",
    "get_settings",
    "settings",
    # Constants
    "CIR_GRI_MAPPING",
    "VALID_GRI_MILESTONES",
    "VALID_CIR_MILESTONES",
    "VALID_MILESTONES",
    "GRI_PHASES",
    "CIR_PHASES",
    # Vector Store
    "GRIHybridStore",
    "SearchResult",
    "get_vector_store",
    # Reranker
    "GRIReranker",
    "RerankedResult",
    "get_reranker",
    "rerank_results",
    # Term Expander
    "GRITermExpander",
    "ExpansionResult",
    "TermDefinition",
    "expand_query_with_terms",
    "detect_gri_terms",
    # Milestone Retriever
    "GRIMilestoneRetriever",
    "MilestoneResult",
    "MilestoneChunk",
    "get_milestone_retriever",
    "get_jalon_complet",
    # Memory
    "GRIMemory",
    "ConversationTurn",
    "MemoryStats",
    "get_memory",
    "reset_memory",
]
