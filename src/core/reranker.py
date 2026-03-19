"""GRI Reranker - Cross-Encoder pour reranking de précision.

Utilise un cross-encoder pour reranker les résultats après retrieval initiale.
Améliore significativement la précision en comparant directement query et documents.

Usage:
    from src.core.reranker import GRIReranker, rerank_results

    reranker = GRIReranker()
    reranked = await reranker.rerank(query, chunks, top_k=5)
"""

import asyncio
from typing import TYPE_CHECKING, Any

import structlog
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from src.core.config import settings

if TYPE_CHECKING:
    from src.core.vector_store import SearchResult

log = structlog.get_logger()


class RerankedResult(BaseModel):
    """Résultat après reranking."""

    id: str
    original_score: float
    rerank_score: float
    content: str
    section_type: str | None = None
    cycle: str | None = None
    milestone_id: str | None = None
    phase_num: int | None = None
    context_prefix: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GRIReranker:
    """Cross-Encoder Reranker pour le GRI.

    Utilise ms-marco-MiniLM-L-6-v2 pour reranker les chunks après
    la retrieval initiale. Le cross-encoder compare directement
    (query, document) pour un score de pertinence plus précis.

    Attributes:
        MODEL: Modèle cross-encoder
        model: Instance du modèle chargé
    """

    MODEL = settings.reranker_model

    def __init__(self, model: str | None = None) -> None:
        """Initialise le reranker.

        Args:
            model: Modèle cross-encoder (défaut: settings)
        """
        self._model_name = model or self.MODEL
        self._model: CrossEncoder | None = None

        log.info("reranker.init", model=self._model_name)

    @property
    def model(self) -> CrossEncoder:
        """Lazy loading du modèle cross-encoder."""
        if self._model is None:
            log.info("reranker.loading_model", model=self._model_name)
            self._model = CrossEncoder(self._model_name)
        return self._model

    async def rerank(
        self,
        query: str,
        results: list["SearchResult"],
        top_k: int | None = None,
    ) -> list[RerankedResult]:
        """Reranke les résultats avec le cross-encoder.

        Args:
            query: Query utilisateur
            results: Résultats de la retrieval initiale
            top_k: Nombre de résultats à retourner (défaut: tous)

        Returns:
            Liste de RerankedResult triés par score
        """
        if not results:
            return []

        top_k = top_k or len(results)

        log.info(
            "reranker.reranking",
            query=query[:50],
            n_results=len(results),
            top_k=top_k,
        )

        # Préparer les paires (query, document)
        pairs = [(query, r.content) for r in results]

        # Scorer avec le cross-encoder (CPU-bound, exécuter dans un thread)
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(
            None,
            lambda: self.model.predict(pairs),
        )

        # Créer les résultats reranked
        reranked = [
            RerankedResult(
                id=r.id,
                original_score=r.score,
                rerank_score=float(scores[i]),
                content=r.content,
                section_type=r.section_type,
                cycle=r.cycle,
                milestone_id=r.milestone_id,
                phase_num=r.phase_num,
                context_prefix=r.context_prefix,
                metadata=r.metadata,
            )
            for i, r in enumerate(results)
        ]

        # Trier par score de reranking
        reranked.sort(key=lambda x: x.rerank_score, reverse=True)

        log.info(
            "reranker.done",
            top_score=reranked[0].rerank_score if reranked else 0,
            n_returned=min(top_k, len(reranked)),
        )

        return reranked[:top_k]

    async def rerank_with_mmr(
        self,
        query: str,
        results: list["SearchResult"],
        top_k: int = 5,
        lambda_mult: float = 0.7,
    ) -> list[RerankedResult]:
        """Reranke avec MMR (Maximal Marginal Relevance) pour la diversité.

        MMR équilibre pertinence et diversité pour éviter les documents
        trop similaires entre eux.

        Args:
            query: Query utilisateur
            results: Résultats de la retrieval initiale
            top_k: Nombre de résultats à retourner
            lambda_mult: Balance pertinence (1.0) vs diversité (0.0)

        Returns:
            Liste diversifiée de RerankedResult
        """
        if not results or len(results) <= top_k:
            return await self.rerank(query, results, top_k)

        log.info(
            "reranker.reranking_mmr",
            query=query[:50],
            n_results=len(results),
            top_k=top_k,
            lambda_mult=lambda_mult,
        )

        # D'abord reranker tous les résultats
        reranked = await self.rerank(query, results)

        if len(reranked) <= top_k:
            return reranked

        # Appliquer MMR
        selected: list[RerankedResult] = []
        candidates = list(reranked)

        # Sélectionner le premier (plus pertinent)
        selected.append(candidates.pop(0))

        while len(selected) < top_k and candidates:
            # Calculer le score MMR pour chaque candidat
            mmr_scores = []

            for cand in candidates:
                # Pertinence = score de reranking
                relevance = cand.rerank_score

                # Diversité = dissimilarité max avec les documents sélectionnés
                max_similarity = max(
                    self._content_similarity(cand.content, sel.content)
                    for sel in selected
                )

                # Score MMR
                mmr = lambda_mult * relevance - (1 - lambda_mult) * max_similarity
                mmr_scores.append((cand, mmr))

            # Sélectionner le candidat avec le meilleur score MMR
            mmr_scores.sort(key=lambda x: x[1], reverse=True)
            best_cand = mmr_scores[0][0]

            selected.append(best_cand)
            candidates.remove(best_cand)

        log.info("reranker.mmr_done", n_selected=len(selected))

        return selected

    def _content_similarity(self, content_a: str, content_b: str) -> float:
        """Calcule une similarité simple entre deux contenus.

        Utilise le coefficient de Jaccard sur les tokens.

        Args:
            content_a: Premier contenu
            content_b: Deuxième contenu

        Returns:
            Score de similarité [0, 1]
        """
        tokens_a = set(content_a.lower().split())
        tokens_b = set(content_b.lower().split())

        if not tokens_a or not tokens_b:
            return 0.0

        intersection = len(tokens_a & tokens_b)
        union = len(tokens_a | tokens_b)

        return intersection / union if union > 0 else 0.0


# Singleton pour usage global
_reranker: GRIReranker | None = None


def get_reranker() -> GRIReranker:
    """Retourne le singleton du reranker."""
    global _reranker
    if _reranker is None:
        _reranker = GRIReranker()
    return _reranker


async def rerank_results(
    query: str,
    results: list["SearchResult"],
    top_k: int | None = None,
    use_mmr: bool = False,
    lambda_mult: float = 0.7,
) -> list[RerankedResult]:
    """Fonction helper pour reranker des résultats.

    Args:
        query: Query utilisateur
        results: Résultats à reranker
        top_k: Nombre de résultats à retourner
        use_mmr: Utiliser MMR pour la diversité
        lambda_mult: Balance pertinence vs diversité (si MMR)

    Returns:
        Liste de RerankedResult
    """
    reranker = get_reranker()

    if use_mmr:
        return await reranker.rerank_with_mmr(
            query, results, top_k=top_k or 5, lambda_mult=lambda_mult
        )

    return await reranker.rerank(query, results, top_k=top_k)
