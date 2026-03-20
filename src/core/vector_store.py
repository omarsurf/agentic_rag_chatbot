"""GRI Hybrid Vector Store - Qdrant + BM25 avec RRF Fusion.

Double index pour le GRI :
- Dense : Qdrant avec embeddings multilingues FR+EN
- Sparse : BM25 pour exact match des termes ISO

Usage:
    from src.core.vector_store import GRIHybridStore

    store = GRIHybridStore()
    results = await store.hybrid_search("critères du CDR", n_results=5)
    definition = await store.glossary_lookup("artefact")
"""

import asyncio
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, TypedDict, cast

import structlog
from pydantic import BaseModel, Field
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchText,
    MatchValue,
    PointStruct,
    VectorParams,
)
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.core.config import settings

log = structlog.get_logger()


class CorpusDoc(TypedDict):
    """Document brut utilisé pour l'index BM25."""

    id: str | int
    payload: dict[str, Any]


class SparseHit(TypedDict):
    """Résultat sparse interne."""

    id: str | int
    score: float
    payload: dict[str, Any]


class SearchResult(BaseModel):
    """Résultat de recherche hybride."""

    id: str
    score: float
    content: str
    section_type: str | None = None
    cycle: str | None = None
    milestone_id: str | None = None
    phase_num: int | None = None
    context_prefix: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GRIHybridStore:
    """Store hybride Qdrant + BM25 avec RRF Fusion pour le GRI.

    Deux collections Qdrant séparées :
    - "gri_main"     : tous les chunks sauf définitions
    - "gri_glossary" : 200+ définitions ISO bilingues (lookup rapide)

    Alpha RRF = 0.6 dense / 0.4 sparse
    → On favorise légèrement le dense pour la sémantique,
      mais le sparse est critique pour les termes ISO exacts.

    Attributes:
        EMBED_MODEL: Modèle d'embeddings multilingue FR+EN
        COLLECTIONS: Mapping des noms de collections
        RRF_ALPHA: Poids du dense dans la fusion RRF
        RRF_K: Constante RRF standard (Cormack 2009)
    """

    EMBED_MODEL = settings.embedding_model
    COLLECTIONS = {
        "main": settings.qdrant_collection_main,
        "glossary": settings.qdrant_collection_glossary,
    }
    RRF_ALPHA = settings.rrf_alpha
    RRF_K = settings.rrf_k

    def __init__(
        self,
        qdrant_url: str | None = None,
        qdrant_port: int | None = None,
        qdrant_api_key: str | None = None,
        use_async: bool = True,
    ) -> None:
        """Initialise le store hybride.

        Args:
            qdrant_url: URL du serveur Qdrant (défaut: settings)
            qdrant_port: Port Qdrant (défaut: settings)
            qdrant_api_key: Clé API optionnelle
            use_async: Utiliser le client async (défaut: True)
        """
        self._qdrant_url = qdrant_url or settings.qdrant_url
        self._qdrant_port = qdrant_port or settings.qdrant_port
        self._qdrant_api_key = qdrant_api_key or settings.qdrant_api_key
        self._use_async = use_async

        api_key = self._qdrant_api_key if self._qdrant_api_key else None
        self._local_qdrant = self._qdrant_url in {":memory:", "memory"}

        if self._local_qdrant:
            qdrant_target = ":memory:"
            # Le backend local de Qdrant n'est pas partagé entre clients sync/async.
            # On force donc un client sync unique pour garder un état cohérent.
            self._async_client = None
            self.client = QdrantClient(location=qdrant_target)
        else:
            # Utiliser URL HTTP explicite pour éviter HTTPS par défaut.
            if self._qdrant_url.startswith(("http://", "https://")):
                qdrant_target = self._qdrant_url
            else:
                qdrant_target = f"http://{self._qdrant_url}:{self._qdrant_port}"

            # Client async pour les opérations non-bloquantes
            if self._use_async:
                self._async_client = AsyncQdrantClient(url=qdrant_target, api_key=api_key)
            else:
                self._async_client = None

            # Client sync pour BM25 corpus loading et opérations de fallback
            self.client = QdrantClient(url=qdrant_target, api_key=api_key)

        # Modèle d'embeddings (chargé à la première utilisation)
        self._embed_model: SentenceTransformer | None = None

        # Index BM25 par collection (chargé à la demande)
        self._bm25: dict[str, BM25Okapi] = {}
        self._corpus: dict[str, list[CorpusDoc]] = {}
        self._corpus_loaded: dict[str, bool] = {"main": False, "glossary": False}
        self._bm25_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="bm25",
        )

        log.info(
            "vector_store.init",
            qdrant_url=qdrant_target,
            qdrant_port=self._qdrant_port,
            embed_model=self.EMBED_MODEL,
            async_enabled=self._async_client is not None,
            local_qdrant=self._local_qdrant,
        )

    @property
    def embed_model(self) -> SentenceTransformer:
        """Lazy loading du modèle d'embeddings."""
        if self._embed_model is None:
            log.info("vector_store.loading_embed_model", model=self.EMBED_MODEL)
            self._embed_model = SentenceTransformer(self.EMBED_MODEL)
        return self._embed_model

    async def ensure_collections(self) -> None:
        """Crée les collections Qdrant si elles n'existent pas."""
        for _name, collection_name in self.COLLECTIONS.items():
            try:
                if self._async_client:
                    await self._async_client.get_collection(collection_name)
                else:
                    self.client.get_collection(collection_name)
                log.info("vector_store.collection_exists", collection=collection_name)
            except Exception:
                if self._async_client:
                    await self._async_client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=settings.embedding_dimension,
                            distance=Distance.COSINE,
                        ),
                    )
                else:
                    self.client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=settings.embedding_dimension,
                            distance=Distance.COSINE,
                        ),
                    )
                log.info("vector_store.collection_created", collection=collection_name)

    async def index_chunks(
        self,
        chunks: list[dict[str, Any]],
        collection: Literal["main", "glossary"] = "main",
        batch_size: int = 100,
    ) -> dict[str, int | str]:
        """Indexe des chunks dans Qdrant et BM25.

        Args:
            chunks: Liste de chunks avec 'content' et 'metadata'
            collection: Collection cible
            batch_size: Taille des batches pour l'indexation

        Returns:
            Stats d'indexation
        """
        collection_name = self.COLLECTIONS[collection]
        total = len(chunks)
        indexed = 0

        log.info(
            "vector_store.indexing_start",
            collection=collection_name,
            total_chunks=total,
        )

        # Préparer les points pour Qdrant
        for i in range(0, total, batch_size):
            batch = chunks[i : i + batch_size]
            points = []

            for _idx, chunk in enumerate(batch):
                content = chunk.get("content", "")
                metadata = chunk.get("metadata", {})

                # Générer l'ID si absent
                chunk_id = (
                    chunk.get("chunk_id") or hashlib.sha256(content.encode()).hexdigest()[:16]
                )

                # Générer un UUID à partir du chunk_id (Qdrant requiert UUID ou int)
                import uuid

                point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))

                # Générer l'embedding
                vector = self.embed_model.encode(content).tolist()

                # Payload complet
                payload = {
                    "content": content,
                    "chunk_id": chunk_id,
                    **metadata,
                }

                points.append(
                    PointStruct(
                        id=point_uuid,
                        vector=vector,
                        payload=payload,
                    )
                )

            # Upsert dans Qdrant (async si disponible)
            if self._async_client:
                await self._async_client.upsert(collection_name=collection_name, points=points)
            else:
                self.client.upsert(collection_name=collection_name, points=points)
            indexed += len(batch)

            log.info(
                "vector_store.batch_indexed",
                collection=collection_name,
                indexed=indexed,
                total=total,
            )

        # Invalider le cache BM25 pour cette collection
        self._corpus_loaded[collection] = False

        return {"total": total, "indexed": indexed, "collection": collection_name}

    def _load_bm25_corpus(self, collection: str) -> None:
        """Charge le corpus BM25 depuis Qdrant."""
        if self._corpus_loaded.get(collection, False):
            return

        collection_name = self.COLLECTIONS[collection]
        log.info("vector_store.loading_bm25_corpus", collection=collection_name)

        # Scroll tous les documents
        all_docs: list[CorpusDoc] = []
        offset = None

        while True:
            results, offset = self.client.scroll(
                collection_name=collection_name,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            if not results:
                break

            for point in results:
                payload = cast(dict[str, Any], point.payload or {})
                all_docs.append(
                    {
                        "id": cast(str | int, point.id),
                        "payload": payload,
                    }
                )

            if offset is None:
                break

        # Construire l'index BM25
        if all_docs:
            tokenized_corpus = [
                self._tokenize(doc["payload"].get("content", "")) for doc in all_docs
            ]
            self._bm25[collection] = BM25Okapi(tokenized_corpus)
            self._corpus[collection] = all_docs
            self._corpus_loaded[collection] = True

            log.info(
                "vector_store.bm25_loaded",
                collection=collection_name,
                n_docs=len(all_docs),
            )

    def _tokenize(self, text: str) -> list[str]:
        """Tokenisation simple pour BM25."""
        import re

        # Lowercase + split sur les non-alphanumériques
        text = text.lower()
        tokens = re.split(r"[^a-zàâäéèêëïîôùûüç0-9]+", text)
        return [t for t in tokens if len(t) > 1]

    def _bm25_search(
        self,
        query: str,
        collection: str,
        n_results: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SparseHit]:
        """Recherche BM25 sparse.

        Args:
            query: Query de recherche
            collection: Collection cible
            n_results: Nombre de résultats
            filters: Filtres metadata optionnels

        Returns:
            Liste de résultats avec scores
        """
        self._load_bm25_corpus(collection)

        if collection not in self._bm25:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25[collection].get_scores(tokenized_query)

        # Appliquer les filtres si présents
        corpus = self._corpus[collection]
        filtered_indices: list[int] = []

        for idx, doc in enumerate(corpus):
            if filters:
                payload = doc["payload"]
                match = all(payload.get(k) == v for k, v in filters.items() if v is not None)
                if not match:
                    continue
            filtered_indices.append(idx)

        # Trier par score
        scored = [(idx, scores[idx]) for idx in filtered_indices]
        scored.sort(key=lambda x: x[1], reverse=True)

        results: list[SparseHit] = []
        for idx, score in scored[:n_results]:
            doc = corpus[idx]
            results.append(
                {
                    "id": doc["id"],
                    "score": float(score),
                    "payload": doc["payload"],
                }
            )

        return results

    async def _bm25_search_async(
        self,
        query: str,
        collection: str,
        n_results: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SparseHit]:
        """Exécute la recherche BM25 en dehors de l'event loop."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._bm25_executor,
            self._bm25_search,
            query,
            collection,
            n_results,
            filters,
        )

    async def hybrid_search(
        self,
        query: str,
        collection: Literal["main", "glossary"] = "main",
        n_results: int = 5,
        filters: dict[str, Any] | None = None,
        alpha: float | None = None,
    ) -> list[SearchResult]:
        """Recherche hybride dense + sparse avec RRF Fusion.

        Args:
            query: Query de recherche
            collection: Collection cible
            n_results: Nombre de résultats finaux
            filters: Filtres metadata (section_type, cycle, phase_num, etc.)
            alpha: Poids du dense (défaut: RRF_ALPHA)

        Returns:
            Liste de SearchResult triés par score RRF
        """
        alpha = alpha if alpha is not None else self.RRF_ALPHA
        collection_name = self.COLLECTIONS[collection]
        n_fetch = min(n_results * 4, 50)

        log.info(
            "vector_store.hybrid_search",
            query=query[:80],
            collection=collection_name,
            n_results=n_results,
            alpha=alpha,
        )

        # Filtre Qdrant
        qdrant_filter = self._build_filter(filters) if filters else None

        # Dense search (Qdrant) - utilise query_points (async si disponible)
        query_vec = self.embed_model.encode(query).tolist()
        if self._async_client:
            dense_response = await self._async_client.query_points(
                collection_name=collection_name,
                query=query_vec,
                limit=n_fetch,
                query_filter=qdrant_filter,
                with_payload=True,
            )
        else:
            dense_response = self.client.query_points(
                collection_name=collection_name,
                query=query_vec,
                limit=n_fetch,
                query_filter=qdrant_filter,
                with_payload=True,
            )
        dense_hits = list(dense_response.points)

        # Sparse search (BM25)
        sparse_hits = await self._bm25_search_async(
            query,
            collection,
            n_fetch,
            filters,
        )

        # RRF Fusion
        fused = self._rrf_fusion(dense_hits, sparse_hits, alpha, n_results)

        log.info(
            "vector_store.hybrid_search_done",
            n_dense=len(dense_hits),
            n_sparse=len(sparse_hits),
            n_fused=len(fused),
        )

        return fused

    async def dense_search(
        self,
        query: str,
        collection: Literal["main", "glossary"] = "main",
        n_results: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Recherche dense uniquement (Qdrant).

        Args:
            query: Query de recherche
            collection: Collection cible
            n_results: Nombre de résultats
            filters: Filtres metadata optionnels

        Returns:
            Liste de SearchResult
        """
        collection_name = self.COLLECTIONS[collection]
        qdrant_filter = self._build_filter(filters) if filters else None

        query_vec = self.embed_model.encode(query).tolist()
        if self._async_client:
            response = await self._async_client.query_points(
                collection_name=collection_name,
                query=query_vec,
                limit=n_results,
                query_filter=qdrant_filter,
                with_payload=True,
            )
        else:
            response = self.client.query_points(
                collection_name=collection_name,
                query=query_vec,
                limit=n_results,
                query_filter=qdrant_filter,
                with_payload=True,
            )

        return [self._hit_to_result(hit) for hit in response.points]

    async def sparse_search(
        self,
        query: str,
        collection: Literal["main", "glossary"] = "main",
        n_results: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Recherche sparse uniquement (BM25).

        Args:
            query: Query de recherche
            collection: Collection cible
            n_results: Nombre de résultats
            filters: Filtres metadata optionnels

        Returns:
            Liste de SearchResult
        """
        hits = self._bm25_search(query, collection, n_results, filters)

        return [
            SearchResult(
                id=str(hit["id"]),
                score=hit["score"],
                content=hit["payload"].get("content", ""),
                section_type=hit["payload"].get("section_type"),
                cycle=hit["payload"].get("cycle"),
                milestone_id=hit["payload"].get("milestone_id"),
                phase_num=hit["payload"].get("phase_num"),
                context_prefix=hit["payload"].get("context_prefix"),
                metadata=hit["payload"],
            )
            for hit in hits
        ]

    async def glossary_lookup(self, term: str) -> SearchResult | None:
        """Lookup exact dans le glossaire GRI.

        Args:
            term: Terme à rechercher (FR ou EN)

        Returns:
            SearchResult avec la définition, ou None si non trouvé
        """
        log.info("vector_store.glossary_lookup", term=term)
        term_lower = term.lower().strip()

        # Essayer l'exact match d'abord (terme FR) - case-insensitive via scroll + filter
        try:
            if self._async_client:
                scroll_results, _ = await self._async_client.scroll(
                    collection_name=self.COLLECTIONS["glossary"],
                    scroll_filter=Filter(
                        must=[FieldCondition(key="term_fr", match=MatchText(text=term_lower))]
                    ),
                    limit=5,
                    with_payload=True,
                )
            else:
                scroll_results, _ = self.client.scroll(
                    collection_name=self.COLLECTIONS["glossary"],
                    scroll_filter=Filter(
                        must=[FieldCondition(key="term_fr", match=MatchText(text=term_lower))]
                    ),
                    limit=5,
                    with_payload=True,
                )
            # Vérifier un match exact (case-insensitive)
            for r in scroll_results:
                payload = cast(dict[str, Any], r.payload or {})
                if payload.get("term_fr", "").lower() == term_lower:
                    return self._point_to_result(r)
        except Exception as e:
            log.warning("vector_store.glossary_exact_match_failed", error=str(e))

        # Essayer l'exact match terme EN
        try:
            if self._async_client:
                scroll_results, _ = await self._async_client.scroll(
                    collection_name=self.COLLECTIONS["glossary"],
                    scroll_filter=Filter(
                        must=[FieldCondition(key="term_en", match=MatchText(text=term_lower))]
                    ),
                    limit=5,
                    with_payload=True,
                )
            else:
                scroll_results, _ = self.client.scroll(
                    collection_name=self.COLLECTIONS["glossary"],
                    scroll_filter=Filter(
                        must=[FieldCondition(key="term_en", match=MatchText(text=term_lower))]
                    ),
                    limit=5,
                    with_payload=True,
                )
            for r in scroll_results:
                payload = cast(dict[str, Any], r.payload or {})
                if payload.get("term_en", "").lower() == term_lower:
                    return self._point_to_result(r)
        except Exception as e:
            log.warning("vector_store.glossary_en_match_failed", error=str(e))

        # Fallback : recherche hybride sur le glossaire
        hybrid_results = await self.hybrid_search(
            term,
            collection="glossary",
            n_results=1,
            alpha=0.3,
        )
        if not hybrid_results:
            return None

        candidate = hybrid_results[0]
        candidate_fields = [
            candidate.metadata.get("term_fr", ""),
            candidate.metadata.get("term_en", ""),
            candidate.content,
        ]
        if any(term_lower in str(value).lower() for value in candidate_fields if value):
            return candidate

        return None

    def _rrf_fusion(
        self,
        dense_hits: list[Any],
        sparse_hits: list[SparseHit],
        alpha: float,
        n_results: int,
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion des résultats dense et sparse.

        RRF score = α * (1 / (k + rank_dense)) + (1-α) * (1 / (k + rank_sparse))

        Args:
            dense_hits: Résultats Qdrant
            sparse_hits: Résultats BM25
            alpha: Poids du dense
            n_results: Nombre de résultats à retourner

        Returns:
            Liste fusionnée triée par score RRF
        """
        scores: dict[str, float] = {}
        payloads: dict[str, dict[str, Any]] = {}

        # Scores dense
        for rank, hit in enumerate(dense_hits):
            doc_id = str(hit.id)
            scores[doc_id] = scores.get(doc_id, 0) + alpha * (1 / (self.RRF_K + rank + 1))
            payloads[doc_id] = cast(dict[str, Any], hit.payload or {})

        # Scores sparse
        for rank, hit in enumerate(sparse_hits):
            doc_id = str(hit["id"])
            scores[doc_id] = scores.get(doc_id, 0) + (1 - alpha) * (1 / (self.RRF_K + rank + 1))
            if doc_id not in payloads:
                payloads[doc_id] = hit["payload"]

        # Trier par score RRF
        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:n_results]

        return [
            SearchResult(
                id=doc_id,
                score=scores[doc_id],
                content=payloads[doc_id].get("content", ""),
                section_type=payloads[doc_id].get("section_type"),
                cycle=payloads[doc_id].get("cycle"),
                milestone_id=payloads[doc_id].get("milestone_id"),
                phase_num=payloads[doc_id].get("phase_num"),
                context_prefix=payloads[doc_id].get("context_prefix"),
                metadata=payloads[doc_id],
            )
            for doc_id in sorted_ids
        ]

    def _build_filter(self, filters: dict[str, Any]) -> Filter | None:
        """Construit un filtre Qdrant à partir d'un dict.

        Args:
            filters: Dict de filtres {key: value}

        Returns:
            Filtre Qdrant
        """
        conditions: list[FieldCondition] = []
        for key, value in filters.items():
            if value is not None:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=conditions) if conditions else None

    def _hit_to_result(self, hit: Any) -> SearchResult:
        """Convertit un hit Qdrant en SearchResult."""
        payload = cast(dict[str, Any], hit.payload or {})
        return SearchResult(
            id=str(hit.id),
            score=hit.score,
            content=cast(str, payload.get("content", "")),
            section_type=cast(str | None, payload.get("section_type")),
            cycle=cast(str | None, payload.get("cycle")),
            milestone_id=cast(str | None, payload.get("milestone_id")),
            phase_num=cast(int | None, payload.get("phase_num")),
            context_prefix=cast(str | None, payload.get("context_prefix")),
            metadata=payload,
        )

    def _point_to_result(self, point: Any) -> SearchResult:
        """Convertit un point Qdrant (scroll) en SearchResult."""
        payload = cast(dict[str, Any], point.payload or {})
        return SearchResult(
            id=str(point.id),
            score=1.0,  # Exact match
            content=cast(str, payload.get("content", "")),
            section_type=cast(str | None, payload.get("section_type")),
            cycle=cast(str | None, payload.get("cycle")),
            milestone_id=cast(str | None, payload.get("milestone_id")),
            phase_num=cast(int | None, payload.get("phase_num")),
            context_prefix=cast(str | None, payload.get("context_prefix")),
            metadata=payload,
        )

    async def get_collection_stats(self) -> dict[str, Any]:
        """Retourne les statistiques des collections."""
        stats: dict[str, dict[str, Any]] = {}
        for name, collection_name in self.COLLECTIONS.items():
            try:
                if self._async_client:
                    info = await self._async_client.get_collection(collection_name)
                else:
                    info = self.client.get_collection(collection_name)
                stats[name] = {
                    "name": collection_name,
                    "points_count": getattr(info, "points_count", 0),
                    "vectors_count": getattr(info, "vectors_count", None),
                    "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
                    "status": getattr(info, "status", None),
                }
            except Exception as e:
                stats[name] = {"error": str(e)}
        return stats

    async def get_stats(self) -> dict[str, Any]:
        """Alias de compatibilité pour les tests et l'API interne."""
        collection_stats = await self.get_collection_stats()
        normalized: dict[str, dict[str, Any]] = {}
        for name, info in collection_stats.items():
            count = info.get("points_count")
            if count in (None, 0):
                count = info.get("vectors_count", 0)
            normalized[name] = {
                "count": count or 0,
                **info,
            }
        return normalized

    async def delete_collection(self, collection: Literal["main", "glossary"]) -> bool:
        """Supprime une collection.

        Args:
            collection: Nom de la collection

        Returns:
            True si supprimée, False sinon
        """
        collection_name = self.COLLECTIONS[collection]
        try:
            if self._async_client:
                await self._async_client.delete_collection(collection_name)
            else:
                self.client.delete_collection(collection_name)
            self._corpus_loaded[collection] = False
            log.info("vector_store.collection_deleted", collection=collection_name)
            return True
        except Exception as e:
            log.error(
                "vector_store.collection_delete_failed",
                collection=collection_name,
                error=str(e),
            )
            return False

    def close(self) -> None:
        """Libère les ressources locales du store."""
        self._bm25_executor.shutdown(wait=False, cancel_futures=True)


# Singleton pour usage global
_store: GRIHybridStore | None = None


def get_vector_store() -> GRIHybridStore:
    """Retourne le singleton du vector store."""
    global _store
    if _store is None:
        _store = GRIHybridStore()
    return _store
