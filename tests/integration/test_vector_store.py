"""Tests d'intégration pour GRIHybridStore.

Ces tests utilisent Qdrant en mémoire pour éviter une dépendance
à un serveur externe pendant la CI locale.
Lancer avec: pytest tests/integration/test_vector_store.py -v -m integration

Note: Les tests utilisent des collections temporaires qui sont nettoyées après exécution.
"""

import contextlib
import hashlib

import numpy as np
import pytest

from src.core.config import settings
from src.core.vector_store import GRIHybridStore, SearchResult

# Marquer tous les tests comme tests d'intégration
pytestmark = pytest.mark.integration


class DummySentenceTransformer:
    """Encodeur déterministe pour tests offline."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def encode(self, texts):
        if isinstance(texts, str):
            return self._encode_one(texts)
        return np.vstack([self._encode_one(text) for text in texts])

    def _encode_one(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = np.frombuffer(digest, dtype=np.uint8).astype(np.float32)
        repeats = (settings.embedding_dimension + len(values) - 1) // len(values)
        vector = np.tile(values, repeats)[: settings.embedding_dimension]
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector


def make_test_store(*, use_async: bool = True) -> GRIHybridStore:
    """Construit un store isolé en mémoire pour les tests."""
    return GRIHybridStore(qdrant_url=":memory:", use_async=use_async)


@pytest.fixture(autouse=True)
def patch_sentence_transformer(monkeypatch):
    """Évite le téléchargement du vrai modèle d'embeddings."""
    monkeypatch.setattr(
        "src.core.vector_store.SentenceTransformer",
        DummySentenceTransformer,
    )


@pytest.fixture
def sample_chunks():
    """Chunks de test représentatifs du GRI."""
    return [
        {
            "content": "[GRI > Phase 3 > Conception > Définitions] Un artefact est un produit "
            "tangible ou intangible résultant d'une activité d'ingénierie.",
            "chunk_id": "def_artefact_001",
            "metadata": {
                "section_type": "definition",
                "cycle": "GRI",
                "term_fr": "artefact",
                "term_en": "artifact",
                "context_prefix": "[GRI > Phase 3 > Conception > Définitions]",
            },
        },
        {
            "content": "[GRI > Jalon M4 (CDR)] Le CDR valide la conception détaillée. "
            "Critères : 1. Architecture finalisée 2. Interfaces définies 3. Plans de test établis",
            "chunk_id": "milestone_m4_001",
            "metadata": {
                "section_type": "milestone",
                "cycle": "GRI",
                "milestone_id": "M4",
                "milestone_name": "CDR",
                "context_prefix": "[GRI > Jalon M4 (CDR)]",
            },
        },
        {
            "content": "[GRI > Phase 3 > Conception > Processus de Vérification] "
            "Le processus de vérification assure que les exigences sont satisfaites "
            "par l'implémentation proposée.",
            "chunk_id": "process_verif_001",
            "metadata": {
                "section_type": "process",
                "cycle": "GRI",
                "process_name": "Vérification",
                "phase_num": 3,
                "context_prefix": "[GRI > Phase 3 > Conception > Processus de Vérification]",
            },
        },
        {
            "content": "[GRI > Glossaire] CONOPS (Concept of Operations) : Document décrivant "
            "l'emploi opérationnel du système du point de vue de l'utilisateur.",
            "chunk_id": "glossary_conops_001",
            "metadata": {
                "section_type": "definition",
                "cycle": "GRI",
                "term_fr": "CONOPS",
                "term_en": "Concept of Operations",
                "context_prefix": "[GRI > Glossaire]",
            },
        },
        {
            "content": "[CIR > Phase 2 > Jalon J3] Le jalon J3 du CIR correspond à la revue "
            "de conception préliminaire. Il valide l'architecture fonctionnelle.",
            "chunk_id": "cir_j3_001",
            "metadata": {
                "section_type": "milestone",
                "cycle": "CIR",
                "milestone_id": "J3",
                "phase_num": 2,
                "context_prefix": "[CIR > Phase 2 > Jalon J3]",
            },
        },
    ]


@pytest.fixture
def glossary_chunks():
    """Chunks spécifiques pour la collection glossary."""
    return [
        {
            "content": "artefact (artifact) : Produit tangible ou intangible résultant "
            "d'une activité d'ingénierie des systèmes.",
            "chunk_id": "gloss_artefact",
            "metadata": {
                "section_type": "definition",
                "term_fr": "artefact",
                "term_en": "artifact",
                "source": "ISO/IEC/IEEE 15288:2023",
            },
        },
        {
            "content": "SEMP (Systems Engineering Management Plan) : Plan définissant "
            "l'organisation et les processus d'ingénierie des systèmes.",
            "chunk_id": "gloss_semp",
            "metadata": {
                "section_type": "definition",
                "term_fr": "SEMP",
                "term_en": "Systems Engineering Management Plan",
                "source": "ISO/IEC/IEEE 15288:2023",
            },
        },
        {
            "content": "vérification (verification) : Confirmation par des preuves objectives "
            "que les exigences spécifiées ont été satisfaites.",
            "chunk_id": "gloss_verification",
            "metadata": {
                "section_type": "definition",
                "term_fr": "vérification",
                "term_en": "verification",
                "source": "ISO/IEC/IEEE 15288:2023",
            },
        },
    ]


class TestGRIHybridStoreConnection:
    """Tests de connexion et initialisation."""

    def test_store_initialization(self):
        """Le store s'initialise avec les paramètres par défaut."""
        store = make_test_store(use_async=False)
        assert store.client is not None
        assert store._embed_model is None  # Lazy loading
        store.close()

    def test_store_initialization_custom_params(self):
        """Le store accepte des paramètres personnalisés."""
        store = GRIHybridStore(
            qdrant_url="localhost",
            qdrant_port=6333,
            use_async=False,
        )
        assert store._qdrant_url == "localhost"
        assert store._qdrant_port == 6333

    def test_embed_model_lazy_loading(self):
        """Le modèle d'embeddings est chargé à la demande."""
        store = make_test_store(use_async=False)
        assert store._embed_model is None
        # Accéder au modèle déclenche le chargement
        model = store.embed_model
        assert model is not None
        assert store._embed_model is not None
        store.close()


class TestCollectionManagement:
    """Tests de gestion des collections Qdrant."""

    @pytest.fixture
    def store(self):
        """Store configuré pour les tests."""
        store = make_test_store(use_async=True)
        yield store
        store.close()

    @pytest.mark.asyncio
    async def test_ensure_collections_creates_missing(self, store):
        """ensure_collections crée les collections manquantes."""
        # Supprimer les collections de test si elles existent
        for collection_name in store.COLLECTIONS.values():
            with contextlib.suppress(Exception):
                store.client.delete_collection(collection_name)

        # Créer les collections
        await store.ensure_collections()

        # Vérifier qu'elles existent
        for collection_name in store.COLLECTIONS.values():
            info = store.client.get_collection(collection_name)
            assert info is not None

    @pytest.mark.asyncio
    async def test_ensure_collections_idempotent(self, store):
        """ensure_collections est idempotent."""
        # Appeler deux fois ne doit pas lever d'erreur
        await store.ensure_collections()
        await store.ensure_collections()

        for collection_name in store.COLLECTIONS.values():
            info = store.client.get_collection(collection_name)
            assert info is not None


class TestChunkIndexing:
    """Tests d'indexation des chunks."""

    @pytest.fixture
    async def store_with_collections(self):
        """Store avec collections prêtes."""
        store = make_test_store(use_async=True)
        await store.ensure_collections()
        yield store
        store.close()

    @pytest.mark.asyncio
    async def test_index_chunks_main_collection(self, store_with_collections, sample_chunks):
        """L'indexation dans la collection main fonctionne."""
        store = store_with_collections

        result = await store.index_chunks(sample_chunks, collection="main")

        assert result["total"] == len(sample_chunks)
        assert result["indexed"] == len(sample_chunks)
        assert result["collection"] == store.COLLECTIONS["main"]

    @pytest.mark.asyncio
    async def test_index_chunks_glossary_collection(self, store_with_collections, glossary_chunks):
        """L'indexation dans la collection glossary fonctionne."""
        store = store_with_collections

        result = await store.index_chunks(glossary_chunks, collection="glossary")

        assert result["total"] == len(glossary_chunks)
        assert result["indexed"] == len(glossary_chunks)
        assert result["collection"] == store.COLLECTIONS["glossary"]

    @pytest.mark.asyncio
    async def test_index_chunks_invalidates_bm25_cache(self, store_with_collections, sample_chunks):
        """L'indexation invalide le cache BM25."""
        store = store_with_collections

        # Charger le cache BM25
        store._corpus_loaded["main"] = True

        # Indexer de nouveaux chunks
        await store.index_chunks(sample_chunks, collection="main")

        # Le cache doit être invalidé
        assert store._corpus_loaded["main"] is False


class TestHybridSearch:
    """Tests de recherche hybride."""

    @pytest.fixture
    async def indexed_store(self, sample_chunks, glossary_chunks):
        """Store avec données indexées."""
        store = make_test_store(use_async=True)
        await store.ensure_collections()
        await store.index_chunks(sample_chunks, collection="main")
        await store.index_chunks(glossary_chunks, collection="glossary")
        yield store
        store.close()

    @pytest.mark.asyncio
    async def test_hybrid_search_returns_results(self, indexed_store):
        """La recherche hybride retourne des résultats."""
        results = await indexed_store.hybrid_search(
            query="Qu'est-ce qu'un artefact ?",
            n_results=3,
        )

        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_hybrid_search_with_section_filter(self, indexed_store):
        """La recherche hybride respecte les filtres de section."""
        results = await indexed_store.hybrid_search(
            query="critères",
            n_results=5,
            filters={"section_type": "milestone"},
        )

        # Tous les résultats doivent être des jalons
        for r in results:
            assert r.section_type == "milestone"

    @pytest.mark.asyncio
    async def test_hybrid_search_with_cycle_filter(self, indexed_store):
        """La recherche hybride respecte les filtres de cycle."""
        results = await indexed_store.hybrid_search(
            query="jalon",
            n_results=5,
            filters={"cycle": "CIR"},
        )

        # Tous les résultats doivent être du CIR
        for r in results:
            assert r.cycle == "CIR"

    @pytest.mark.asyncio
    async def test_hybrid_search_rrf_fusion(self, indexed_store):
        """La fusion RRF combine dense et sparse correctement."""
        # Query avec terme exact (favorise sparse)
        results = await indexed_store.hybrid_search(
            query="CONOPS",
            n_results=3,
        )

        assert len(results) > 0
        # Le résultat avec CONOPS doit être bien classé
        contents = [r.content for r in results]
        assert any("CONOPS" in c for c in contents)


class TestGlossaryLookup:
    """Tests de recherche de définitions."""

    @pytest.fixture
    async def indexed_store(self, glossary_chunks):
        """Store avec glossaire indexé."""
        store = make_test_store(use_async=True)
        await store.ensure_collections()
        await store.index_chunks(glossary_chunks, collection="glossary")
        yield store
        store.close()

    @pytest.mark.asyncio
    async def test_glossary_lookup_exact_term_fr(self, indexed_store):
        """La recherche trouve un terme français exact."""
        result = await indexed_store.glossary_lookup("artefact")

        assert result is not None
        assert "artefact" in result.content.lower()

    @pytest.mark.asyncio
    async def test_glossary_lookup_exact_term_en(self, indexed_store):
        """La recherche trouve un terme anglais exact."""
        result = await indexed_store.glossary_lookup("artifact")

        assert result is not None
        assert "artifact" in result.content.lower() or "artefact" in result.content.lower()

    @pytest.mark.asyncio
    async def test_glossary_lookup_acronym(self, indexed_store):
        """La recherche trouve un acronyme."""
        result = await indexed_store.glossary_lookup("SEMP")

        assert result is not None
        assert "SEMP" in result.content

    @pytest.mark.asyncio
    async def test_glossary_lookup_not_found(self, indexed_store):
        """La recherche retourne None si le terme n'existe pas."""
        result = await indexed_store.glossary_lookup("terme_inexistant_xyz")

        assert result is None


class TestBM25Search:
    """Tests de recherche BM25 sparse."""

    @pytest.fixture
    async def indexed_store(self, sample_chunks):
        """Store avec données indexées."""
        store = make_test_store(use_async=True)
        await store.ensure_collections()
        await store.index_chunks(sample_chunks, collection="main")
        yield store
        store.close()

    def test_bm25_corpus_loading(self, indexed_store):
        """Le corpus BM25 se charge correctement."""
        indexed_store._load_bm25_corpus("main")

        assert indexed_store._corpus_loaded["main"] is True
        assert "main" in indexed_store._bm25
        assert len(indexed_store._corpus["main"]) > 0

    def test_bm25_search_returns_scored_results(self, indexed_store):
        """La recherche BM25 retourne des résultats avec scores."""
        indexed_store._load_bm25_corpus("main")

        results = indexed_store._bm25_search(
            query="artefact",
            collection="main",
            n_results=3,
        )

        assert len(results) > 0
        assert all("score" in r for r in results)
        # Les scores doivent être décroissants
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_bm25_search_with_filters(self, indexed_store):
        """La recherche BM25 respecte les filtres."""
        indexed_store._load_bm25_corpus("main")

        results = indexed_store._bm25_search(
            query="critères",
            collection="main",
            n_results=5,
            filters={"section_type": "milestone"},
        )

        for r in results:
            assert r["payload"]["section_type"] == "milestone"

    def test_tokenize_handles_french_accents(self, indexed_store):
        """La tokenisation gère les accents français."""
        tokens = indexed_store._tokenize("Vérification des exigences système")

        assert "vérification" in tokens
        assert "exigences" in tokens
        assert "système" in tokens


class TestStoreStatistics:
    """Tests des statistiques du store."""

    @pytest.fixture
    async def indexed_store(self, sample_chunks, glossary_chunks):
        """Store avec données indexées."""
        store = make_test_store(use_async=True)
        await store.ensure_collections()
        await store.index_chunks(sample_chunks, collection="main")
        await store.index_chunks(glossary_chunks, collection="glossary")
        yield store
        store.close()

    @pytest.mark.asyncio
    async def test_get_stats_returns_counts(self, indexed_store):
        """get_stats retourne les comptages par collection."""
        stats = await indexed_store.get_stats()

        assert "main" in stats
        assert "glossary" in stats
        assert stats["main"]["count"] > 0
        assert stats["glossary"]["count"] > 0


# =============================================================================
# Tests de performance (optionnels)
# =============================================================================


@pytest.mark.slow
class TestSearchPerformance:
    """Tests de performance pour la recherche."""

    @pytest.fixture
    async def large_indexed_store(self):
        """Store avec beaucoup de données."""
        store = make_test_store(use_async=True)
        await store.ensure_collections()

        # Générer 500 chunks de test
        chunks = []
        for i in range(500):
            chunks.append(
                {
                    "content": f"[GRI > Phase {i % 7 + 1}] Contenu de test numéro {i} "
                    f"avec des termes variés comme artefact, vérification, système.",
                    "chunk_id": f"perf_chunk_{i:04d}",
                    "metadata": {
                        "section_type": ["definition", "milestone", "process"][i % 3],
                        "cycle": "GRI",
                        "phase_num": i % 7 + 1,
                    },
                }
            )

        await store.index_chunks(chunks, collection="main")
        yield store
        store.close()

    @pytest.mark.asyncio
    async def test_hybrid_search_latency(self, large_indexed_store):
        """La recherche hybride doit être rapide (< 2s)."""
        import time

        start = time.time()
        results = await large_indexed_store.hybrid_search(
            query="processus de vérification",
            n_results=10,
        )
        elapsed = time.time() - start

        assert elapsed < 2.0, f"Recherche trop lente: {elapsed:.2f}s"
        assert len(results) > 0
