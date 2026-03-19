---
name: rag-retrieval-gri
description: Retrieval hybride spécialisé pour le GRI/FAR avec query routing par intent, double index dense+sparse, reranking cross-encoder, et gestion des types de contenu GRI (jalons, processus IS 15288, CIR). Utiliser pour implémenter ou améliorer la couche retrieval du RAG GRI. Déclencher pour toute mention de "retrieval GRI", "recherche dans le GRI", "hybrid search", "query router", "BM25 termes ISO", "reranking jalons", "recherche vectorielle", ou "améliorer le recall GRI".
---

# Retrieval Skill — GRI/FAR

## Architecture de Retrieval GRI

```
Query utilisateur
       │
       ▼
[Query Router]  ──── 6 intents GRI ────►  stratégie + filtres metadata
       │
       ▼
[Term Expansion]  ── glossaire GRI ──►  enrichissement query avec termes ISO
       │
       ├──► [Dense Search]   Qdrant — cosine similarity (embeddings multilingues)
       │
       ├──► [Sparse Search]  BM25 — exact match termes normés ISO
       │
       └──► [Glossary Lookup] Index dédié — définitions exactes FR/EN
                │
                ▼
          [RRF Fusion]  α=0.6 dense / 0.4 sparse
                │
                ▼
          [Reranking]   cross-encoder — précision finale
                │
                ▼
          [MMR Filter]  λ=0.7 — diversité sections
                │
                ▼
          Top-5 Chunks contextualisés
```

## Query Router — 6 Intents GRI

C'est la pièce la plus critique du retrieval. Sans routing, la qualité s'effondre.

```python
# src/agents/query_router.py
from huggingface_hub import InferenceClient
import os
import json

# Modèle HF recommandé pour le routing (léger et rapide)
HF_ROUTER_MODEL = os.getenv("HF_ROUTER_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")

ROUTER_PROMPT = """Classifie cette question sur le GRI des FAR dans exactement un des 6 intents.

INTENTS :
- DEFINITION    : demande de définition d'un terme ISO ou GRI
- PROCESSUS     : question sur un processus IS 15288 (activités, inputs, outputs)
- JALON         : critères de passage d'un jalon (M0-M9 ou J1-J6)
- PHASE_COMPLETE: résumé ou objectifs d'une phase complète (1-7 GRI ou 1-4 CIR)
- COMPARAISON   : comparaison entre deux éléments du GRI
- CIR           : question spécifique au Cycle d'Innovation Rapide

Retourne JSON : {{"intent": "...", "cycle": "GRI"|"CIR"|"BOTH", "entities": ["..."]}}

Question : {query}"""

ROUTING_TABLE = {
    "DEFINITION": {
        "search_mode":    "sparse",     # BM25 suffit pour les termes exacts
        "primary_index":  "glossary",   # Index glossaire dédié
        "fallback_index": "main",
        "filters":        {"section_type": "definition"},
        "n_initial":      5,
        "use_reranker":   False,        # Pas besoin — exact match
        "temperature":    0.0,          # Définitions ISO : fidélité absolue
    },
    "PROCESSUS": {
        "search_mode":    "hybrid",
        "primary_index":  "main",
        "filters":        {"section_type": "process"},
        "n_initial":      20,
        "use_reranker":   True,         # Critique pour les processus IS 15288
        "use_parent":     True,         # Remonter au processus parent si besoin
        "temperature":    0.1,
    },
    "JALON": {
        "search_mode":    "hybrid",
        "primary_index":  "main",
        "filters":        {"section_type": "milestone"},
        "n_initial":      5,            # Peu de jalons → fetch direct
        "use_reranker":   False,
        "return_complete": True,        # TOUJOURS retourner le jalon entier
        "temperature":    0.0,
    },
    "PHASE_COMPLETE": {
        "search_mode":    "dense",
        "primary_index":  "main",
        "filters":        {"section_type": "phase"},
        "n_initial":      15,
        "use_parent":     True,         # Parent Document Retriever obligatoire
        "use_reranker":   True,
        "use_mmr":        True,         # Diversité sous-sections
        "temperature":    0.1,
    },
    "COMPARAISON": {
        "search_mode":    "hybrid",
        "primary_index":  "main",
        "filters":        None,         # Multi-filter selon entities
        "n_initial":      30,           # Large pour couvrir les deux sujets
        "use_reranker":   True,
        "use_mmr":        True,
        "multi_query":    True,         # 1 query par entité à comparer
        "temperature":    0.1,
    },
    "CIR": {
        "search_mode":    "hybrid",
        "primary_index":  "main",
        "filters":        {"cycle": "CIR"},
        "fallback_filter":{"cycle": "GRI"},  # Fallback sur GRI pour mapping
        "n_initial":      20,
        "use_reranker":   True,
        "include_gri_mapping": True,    # Ajouter le mapping J→M dans le contexte
        "temperature":    0.1,
    },
}

async def route_query(query: str, client: InferenceClient) -> dict:
    response = await client.text_generation(
        prompt=ROUTER_PROMPT.format(query=query),
        model=HF_ROUTER_MODEL,
        max_new_tokens=256,
        temperature=0.1,
    )
    routing = json.loads(response)
    strategy = ROUTING_TABLE[routing["intent"]]
    return {**routing, **strategy}
```

## Double Index — Dense + Sparse

```python
# src/core/vector_store.py
import qdrant_client
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np

class GRIHybridStore:
    """
    Deux index Qdrant séparés :
    - "gri_main"     : tous les chunks sauf définitions
    - "gri_glossary" : 200+ définitions ISO bilingues (lookup rapide)

    Alpha RRF = 0.6 dense / 0.4 sparse
    → On favorise légèrement le dense pour la sémantique,
      mais le sparse est critique pour les termes ISO exacts.
    """

    EMBED_MODEL = "paraphrase-multilingual-mpnet-base-v2"  # FR+EN natif
    COLLECTIONS = {
        "main":     "gri_main",
        "glossary": "gri_glossary",
    }
    RRF_ALPHA = 0.6   # Dense
    RRF_K     = 60    # Constante RRF standard (Cormack 2009)

    def __init__(self, qdrant_url: str = "localhost:6333"):
        self.client = qdrant_client.QdrantClient(url=qdrant_url)
        self.embed_model = SentenceTransformer(self.EMBED_MODEL)
        self._bm25: dict[str, BM25Okapi] = {}
        self._corpus: dict[str, list] = {}

    def hybrid_search(
        self,
        query: str,
        collection: str = "main",
        n_results: int = 5,
        filters: dict | None = None,
        alpha: float | None = None,
    ) -> list[dict]:
        alpha = alpha or self.RRF_ALPHA
        coll_name = self.COLLECTIONS[collection]
        n_fetch = min(n_results * 4, 50)

        # Filtre Qdrant
        qdrant_filter = self._build_filter(filters) if filters else None

        # Dense search
        query_vec = self.embed_model.encode([query])[0].tolist()
        dense_hits = self.client.search(
            collection_name=coll_name,
            query_vector=query_vec,
            limit=n_fetch,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        # Sparse search (BM25)
        sparse_hits = self._bm25_search(query, collection, n_fetch, filters)

        # RRF Fusion
        return self._rrf_fusion(dense_hits, sparse_hits, alpha, n_results)

    def glossary_lookup(self, term: str) -> dict | None:
        """Lookup exact dans le glossaire GRI. Retourne FR + EN + norme."""
        # Essayer l'exact match d'abord
        results = self.client.scroll(
            collection_name=self.COLLECTIONS["glossary"],
            scroll_filter=Filter(must=[
                FieldCondition(key="term_fr", match=MatchValue(value=term.lower()))
            ]),
            limit=1,
            with_payload=True,
        )
        if results[0]:
            return results[0][0].payload

        # Fallback : recherche sémantique
        hits = self.hybrid_search(term, collection="glossary", n_results=1)
        return hits[0] if hits else None

    def _rrf_fusion(self, dense_hits, sparse_hits, alpha, n_results):
        scores = {}
        payloads = {}

        for rank, hit in enumerate(dense_hits):
            doc_id = hit.id
            scores[doc_id] = scores.get(doc_id, 0) + alpha * (1 / (self.RRF_K + rank + 1))
            payloads[doc_id] = hit.payload

        for rank, hit in enumerate(sparse_hits):
            doc_id = hit["id"]
            scores[doc_id] = scores.get(doc_id, 0) + (1 - alpha) * (1 / (self.RRF_K + rank + 1))
            payloads[doc_id] = hit["payload"]

        sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:n_results]
        return [
            {"id": doc_id, "score": scores[doc_id], **payloads[doc_id]}
            for doc_id in sorted_ids
        ]

    def _build_filter(self, filters: dict) -> Filter:
        conditions = []
        for key, value in filters.items():
            conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=conditions)
```

## Pre-Retrieval : Enrichissement Terminologique

Avant la retrieval principale, détecter et injecter les définitions GRI connues dans le contexte LLM.

```python
# src/core/term_expander.py

# Termes GRI détectables dans les queries (subset des 200+)
GRI_KEY_TERMS = [
    "artefact", "artifact", "CONOPS", "SEMP", "SRR", "PDR", "CDR",
    "IRR", "TRR", "SAR", "ORR", "MNR", "CIR", "TRL", "MRL",
    "exigence système", "architecture système", "vérification",
    "validation", "intégration", "jalon", "cycle de vie",
    "ingénierie système", "parties prenantes", "traçabilité",
]

async def expand_query_with_terms(
    query: str,
    store: GRIHybridStore,
) -> tuple[str, str]:
    """
    Détecte les termes GRI dans la query et récupère leurs définitions.
    Retourne (query_originale, contexte_terminologique).
    Le contexte terminologique est injecté dans le system prompt, pas dans la query vectorielle.
    """
    detected = [t for t in GRI_KEY_TERMS if t.lower() in query.lower()]
    if not detected:
        return query, ""

    definitions = []
    for term in detected[:3]:  # Max 3 définitions injectées
        defn = store.glossary_lookup(term)
        if defn:
            definitions.append(
                f"• {defn.get('term_fr', term)} ({defn.get('term_en', '')}): "
                f"{defn.get('definition_fr', '')}"
            )

    term_context = (
        "## Définitions GRI applicables\n" + "\n".join(definitions)
        if definitions else ""
    )
    return query, term_context
```

## Retrieval Spéciale Jalons

Les jalons ont une logique de retrieval particulière : on veut TOUJOURS le jalon complet.

```python
async def get_jalon_complet(milestone_id: str, store: GRIHybridStore) -> dict:
    """
    Récupère un jalon complet (M0-M9 ou J1-J6) avec TOUS ses critères.
    Pour les jalons CIR, ajoute automatiquement le mapping GRI.
    """
    # Lookup direct par milestone_id
    results = store.client.scroll(
        collection_name=store.COLLECTIONS["main"],
        scroll_filter=Filter(must=[
            FieldCondition(key="milestone_id", match=MatchValue(value=milestone_id))
        ]),
        limit=5,
        with_payload=True,
    )

    chunks = [r.payload for r in results[0]]

    # Pour les jalons CIR, ajouter le mapping GRI équivalent
    if milestone_id.startswith("J"):
        gri_equivalents = CIR_GRI_MAPPING.get(milestone_id, [])
        for gri_id in gri_equivalents:
            gri_chunk = await get_jalon_complet(gri_id, store)
            chunks.append({**gri_chunk, "role": f"GRI équivalent de {milestone_id}"})

    return {"milestone_id": milestone_id, "chunks": chunks}

CIR_GRI_MAPPING = {
    "J1": ["M0", "M1"],
    "J2": ["M2", "M3", "M4"],
    "J3": ["M5", "M6"],
    "J4": ["SAR"],
    "J5": ["SAR"],
    "J6": ["M8"],
}
```

## Paramètres de Retrieval par Intent

| Intent | n_initial | n_final | Alpha RRF | Reranker | MMR |
|--------|-----------|---------|-----------|----------|-----|
| DEFINITION | 5 | 1-2 | 0.3/0.7 sparse | Non | Non |
| PROCESSUS | 20 | 5 | 0.6/0.4 | Oui | Non |
| JALON | 3 | 1 | 0.5/0.5 | Non | Non |
| PHASE_COMPLETE | 30 | 8 | 0.7/0.3 | Oui | Oui λ=0.7 |
| COMPARAISON | 30 | 10 | 0.6/0.4 | Oui | Oui λ=0.6 |
| CIR | 20 | 5 + mapping | 0.5/0.5 | Oui | Non |

## Tests Prioritaires

```python
# tests/test_retrieval_gri.py

async def test_definition_lookup_exact():
    # "Qu'est-ce qu'un artefact ?" → chunk définition avec term_fr='artefact'
    results = await retrieve("Qu'est-ce qu'un artefact ?")
    assert results[0]["section_type"] == "definition"
    assert "artefact" in results[0]["term_fr"].lower()

async def test_milestone_retrieval_complete():
    # La query CDR doit retourner le jalon M4 entier, pas un fragment
    results = await retrieve("Critères du CDR")
    assert results[0]["milestone_id"] == "M4"
    assert "critères" in results[0]["content"].lower()

async def test_cir_includes_gri_mapping():
    # Une question CIR doit aussi retourner le mapping GRI
    results = await retrieve("Critères du jalon J3")
    ids = [r.get("milestone_id") for r in results]
    assert "J3" in ids
    assert any(m in ids for m in ["M5", "M6"])  # Mapping GRI présent

async def test_sparse_wins_on_iso_terms():
    # Les termes ISO exacts doivent être trouvés même sans contexte sémantique
    dense_only = await retrieve("SEMP", alpha=1.0)
    hybrid = await retrieve("SEMP", alpha=0.6)
    assert hybrid[0]["score"] >= dense_only[0]["score"]

async def test_phase_complete_uses_parent():
    # Phase 3 complète → les chunks retournés doivent avoir le même parent_chunk_id
    results = await retrieve("Objectifs de la Phase 3 Conception")
    if len(results) > 1:
        parent_ids = set(r.get("parent_chunk_id") for r in results)
        assert len(parent_ids) <= 2  # Max 2 parents différents
```
