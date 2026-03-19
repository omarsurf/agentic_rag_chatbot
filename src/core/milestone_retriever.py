"""Milestone Retriever - Retrieval spécialisée pour les jalons GRI/CIR.

Les jalons ont une logique de retrieval particulière :
- On veut TOUJOURS le jalon complet, pas un fragment
- Pour les jalons CIR, on ajoute automatiquement le mapping GRI équivalent

Mapping CIR → GRI :
- J1 → M0 + M1
- J2 → M2 + M3 + M4
- J3 → M5 + M6
- J4 → SAR
- J5 → SAR
- J6 → M8

Usage:
    from src.core.milestone_retriever import GRIMilestoneRetriever

    retriever = GRIMilestoneRetriever(store)
    result = await retriever.get_milestone("M4")
    result_cir = await retriever.get_milestone("J3", include_gri_mapping=True)
"""

import asyncio
from typing import TYPE_CHECKING, Any, Literal

import structlog
from pydantic import BaseModel, Field
from qdrant_client.models import FieldCondition, Filter, MatchValue

from src.core.config import (
    CIR_GRI_MAPPING,
    MILESTONE_ACRONYM_TO_ID,
    VALID_CIR_MILESTONES,
    VALID_MILESTONES,
)

if TYPE_CHECKING:
    from src.core.vector_store import GRIHybridStore

log = structlog.get_logger()


class MilestoneChunk(BaseModel):
    """Chunk de jalon avec ses métadonnées."""

    content: str
    milestone_id: str
    milestone_name: str | None = None
    cycle: Literal["GRI", "CIR"]
    phase_num: int | None = None
    criteria_count: int | None = None
    context_prefix: str | None = None
    score: float = 1.0
    role: str = "primary"  # "primary" | "gri_equivalent"
    metadata: dict[str, Any] = Field(default_factory=dict)


class MilestoneResult(BaseModel):
    """Résultat complet de retrieval de jalon."""

    milestone_id: str
    cycle: Literal["GRI", "CIR"]
    is_cir: bool = False
    chunks: list[MilestoneChunk] = Field(default_factory=list)
    gri_equivalents: list[str] = Field(default_factory=list)
    gri_chunks: list[MilestoneChunk] = Field(default_factory=list)
    found: bool = False
    is_complete: bool = False


# Noms des jalons GRI
MILESTONE_NAMES: dict[str, str] = {
    "M0": "Mission Needs Statement (MNS)",
    "M1": "System Requirements Review (SRR)",
    "M2": "Preliminary Design Review (PDR)",
    "M3": "Critical Design Review (CDR)",
    "M4": "Integration Readiness Review (IRR)",
    "M5": "Test Readiness Review (TRR)",
    "M6": "System Acceptance Review (SAR)",
    "M7": "Operational Readiness Review (ORR)",
    "M8": "Mission Needs Review (MNR)",
    "M9": "System Retirement Review",
    "J1": "Jalon CIR 1 - Lancement",
    "J2": "Jalon CIR 2 - Conception",
    "J3": "Jalon CIR 3 - Intégration",
    "J4": "Jalon CIR 4 - Qualification",
    "J5": "Jalon CIR 5 - Acceptation",
    "J6": "Jalon CIR 6 - Clôture",
}


class GRIMilestoneRetriever:
    """Retriever spécialisé pour les jalons GRI/CIR.

    Garantit de récupérer le jalon complet et gère automatiquement
    le mapping entre jalons CIR et GRI.

    Attributes:
        store: GRIHybridStore pour les lookups
    """

    def __init__(self, store: "GRIHybridStore") -> None:
        """Initialise le retriever de jalons.

        Args:
            store: Vector store pour les lookups
        """
        self.store = store

        log.info("milestone_retriever.init")

    def validate_milestone_id(self, milestone_id: str) -> tuple[bool, str]:
        """Valide un ID de jalon.

        Args:
            milestone_id: ID du jalon (ex: "M4", "J3")

        Returns:
            Tuple (is_valid, normalized_id)
        """
        # Normaliser (majuscules)
        normalized = milestone_id.upper().strip()

        # Gérer les acronymes de revue (CDR -> M3, PDR -> M2, etc.)
        # Utilise le mapping centralisé de config.py
        if normalized in MILESTONE_ACRONYM_TO_ID:
            normalized = MILESTONE_ACRONYM_TO_ID[normalized]

        is_valid = normalized in VALID_MILESTONES
        return is_valid, normalized

    def get_gri_equivalents(self, milestone_id: str) -> list[str]:
        """Retourne les jalons GRI équivalents pour un jalon CIR.

        Args:
            milestone_id: ID du jalon CIR (ex: "J3")

        Returns:
            Liste des IDs GRI équivalents
        """
        return CIR_GRI_MAPPING.get(milestone_id.upper(), [])

    async def get_milestone(
        self,
        milestone_id: str,
        include_gri_mapping: bool = True,
    ) -> MilestoneResult:
        """Récupère un jalon complet avec tous ses critères.

        Pour les jalons CIR, inclut automatiquement le mapping vers
        les jalons GRI équivalents si demandé.

        Args:
            milestone_id: ID du jalon (M0-M9 ou J1-J6)
            include_gri_mapping: Inclure les équivalents GRI pour les jalons CIR

        Returns:
            MilestoneResult avec chunks et mapping
        """
        # Valider l'ID
        is_valid, normalized_id = self.validate_milestone_id(milestone_id)

        if not is_valid:
            log.warning(
                "milestone_retriever.invalid_id",
                milestone_id=milestone_id,
            )
            return MilestoneResult(
                milestone_id=milestone_id,
                cycle="GRI",
                found=False,
            )

        is_cir = normalized_id in VALID_CIR_MILESTONES
        cycle: Literal["GRI", "CIR"] = "CIR" if is_cir else "GRI"

        log.info(
            "milestone_retriever.get_milestone",
            milestone_id=normalized_id,
            cycle=cycle,
            include_gri_mapping=include_gri_mapping,
        )

        # Récupérer le jalon principal
        main_chunks = await self._fetch_milestone_chunks(normalized_id)

        # Récupérer les équivalents GRI si CIR
        gri_equivalents: list[str] = []
        gri_chunks: list[MilestoneChunk] = []

        if is_cir and include_gri_mapping:
            gri_equivalents = self.get_gri_equivalents(normalized_id)

            if gri_equivalents:
                # Récupérer tous les jalons GRI en parallèle
                tasks = [
                    self._fetch_milestone_chunks(gri_id) for gri_id in gri_equivalents
                ]
                results = await asyncio.gather(*tasks)

                for gri_id, chunks in zip(gri_equivalents, results, strict=False):
                    for chunk in chunks:
                        chunk.role = f"gri_equivalent ({gri_id})"
                        gri_chunks.append(chunk)

        return MilestoneResult(
            milestone_id=normalized_id,
            cycle=cycle,
            is_cir=is_cir,
            chunks=main_chunks,
            gri_equivalents=gri_equivalents,
            gri_chunks=gri_chunks,
            found=len(main_chunks) > 0,
            is_complete=self._check_completeness(main_chunks),
        )

    async def _fetch_milestone_chunks(
        self,
        milestone_id: str,
    ) -> list[MilestoneChunk]:
        """Récupère les chunks d'un jalon depuis le store.

        Args:
            milestone_id: ID du jalon

        Returns:
            Liste de MilestoneChunk
        """
        try:
            # Lookup direct par milestone_id
            results, _ = self.store.client.scroll(
                collection_name=self.store.COLLECTIONS["main"],
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="milestone_id",
                            match=MatchValue(value=milestone_id),
                        )
                    ]
                ),
                limit=10,
                with_payload=True,
            )

            chunks = []
            for point in results:
                payload = point.payload
                chunks.append(
                    MilestoneChunk(
                        content=payload.get("content", ""),
                        milestone_id=milestone_id,
                        milestone_name=MILESTONE_NAMES.get(milestone_id),
                        cycle="CIR" if milestone_id.startswith("J") else "GRI",
                        phase_num=payload.get("phase_num"),
                        context_prefix=payload.get("context_prefix"),
                        score=1.0,
                        role="primary",
                        metadata=payload,
                    )
                )

            log.info(
                "milestone_retriever.fetched",
                milestone_id=milestone_id,
                n_chunks=len(chunks),
            )

            return chunks

        except Exception as e:
            log.error(
                "milestone_retriever.fetch_failed",
                milestone_id=milestone_id,
                error=str(e),
            )
            return []

    def _check_completeness(self, chunks: list[MilestoneChunk]) -> bool:
        """Vérifie si les chunks contiennent le jalon complet.

        Un jalon est complet s'il contient au moins un chunk avec
        "critères" dans le contenu.

        Args:
            chunks: Liste de chunks

        Returns:
            True si complet
        """
        if not chunks:
            return False

        for chunk in chunks:
            content_lower = chunk.content.lower()
            if "critère" in content_lower or "criteria" in content_lower:
                return True

        return False

    async def search_milestones_by_query(
        self,
        query: str,
        n_results: int = 3,
    ) -> list[MilestoneResult]:
        """Recherche des jalons par query sémantique.

        Utile quand l'utilisateur ne spécifie pas d'ID de jalon précis.

        Args:
            query: Query de recherche
            n_results: Nombre de jalons à retourner

        Returns:
            Liste de MilestoneResult
        """
        # Recherche dans les chunks de type milestone
        results = await self.store.hybrid_search(
            query=query,
            collection="main",
            n_results=n_results * 2,  # Over-fetch pour dédupliquer
            filters={"section_type": "milestone"},
            alpha=0.5,
        )

        # Dédupliquer par milestone_id
        seen_ids: set[str] = set()
        milestone_results: list[MilestoneResult] = []

        for result in results:
            milestone_id = result.milestone_id
            if milestone_id and milestone_id not in seen_ids:
                seen_ids.add(milestone_id)

                # Récupérer le jalon complet
                full_result = await self.get_milestone(milestone_id)
                milestone_results.append(full_result)

                if len(milestone_results) >= n_results:
                    break

        return milestone_results


# Singleton pour usage global
_retriever: GRIMilestoneRetriever | None = None


def get_milestone_retriever(store: "GRIHybridStore") -> GRIMilestoneRetriever:
    """Retourne le singleton du milestone retriever."""
    global _retriever
    if _retriever is None:
        _retriever = GRIMilestoneRetriever(store)
    return _retriever


async def get_jalon_complet(
    milestone_id: str,
    store: "GRIHybridStore",
    include_gri_mapping: bool = True,
) -> MilestoneResult:
    """Fonction helper pour récupérer un jalon complet.

    Args:
        milestone_id: ID du jalon
        store: Vector store
        include_gri_mapping: Inclure les équivalents GRI pour CIR

    Returns:
        MilestoneResult
    """
    retriever = get_milestone_retriever(store)
    return await retriever.get_milestone(milestone_id, include_gri_mapping)
