"""Pipeline d'ingestion complet pour le GRI.

Orchestration:
1. Parse DOCX → sections + tables
2. Extraction glossaire
3. Chunking par type (7 stratégies)
4. Validation des chunks
5. Double indexation (Qdrant dense + BM25 sparse)
"""

import hashlib
import time
from pathlib import Path

from src.core.config import settings
from src.core.logging import get_logger
from src.ingestion.chunker import GRIChunker
from src.ingestion.glossary_extractor import GRIGlossaryExtractor
from src.ingestion.models import (
    GRIChunk,
    IngestionResult,
    ParsedSection,
    ParsedTable,
    SectionType,
)
from src.ingestion.parser import GRIDocxParser
from src.ingestion.table_extractor import GRITableExtractor

log = get_logger(__name__)


class GRIIngestionPipeline:
    """Pipeline d'ingestion complet pour le document GRI.

    Usage:
        pipeline = GRIIngestionPipeline()
        result = pipeline.run("data/raw/IRF20251211_last_FF.docx")
        print(f"Indexed {result.valid_chunks} chunks")
    """

    def __init__(
        self,
        vector_store=None,
        output_dir: str | Path | None = None,
    ) -> None:
        """Initialise le pipeline.

        Args:
            vector_store: Instance du vector store (optionnel)
            output_dir: Répertoire de sortie pour les fichiers intermédiaires
        """
        self.parser = GRIDocxParser()
        self.table_extractor = GRITableExtractor()
        self.glossary_extractor = GRIGlossaryExtractor()
        self.vector_store = vector_store
        self.output_dir = Path(output_dir) if output_dir else Path("data/processed")

    def run(self, docx_path: str | Path) -> IngestionResult:
        """Exécute le pipeline d'ingestion complet.

        Args:
            docx_path: Chemin vers le document GRI (.docx)

        Returns:
            IngestionResult avec les statistiques d'ingestion
        """
        start_time = time.time()
        path = Path(docx_path)

        log.info("gri.pipeline.start", doc_path=str(path))

        # Calculer le hash du document
        doc_id = self._hash_file(path)
        log.info("gri.pipeline.doc_id", doc_id=doc_id)

        warnings: list[str] = []
        errors: list[str] = []

        # === ÉTAPE 1: Parse DOCX ===
        log.info("gri.pipeline.step", step="parse_docx")
        try:
            sections, tables = self.parser.parse(path)
        except Exception as e:
            errors.append(f"Erreur parsing DOCX: {e}")
            return self._error_result(doc_id, str(path), errors, start_time)

        # === ÉTAPE 2: Extraction tables séparée ===
        log.info("gri.pipeline.step", step="extract_tables")
        try:
            tables_detailed = self.table_extractor.extract(path)
            # Fusionner avec les tables du parser si nécessaire
            if len(tables_detailed) > len(tables):
                tables = tables_detailed
        except Exception as e:
            warnings.append(f"Erreur extraction tables: {e}")

        # === ÉTAPE 3: Extraction glossaire ===
        log.info("gri.pipeline.step", step="extract_glossary")
        try:
            glossary_entries = self.glossary_extractor.extract_from_sections(sections)
            # Sauvegarder le glossaire
            glossary_path = self.output_dir / "glossary_gri.json"
            self.glossary_extractor.save_to_json(glossary_entries, glossary_path)
        except Exception as e:
            warnings.append(f"Erreur extraction glossaire: {e}")
            glossary_entries = []

        # === ÉTAPE 4: Chunking ===
        log.info("gri.pipeline.step", step="chunking")
        chunker = GRIChunker(doc_id)

        all_chunks: list[GRIChunk] = []

        # Chunks des sections
        section_chunks = chunker.chunk_sections(sections)
        all_chunks.extend(section_chunks)

        # Chunks des tables
        table_chunks = chunker.chunk_tables(tables)
        all_chunks.extend(table_chunks)

        # Chunks du glossaire
        glossary_chunks = chunker.chunk_glossary(glossary_entries)
        all_chunks.extend(glossary_chunks)

        # === ÉTAPE 5: Validation ===
        log.info("gri.pipeline.step", step="validation")
        valid_chunks = []
        invalid_count = 0

        for chunk in all_chunks:
            is_valid, reason = self._validate_chunk(chunk)
            if is_valid:
                valid_chunks.append(chunk)
            else:
                invalid_count += 1
                if invalid_count <= 10:  # Limiter les warnings
                    warnings.append(f"Chunk invalide ({chunk.chunk_id}): {reason}")

        # === ÉTAPE 6: Indexation ===
        log.info("gri.pipeline.step", step="indexation")
        if self.vector_store:
            try:
                self._index_chunks(valid_chunks)
            except Exception as e:
                errors.append(f"Erreur indexation: {e}")

        # === ÉTAPE 7: Sauvegarder les chunks ===
        self._save_chunks(valid_chunks)

        # === Statistiques ===
        duration = time.time() - start_time

        chunks_by_type = self._count_by_type(valid_chunks)
        milestones_found = self._extract_milestones(valid_chunks)

        result = IngestionResult(
            doc_id=doc_id,
            doc_path=str(path),
            total_chunks=len(all_chunks),
            valid_chunks=len(valid_chunks),
            invalid_chunks=invalid_count,
            chunks_by_type=chunks_by_type,
            glossary_terms=len(glossary_entries),
            tables_extracted=len(tables),
            milestones_found=milestones_found,
            warnings=warnings,
            errors=errors,
            duration_seconds=duration,
            chunks=valid_chunks,
        )

        log.info(
            "gri.pipeline.complete",
            doc_id=doc_id,
            total_chunks=result.total_chunks,
            valid_chunks=result.valid_chunks,
            milestones=len(milestones_found),
            duration_s=f"{duration:.2f}",
        )

        return result

    def _validate_chunk(self, chunk: GRIChunk) -> tuple[bool, str]:
        """Valide un chunk selon les règles GRI.

        Returns:
            Tuple (is_valid, reason)
        """
        # Règle 1: Longueur minimale
        if len(chunk.content) < 80:
            return False, "Contenu trop court (<80 chars)"

        # Règle 2: Context prefix obligatoire
        if not chunk.content.startswith(("[GRI", "[CIR")):
            return False, "Context prefix manquant"

        # Règle 3: Taille maximale
        if chunk.metadata.token_estimate > 1000:
            return False, f"Trop long ({chunk.metadata.token_estimate} tokens)"

        # Règle 4: Métadonnées requises
        required_fields = ["doc_id", "source", "section_type", "context_prefix", "cycle"]
        for field in required_fields:
            if not getattr(chunk.metadata, field, None):
                return False, f"Métadonnée manquante: {field}"

        # Règle 5: Cohérence cycle/prefix
        if chunk.metadata.cycle.value == "CIR" and "[GRI" in chunk.content[:20]:
            return False, "Incohérence cycle CIR avec prefix GRI"

        return True, ""

    def _index_chunks(self, chunks: list[GRIChunk]) -> None:
        """Indexe les chunks dans le vector store."""
        if not self.vector_store:
            log.warning("gri.pipeline.no_vector_store")
            return

        import asyncio

        # Convertir les chunks en format dict pour le vector store
        chunk_dicts = [
            {
                "chunk_id": c.chunk_id,
                "content": c.content,
                "metadata": c.metadata.model_dump(mode="json"),
            }
            for c in chunks
        ]

        # Séparer les définitions du reste
        main_chunks = [c for c in chunk_dicts if c["metadata"].get("section_type") != "definition"]
        glossary_chunks = [c for c in chunk_dicts if c["metadata"].get("section_type") == "definition"]

        # Indexer dans Qdrant (async)
        async def _do_index():
            # Créer les collections si elles n'existent pas
            await self.vector_store.ensure_collections()

            if main_chunks:
                await self.vector_store.index_chunks(main_chunks, collection="main")
                log.info("gri.pipeline.indexed_main", count=len(main_chunks))
            if glossary_chunks:
                await self.vector_store.index_chunks(glossary_chunks, collection="glossary")
                log.info("gri.pipeline.indexed_glossary", count=len(glossary_chunks))

        asyncio.run(_do_index())

    def _save_chunks(self, chunks: list[GRIChunk]) -> None:
        """Sauvegarde les chunks en JSON."""
        import json

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "chunks.json"

        data = [
            {
                "chunk_id": c.chunk_id,
                "content": c.content,
                "metadata": c.metadata.model_dump(mode="json"),
            }
            for c in chunks
        ]

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        log.info("gri.pipeline.chunks_saved", path=str(output_path), count=len(chunks))

    def _hash_file(self, path: Path) -> str:
        """Calcule le hash SHA256 du fichier (16 chars)."""
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

    def _count_by_type(self, chunks: list[GRIChunk]) -> dict[str, int]:
        """Compte les chunks par type de section."""
        counts: dict[str, int] = {}
        for chunk in chunks:
            section_type = chunk.metadata.section_type.value
            counts[section_type] = counts.get(section_type, 0) + 1
        return counts

    def _extract_milestones(self, chunks: list[GRIChunk]) -> list[str]:
        """Extrait la liste des jalons trouvés."""
        milestones: set[str] = set()
        for chunk in chunks:
            if chunk.metadata.milestone_id:
                milestones.add(chunk.metadata.milestone_id)
        return sorted(milestones)

    def _error_result(
        self,
        doc_id: str,
        doc_path: str,
        errors: list[str],
        start_time: float,
    ) -> IngestionResult:
        """Crée un résultat d'erreur."""
        return IngestionResult(
            doc_id=doc_id,
            doc_path=doc_path,
            total_chunks=0,
            valid_chunks=0,
            invalid_chunks=0,
            chunks_by_type={},
            glossary_terms=0,
            tables_extracted=0,
            milestones_found=[],
            warnings=[],
            errors=errors,
            duration_seconds=time.time() - start_time,
            chunks=[],
        )


def main() -> None:
    """Point d'entrée CLI pour l'ingestion."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Pipeline d'ingestion GRI",
    )
    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Chemin vers le document DOCX GRI",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/processed",
        help="Répertoire de sortie",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Afficher les statistiques détaillées",
    )

    args = parser.parse_args()

    # Configurer le logging
    from src.core.logging import setup_logging
    setup_logging(level="INFO", log_format="console")

    # Initialiser le vector store pour l'indexation
    from src.core.vector_store import GRIHybridStore
    vector_store = GRIHybridStore()

    # Exécuter le pipeline avec vector store
    pipeline = GRIIngestionPipeline(vector_store=vector_store, output_dir=args.output)
    result = pipeline.run(args.input)

    # Afficher le résultat
    print("\n" + "=" * 60)
    print("RÉSULTAT DE L'INGESTION")
    print("=" * 60)
    print(f"Document: {result.doc_path}")
    print(f"Doc ID: {result.doc_id}")
    print(f"Durée: {result.duration_seconds:.2f}s")
    print()
    print(f"Chunks totaux: {result.total_chunks}")
    print(f"Chunks valides: {result.valid_chunks}")
    print(f"Chunks invalides: {result.invalid_chunks}")
    print()
    print(f"Termes glossaire: {result.glossary_terms}")
    print(f"Tables extraites: {result.tables_extracted}")
    print(f"Jalons trouvés: {result.milestones_found}")
    print()

    if args.stats:
        print("Chunks par type:")
        for stype, count in sorted(result.chunks_by_type.items()):
            print(f"  - {stype}: {count}")
        print()

    if result.warnings:
        print(f"Warnings ({len(result.warnings)}):")
        for w in result.warnings[:5]:
            print(f"  - {w}")
        if len(result.warnings) > 5:
            print(f"  ... et {len(result.warnings) - 5} autres")
        print()

    if result.errors:
        print(f"ERREURS ({len(result.errors)}):")
        for e in result.errors:
            print(f"  - {e}")
        print()

    # Exit code
    exit(1 if result.errors else 0)


if __name__ == "__main__":
    main()
