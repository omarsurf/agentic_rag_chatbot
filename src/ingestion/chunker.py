"""Chunker multi-stratégies pour le document GRI.

7 stratégies de chunking selon le type de contenu :
- definition: 1 terme = 1 chunk (150-300 tokens)
- principle: 1 principe = 1 chunk (400-600 tokens)
- phase: Parent Document Retriever (parent 2048, enfants 512)
- milestone: Jalon COMPLET = 1 chunk (600-900 tokens) - NE JAMAIS COUPER
- process: 1 activité = 1 chunk (400-600 tokens)
- cir: Section CIR avec mapping GRI (400-600 tokens)
- table: Row-by-row + synthèse

RÈGLE N°1: Chaque chunk DOIT commencer par son context_prefix.
"""

import hashlib
import re
import uuid
from datetime import datetime

from src.core.logging import get_logger
from src.core.milestone_utils import extract_milestone_id
from src.ingestion.models import (
    CIR_GRI_MAPPING,
    Cycle,
    GlossaryEntry,
    GRIChunk,
    GRIMetadata,
    ParsedSection,
    ParsedTable,
    SectionType,
)

log = get_logger(__name__)


class GRIChunker:
    """Chunker spécialisé pour le document GRI.

    Implémente les 7 stratégies de chunking définies dans ingestion_skill.md.
    """

    # Configuration par type de section
    CHUNK_CONFIG = {
        SectionType.DEFINITION: {
            "min_size": 150,
            "max_size": 300,
            "overlap": 0,
        },
        SectionType.PRINCIPLE: {
            "min_size": 400,
            "max_size": 600,
            "overlap": 0,
        },
        SectionType.PHASE: {
            "parent_size": 2048,
            "child_size": 512,
            "child_overlap": 64,
        },
        SectionType.MILESTONE: {
            "min_size": 600,
            "max_size": 900,
            "overlap": 0,
            "never_split": True,  # CRITIQUE
        },
        SectionType.PROCESS: {
            "min_size": 400,
            "max_size": 600,
            "overlap": 0,
        },
        SectionType.CIR: {
            "min_size": 400,
            "max_size": 600,
            "overlap": 0,
        },
        SectionType.TABLE: {
            "row_min_size": 100,
            "row_max_size": 250,
            "summary_min_size": 300,
            "summary_max_size": 500,
        },
    }

    def __init__(self, doc_id: str | None = None) -> None:
        self.doc_id = doc_id or hashlib.sha256(
            uuid.uuid4().hex.encode("utf-8")
        ).hexdigest()[:16]
        self._chunk_index = 0

    def chunk_section(self, section: ParsedSection) -> list[GRIChunk]:
        """API publique: chunk une section unique."""
        return self._chunk_section(section)

    def chunk_sections(self, sections: list[ParsedSection]) -> list[GRIChunk]:
        """Chunk toutes les sections selon leur type."""
        chunks: list[GRIChunk] = []

        for section in sections:
            section_chunks = self._chunk_section(section)
            chunks.extend(section_chunks)

        log.info("gri.chunker.sections_chunked", total_chunks=len(chunks))
        return chunks

    def chunk_tables(self, tables: list[ParsedTable]) -> list[GRIChunk]:
        """Chunk les tables (jalons = chunk unique)."""
        chunks: list[GRIChunk] = []

        for table in tables:
            table_chunks = self._chunk_table(table)
            chunks.extend(table_chunks)

        log.info("gri.chunker.tables_chunked", total_chunks=len(chunks))
        return chunks

    def chunk_glossary(self, entries: list[GlossaryEntry]) -> list[GRIChunk]:
        """Chunk le glossaire (1 terme = 1 chunk)."""
        chunks: list[GRIChunk] = []

        for entry in entries:
            chunk = self._chunk_glossary_entry(entry)
            if chunk:
                chunks.append(chunk)

        log.info("gri.chunker.glossary_chunked", total_chunks=len(chunks))
        return chunks

    def _get_config_for_type(self, section_type: SectionType) -> dict[str, int | bool]:
        """Expose une configuration compatible pour les tests et l'inspection."""
        config = self.CHUNK_CONFIG[section_type]

        if section_type == SectionType.PHASE:
            return {
                "min_tokens": config["child_size"],
                "max_tokens": config["child_size"],
                "overlap": config["child_overlap"],
            }

        if section_type == SectionType.TABLE:
            return {
                "min_tokens": config["row_min_size"],
                "max_tokens": config["row_max_size"],
                "overlap": 0,
            }

        result: dict[str, int | bool] = {
            "min_tokens": config["min_size"],
            "max_tokens": config["max_size"],
            "overlap": config["overlap"],
        }
        if "never_split" in config:
            result["never_split"] = config["never_split"]
        return result

    def _chunk_section(self, section: ParsedSection) -> list[GRIChunk]:
        """Chunk une section selon son type."""
        if section.section_type == SectionType.DEFINITION:
            return self._chunk_definition_section(section)
        elif section.section_type == SectionType.PRINCIPLE:
            return self._chunk_principle_section(section)
        elif section.section_type == SectionType.PHASE:
            return self._chunk_phase_section(section)
        elif section.section_type == SectionType.MILESTONE:
            return self._chunk_milestone_section(section)
        elif section.section_type == SectionType.PROCESS:
            return self._chunk_process_section(section)
        elif section.section_type == SectionType.CIR:
            return self._chunk_cir_section(section)
        else:
            return self._chunk_generic_section(section)

    def _chunk_definition_section(self, section: ParsedSection) -> list[GRIChunk]:
        """Chunk les définitions (sera géré par chunk_glossary)."""
        # Les définitions sont extraites séparément via le glossaire
        # Ici on garde juste le titre de la section
        if not section.content:
            return []

        context_prefix = self._build_prefix(section.hierarchy)
        content = f"{context_prefix}\n\n{section.title}\n\n{section.content}"

        # Skip chunks too short (min 80 chars required by GRIChunk)
        if len(content) < 80:
            return []

        return [self._create_chunk(content, section, SectionType.DEFINITION)]

    def _chunk_principle_section(self, section: ParsedSection) -> list[GRIChunk]:
        """Chunk les principes (1 principe = 1 chunk complet)."""
        if not section.content and not section.title:
            return []

        # Extraire le numéro de principe
        principle_num = self._extract_principle_num(section.title)

        context_prefix = self._build_prefix(section.hierarchy)
        content = f"{context_prefix}\n\n{section.title}\n\n{section.content}"

        # Skip chunks too short (min 80 chars required by GRIChunk)
        if len(content) < 80:
            return []

        chunk = self._create_chunk(
            content,
            section,
            SectionType.PRINCIPLE,
            principle_num=principle_num,
        )
        return [chunk]

    def _chunk_phase_section(self, section: ParsedSection) -> list[GRIChunk]:
        """Chunk les phases avec Parent Document Retriever."""
        chunks: list[GRIChunk] = []

        if not section.content and not section.title:
            return chunks

        phase_num = self._extract_phase_num(section.title)
        context_prefix = self._build_prefix(section.hierarchy)

        # Parent chunk (phase entière)
        full_content = f"{context_prefix}\n\n{section.title}\n\n{section.content}"

        # Skip chunks too short (min 80 chars required by GRIChunk)
        if len(full_content) < 80:
            return chunks

        parent_chunk = self._create_chunk(
            full_content,
            section,
            SectionType.PHASE,
            phase_num=phase_num,
        )
        chunks.append(parent_chunk)

        # Enfants (sous-sections) si le contenu est assez long
        config = self.CHUNK_CONFIG[SectionType.PHASE]
        if len(section.content) > config["child_size"] * 2:
            child_chunks = self._split_into_children(
                section,
                parent_chunk.chunk_id,
                config["child_size"],
                config["child_overlap"],
            )
            chunks.extend(child_chunks)

        return chunks

    def _chunk_milestone_section(self, section: ParsedSection) -> list[GRIChunk]:
        """Chunk les jalons - NE JAMAIS FRAGMENTER."""
        if not section.content and not section.title:
            return []

        milestone_id = extract_milestone_id(
            section.title + " " + section.content,
            hierarchy=section.hierarchy,
        )
        context_prefix = self._build_prefix(section.hierarchy)

        # RÈGLE CRITIQUE: Le jalon entier = 1 seul chunk
        content = f"{context_prefix}\n\n{section.title}\n\n{section.content}"

        # Skip chunks too short (min 80 chars required by GRIChunk)
        if len(content) < 80:
            return []

        chunk = self._create_chunk(
            content,
            section,
            SectionType.MILESTONE,
            milestone_id=milestone_id,
        )

        log.info(
            "gri.chunker.milestone",
            milestone_id=milestone_id,
            chunk_id=chunk.chunk_id,
        )

        return [chunk]

    def _chunk_process_section(self, section: ParsedSection) -> list[GRIChunk]:
        """Chunk les processus IS 15288."""
        if not section.content and not section.title:
            return []

        process_name = self._extract_process_name(section.title)
        context_prefix = self._build_prefix(section.hierarchy)
        content = f"{context_prefix}\n\n{section.title}\n\n{section.content}"

        # Skip chunks too short (min 80 chars required by GRIChunk)
        if len(content) < 80:
            return []

        chunk = self._create_chunk(
            content,
            section,
            SectionType.PROCESS,
            process_name=process_name,
        )
        return [chunk]

    def _chunk_cir_section(self, section: ParsedSection) -> list[GRIChunk]:
        """Chunk les sections CIR avec mapping GRI."""
        if not section.content and not section.title:
            return []

        cir_phase = self._extract_cir_phase(section.title)
        milestone_id = extract_milestone_id(
            section.title + " " + section.content,
            hierarchy=section.hierarchy,
        )

        # Mapping GRI automatique
        gri_equivalent = None
        if milestone_id and milestone_id in CIR_GRI_MAPPING:
            gri_equivalent = CIR_GRI_MAPPING[milestone_id]

        context_prefix = self._build_prefix(section.hierarchy, cycle=Cycle.CIR)
        content = f"{context_prefix}\n\n{section.title}\n\n{section.content}"

        # Ajouter le mapping dans le contenu
        if gri_equivalent:
            content += f"\n\nÉquivalent GRI : {', '.join(gri_equivalent)}"

        # Skip chunks too short (min 80 chars required by GRIChunk)
        if len(content) < 80:
            return []

        chunk = self._create_chunk(
            content,
            section,
            SectionType.CIR,
            cycle=Cycle.CIR,
            cir_phase=cir_phase,
            milestone_id=milestone_id,
            gri_equivalent=gri_equivalent,
        )
        return [chunk]

    def _chunk_generic_section(self, section: ParsedSection) -> list[GRIChunk]:
        """Chunk générique pour les autres types."""
        if not section.content and not section.title:
            return []

        context_prefix = self._build_prefix(section.hierarchy)
        content = f"{context_prefix}\n\n{section.title}\n\n{section.content}"

        # Skip chunks too short (min 80 chars required by GRIChunk)
        if len(content) < 80:
            return []

        chunk = self._create_chunk(content, section, SectionType.CONTENT)
        return [chunk]

    def _chunk_table(self, table: ParsedTable) -> list[GRIChunk]:
        """Chunk une table."""
        chunks: list[GRIChunk] = []

        if table.table_type == "milestone_criteria":
            # Table de jalon = 1 seul chunk complet
            chunk = self._chunk_milestone_table(table)
            if chunk:
                chunks.append(chunk)
        else:
            # Autres tables: synthèse + lignes si nécessaire
            summary_chunk = self._chunk_table_summary(table)
            if summary_chunk:
                chunks.append(summary_chunk)

        return chunks

    def _chunk_milestone_table(self, table: ParsedTable) -> GRIChunk | None:
        """Chunk une table de jalon (COMPLET, jamais fragmenté)."""
        if not table.rows:
            return None

        milestone_id = table.milestone_id
        hierarchy = ["GRI", "Jalons", f"{milestone_id}" if milestone_id else "Table"]
        context_prefix = self._build_prefix(hierarchy)

        # Formater les critères
        lines = [f"{context_prefix}\n"]
        lines.append(f"Critères de passage du jalon {milestone_id or 'N/A'}\n")

        for i, row in enumerate(table.rows, 1):
            criterion = (
                row.get("Critère")
                or row.get("Critères")
                or row.get("Description")
                or " | ".join(str(v) for v in row.values())
            )
            lines.append(f"{i}. {criterion}")

        content = "\n".join(lines)

        metadata = GRIMetadata(
            doc_id=self.doc_id,
            chunk_index=self._next_index(),
            section_type=SectionType.MILESTONE,
            hierarchy=hierarchy,
            context_prefix=context_prefix,
            cycle=Cycle.CIR if milestone_id and milestone_id.startswith("J") else Cycle.GRI,
            milestone_id=milestone_id,
            char_count=len(content),
        )

        chunk_id = self._hash_content(content)

        log.info(
            "gri.chunker.milestone_table",
            milestone_id=milestone_id,
            criteria_count=len(table.rows),
        )

        return GRIChunk(
            chunk_id=chunk_id,
            content=content,
            metadata=metadata,
        )

    def _chunk_table_summary(self, table: ParsedTable) -> GRIChunk | None:
        """Crée un chunk de synthèse pour une table."""
        if not table.full_text:
            return None

        parent_section = table.parent_section or "Document"
        hierarchy = ["GRI", parent_section, f"Tableau {table.table_index}"]
        context_prefix = self._build_prefix(hierarchy)
        content = f"{context_prefix}\n\n{table.full_text}"

        metadata = GRIMetadata(
            doc_id=self.doc_id,
            chunk_index=self._next_index(),
            section_type=SectionType.TABLE,
            hierarchy=hierarchy,
            context_prefix=context_prefix,
            table_id=f"table_{table.table_index}",
            char_count=len(content),
        )

        chunk_id = self._hash_content(content)

        return GRIChunk(
            chunk_id=chunk_id,
            content=content,
            metadata=metadata,
        )

    def _chunk_glossary_entry(self, entry: GlossaryEntry) -> GRIChunk | None:
        """Chunk une entrée du glossaire."""
        hierarchy = ["GRI", "Terminologie", entry.term_fr]
        context_prefix = self._build_prefix(hierarchy)

        lines = [context_prefix, ""]
        lines.append(f"**{entry.term_fr}**")
        if entry.term_en:
            lines.append(f"({entry.term_en})")
        lines.append("")
        lines.append(entry.definition_fr)
        if entry.definition_en:
            lines.append("")
            lines.append(f"EN: {entry.definition_en}")
        if entry.standard_ref:
            lines.append("")
            lines.append(f"Source: {entry.standard_ref}")

        content = "\n".join(lines)

        # Ignorer les entrées trop courtes (min 80 chars requis par GRIChunk)
        if len(content) < 80:
            log.debug(
                "gri.chunker.glossary_entry_too_short",
                term=entry.term_fr,
                length=len(content),
            )
            return None

        metadata = GRIMetadata(
            doc_id=self.doc_id,
            chunk_index=self._next_index(),
            section_type=SectionType.DEFINITION,
            hierarchy=hierarchy,
            context_prefix=context_prefix,
            term_fr=entry.term_fr,
            term_en=entry.term_en,
            standard_ref=entry.standard_ref,
            char_count=len(content),
        )

        chunk_id = self._hash_content(content)

        return GRIChunk(
            chunk_id=chunk_id,
            content=content,
            metadata=metadata,
        )

    def _split_into_children(
        self,
        section: ParsedSection,
        parent_id: str,
        child_size: int,
        overlap: int,
    ) -> list[GRIChunk]:
        """Divise une section en chunks enfants."""
        chunks: list[GRIChunk] = []
        text = section.content
        sentences = self._split_sentences(text)

        current_chunk: list[str] = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence) // 4  # Estimation tokens

            if current_size + sentence_size > child_size and current_chunk:
                # Créer le chunk
                chunk_text = " ".join(current_chunk)
                context_prefix = self._build_prefix(section.hierarchy)
                content = f"{context_prefix}\n\n{chunk_text}"

                metadata = GRIMetadata(
                    doc_id=self.doc_id,
                    chunk_index=self._next_index(),
                    section_type=section.section_type,
                    hierarchy=section.hierarchy,
                    context_prefix=context_prefix,
                    parent_chunk_id=parent_id,
                    char_count=len(content),
                )

                chunk_id = self._hash_content(content)
                chunks.append(GRIChunk(
                    chunk_id=chunk_id,
                    content=content,
                    metadata=metadata,
                ))

                # Overlap: garder les dernières phrases
                overlap_sentences = current_chunk[-2:] if overlap > 0 else []
                current_chunk = overlap_sentences
                current_size = sum(len(s) // 4 for s in current_chunk)

            current_chunk.append(sentence)
            current_size += sentence_size

        # Dernier chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            context_prefix = self._build_prefix(section.hierarchy)
            content = f"{context_prefix}\n\n{chunk_text}"

            metadata = GRIMetadata(
                doc_id=self.doc_id,
                chunk_index=self._next_index(),
                section_type=section.section_type,
                hierarchy=section.hierarchy,
                context_prefix=context_prefix,
                parent_chunk_id=parent_id,
                char_count=len(content),
            )

            chunk_id = self._hash_content(content)
            chunks.append(GRIChunk(
                chunk_id=chunk_id,
                content=content,
                metadata=metadata,
            ))

        return chunks

    def _create_chunk(
        self,
        content: str,
        section: ParsedSection,
        section_type: SectionType,
        cycle: Cycle = Cycle.GRI,
        phase_num: int | None = None,
        cir_phase: int | None = None,
        milestone_id: str | None = None,
        process_name: str | None = None,
        principle_num: int | None = None,
        gri_equivalent: list[str] | None = None,
        parent_chunk_id: str | None = None,
    ) -> GRIChunk:
        """Crée un chunk avec ses métadonnées."""
        context_prefix = self._build_prefix(section.hierarchy, cycle)

        metadata = GRIMetadata(
            doc_id=self.doc_id,
            chunk_index=self._next_index(),
            section_type=section_type,
            hierarchy=section.hierarchy,
            context_prefix=context_prefix,
            cycle=cycle,
            phase_num=phase_num,
            cir_phase=cir_phase,
            milestone_id=milestone_id,
            process_name=process_name,
            principle_num=principle_num,
            gri_equivalent=gri_equivalent,
            parent_chunk_id=parent_chunk_id,
            char_count=len(content),
        )

        chunk_id = self._hash_content(content)

        return GRIChunk(
            chunk_id=chunk_id,
            content=content,
            metadata=metadata,
        )

    def _build_prefix(
        self,
        hierarchy: list[str],
        cycle: Cycle = Cycle.GRI,
    ) -> str:
        """Construit le context prefix."""
        if not hierarchy:
            return f"[{cycle.value}]"

        # S'assurer que le cycle est en premier
        if hierarchy[0] not in ("GRI", "CIR"):
            hierarchy = [cycle.value] + hierarchy

        return "[" + " > ".join(hierarchy) + "]"

    def _hash_content(self, content: str) -> str:
        """Hash SHA256 du contenu (16 chars)."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _next_index(self) -> int:
        """Retourne le prochain index de chunk."""
        idx = self._chunk_index
        self._chunk_index += 1
        return idx

    def _split_sentences(self, text: str) -> list[str]:
        """Divise le texte en phrases."""
        # Pattern simple pour le français
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _extract_phase_num(self, text: str) -> int | None:
        """Extrait le numéro de phase."""
        match = re.search(r"Phase\s+(\d+)", text, re.IGNORECASE)
        if match:
            num = int(match.group(1))
            return num if 1 <= num <= 7 else None
        return None

    def _extract_cir_phase(self, text: str) -> int | None:
        """Extrait le numéro de phase CIR."""
        match = re.search(r"Phase\s+(\d+)", text, re.IGNORECASE)
        if match:
            num = int(match.group(1))
            return num if 1 <= num <= 4 else None
        return None

    def _extract_process_name(self, text: str) -> str | None:
        """Extrait le nom du processus."""
        # Pattern: "Processus de X" ou "Process X"
        match = re.search(
            r"(?:Processus|Process)\s+(?:de\s+)?([^:—–\n]+)",
            text,
            re.IGNORECASE,
        )
        return match.group(1).strip() if match else None

    def _extract_principle_num(self, text: str) -> int | None:
        """Extrait le numéro de principe."""
        match = re.search(r"Principe\s+N°\s*(\d+)", text, re.IGNORECASE)
        if match:
            num = int(match.group(1))
            return num if 1 <= num <= 11 else None
        return None
