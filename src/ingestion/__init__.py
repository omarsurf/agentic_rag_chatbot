"""Pipeline d'ingestion GRI - Parser, Chunker, Indexer."""

from src.ingestion.models import (
    CIR_GRI_MAPPING,
    Cycle,
    GlossaryEntry,
    GRIChunk,
    GRIMetadata,
    IngestionResult,
    ParsedSection,
    ParsedTable,
    SectionType,
)
from src.ingestion.chunker import GRIChunker
from src.ingestion.glossary_extractor import GRIGlossaryExtractor
from src.ingestion.parser import GRIDocxParser
from src.ingestion.pipeline import GRIIngestionPipeline
from src.ingestion.table_extractor import GRITableExtractor

__all__ = [
    # Models
    "CIR_GRI_MAPPING",
    "Cycle",
    "GlossaryEntry",
    "GRIChunk",
    "GRIMetadata",
    "IngestionResult",
    "ParsedSection",
    "ParsedTable",
    "SectionType",
    # Components
    "GRIChunker",
    "GRIDocxParser",
    "GRIGlossaryExtractor",
    "GRIIngestionPipeline",
    "GRITableExtractor",
]
