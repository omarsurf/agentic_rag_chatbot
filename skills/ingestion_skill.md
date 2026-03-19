---
name: rag-ingestion-gri
description: Pipeline d'ingestion spécialisé pour le GRI des FAR (ISO/IEC/IEEE 15288:2023). Utiliser pour parser, chunker et indexer le document GRI avec préservation de la hiérarchie à 7 niveaux, extraction des tables de jalons, et enrichissement metadata domaine-spécifique. Déclencher pour toute mention de "ingestion GRI", "parser le DOCX", "indexer les sections", "chunking hiérarchique", "extraire les tables de jalons", "pipeline de traitement documents", ou "préparer le RAG GRI".
---

# Ingestion Skill — GRI/FAR

## Spécificités du Document GRI

Le GRI est un document structuré de 22 755 lignes avec des contraintes d'ingestion très précises :

| Caractéristique | Valeur | Impact sur l'ingestion |
|----------------|--------|----------------------|
| Lignes totales | 22 755 | Parsing par batch obligatoire |
| Lignes de tables | 5 092 | Extraction structurée séparée |
| Sections (##) | 430 | Chunking par section, pas fixe |
| Sous-sections (###) | 395 | Hiérarchie 7 niveaux à préserver |
| Termes bilingues | 200+ | Index glossaire dédié |
| Cycles | GRI (7 phases) + CIR (4 phases) | Metadata `cycle` obligatoire |

## Pipeline d'Ingestion GRI

```
IRF20251211_last_FF.docx
        │
        ▼
[1. Parse DOCX]         → python-docx : extraction hiérarchie complète
        │
        ▼
[2. Séparation flux]    → texte courant  /  tables  /  glossaire
        │         │              │
        ▼         ▼              ▼
[3. Chunk par]   [Table     [Glossaire :
   section       extractor]  1 chunk/terme]
        │         │              │
        └─────────┴──────────────┘
                  │
                  ▼
[4. Context Prefix]     → "[GRI > Phase N > Processus X]" préfixé sur chaque chunk
                  │
                  ▼
[5. Metadata enrichment]→ section_type, phase_num, milestone_id, cycle, hierarchy
                  │
                  ▼
[6. Validation]         → longueur, metadata complète, prefix présent
                  │
                  ▼
[7. Double indexation]  → Qdrant (dense) + BM25 (sparse) + glossaire séparé
```

## Stratégies de Chunking par Type de Contenu GRI

### Règle N°1 : Context Prefix OBLIGATOIRE

Chaque chunk doit commencer par son chemin hiérarchique. C'est la règle la plus importante.

```python
# Format du context prefix
context_prefix = "[GRI > Phase 3 > Conception et Développement > Processus de Vérification]"
chunk_content = f"{context_prefix}\n\n{section_text}"
```

Sans ce prefix, le LLM ne peut pas localiser l'information dans le document.

### Stratégie par Type de Section

```python
CHUNKING_STRATEGY = {

    "definition": {
        # Glossaire : 1 terme ISO = 1 chunk avec FR + EN + norme
        "size": (150, 300),    # tokens
        "overlap": 0,          # Pas d'overlap — les définitions sont atomiques
        "boundary": "term",    # Découper sur les frontières de terme
        "prefix_format": "[GRI > Terminologie > {term_fr}]",
        "metadata": ["term_fr", "term_en", "standard_ref"],
    },

    "principle": {
        # Principes 1-11 : 1 principe = 1 chunk complet
        "size": (400, 600),
        "overlap": 0,
        "boundary": "section",  # Un principe ne se coupe jamais
        "prefix_format": "[GRI > Principes > Principe N°{num}]",
        "metadata": ["principle_num", "title"],
    },

    "phase": {
        # Phases 1-7 : Parent Document Retriever
        # Parent = phase entière (pour contexte large)
        # Enfants = sous-sections (pour retrieval précis)
        "parent_size": 2048,
        "child_size": 512,
        "child_overlap": 64,
        "boundary": "sentence",
        "prefix_format": "[GRI > Phase {num} > {phase_title} > {subsection}]",
        "metadata": ["phase_num", "phase_title", "subsection_type"],
        "subsection_types": ["objectives", "activities", "deliverables", "entry_criteria"],
    },

    "milestone": {
        # Jalons M0-M9 (GRI) et J1-J6 (CIR) : TOUJOURS le jalon COMPLET
        # Ne jamais couper un jalon — ses critères forment un tout cohérent
        "size": (600, 900),
        "overlap": 0,
        "boundary": "milestone",  # Jalon entier = 1 seul chunk
        "prefix_format": "[GRI > Jalons > {milestone_id} : {milestone_name}]",
        "metadata": ["milestone_id", "milestone_name", "phase_num", "cycle"],
        "warning": "Ne JAMAIS chunker à l'intérieur d'un jalon — les critères sont interdépendants"
    },

    "process": {
        # Processus IS 15288 : 1 activité = 1 chunk (avec ses inputs/outputs)
        "size": (400, 600),
        "overlap": 0,
        "boundary": "activity",
        "prefix_format": "[GRI > Processus IS 15288 > {process_name} > {activity_type}]",
        "metadata": ["process_name", "activity_type", "inputs", "outputs", "version"],
    },

    "cir": {
        # CIR : inclure mapping vers GRI DANS le chunk
        "size": (400, 600),
        "overlap": 0,
        "boundary": "section",
        "prefix_format": "[CIR > Phase {cir_phase} > Jalon {jalon_id}]",
        "metadata": ["cir_phase", "jalon_id", "gri_equivalent", "duration_weeks"],
        "gri_mapping": {
            "J1": ["M0", "M1"],
            "J2": ["M2", "M3", "M4"],
            "J3": ["M5", "M6"],
            "J4": ["SAR"],
            "J5": ["SAR"],
            "J6": ["M8"],
        }
    },

    "table": {
        # Tables : row-by-row + chunk de synthèse global
        "strategy": "row_plus_summary",
        "row_size": (100, 250),
        "summary_size": (300, 500),
        "prefix_format": "[GRI > {parent_section} > Tableau {table_id}]",
        "metadata": ["table_id", "table_title", "columns", "parent_section"],
    },
}
```

## Metadata Obligatoire

```python
@dataclass
class GRIChunk:
    content: str              # context_prefix + "\n\n" + text
    chunk_id: str             # sha256(content)[:16]
    metadata: GRIMetadata

@dataclass
class GRIMetadata:
    # Identifiants
    doc_id: str               # sha256(fichier source)[:16]
    source: str               # "GRI_FAR_2025"
    chunk_index: int

    # Localisation GRI
    section_type: str         # 'definition' | 'principle' | 'phase' | 'milestone'
                              # | 'process' | 'cir' | 'table' | 'intro'
    hierarchy: list[str]      # ['GRI', 'Phase 3', 'Conception', 'Vérification']
    context_prefix: str       # "[GRI > Phase 3 > Conception > Vérification]"

    # Identifiants domaine
    cycle: str                # 'GRI' | 'CIR'
    phase_num: int | None     # 1-7 pour GRI, 1-4 pour CIR
    cir_phase: int | None     # 1-4 (CIR uniquement)
    milestone_id: str | None  # 'M0' à 'M9', 'J1' à 'J6'
    process_name: str | None  # Nom du processus IS 15288
    principle_num: int | None # 1-11

    # Technique
    language: str             # 'fr'
    char_count: int
    token_estimate: int       # char_count // 4
    created_at: str           # ISO timestamp
    parent_chunk_id: str | None  # Pour Parent Document Retriever
```

## Implémentation

```python
# src/ingestion/pipeline.py
import hashlib
from pathlib import Path
from datetime import datetime
from docx import Document as DocxDocument
import re

class GRIIngestionPipeline:

    SECTION_TYPE_PATTERNS = {
        "definition":  r"^#{1,2}\s+(Liste des abréviations|définition de la terminologie)",
        "principle":   r"^#{2}\s+Principe N°\d+",
        "phase":       r"^#{3}\s+Phase [1-7]",
        "milestone":   r"Critères du passage du jalon|Critères du passage du M\d|Jalon [MJ]\d",
        "process":     r"#{4,5}.*(Process|Processus).*(Version réduite|standard|allégée)",
        "cir":         r"^#{2,3}.*(CIR|Cycle d.Innovation Rapide|Phase [1-4].*J\d)",
    }

    def run(self, docx_path: str) -> dict:
        path = Path(docx_path)
        doc_id = self._hash_file(path)

        # 3 flux séparés
        sections  = self._extract_sections(path)
        tables    = self._extract_tables(path)
        glossary  = self._extract_glossary(sections)

        chunks = []
        chunks += self._chunk_sections(sections, doc_id)
        chunks += self._chunk_tables(tables, doc_id)
        chunks += self._chunk_glossary(glossary, doc_id)

        # Valider et indexer
        valid = [c for c in chunks if self._validate(c)]
        return {"total": len(chunks), "valid": len(valid), "chunks": valid}

    def _detect_section_type(self, text: str) -> str:
        for stype, pattern in self.SECTION_TYPE_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                return stype
        return "content"

    def _build_prefix(self, hierarchy: list[str]) -> str:
        return "[" + " > ".join(hierarchy) + "]"

    def _validate(self, chunk) -> bool:
        # Chunk non vide
        if len(chunk.content) < 80:
            return False
        # Context prefix présent
        if not chunk.content.startswith("[GRI") and not chunk.content.startswith("[CIR"):
            return False
        # Metadata complète
        required = ["doc_id", "source", "section_type", "context_prefix", "cycle"]
        if not all(hasattr(chunk.metadata, f) for f in required):
            return False
        # Taille raisonnable
        if chunk.metadata.token_estimate > 1000:
            return False
        return True

    def _hash_file(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
```

## Extraction des Tables (Jalons & Livrables)

Les tables de critères de jalons sont la partie la plus critique — ne jamais les perdre.

```python
# src/ingestion/table_extractor.py
from docx import Document
from docx.table import Table

class GRITableExtractor:

    JALON_TABLE_MARKERS = [
        "Critères du passage", "Jalon M", "Jalon J",
        "CDR", "TRR", "SRR", "PDR", "IRR", "SAR", "ORR"
    ]

    def extract(self, docx_path: str) -> list[dict]:
        doc = Document(docx_path)
        tables_data = []

        for i, table in enumerate(doc.tables):
            table_text = self._table_to_text(table)
            is_milestone = any(m in table_text for m in self.JALON_TABLE_MARKERS)

            tables_data.append({
                "table_index": i,
                "table_type": "milestone_criteria" if is_milestone else "general",
                "rows": self._extract_rows(table),
                "full_text": table_text,
            })

        return tables_data

    def _table_to_text(self, table: Table) -> str:
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                rows.append(" | ".join(cells))
        return "\n".join(rows)

    def _extract_rows(self, table: Table) -> list[dict]:
        headers = [cell.text.strip() for cell in table.rows[0].cells] if table.rows else []
        rows = []
        for row in table.rows[1:]:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                rows.append(dict(zip(headers, cells)))
        return rows
```

## Tests à Écrire

```python
# tests/test_ingestion_gri.py

def test_all_milestones_indexed():
    # M0, M1, M2, M3, M4, M5, M6, M7, M8, M9, J1, J2, J3, J4, J5, J6
    # Chacun doit avoir exactement 1 chunk 'milestone' dans l'index
    milestone_ids = [f"M{i}" for i in range(10)] + [f"J{i}" for i in range(1, 7)]
    for mid in milestone_ids:
        chunks = store.filter(milestone_id=mid)
        assert len(chunks) >= 1, f"Jalon {mid} manquant dans l'index"

def test_context_prefix_on_all_chunks():
    chunks = store.get_all()
    for chunk in chunks:
        assert chunk.content.startswith("[GRI") or chunk.content.startswith("[CIR"), \
            f"Context prefix manquant sur chunk {chunk.chunk_id}"

def test_glossary_index_completeness():
    # Le glossaire GRI contient ~200 termes bilingues
    glossary_chunks = store.filter(section_type="definition")
    assert len(glossary_chunks) >= 150, "Glossaire incomplet"

def test_cir_chunks_have_gri_mapping():
    cir_chunks = store.filter(cycle="CIR")
    for chunk in cir_chunks:
        if chunk.metadata.milestone_id:
            assert chunk.metadata.gri_equivalent is not None, \
                f"Mapping GRI manquant pour {chunk.metadata.milestone_id}"

def test_milestone_chunks_not_split():
    # Un jalon = 1 seul chunk — vérifier qu'il n'est pas fragmenté
    m4_chunks = store.filter(milestone_id="M4")
    assert len(m4_chunks) == 1, "Le jalon M4 (CDR) ne doit pas être fragmenté"

def test_no_empty_chunks():
    chunks = store.get_all()
    assert all(len(c.content) >= 80 for c in chunks)
```
