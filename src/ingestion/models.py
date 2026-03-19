"""Modèles Pydantic pour l'ingestion GRI.

Définit les structures de données pour les chunks et leurs métadonnées.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, field_validator


class SectionType(StrEnum):
    """Types de sections dans le document GRI."""

    DEFINITION = "definition"
    PRINCIPLE = "principle"
    PHASE = "phase"
    MILESTONE = "milestone"
    PROCESS = "process"
    CIR = "cir"
    TABLE = "table"
    INTRO = "intro"
    CONTENT = "content"


class Cycle(StrEnum):
    """Cycles de développement GRI."""

    GRI = "GRI"
    CIR = "CIR"


# Mapping CIR → GRI (jalons)
CIR_GRI_MAPPING: dict[str, list[str]] = {
    "J1": ["M0", "M1"],
    "J2": ["M2", "M3", "M4"],
    "J3": ["M5", "M6"],
    "J4": ["SAR"],
    "J5": ["SAR"],
    "J6": ["M8"],
}


class GRIMetadata(BaseModel):
    """Métadonnées obligatoires pour chaque chunk GRI.

    Voir ingestion_skill.md pour les règles de chunking.
    """

    # === Identifiants ===
    doc_id: str = Field(
        ...,
        description="SHA256 du fichier source (16 chars)",
        min_length=16,
        max_length=16,
    )
    source: str = Field(
        default="GRI_FAR_2025",
        description="Identifiant de la source",
    )
    chunk_index: int = Field(
        ...,
        ge=0,
        description="Index du chunk dans le document",
    )

    # === Localisation GRI ===
    section_type: SectionType = Field(
        ...,
        description="Type de section GRI",
    )
    hierarchy: list[str] = Field(
        default_factory=list,
        description="Chemin hiérarchique ['GRI', 'Phase 3', 'Conception', ...]",
    )
    context_prefix: str = Field(
        ...,
        description="Prefix de contexte [GRI > Phase 3 > ...]",
    )

    # === Identifiants domaine ===
    cycle: Cycle = Field(
        default=Cycle.GRI,
        description="Cycle de développement (GRI ou CIR)",
    )
    phase_num: int | None = Field(
        default=None,
        ge=1,
        le=7,
        description="Numéro de phase (1-7 GRI, 1-4 CIR)",
    )
    cir_phase: int | None = Field(
        default=None,
        ge=1,
        le=4,
        description="Phase CIR (1-4)",
    )
    milestone_id: str | None = Field(
        default=None,
        pattern=r"^[MJ]\d$",
        description="ID du jalon (M0-M9 ou J1-J6)",
    )
    milestone_name: str | None = Field(
        default=None,
        description="Nom du jalon (CDR, TRR, etc.)",
    )
    process_name: str | None = Field(
        default=None,
        description="Nom du processus IS 15288",
    )
    principle_num: int | None = Field(
        default=None,
        ge=1,
        le=11,
        description="Numéro de principe (1-11)",
    )

    # === Glossaire ===
    term_fr: str | None = Field(
        default=None,
        description="Terme français (pour définitions)",
    )
    term_en: str | None = Field(
        default=None,
        description="Terme anglais (pour définitions)",
    )
    standard_ref: str | None = Field(
        default=None,
        description="Référence normative (ISO/IEC/IEEE 15288:2023)",
    )

    # === CIR spécifique ===
    gri_equivalent: list[str] | None = Field(
        default=None,
        description="Jalons GRI équivalents (pour CIR)",
    )

    # === Table spécifique ===
    table_id: str | None = Field(
        default=None,
        description="Identifiant de table",
    )
    table_title: str | None = Field(
        default=None,
        description="Titre de la table",
    )

    # === Technique ===
    language: Literal["fr"] = Field(
        default="fr",
        description="Langue du contenu",
    )
    char_count: int = Field(
        ...,
        ge=0,
        description="Nombre de caractères",
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp de création",
    )
    parent_chunk_id: str | None = Field(
        default=None,
        description="ID du chunk parent (Parent Document Retriever)",
    )

    @computed_field
    @property
    def token_estimate(self) -> int:
        """Estimation du nombre de tokens (chars / 4)."""
        return self.char_count // 4

    def model_post_init(self, __context) -> None:
        """Enrichir automatiquement les métadonnées CIR."""
        if self.cycle == Cycle.CIR and self.milestone_id in CIR_GRI_MAPPING:
            self.gri_equivalent = CIR_GRI_MAPPING[self.milestone_id]

    def get(self, key: str, default: Any = None) -> Any:
        """Compatibilité dict-like pour certains appels historiques."""
        return getattr(self, key, default)


class GRIChunk(BaseModel):
    """Chunk GRI avec contenu et métadonnées.

    RÈGLE CRITIQUE: Le contenu DOIT commencer par le context_prefix.
    Format: "[GRI > Section > ...]\n\n{texte}"
    """

    chunk_id: str = Field(
        ...,
        description="SHA256 du contenu (16 chars)",
        min_length=16,
        max_length=16,
    )
    content: str = Field(
        ...,
        min_length=80,
        description="Contenu du chunk (avec context_prefix)",
    )
    metadata: GRIMetadata = Field(
        ...,
        description="Métadonnées du chunk",
    )

    def model_post_init(self, __context) -> None:
        """Valider que le contenu commence par le prefix."""
        if not (self.content.startswith("[GRI") or self.content.startswith("[CIR")):
            raise ValueError(
                f"Le chunk doit commencer par [GRI ou [CIR, " f"reçu: {self.content[:50]}..."
            )

    @property
    def is_valid(self) -> bool:
        """Vérifie si le chunk est valide pour l'indexation."""
        return (
            len(self.content) >= 80
            and self.metadata.token_estimate <= 1000
            and self.content.startswith(("[GRI", "[CIR"))
        )

    @property
    def section_type(self) -> SectionType:
        """Compatibilité d'accès plat au type de section."""
        return self.metadata.section_type

    @property
    def cycle(self) -> str:
        """Compatibilité d'accès plat au cycle."""
        return self.metadata.cycle.value

    @property
    def context_prefix(self) -> str:
        """Compatibilité d'accès plat au préfixe de contexte."""
        return self.metadata.context_prefix


class ParsedSection(BaseModel):
    """Section extraite du document DOCX."""

    level: int = Field(
        ...,
        ge=1,
        le=7,
        description="Niveau de heading (1-7)",
    )
    title: str = Field(
        ...,
        description="Titre de la section",
    )
    content: str = Field(
        default="",
        description="Contenu texte de la section",
    )
    hierarchy: list[str] = Field(
        default_factory=list,
        description="Chemin hiérarchique",
    )
    section_type: SectionType = Field(
        default=SectionType.CONTENT,
        description="Type détecté de la section",
    )
    tables: list[dict] = Field(
        default_factory=list,
        description="Tables contenues dans la section",
    )
    start_index: int = Field(
        default=0,
        description="Index de début dans le document",
    )
    end_index: int = Field(
        default=0,
        description="Index de fin dans le document",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Métadonnées de section complémentaires",
    )


class ParsedTable(BaseModel):
    """Table extraite du document DOCX."""

    table_index: int = Field(
        default=0,
        description="Index de la table dans le document",
    )
    table_type: Literal["milestone_criteria", "general", "deliverables"] = Field(
        default="general",
        description="Type de table",
    )
    headers: list[str] = Field(
        default_factory=list,
        description="En-têtes de colonnes",
    )
    rows: list[dict[str, str]] = Field(
        default_factory=list,
        description="Lignes de la table (dict par ligne)",
    )
    full_text: str = Field(
        default="",
        description="Texte complet de la table",
    )
    parent_section: str | None = Field(
        default=None,
        description="Section parente",
    )
    milestone_id: str | None = Field(
        default=None,
        description="ID du jalon associé",
    )
    title: str | None = Field(
        default=None,
        description="Titre lisible de la table",
    )
    section_type: SectionType = Field(
        default=SectionType.TABLE,
        description="Type de section associé",
    )
    hierarchy: list[str] = Field(
        default_factory=list,
        description="Chemin hiérarchique de la table",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Métadonnées de compatibilité",
    )

    @field_validator("rows", mode="before")
    @classmethod
    def normalize_rows(cls, v: Any, info) -> list[dict[str, str]]:
        """Accepte list[dict] et list[list] pour compatibilité tests/fixtures."""
        if not isinstance(v, list) or not v:
            return v

        first = v[0]
        if isinstance(first, dict):
            return v

        headers = info.data.get("headers", [])
        normalized: list[dict[str, str]] = []
        for row in v:
            if isinstance(row, list):
                normalized.append(
                    {header: str(value) for header, value in zip(headers, row, strict=False)}
                )
            else:
                normalized.append({"value": str(row)})
        return normalized


class GlossaryEntry(BaseModel):
    """Entrée du glossaire GRI."""

    term_fr: str = Field(
        ...,
        description="Terme en français",
    )
    term_en: str | None = Field(
        default=None,
        description="Terme en anglais",
    )
    definition_fr: str = Field(
        ...,
        description="Définition en français",
    )
    definition_en: str | None = Field(
        default=None,
        description="Définition en anglais",
    )
    standard_ref: str | None = Field(
        default=None,
        description="Référence normative",
    )
    domain: str | None = Field(
        default=None,
        description="Domaine (ingénierie système, etc.)",
    )


class IngestionResult(BaseModel):
    """Résultat de l'ingestion."""

    doc_id: str
    doc_path: str
    total_chunks: int
    valid_chunks: int
    invalid_chunks: int
    chunks_by_type: dict[str, int]
    glossary_terms: int
    tables_extracted: int
    milestones_found: list[str]
    warnings: list[str]
    errors: list[str]
    duration_seconds: float
    chunks: list[GRIChunk] = Field(default_factory=list)
