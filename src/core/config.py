"""Configuration centralisée avec Pydantic Settings.

Usage:
    from src.core.config import settings
    print(settings.hf_api_key)
    print(settings.qdrant_url)
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration globale du projet GRI RAG.

    Les valeurs sont chargées dans cet ordre de priorité :
    1. Variables d'environnement
    2. Fichier .env
    3. Valeurs par défaut
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # === Hugging Face ===
    hf_api_key: str = Field(
        default="",
        description="Clé API Hugging Face",
    )
    hf_router_model: str = Field(
        default="mistralai/Mistral-7B-Instruct-v0.3",
        description="Modèle pour le routing des queries",
    )
    hf_orchestrator_model: str = Field(
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        description="Modèle pour l'orchestration",
    )
    hf_generation_model: str = Field(
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        description="Modèle pour la génération",
    )
    hf_eval_model: str = Field(
        default="mistralai/Mixtral-8x7B-Instruct-v0.1",
        description="Modèle pour l'évaluation",
    )

    # === Vector Store (Qdrant) ===
    qdrant_url: str = Field(
        default="localhost",
        description="URL du serveur Qdrant",
    )
    qdrant_port: int = Field(
        default=6333,
        description="Port du serveur Qdrant",
    )
    qdrant_api_key: str = Field(
        default="",
        description="Clé API Qdrant (optionnel)",
    )
    qdrant_collection_main: str = Field(
        default="gri_main",
        description="Collection principale",
    )
    qdrant_collection_glossary: str = Field(
        default="gri_glossary",
        description="Collection glossaire",
    )

    # === Embeddings ===
    embedding_model: str = Field(
        default="paraphrase-multilingual-mpnet-base-v2",
        description="Modèle d'embeddings (FR+EN)",
    )
    embedding_dimension: int = Field(
        default=768,
        description="Dimension des vecteurs",
    )

    # === Reranker ===
    reranker_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Modèle de reranking",
    )

    # === API ===
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_reload: bool = Field(default=False)
    api_workers: int = Field(default=1)

    # === Security ===
    api_auth_enabled: bool = Field(
        default=False,
        description="Enable API authentication (auto-enabled when host != 127.0.0.1)",
    )
    api_bearer_token: str = Field(
        default="",
        description="Bearer token for API authentication",
    )
    cors_allowed_origins: str = Field(
        default="http://localhost:3000,http://localhost:8080",
        description="Comma-separated CORS origins. Use '*' ONLY for development.",
    )
    session_ttl_seconds: int = Field(
        default=3600,
        description="Session TTL in seconds (default: 1 hour)",
    )

    # === State Management ===
    session_backend: str = Field(
        default="memory",
        description="Session backend: memory, redis, or postgres",
    )
    redis_url: str = Field(
        default="",
        description="Redis URL for distributed sessions/rate limiting",
    )
    postgres_dsn: str = Field(
        default="",
        description="PostgreSQL DSN for persistent sessions/feedback",
    )

    # === Logging ===
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json")

    # === Paths ===
    data_dir: Path = Field(default=Path("./data"))
    reports_dir: Path = Field(default=Path("./reports"))
    config_path: Path = Field(default=Path("./configs/config.yaml"))

    # === Agent ===
    agent_max_iterations: int = Field(default=5)
    agent_timeout_seconds: int = Field(default=120)

    # === Retrieval ===
    rrf_alpha: float = Field(default=0.6)
    rrf_k: int = Field(default=60)
    default_n_results: int = Field(default=5)
    rerank_top_k: int = Field(default=10)

    # === Generation ===
    generation_max_tokens: int = Field(default=2048)
    definition_temperature: float = Field(default=0.0)
    general_temperature: float = Field(default=0.1)

    # === Evaluation Thresholds ===
    eval_faithfulness_threshold: float = Field(default=0.85)
    eval_answer_relevance_threshold: float = Field(default=0.80)
    eval_context_recall_threshold: float = Field(default=0.75)
    eval_term_accuracy_threshold: float = Field(default=0.95)
    eval_latency_p95_ms: int = Field(default=8000)

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return v_upper

    @field_validator("log_format")
    @classmethod
    def validate_log_format(cls, v: str) -> str:
        valid = {"json", "console"}
        v_lower = v.lower()
        if v_lower not in valid:
            raise ValueError(f"log_format must be one of {valid}")
        return v_lower

    @property
    def qdrant_connection_url(self) -> str:
        """URL complète de connexion Qdrant."""
        return f"http://{self.qdrant_url}:{self.qdrant_port}"

    def get_yaml_config(self) -> dict[str, Any]:
        """Charge la configuration YAML complète."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}


@lru_cache
def get_settings() -> Settings:
    """Singleton pour les settings (mis en cache)."""
    return Settings()


# Export global pour import facile
settings = get_settings()


# === CIR / GRI Mapping (constantes) ===
CIR_GRI_MAPPING: dict[str, list[str]] = {
    "J1": ["M0", "M1"],
    "J2": ["M2", "M3", "M4"],
    "J3": ["M5", "M6"],
    "J4": ["SAR"],
    "J5": ["SAR"],
    "J6": ["M8"],
}

VALID_GRI_MILESTONES = {f"M{i}" for i in range(10)}
VALID_CIR_MILESTONES = {f"J{i}" for i in range(1, 7)}
VALID_MILESTONES = VALID_GRI_MILESTONES | VALID_CIR_MILESTONES

GRI_PHASES = range(1, 8)  # 1-7
CIR_PHASES = range(1, 5)  # 1-4

# Mapping des acronymes de revue vers les IDs de jalons
# Utilisé par l'extraction (ingestion) et la validation (retrieval)
MILESTONE_ACRONYM_TO_ID: dict[str, str] = {
    "ASR": "M0",  # Advance Study Review
    "MNS": "M0",  # Mission Needs Statement
    "SRR": "M1",  # System Requirements Review
    "SFR": "M2",  # System Functional Review
    "PDR": "M2",  # Preliminary Design Review (alias SFR)
    "CDR": "M3",  # Critical Design Review
    "IRR": "M4",  # Integration Readiness Review
    "TRR": "M5",  # Test Readiness Review
    "SAR": "M6",  # System Acceptance Review
    "ORR": "M7",  # Operational Readiness Review
    "MNR": "M8",  # Mission Needs Review
}
