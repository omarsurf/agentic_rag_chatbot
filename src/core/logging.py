"""Configuration du logging structuré avec structlog.

Usage:
    from src.core.logging import get_logger
    log = get_logger(__name__)
    log.info("action.started", key="value", count=42)
"""

import logging
import sys
from typing import Any, cast

import structlog
from structlog.types import Processor


def setup_logging(
    level: str = "INFO",
    log_format: str = "json",
    include_timestamp: bool = True,
    include_caller: bool = True,
) -> None:
    """Configure structlog pour le projet GRI RAG.

    Args:
        level: Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format de sortie ("json" ou "console")
        include_timestamp: Inclure le timestamp dans les logs
        include_caller: Inclure le fichier/ligne d'appel

    Example:
        setup_logging(level="DEBUG", log_format="console")
        log = get_logger(__name__)
        log.info("gri.ingestion.started", doc_path="/path/to/doc.docx")
    """
    # Processors communs
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if include_timestamp:
        shared_processors.insert(0, structlog.processors.TimeStamper(fmt="iso", utc=True))

    if include_caller:
        shared_processors.append(
            structlog.processors.CallsiteParameterAdder(
                [
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                ]
            )
        )

    # Processor final selon le format
    if log_format == "json":
        final_processor: Processor = structlog.processors.JSONRenderer()
    else:
        final_processor = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        )

    # Configuration structlog
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configuration du handler stdlib
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                final_processor,
            ],
        )
    )

    # Configuration du root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper()))

    # Réduire le bruit des loggers tiers
    for logger_name in ["httpx", "httpcore", "urllib3", "asyncio", "transformers"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Obtenir un logger structuré.

    Args:
        name: Nom du logger (typiquement __name__)

    Returns:
        Logger structuré prêt à l'emploi

    Example:
        log = get_logger(__name__)
        log.info("gri.retrieval.search", query="CDR critères", n_results=5)
        log.warning("gri.agent.max_iter", query="...", iterations=5)
        log.error("gri.tool.error", tool="retrieve_gri_chunks", error="timeout")
    """
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))


def bind_context(**kwargs: Any) -> None:
    """Ajouter du contexte à tous les logs suivants (thread-local).

    Example:
        bind_context(request_id="abc123", user_id="user456")
        log.info("processing")  # Inclura request_id et user_id
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """Effacer le contexte thread-local."""
    structlog.contextvars.clear_contextvars()


# === Conventions de nommage GRI ===
#
# Format: {module}.{action}
#
# Modules:
#   - gri.ingestion    : Pipeline d'ingestion
#   - gri.retrieval    : Recherche hybride
#   - gri.router       : Query router
#   - gri.agent        : Orchestrateur ReAct
#   - gri.tool         : Tools de l'agent
#   - gri.generation   : Génération de réponses
#   - gri.evaluation   : Évaluation RAGAS
#   - gri.api          : Endpoints FastAPI
#
# Actions:
#   - started, completed, failed
#   - search, found, not_found
#   - call, result, error
#
# Exemples:
#   log.info("gri.ingestion.started", doc_path="...")
#   log.info("gri.retrieval.search", query="...", n_results=5)
#   log.info("gri.agent.tool_call", tool="lookup_gri_glossary", input={...})
#   log.warning("gri.agent.max_iter_reached", query="...", iterations=5)
#   log.error("gri.tool.error", tool="...", error="...")
