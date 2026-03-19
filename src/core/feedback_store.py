"""Feedback Store for GRI RAG.

Persists user feedback for analysis and model improvement.

Usage:
    from src.core.feedback_store import save_feedback

    await save_feedback(query_id, rating, comment, incorrect_info)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import structlog

from src.core.config import settings

log = structlog.get_logger()


async def save_feedback(
    query_id: str,
    rating: int,
    comment: str | None = None,
    incorrect_info: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Save feedback to persistent storage.

    Tries PostgreSQL first, falls back to JSONL file.

    Args:
        query_id: The query ID this feedback is for
        rating: Rating 1-5
        comment: Optional user comment
        incorrect_info: Optional incorrect information report
        metadata: Additional metadata

    Returns:
        True if saved successfully
    """
    # Always log feedback
    log.info(
        "feedback.received",
        query_id=query_id,
        rating=rating,
        has_comment=comment is not None,
        has_incorrect_info=incorrect_info is not None,
    )

    # Try PostgreSQL if configured
    if hasattr(settings, "postgres_dsn") and settings.postgres_dsn:
        success = await _save_to_postgres(query_id, rating, comment, incorrect_info, metadata)
        if success:
            return True
        log.warning("feedback.postgres_failed_fallback_to_file", query_id=query_id)

    # Fallback: append to JSON file
    return await _save_to_file(query_id, rating, comment, incorrect_info, metadata)


async def _save_to_postgres(
    query_id: str,
    rating: int,
    comment: str | None,
    incorrect_info: str | None,
    metadata: dict[str, Any] | None,
) -> bool:
    """Save feedback to PostgreSQL."""
    try:
        import asyncpg

        pool = await asyncpg.create_pool(settings.postgres_dsn, min_size=1, max_size=5)

        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO feedback (query_id, rating, comment, incorrect_info, metadata)
                VALUES ($1, $2, $3, $4, $5)
                """,
                query_id,
                rating,
                comment,
                incorrect_info,
                json.dumps(metadata or {}),
            )

        await pool.close()
        log.info("feedback.saved_postgres", query_id=query_id)
        return True

    except Exception as e:
        log.error("feedback.postgres_save_failed", error=str(e))
        return False


async def _save_to_file(
    query_id: str,
    rating: int,
    comment: str | None,
    incorrect_info: str | None,
    metadata: dict[str, Any] | None,
) -> bool:
    """Save feedback to JSONL file."""
    try:
        feedback_file = Path(settings.data_dir) / "feedback.jsonl"
        feedback_file.parent.mkdir(parents=True, exist_ok=True)

        entry = {
            "query_id": query_id,
            "rating": rating,
            "comment": comment,
            "incorrect_info": incorrect_info,
            "metadata": metadata or {},
            "timestamp": datetime.utcnow().isoformat(),
        }

        with open(feedback_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

        log.info("feedback.saved_file", query_id=query_id, file=str(feedback_file))
        return True

    except Exception as e:
        log.error("feedback.file_save_failed", error=str(e))
        return False


async def get_feedback_stats() -> dict[str, Any]:
    """Get feedback statistics.

    Returns:
        Dictionary with feedback statistics
    """
    stats = {
        "total_feedback": 0,
        "average_rating": 0.0,
        "rating_distribution": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
        "source": "unknown",
    }

    # Try PostgreSQL first
    if hasattr(settings, "postgres_dsn") and settings.postgres_dsn:
        try:
            import asyncpg

            pool = await asyncpg.create_pool(settings.postgres_dsn, min_size=1, max_size=2)

            async with pool.acquire() as conn:
                # Total and average
                row = await conn.fetchrow(
                    "SELECT COUNT(*) as total, AVG(rating) as avg FROM feedback"
                )
                if row:
                    stats["total_feedback"] = row["total"]
                    stats["average_rating"] = float(row["avg"]) if row["avg"] else 0.0

                # Distribution
                rows = await conn.fetch(
                    "SELECT rating, COUNT(*) as count FROM feedback GROUP BY rating"
                )
                for row in rows:
                    stats["rating_distribution"][row["rating"]] = row["count"]

            await pool.close()
            stats["source"] = "postgres"
            return stats

        except Exception as e:
            log.warning("feedback.postgres_stats_failed", error=str(e))

    # Fallback: read from file
    feedback_file = Path(settings.data_dir) / "feedback.jsonl"
    if feedback_file.exists():
        try:
            ratings = []
            with open(feedback_file) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    rating = entry.get("rating")
                    if rating:
                        ratings.append(rating)
                        stats["rating_distribution"][rating] = (
                            stats["rating_distribution"].get(rating, 0) + 1
                        )

            stats["total_feedback"] = len(ratings)
            stats["average_rating"] = sum(ratings) / len(ratings) if ratings else 0.0
            stats["source"] = "file"

        except Exception as e:
            log.warning("feedback.file_stats_failed", error=str(e))

    return stats
