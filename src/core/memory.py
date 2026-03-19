"""GRI Memory - Mémoire conversationnelle pour l'orchestrateur.

Gère l'historique des échanges question/réponse pour maintenir
le contexte dans une conversation multi-tours.

Usage:
    from src.core.memory import GRIMemory

    memory = GRIMemory()
    memory.add_turn("Qu'est-ce qu'un artefact ?", "Un artefact est...")
    memory.add_turn("Et pour le CONOPS ?", "Le CONOPS est...")

    context = memory.get_context()
    # Retourne le contexte formaté pour injection dans le prompt
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog
from pydantic import BaseModel, Field

log = structlog.get_logger()


@dataclass
class ConversationTurn:
    """Un tour de conversation (question + réponse)."""

    query: str
    answer: str
    timestamp: datetime = field(default_factory=datetime.now)
    intent: str | None = None
    cycle: str | None = None
    tool_calls: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convertit le tour en dictionnaire."""
        return {
            "query": self.query,
            "answer": self.answer,
            "timestamp": self.timestamp.isoformat(),
            "intent": self.intent,
            "cycle": self.cycle,
            "tool_calls": self.tool_calls,
            "citations": self.citations,
        }


class MemoryStats(BaseModel):
    """Statistiques de la mémoire."""

    total_turns: int = 0
    oldest_turn: datetime | None = None
    newest_turn: datetime | None = None
    intents_used: list[str] = Field(default_factory=list)
    cycles_used: list[str] = Field(default_factory=list)


class GRIMemory:
    """Mémoire conversationnelle pour l'agent GRI.

    Stocke l'historique des échanges et permet de reconstruire
    le contexte conversationnel pour les tours suivants.

    Attributes:
        MAX_TURNS: Nombre maximum de tours conservés
        turns: Liste des tours de conversation
        session_id: Identifiant de session (optionnel)
    """

    MAX_TURNS: int = 10

    def __init__(
        self, session_id: str | None = None, max_turns: int | None = None
    ) -> None:
        """Initialise la mémoire.

        Args:
            session_id: Identifiant de session (optionnel)
            max_turns: Nombre max de tours (défaut: MAX_TURNS)
        """
        self._session_id = session_id
        self._max_turns = max_turns or self.MAX_TURNS
        self._turns: list[ConversationTurn] = []

        log.info("memory.init", session_id=session_id, max_turns=self._max_turns)

    @property
    def session_id(self) -> str | None:
        """Retourne l'identifiant de session."""
        return self._session_id

    @property
    def turns(self) -> list[ConversationTurn]:
        """Retourne les tours de conversation."""
        return self._turns

    @property
    def is_empty(self) -> bool:
        """Vérifie si la mémoire est vide."""
        return len(self._turns) == 0

    def add_turn(
        self,
        query: str,
        answer: str,
        intent: str | None = None,
        cycle: str | None = None,
        tool_calls: list[str] | None = None,
        citations: list[str] | None = None,
    ) -> None:
        """Ajoute un tour de conversation.

        Args:
            query: Question utilisateur
            answer: Réponse de l'agent
            intent: Intent détecté (optionnel)
            cycle: Cycle GRI/CIR (optionnel)
            tool_calls: Liste des tools appelés (optionnel)
            citations: Liste des citations (optionnel)
        """
        turn = ConversationTurn(
            query=query,
            answer=answer,
            intent=intent,
            cycle=cycle,
            tool_calls=tool_calls or [],
            citations=citations or [],
        )

        self._turns.append(turn)

        # Élaguer si nécessaire
        if len(self._turns) > self._max_turns:
            removed = self._turns.pop(0)
            log.info("memory.turn_evicted", query=removed.query[:50])

        log.info(
            "memory.turn_added",
            query=query[:50],
            n_turns=len(self._turns),
        )

    def get_context_for_llm(self, max_length: int = 4000) -> str:
        """Génère le contexte conversationnel formaté pour le LLM.

        Alias de get_context() pour compatibilité API.

        Args:
            max_length: Limite de caractères pour le contexte

        Returns:
            Contexte formaté ou chaîne vide si pas d'historique
        """
        return self.get_context(max_chars=max_length)

    def get_context(self, max_chars: int = 4000) -> str:
        """Génère le contexte conversationnel formaté.

        Retourne l'historique formaté pour injection dans le system prompt.
        Tronque si nécessaire pour respecter la limite de caractères.

        Args:
            max_chars: Limite de caractères pour le contexte

        Returns:
            Contexte formaté ou chaîne vide si pas d'historique
        """
        if not self._turns:
            return ""

        lines = ["## Historique de la conversation\n"]

        for i, turn in enumerate(self._turns, 1):
            turn_text = f"**Tour {i}**\n"
            turn_text += f"- Question : {turn.query}\n"

            # Tronquer la réponse si trop longue
            answer_preview = turn.answer
            if len(answer_preview) > 500:
                answer_preview = answer_preview[:500] + "..."

            turn_text += f"- Réponse : {answer_preview}\n"

            if turn.intent:
                turn_text += f"- Intent : {turn.intent}\n"

            lines.append(turn_text)

        context = "\n".join(lines)

        # Tronquer si dépasse la limite
        if len(context) > max_chars:
            context = context[:max_chars] + "\n\n[... historique tronqué ...]"

        return context

    def get_last_turn(self) -> ConversationTurn | None:
        """Retourne le dernier tour de conversation."""
        return self._turns[-1] if self._turns else None

    def get_last_n_turns(self, n: int) -> list[ConversationTurn]:
        """Retourne les N derniers tours.

        Args:
            n: Nombre de tours à retourner

        Returns:
            Liste des N derniers tours
        """
        return self._turns[-n:] if self._turns else []

    def clear(self) -> None:
        """Efface toute la mémoire."""
        n_cleared = len(self._turns)
        self._turns.clear()
        log.info("memory.cleared", n_turns_cleared=n_cleared)

    def get_stats(self) -> dict[str, Any]:
        """Retourne les statistiques de la mémoire sous forme de dictionnaire.

        Returns:
            Dictionnaire contenant session_id, total_turns, oldest_turn,
            newest_turn, intents_used, cycles_used
        """
        intents = list({t.intent for t in self._turns if t.intent})
        cycles = list({t.cycle for t in self._turns if t.cycle})

        return {
            "session_id": self._session_id,
            "total_turns": len(self._turns),
            "oldest_turn": self._turns[0].timestamp.isoformat() if self._turns else None,
            "newest_turn": self._turns[-1].timestamp.isoformat() if self._turns else None,
            "intents_used": intents,
            "cycles_used": cycles,
        }

    def get_stats_model(self) -> MemoryStats:
        """Retourne les statistiques sous forme de modèle Pydantic."""
        if not self._turns:
            return MemoryStats()

        intents = list({t.intent for t in self._turns if t.intent})
        cycles = list({t.cycle for t in self._turns if t.cycle})

        return MemoryStats(
            total_turns=len(self._turns),
            oldest_turn=self._turns[0].timestamp,
            newest_turn=self._turns[-1].timestamp,
            intents_used=intents,
            cycles_used=cycles,
        )

    def search_by_intent(self, intent: str) -> list[ConversationTurn]:
        """Recherche les tours par intent.

        Args:
            intent: Intent à rechercher

        Returns:
            Liste des tours correspondants
        """
        return [t for t in self._turns if t.intent == intent]

    def has_discussed(self, topic: str) -> bool:
        """Vérifie si un sujet a été discuté.

        Recherche simple dans les questions précédentes.

        Args:
            topic: Sujet à rechercher

        Returns:
            True si le sujet a été mentionné
        """
        topic_lower = topic.lower()
        return any(topic_lower in t.query.lower() for t in self._turns)

    def get_referenced_milestones(self) -> set[str]:
        """Retourne les jalons mentionnés dans la conversation."""
        import re

        milestones: set[str] = set()

        for turn in self._turns:
            # Chercher dans query et answer
            text = f"{turn.query} {turn.answer}"
            found = re.findall(r"\b([MJ]\d)\b", text, re.IGNORECASE)
            milestones.update(m.upper() for m in found)

        return milestones

    def get_referenced_phases(self) -> set[int]:
        """Retourne les numéros de phases mentionnés."""
        import re

        phases: set[int] = set()

        for turn in self._turns:
            text = f"{turn.query} {turn.answer}"
            found = re.findall(r"[Pp]hase\s+(\d+)", text)
            phases.update(int(p) for p in found if 1 <= int(p) <= 7)

        return phases

    def to_messages(self) -> list[dict[str, str]]:
        """Convertit l'historique en format messages (pour API chat).

        Returns:
            Liste de messages {"role": "user"|"assistant", "content": ...}
        """
        messages = []
        for turn in self._turns:
            messages.append({"role": "user", "content": turn.query})
            messages.append({"role": "assistant", "content": turn.answer})
        return messages

    def __len__(self) -> int:
        """Retourne le nombre de tours."""
        return len(self._turns)

    def __repr__(self) -> str:
        """Représentation string."""
        return f"GRIMemory(turns={len(self._turns)}, max={self._max_turns})"


# Singleton pour usage global
_memory: GRIMemory | None = None


def get_memory() -> GRIMemory:
    """Retourne le singleton de la mémoire."""
    global _memory
    if _memory is None:
        _memory = GRIMemory()
    return _memory


def reset_memory() -> GRIMemory:
    """Réinitialise le singleton de la mémoire."""
    global _memory
    _memory = GRIMemory()
    return _memory
