"""Conversation memory with access rules.

* ``conversation_id=None`` starts a new conversation.
* A conversation that a member has used is **member-only**: an unauthenticated
  caller who knows its id gets "not found", never the history.
* An anonymous conversation that a member continues is upgraded to member-only,
  because its stored history may then contain member-only content.
"""

from __future__ import annotations

from mc_rag.domain.exceptions import ConversationNotFoundError
from mc_rag.infrastructure.persistence.conversation_repository import ConversationRepository

DEFAULT_HISTORY_MESSAGES = 6


class ConversationService:
    def __init__(
        self,
        repository: ConversationRepository,
        *,
        history_messages: int = DEFAULT_HISTORY_MESSAGES,
    ) -> None:
        self._repo = repository
        self._history_messages = history_messages

    def open(
        self, conversation_id: str | None, *, is_member: bool
    ) -> tuple[str, list[dict[str, str]], str | None]:
        """Return ``(conversation_id, recent_history, title)`` for this caller."""
        if conversation_id is None:
            return self._repo.create(is_member=is_member), [], None

        record = self._repo.get(conversation_id)
        if record is None or (record.is_member and not is_member):
            raise ConversationNotFoundError("Conversation not found.")
        if is_member and not record.is_member:
            self._repo.mark_member(conversation_id)
        history = self._repo.recent_messages(conversation_id, self._history_messages)
        return conversation_id, history, record.title

    def add_turn(self, conversation_id: str, question: str, answer: str) -> None:
        self._repo.add_messages(conversation_id, [("user", question), ("assistant", answer)])

    def set_title(self, conversation_id: str, title: str) -> None:
        """Persist the auto-generated title of a conversation."""
        self._repo.set_title(conversation_id, title)