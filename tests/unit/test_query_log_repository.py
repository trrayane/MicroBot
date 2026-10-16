from __future__ import annotations

from pathlib import Path

from mc_rag.infrastructure.persistence.query_log_repository import QueryLogRepository


def test_list_repetitive_groups_normalized_queries_and_orders_by_counts(tmp_path: Path) -> None:
    repository = QueryLogRepository(tmp_path / "queries.sqlite3")

    for _ in range(2):
        query_id = repository.add_query("conversation-a", "How to join MC?")
        repository.set_feedback(
            "down", query_log_id=query_id, conversation_id="conversation-a"
        )
    for _ in range(2):
        repository.add_query("conversation-b", "HOW TO JOIN MC")
    for _ in range(4):
        repository.add_query("conversation-c", "What are the departments?")
    repository.add_query("conversation-d", "Is the office open?")

    results = repository.list_repetitive(min_frequency=2)

    assert [(r.normalized_query, r.total_count, r.unuseful_count) for r in results] == [
        ("how to join mc", 4, 2),
        ("what are the departments", 4, 0),
    ]
    assert repository.list_repetitive(min_frequency=4) == results
    assert repository.list_repetitive(min_frequency=5) == []