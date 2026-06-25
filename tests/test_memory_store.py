"""
Regression tests for reflexion_core/memory_store.py logic fixes.

These exist specifically to prevent silent regressions of two bugs
found during the hardening pass:
  1. CROSS_AGENT_SIMILARITY_THRESHOLD / RULE_DECAY_DAYS were hardcoded
     constants that silently ignored .env / settings overrides.
  2. retrieve_rules() had no relevance floor, so an unrelated "closest"
     concept match could surface irrelevant rules instead of returning none.
"""

import os
os.environ.setdefault("GROQ_API_KEY", "test-dummy-key")
os.environ.setdefault("API_ACCESS_KEY", "dev-secret-key")

import importlib
from unittest.mock import patch, MagicMock

import reflexion_core.memory_store as memory_store_module
from reflexion_core.config import settings


# ── Test 1: thresholds are sourced from settings, not hardcoded ──────────────

def test_thresholds_are_driven_by_settings():
    """
    Guards against regressing to hardcoded constants that ignore .env.
    If someone reverts to `CROSS_AGENT_SIMILARITY_THRESHOLD = 0.85` as a
    bare literal, this test still passes only if settings.* happens to
    equal 0.85/7 — so we also assert equality is *not* coincidental by
    checking it tracks a non-default override.
    """
    # Sanity: current module-level values must match settings right now
    assert memory_store_module.CROSS_AGENT_SIMILARITY_THRESHOLD == settings.cross_agent_similarity_threshold
    assert memory_store_module.RULE_DECAY_DAYS == settings.rule_decay_days

    # Stronger check: override settings, reload the module, confirm the
    # module-level constants follow the override rather than staying fixed.
    with patch.object(settings, "cross_agent_similarity_threshold", 0.42), \
         patch.object(settings, "rule_decay_days", 99):
        importlib.reload(memory_store_module)
        assert memory_store_module.CROSS_AGENT_SIMILARITY_THRESHOLD == 0.42
        assert memory_store_module.RULE_DECAY_DAYS == 99

    # Reload again to restore normal state for any tests that run after this
    importlib.reload(memory_store_module)


# ── Test 2: relevance floor in retrieve_rules ────────────────────────────────

@patch("reflexion_core.memory_store.GraphDatabase")
@patch("reflexion_core.memory_store.embedding_functions")
@patch("reflexion_core.memory_store.chromadb")
def test_retrieve_rules_returns_empty_when_closest_match_too_far(
    mock_chromadb, mock_embedding_functions, mock_graphdb
):
    """
    If even the closest semantic match has a cosine distance beyond the
    relevance floor (MAX_RELEVANT_DISTANCE), retrieve_rules must return []
    instead of surfacing an unrelated concept's rules.
    """
    mock_collection = MagicMock()
    mock_collection.count.return_value = 5
    mock_collection.query.return_value = {
        "documents": [["some unrelated rule text"]],
        "metadatas": [[{"concept": "UNRELATED_CONCEPT"}]],
        "distances": [[1.8]],  # well beyond MAX_RELEVANT_DISTANCE = 1.2
        "ids": [["rule_x_1"]],
    }

    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_chromadb.PersistentClient.return_value = mock_client

    from reflexion_core.memory_store import MemoryRepository

    repo = MemoryRepository(agent_id="test_agent")
    repo.collection = mock_collection  # ensure our mock collection is used

    result = repo.retrieve_rules("some query that matches nothing relevant")

    assert result == []
    # Graph should never even be queried if we bail out early on relevance
    repo.graph.session.assert_not_called()


@patch("reflexion_core.memory_store.GraphDatabase")
@patch("reflexion_core.memory_store.embedding_functions")
@patch("reflexion_core.memory_store.chromadb")
def test_retrieve_rules_proceeds_when_closest_match_is_relevant(
    mock_chromadb, mock_embedding_functions, mock_graphdb
):
    """
    Sanity check: a close match (small distance) should NOT be filtered out
    by the relevance floor, and should proceed to graph traversal.
    """
    mock_collection = MagicMock()
    mock_collection.count.return_value = 5
    mock_collection.query.return_value = {
        "documents": [["a relevant rule"]],
        "metadatas": [[{"concept": "HTTP_REQUEST_BEST_PRACTICES"}]],
        "distances": [[0.1]],  # well within MAX_RELEVANT_DISTANCE = 1.2
        "ids": [["rule_x_2"]],
    }

    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_chromadb.PersistentClient.return_value = mock_client

    mock_session = MagicMock()
    mock_session.run.return_value = []  # no graph rules found, that's fine
    mock_graph_driver = MagicMock()
    mock_graph_driver.session.return_value.__enter__.return_value = mock_session
    mock_graphdb.driver.return_value = mock_graph_driver

    from reflexion_core.memory_store import MemoryRepository

    repo = MemoryRepository(agent_id="test_agent")
    repo.collection = mock_collection
    repo.graph = mock_graph_driver

    result = repo.retrieve_rules("a query that matches well")

    # Graph traversal SHOULD have been attempted since relevance floor passed
    mock_graph_driver.session.assert_called()
    assert result == []  # empty because our mocked session.run returns []