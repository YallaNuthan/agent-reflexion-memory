"""
Regression tests for reflexion_core/memory_store.py logic fixes.

These exist specifically to prevent silent regressions of bugs found
during the hardening pass:
  1. CROSS_AGENT_SIMILARITY_THRESHOLD / RULE_DECAY_DAYS were hardcoded
     constants that silently ignored .env / settings overrides.
  2. retrieve_rules() had no relevance floor, so an unrelated "closest"
     concept match could surface irrelevant rules instead of returning none.
  3. link_concept_parent() had no cycle protection — a concept could be
     linked as its own ancestor, producing a nonsensical hierarchy.
  4. _cross_agent_reinforce() used a read-modify-write pattern on Neo4j
     confidence values, which could lose increments under concurrent calls.
"""

import os
os.environ.setdefault("GROQ_API_KEY", "test-dummy-key")
os.environ.setdefault("API_ACCESS_KEY", "dev-secret-key")

import importlib
import pytest
from unittest.mock import patch, MagicMock

import reflexion_core.memory_store as memory_store_module
from reflexion_core.config import settings


# ── Test 1: thresholds are sourced from settings, not hardcoded ──────────────

def test_thresholds_are_driven_by_settings():
    """
    Guards against regressing to hardcoded constants that ignore .env.
    """
    assert memory_store_module.CROSS_AGENT_SIMILARITY_THRESHOLD == settings.cross_agent_similarity_threshold
    assert memory_store_module.RULE_DECAY_DAYS == settings.rule_decay_days

    with patch.object(settings, "cross_agent_similarity_threshold", 0.42), \
         patch.object(settings, "rule_decay_days", 99):
        importlib.reload(memory_store_module)
        assert memory_store_module.CROSS_AGENT_SIMILARITY_THRESHOLD == 0.42
        assert memory_store_module.RULE_DECAY_DAYS == 99

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
    repo.collection = mock_collection

    result = repo.retrieve_rules("some query that matches nothing relevant")

    assert result == []
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
    mock_session.run.return_value = []
    mock_graph_driver = MagicMock()
    mock_graph_driver.session.return_value.__enter__.return_value = mock_session
    mock_graphdb.driver.return_value = mock_graph_driver

    from reflexion_core.memory_store import MemoryRepository

    repo = MemoryRepository(agent_id="test_agent")
    repo.collection = mock_collection
    repo.graph = mock_graph_driver

    result = repo.retrieve_rules("a query that matches well")

    mock_graph_driver.session.assert_called()
    assert result == []


# ── Test 3: cycle protection in link_concept_parent (NOVEL-1) ────────────────

@patch("reflexion_core.memory_store.GraphDatabase")
@patch("reflexion_core.memory_store.embedding_functions")
@patch("reflexion_core.memory_store.chromadb")
def test_link_concept_parent_rejects_self_reference(
    mock_chromadb, mock_embedding_functions, mock_graphdb
):
    """
    A concept cannot be its own parent. This must be rejected before
    even touching the graph (cheap, deterministic check).
    """
    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_chromadb.PersistentClient.return_value = mock_client

    from reflexion_core.memory_store import MemoryRepository

    repo = MemoryRepository(agent_id="test_agent")

    with pytest.raises(ValueError, match="cannot be its own parent"):
        repo.link_concept_parent("HTTP_REQUESTS", "HTTP_REQUESTS")

    repo.graph.session.assert_not_called()


@patch("reflexion_core.memory_store.GraphDatabase")
@patch("reflexion_core.memory_store.embedding_functions")
@patch("reflexion_core.memory_store.chromadb")
def test_link_concept_parent_rejects_cycle(
    mock_chromadb, mock_embedding_functions, mock_graphdb
):
    """
    If 'parent_concept' is already a descendant of 'child_concept' in the
    graph, linking child -> parent here would close a cycle. This must be
    rejected with a ValueError, and the MERGE/CREATE edge must never run.
    """
    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_chromadb.PersistentClient.return_value = mock_client

    mock_session = MagicMock()
    cycle_check_result = MagicMock()
    cycle_check_result.single.return_value = {"path_count": 1}  # path exists -> cycle
    mock_session.run.return_value = cycle_check_result

    mock_graph_driver = MagicMock()
    mock_graph_driver.session.return_value.__enter__.return_value = mock_session
    mock_graphdb.driver.return_value = mock_graph_driver

    from reflexion_core.memory_store import MemoryRepository

    repo = MemoryRepository(agent_id="test_agent")
    repo.graph = mock_graph_driver

    with pytest.raises(ValueError, match="would create a cycle"):
        repo.link_concept_parent(
            child_concept="ASYNC_HTTP_TIMEOUT",
            parent_concept="HTTP_REQUEST_BEST_PRACTICES"
        )

    assert mock_session.run.call_count == 1


@patch("reflexion_core.memory_store.GraphDatabase")
@patch("reflexion_core.memory_store.embedding_functions")
@patch("reflexion_core.memory_store.chromadb")
def test_link_concept_parent_allows_valid_link(
    mock_chromadb, mock_embedding_functions, mock_graphdb
):
    """
    Sanity check: a legitimate, non-cyclic link must still succeed and
    should issue exactly two queries — the cycle check, then the MERGE.
    """
    mock_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection
    mock_chromadb.PersistentClient.return_value = mock_client

    mock_session = MagicMock()
    cycle_check_result = MagicMock()
    cycle_check_result.single.return_value = {"path_count": 0}  # no cycle
    mock_session.run.return_value = cycle_check_result

    mock_graph_driver = MagicMock()
    mock_graph_driver.session.return_value.__enter__.return_value = mock_session
    mock_graphdb.driver.return_value = mock_graph_driver

    from reflexion_core.memory_store import MemoryRepository

    repo = MemoryRepository(agent_id="test_agent")
    repo.graph = mock_graph_driver

    repo.link_concept_parent(
        child_concept="ASYNC_HTTP_TIMEOUT",
        parent_concept="HTTP_REQUEST_BEST_PRACTICES"
    )

    assert mock_session.run.call_count == 2


# ── Test 4: atomic increment in cross-agent reinforcement (NOVEL-2) ─────────

@patch("reflexion_core.memory_store.GraphDatabase")
@patch("reflexion_core.memory_store.embedding_functions")
@patch("reflexion_core.memory_store.chromadb")
def test_cross_agent_reinforce_uses_atomic_neo4j_increment(
    mock_chromadb, mock_embedding_functions, mock_graphdb
):
    """
    Guards against regressing to a read-modify-write pattern for confidence
    increments. The Neo4j query must perform 'r.confidence = r.confidence + 1'
    server-side and return the new value — not read a Python-side confidence
    value and SET it back, which would lose concurrent increments.
    """
    mock_source_collection = MagicMock()
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_source_collection
    mock_chromadb.PersistentClient.return_value = mock_client

    mock_peer_collection = MagicMock()
    mock_peer_collection.count.return_value = 1
    mock_peer_collection.query.return_value = {
        "ids": [["rule_peer_1"]],
        "metadatas": [[{"confidence": 3, "concept": "HTTP_REQUEST_BEST_PRACTICES"}]],
        "distances": [[0.1]],  # similarity = 1 - (0.1/2) = 0.95, above threshold
    }

    mock_collection_info = MagicMock()
    mock_collection_info.name = "agent_peer_agent_rules"
    mock_client.list_collections.return_value = [mock_collection_info]
    mock_client.get_collection.return_value = mock_peer_collection

    mock_session = MagicMock()
    increment_result = MagicMock()
    increment_result.single.return_value = {"new_confidence": 4}
    mock_session.run.return_value = increment_result

    mock_graph_driver = MagicMock()
    mock_graph_driver.session.return_value.__enter__.return_value = mock_session
    mock_graphdb.driver.return_value = mock_graph_driver

    from reflexion_core.memory_store import MemoryRepository

    repo = MemoryRepository(agent_id="source_agent")
    repo.graph = mock_graph_driver
    repo.chroma_client = mock_client

    repo._cross_agent_reinforce(
        rule_text="Always include a timeout parameter in HTTP requests",
        source_agent_id="source_agent"
    )

    cypher_query = mock_session.run.call_args[0][0]
    assert "r.confidence + 1" in cypher_query
    assert "RETURN" in cypher_query

    mock_peer_collection.update.assert_called_once()
    update_kwargs = mock_peer_collection.update.call_args
    updated_metadata = update_kwargs.kwargs["metadatas"][0]
    assert updated_metadata["confidence"] == 4
    
    
    
