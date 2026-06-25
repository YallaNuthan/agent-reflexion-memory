import os
os.environ.setdefault("GROQ_API_KEY", "test-dummy-key")
os.environ.setdefault("API_ACCESS_KEY", "dev-secret-key")

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)
VALID_HEADERS = {"X-API-Key": "dev-secret-key"}


# ── /health ───────────────────────────────────────────────────────────────────

def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["version"] == "2.0.0"
    assert len(data["novel_features"]) == 3


# ── /v1/reflect ───────────────────────────────────────────────────────────────

@patch("api.ReflexionEngine")
def test_reflect_success(mock_engine_cls):
    instance = MagicMock()
    instance.reflect_on_failure.return_value = "Always include a 10s timeout."
    mock_engine_cls.return_value = instance

    resp = client.post("/v1/reflect", headers=VALID_HEADERS, json={
        "agent_id": "test_agent",
        "task_description": "Fetch user data from REST API",
        "failure_reason": "Request timed out after 30s"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "learned_rule" in data
    instance.repo.graph.close.assert_called_once()


@patch("api.ReflexionEngine")
def test_reflect_missing_api_key(mock_engine_cls):
    resp = client.post("/v1/reflect", json={
        "agent_id": "test_agent",
        "task_description": "Fetch data",
        "failure_reason": "Failed"
    })
    assert resp.status_code == 401  # missing header = 401, wrong key = 403


@patch("api.ReflexionEngine")
def test_reflect_wrong_api_key(mock_engine_cls):
    resp = client.post("/v1/reflect",
                       headers={"X-API-Key": "totally-wrong"},
                       json={"agent_id": "x", "task_description": "x", "failure_reason": "x"})
    assert resp.status_code == 403


@patch("api.ReflexionEngine")
def test_reflect_internal_error_is_sanitized(mock_engine_cls):
    """Raw exception text (e.g. DB credentials) must never reach the client."""
    instance = MagicMock()
    instance.reflect_on_failure.side_effect = Exception(
        "bolt+ssc://neo4j:supersecretpassword@db:7687 connection refused"
    )
    mock_engine_cls.return_value = instance

    resp = client.post("/v1/reflect", headers=VALID_HEADERS, json={
        "agent_id": "test_agent",
        "task_description": "Fetch data",
        "failure_reason": "Failed"
    })
    assert resp.status_code == 500
    assert "supersecretpassword" not in resp.json()["detail"]
    assert "Internal server error" in resp.json()["detail"]
    instance.repo.graph.close.assert_called_once()


# ── /v1/rules ─────────────────────────────────────────────────────────────────

@patch("api.ReflexionEngine")
def test_get_rules_success(mock_engine_cls):
    instance = MagicMock()
    instance.get_relevant_rules_prompt.return_value = {
        "prompt": "CRITICAL RULES: Always include timeout.",
        "rule_ids": ["rule_agent_123"]
    }
    mock_engine_cls.return_value = instance

    resp = client.post("/v1/rules", headers=VALID_HEADERS, json={
        "agent_id": "test_agent",
        "task_description": "Fetch data from weather API"
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "prompt" in data
    assert "rule_ids" in data
    instance.repo.graph.close.assert_called_once()


@patch("api.ReflexionEngine")
def test_get_rules_no_memory(mock_engine_cls):
    instance = MagicMock()
    instance.get_relevant_rules_prompt.return_value = {
        "prompt": "No prior rules learned.",
        "rule_ids": []
    }
    mock_engine_cls.return_value = instance

    resp = client.post("/v1/rules", headers=VALID_HEADERS, json={
        "agent_id": "brand_new_agent",
        "task_description": "Some task"
    })
    assert resp.status_code == 200
    assert resp.json()["rule_ids"] == []


# ── /v1/reinforce ─────────────────────────────────────────────────────────────

@patch("api.ReflexionEngine")
def test_reinforce_success(mock_engine_cls):
    instance = MagicMock()
    mock_engine_cls.return_value = instance

    resp = client.post("/v1/reinforce", headers=VALID_HEADERS, json={
        "agent_id": "test_agent",
        "rule_ids": ["rule_agent_123"],
        "success": True
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    instance.reinforce_rules.assert_called_once_with(["rule_agent_123"], True)
    instance.repo.graph.close.assert_called_once()


@patch("api.ReflexionEngine")
def test_reinforce_failure_decrements(mock_engine_cls):
    instance = MagicMock()
    mock_engine_cls.return_value = instance

    resp = client.post("/v1/reinforce", headers=VALID_HEADERS, json={
        "agent_id": "test_agent",
        "rule_ids": ["rule_agent_123"],
        "success": False
    })
    assert resp.status_code == 200
    instance.reinforce_rules.assert_called_once_with(["rule_agent_123"], False)
    instance.repo.graph.close.assert_called_once()


# ── /v1/decay  [NOVEL-3] ──────────────────────────────────────────────────────

@patch("api.MemoryRepository")
def test_decay_success(mock_repo_cls):
    instance = MagicMock()
    instance.run_temporal_decay.return_value = {"decayed": 2, "deleted": 1}
    mock_repo_cls.return_value = instance

    resp = client.post("/v1/decay", headers=VALID_HEADERS, json={"agent_id": "test_agent"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["decay_result"]["decayed"] == 2
    assert data["decay_result"]["deleted"] == 1
    instance.graph.close.assert_called_once()


@patch("api.MemoryRepository")
def test_decay_empty_returns_zeros(mock_repo_cls):
    instance = MagicMock()
    instance.run_temporal_decay.return_value = {"decayed": 0, "deleted": 0}
    mock_repo_cls.return_value = instance

    resp = client.post("/v1/decay", headers=VALID_HEADERS, json={"agent_id": "fresh_agent"})
    assert resp.status_code == 200
    assert resp.json()["decay_result"] == {"decayed": 0, "deleted": 0}


# ── /v1/concepts/link  [NOVEL-1] ──────────────────────────────────────────────

@patch("api.MemoryRepository")
def test_link_concepts_success(mock_repo_cls):
    instance = MagicMock()
    mock_repo_cls.return_value = instance

    resp = client.post("/v1/concepts/link", headers=VALID_HEADERS, json={
        "agent_id": "test_agent",
        "child_concept": "ASYNC_HTTP_TIMEOUT",
        "parent_concept": "HTTP_REQUEST_BEST_PRACTICES"
    })
    assert resp.status_code == 200
    assert resp.json()["status"] == "success"
    instance.link_concept_parent.assert_called_once_with(
        "ASYNC_HTTP_TIMEOUT", "HTTP_REQUEST_BEST_PRACTICES"
    )
    instance.graph.close.assert_called_once()


# ── /v1/concepts/hierarchy  [NOVEL-1] ─────────────────────────────────────────

@patch("api.MemoryRepository")
def test_get_hierarchy_success(mock_repo_cls):
    instance = MagicMock()
    instance.get_concept_hierarchy.return_value = [
        {"child": "ASYNC_HTTP_TIMEOUT", "parent": "HTTP_REQUEST_BEST_PRACTICES"}
    ]
    mock_repo_cls.return_value = instance

    resp = client.get("/v1/concepts/hierarchy?agent_id=test_agent", headers=VALID_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["agent_id"] == "test_agent"
    assert len(data["hierarchy"]) == 1
    assert data["hierarchy"][0]["child"] == "ASYNC_HTTP_TIMEOUT"
    instance.graph.close.assert_called_once()


@patch("api.MemoryRepository")
def test_get_hierarchy_empty(mock_repo_cls):
    instance = MagicMock()
    instance.get_concept_hierarchy.return_value = []
    mock_repo_cls.return_value = instance

    resp = client.get("/v1/concepts/hierarchy?agent_id=no_concepts_yet", headers=VALID_HEADERS)
    assert resp.status_code == 200
    assert resp.json()["hierarchy"] == []