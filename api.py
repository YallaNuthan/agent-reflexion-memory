"""
Agent Reflexion Memory — FastAPI Application
=============================================
Endpoints:
  POST /v1/reflect          — Distill failure into rule (all 3 novel features active)
  POST /v1/rules            — Hierarchical rule retrieval         [NOVEL-1]
  POST /v1/reinforce        — Confidence update + timestamp reset  [NOVEL-3]
  POST /v1/decay            — Manual temporal decay trigger        [NOVEL-3]
  POST /v1/concepts/link    — Manually link concept parent         [NOVEL-1]
  GET  /v1/concepts/hierarchy — View concept hierarchy for agent   [NOVEL-1]
  GET  /health
"""

import re
import logging
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, field_validator
from typing import Optional
from reflexion_core.reflexion_engine import ReflexionEngine
from reflexion_core.memory_store import MemoryRepository
import os

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agent Reflexion Memory API",
    description=(
        "Production-grade Graph-Vector Hybrid memory microservice for autonomous AI agents.\n\n"
        "Novel contributions:\n"
        "[NOVEL-1] Hierarchical Concept Inheritance Retrieval\n"
        "[NOVEL-2] Cross-Agent Confidence Reinforcement\n"
        "[NOVEL-3] Temporal Decay on Distilled Behavioral Rules"
    ),
    version="2.0.0"
)

api_key_header = APIKeyHeader(name="X-API-Key")

# ── Auth ──────────────────────────────────────────────────────────────────────

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == os.getenv("API_ACCESS_KEY", "dev-secret-key"):
        return api_key
    raise HTTPException(status_code=403, detail="Invalid or missing API Key")

# ── agent_id validator (shared by all models) ─────────────────────────────────

AGENT_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

def validate_agent_id(v: str) -> str:
    if not AGENT_ID_PATTERN.match(v):
        raise ValueError(
            "agent_id must be 1-64 characters and contain only "
            "letters, digits, underscores, or hyphens."
        )
    return v

# ── Pydantic Models ───────────────────────────────────────────────────────────

class ReflectRequest(BaseModel):
    agent_id: str = "default"
    task_description: str
    failure_reason: str

    @field_validator("agent_id")
    @classmethod
    def check_agent_id(cls, v): return validate_agent_id(v)

class RetrieveRequest(BaseModel):
    agent_id: str = "default"
    task_description: str
    top_k: int = 5

    @field_validator("agent_id")
    @classmethod
    def check_agent_id(cls, v): return validate_agent_id(v)

class ReinforceRequest(BaseModel):
    agent_id: str = "default"
    rule_ids: list
    success: bool = True

    @field_validator("agent_id")
    @classmethod
    def check_agent_id(cls, v): return validate_agent_id(v)

class DecayRequest(BaseModel):
    agent_id: str = "default"

    @field_validator("agent_id")
    @classmethod
    def check_agent_id(cls, v): return validate_agent_id(v)

class LinkConceptRequest(BaseModel):
    agent_id: str = "default"
    child_concept: str
    parent_concept: str

    @field_validator("agent_id")
    @classmethod
    def check_agent_id(cls, v): return validate_agent_id(v)

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/v1/reflect", tags=["Memory"], dependencies=[Depends(get_api_key)])
def reflect_on_failure(request: ReflectRequest):
    """Distills failure → rule. [NOVEL-1] hierarchy, [NOVEL-2] cross-agent, [NOVEL-3] timestamp."""
    eng = ReflexionEngine(agent_id=request.agent_id)
    try:
        rule = eng.reflect_on_failure(request.task_description, request.failure_reason)
        return {"status": "success", "learned_rule": rule}
    except Exception:
        logger.exception("Error in /v1/reflect for agent_id=%s", request.agent_id)
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs.")
    finally:
        eng.repo.graph.close()

@app.post("/v1/rules", tags=["Memory"], dependencies=[Depends(get_api_key)])
def get_rules(request: RetrieveRequest):
    """[NOVEL-1] Hierarchical retrieval — direct concept + ancestor concept rules."""
    eng = ReflexionEngine(agent_id=request.agent_id)
    try:
        result = eng.get_relevant_rules_prompt(request.task_description)
        return result
    except Exception:
        logger.exception("Error in /v1/rules for agent_id=%s", request.agent_id)
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs.")
    finally:
        eng.repo.graph.close()

@app.post("/v1/reinforce", tags=["Memory"], dependencies=[Depends(get_api_key)])
def reinforce_rules(request: ReinforceRequest):
    """[NOVEL-3] Updates confidence + resets decay clock timestamp."""
    eng = ReflexionEngine(agent_id=request.agent_id)
    try:
        eng.reinforce_rules(request.rule_ids, request.success)
        return {"status": "success", "message": "Rule confidence updated and decay clock reset."}
    except Exception:
        logger.exception("Error in /v1/reinforce for agent_id=%s", request.agent_id)
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs.")
    finally:
        eng.repo.graph.close()

@app.post("/v1/decay", tags=["Temporal Decay [NOVEL-3]"], dependencies=[Depends(get_api_key)])
def run_temporal_decay(request: DecayRequest):
    """[NOVEL-3] Manual trigger: decrement confidence on stale rules, delete at zero."""
    repo = MemoryRepository(agent_id=request.agent_id)
    try:
        result = repo.run_temporal_decay()
        return {"status": "success", "decay_result": result}
    except Exception:
        logger.exception("Error in /v1/decay for agent_id=%s", request.agent_id)
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs.")
    finally:
        repo.graph.close()

@app.post("/v1/concepts/link", tags=["Hierarchy [NOVEL-1]"], dependencies=[Depends(get_api_key)])
def link_concept(request: LinkConceptRequest):
    """[NOVEL-1] Creates PARENT_CONCEPT edge in Neo4j between two concept nodes."""
    repo = MemoryRepository(agent_id=request.agent_id)
    try:
        repo.link_concept_parent(request.child_concept, request.parent_concept)
        return {"status": "success", "message": f"Linked '{request.child_concept}' → parent '{request.parent_concept}'"}
    except Exception:
        logger.exception("Error in /v1/concepts/link for agent_id=%s", request.agent_id)
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs.")
    finally:
        repo.graph.close()

@app.get("/v1/concepts/hierarchy", tags=["Hierarchy [NOVEL-1]"], dependencies=[Depends(get_api_key)])
def get_hierarchy(agent_id: str = "default"):
    """[NOVEL-1] Returns full concept hierarchy tree for the given agent."""
    validate_agent_id(agent_id)
    repo = MemoryRepository(agent_id=agent_id)
    try:
        result = repo.get_concept_hierarchy()
        return {"agent_id": agent_id, "hierarchy": result}
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("Error in /v1/concepts/hierarchy for agent_id=%s", agent_id)
        raise HTTPException(status_code=500, detail="Internal server error. Check server logs.")
    finally:
        repo.graph.close()

@app.get("/health", tags=["System"])
def health_check():
    return {
        "status":  "healthy",
        "version": "2.0.0",
        "novel_features": [
            "NOVEL-1: Hierarchical Concept Inheritance Retrieval",
            "NOVEL-2: Cross-Agent Confidence Reinforcement",
            "NOVEL-3: Temporal Decay on Distilled Behavioral Rules"
        ]
    }