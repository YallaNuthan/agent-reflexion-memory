from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from reflexion_core.reflexion_engine import ReflexionEngine
import os

app = FastAPI(
    title="Agent Reflexion Memory API",
    description="Enterprise-grade memory layer for autonomous AI agents using Semantic Distillation, Multi-Tenancy, and Rule Decay.",
    version="2.0.0"
)

# --- API Authentication ---
api_key_header = APIKeyHeader(name="X-API-Key")

def get_api_key(api_key: str = Security(api_key_header)):
    if api_key == os.getenv("API_ACCESS_KEY", "dev-secret-key"):
        return api_key
    raise HTTPException(status_code=403, detail="Invalid or missing API Key")

# --- Pydantic Models ---
class ReflectRequest(BaseModel):
    agent_id: str = "default"
    task_description: str
    failure_reason: str

class RetrieveRequest(BaseModel):
    agent_id: str = "default"
    task_description: str
    top_k: int = 3

class ReinforceRequest(BaseModel):
    agent_id: str = "default"
    rule_ids: list
    success: bool = True

# --- Secured API Endpoints ---
@app.post("/v1/reflect", tags=["Memory"], dependencies=[Depends(get_api_key)])
def reflect_on_failure(request: ReflectRequest):
    """Distills a failure into a dense rule and stores it in the agent's specific Vector DB."""
    try:
        # Instantiate engine directly on the request to avoid httpx client closure bugs
        eng = ReflexionEngine(agent_id=request.agent_id)
        rule = eng.reflect_on_failure(request.task_description, request.failure_reason)
        eng.repo.graph.close() # Cleanly close Neo4j connection
        return {"status": "success", "learned_rule": rule}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/rules", tags=["Memory"], dependencies=[Depends(get_api_key)])
def get_rules(request: RetrieveRequest):
    """Retrieve top-K relevant rules for a given task to inject into system prompts."""
    try:
        eng = ReflexionEngine(agent_id=request.agent_id)
        result = eng.get_relevant_rules_prompt(request.task_description)
        eng.repo.graph.close()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/reinforce", tags=["Memory"], dependencies=[Depends(get_api_key)])
def reinforce_rules(request: ReinforceRequest):
    """Rule Decay endpoint: Increase confidence if rule helped, decrease if it failed."""
    try:
        eng = ReflexionEngine(agent_id=request.agent_id)
        eng.reinforce_rules(request.rule_ids, request.success)
        eng.repo.graph.close()
        return {"status": "success", "message": "Rule confidence updated."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy", "version": "2.0.0"}