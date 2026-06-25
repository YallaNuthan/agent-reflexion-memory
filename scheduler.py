"""
scheduler.py — Temporal Decay Background Job  [NOVEL-3]
========================================================
Run this instead of uvicorn directly:
    python scheduler.py

Starts APScheduler (decay every 24h) + FastAPI server together.
"""

import chromadb
import uvicorn
from apscheduler.schedulers.background import BackgroundScheduler
from reflexion_core.memory_store import MemoryRepository

DECAY_INTERVAL_HOURS = 24


def run_decay_all_agents():
    """[NOVEL-3] Auto-discovers all agent collections and runs temporal decay on each."""
    print("[Scheduler] Running temporal decay across all agents...")
    client = chromadb.PersistentClient(path="./chroma_db")
    collections = client.list_collections()

    total_decayed = 0
    total_deleted = 0

    for col in collections:
        col_name = col.name if hasattr(col, "name") else str(col)
        if not col_name.startswith("agent_") or not col_name.endswith("_rules"):
            continue

        agent_id = col_name[len("agent_"):-len("_rules")]

        try:
            repo   = MemoryRepository(agent_id=agent_id)
            result = repo.run_temporal_decay()
            repo.graph.close()
            total_decayed += result.get("decayed", 0)
            total_deleted += result.get("deleted", 0)
            print(f"  Agent '{agent_id}': decayed={result['decayed']}, deleted={result['deleted']}")
        except Exception as e:
            print(f"  Agent '{agent_id}': ERROR — {e}")

    print(f"[Scheduler] Done. Total decayed={total_decayed}, deleted={total_deleted}")


if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.add_job(
        run_decay_all_agents,
        trigger="interval",
        hours=DECAY_INTERVAL_HOURS,
        id="temporal_decay_job"
    )
    scheduler.start()
    print(f"[Scheduler] Temporal decay started — runs every {DECAY_INTERVAL_HOURS}h")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)