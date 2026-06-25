# agent-reflexion-memory

> A production-grade, Graph-Vector Hybrid memory microservice for autonomous AI agents. Stop feeding your agents massive conversation logs. Teach them dense, actionable rules.

[![CI](https://github.com/YallaNuthan/agent-reflexion-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/YallaNuthan/agent-reflexion-memory/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-009688.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![Neo4j](https://img.shields.io/badge/Neo4j-GraphDB-008CC1.svg)](https://neo4j.com/)

## The Bottleneck: Agent Amnesia

Frameworks like CrewAI, LangGraph, and AutoGen are powerful, but they have amnesia. If an agent fails a task on Tuesday, it will make the exact same mistake on Wednesday.

Current memory solutions (like Mem0) solve this by dumping past conversation logs into a Vector DB.

**The Problem:** Vector DBs store *what happened*, not *what was learned*. The agent retrieves a massive wall of text, burning tokens and context window without actually improving behavior.

## The Solution: Semantic Distillation + Graph DB

Instead of saving logs, this microservice uses a Reflexion Loop:

1. **Evaluate:** A strict reviewer checks the agent's output for failures.
2. **Distill:** A lightweight LLM extracts the failure into a single, dense Rule (e.g., *"Always include a 10s timeout"*).
3. **Categorize:** The LLM extracts a Concept Category (e.g., `HTTP_REQUEST_BEST_PRACTICES`), with an optional parent concept for hierarchical inheritance.
4. **Store:** The rule is stored in ChromaDB (Vector) and Neo4j (Graph), linking related rules together conceptually.
5. **Retrieve:** Before any new task, the agent retrieves the top relevant rules — including inherited rules from parent concepts — saving the bulk of context tokens.

## Architecture

- **Graph-Vector Hybrid:** ChromaDB for semantic search, Neo4j for conceptual hierarchies.
- **Multi-Tenancy:** Pass an `agent_id` to give every distinct agent its own isolated memory namespace.
- **Hierarchical Concept Inheritance [NOVEL-1]:** Rules stored under a concept automatically surface to queries on related parent/child concepts.
- **Cross-Agent Confidence Reinforcement [NOVEL-2]:** When one agent learns a rule, semantically matching rules in other agents' namespaces get their confidence reinforced — without duplicating storage.
- **Temporal Rule Decay [NOVEL-3]:** Rules have a confidence score. It increases on success, decreases on failure or staleness. Rules hitting 0 are automatically deleted.
- **Secured Microservice:** Built with FastAPI, Dockerized, and secured with API Key authentication and `agent_id` input validation.

## Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- A free [Groq API key](https://console.groq.com/keys)

### 1. Clone the repository

    git clone https://github.com/YallaNuthan/agent-reflexion-memory.git
    cd agent-reflexion-memory

### 2. Set up your environment

    cp .env.example .env

Open `.env` and fill in:

    GROQ_API_KEY=your_groq_key_here
    API_ACCESS_KEY=choose_a_secret_key
    NEO4J_URI=bolt://neo4j:7687
    NEO4J_USER=neo4j
    NEO4J_PASSWORD=password123

### 3a. Run with Docker (recommended)

    docker-compose up --build

### 3b. Run locally without Docker

    python -m venv myenv
    myenv\Scripts\activate          # Windows
    source myenv/bin/activate       # macOS/Linux

    pip install -r requirements.txt

    # Start Neo4j separately (or via docker-compose up -d neo4j)
    uvicorn api:app --reload

### 4. Verify it's running

Open the interactive API docs:

    http://localhost:8000/docs

Or check health:

    curl http://localhost:8000/health

### 5. Run the test suite

    python -m pytest tests/ -v

All 24 tests should pass.

## API Usage

All endpoints (except `/health`) require an `X-API-Key` header matching your `API_ACCESS_KEY`.

### `POST /v1/reflect` — Learn from a failure

    curl -X POST http://localhost:8000/v1/reflect \
      -H "X-API-Key: your_key" \
      -H "Content-Type: application/json" \
      -d '{
            "agent_id": "agent_alpha",
            "task_description": "Fetch user data from REST API",
            "failure_reason": "Request timed out after 30s"
          }'

### `POST /v1/rules` — Retrieve relevant rules for a task

    curl -X POST http://localhost:8000/v1/rules \
      -H "X-API-Key: your_key" \
      -H "Content-Type: application/json" \
      -d '{
            "agent_id": "agent_alpha",
            "task_description": "Call a third-party weather API"
          }'

### `POST /v1/reinforce` — Update rule confidence

    curl -X POST http://localhost:8000/v1/reinforce \
      -H "X-API-Key: your_key" \
      -H "Content-Type: application/json" \
      -d '{
            "agent_id": "agent_alpha",
            "rule_ids": ["rule_agent_alpha_123"],
            "success": true
          }'

### `POST /v1/decay` — Trigger temporal decay manually

    curl -X POST http://localhost:8000/v1/decay \
      -H "X-API-Key: your_key" \
      -H "Content-Type: application/json" \
      -d '{"agent_id": "agent_alpha"}'

### `POST /v1/concepts/link` — Link a concept to a parent

    curl -X POST http://localhost:8000/v1/concepts/link \
      -H "X-API-Key: your_key" \
      -H "Content-Type: application/json" \
      -d '{
            "agent_id": "agent_alpha",
            "child_concept": "ASYNC_HTTP_TIMEOUT",
            "parent_concept": "HTTP_REQUEST_BEST_PRACTICES"
          }'

### `GET /v1/concepts/hierarchy` — View an agent's concept tree

    curl "http://localhost:8000/v1/concepts/hierarchy?agent_id=agent_alpha" \
      -H "X-API-Key: your_key"

### `GET /health` — Health check (no auth required)

    curl http://localhost:8000/health

## Tech Stack

- **FastAPI** — REST API layer
- **ChromaDB** — vector embeddings + semantic search
- **Neo4j** — concept graph + hierarchy traversal
- **Groq (Llama)** — LLM-powered rule distillation and concept categorization
- **APScheduler** — autonomous temporal decay job
- **pytest** — 14-test suite covering all endpoints, auth, and error sanitization
- **GitHub Actions** — CI pipeline running on every push/PR
- **Docker / Docker Compose** — containerized deployment

## Benchmark Results

Agent Reflexion Memory achieves **37–54% token savings** vs raw-log memory (Mem0-style),
with all 3 novel mechanisms verified live against real Neo4j + ChromaDB.

See [benchmarks/RESULTS.md](benchmarks/RESULTS.md) for the full breakdown including NOVEL-1
hierarchy retrieval, NOVEL-2 cross-agent reinforcement, and NOVEL-3 temporal decay
verification.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for setup details, coding guidelines, and how the three novel mechanisms (NOVEL-1, NOVEL-2, NOVEL-3) are protected from breaking changes.

## Credits

Built by [YallaNuthan](https://github.com/YallaNuthan).

