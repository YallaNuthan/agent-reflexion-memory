# Contributing to Agent Reflexion Memory

Thank you for your interest in contributing! This project implements three novel
mechanisms for autonomous AI agent memory — please read below before submitting.

---

## Before You Start

- Check existing [Issues](https://github.com/YallaNuthan/agent-reflexion-memory-v2/issues) to avoid duplicates
- For large changes, open an issue first to discuss the approach
- All PRs must pass the CI pipeline (14 tests) before review

---

## Setup

\`\`\`bash
git clone https://github.com/YallaNuthan/agent-reflexion-memory-v2.git
cd agent-reflexion-memory-v2
python -m venv myenv
myenv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env          # fill in your keys
\`\`\`

Start Neo4j (required for graph features):

\`\`\`bash
docker-compose up -d neo4j
\`\`\`

Run tests:

\`\`\`bash
python -m pytest tests/ -v
\`\`\`

---

## What You Can Contribute

**Good first issues:**
- Add more test cases to `tests/test_api.py`
- Improve error messages in `reflexion_core/`
- Add logging to `reflexion_engine.py`

**Intermediate:**
- Add APScheduler integration for automatic temporal decay
- Add pagination to `/v1/rules` endpoint
- Write integration tests with a real Neo4j instance

**Advanced (discuss first):**
- Changes to NOVEL-1, NOVEL-2, or NOVEL-3 core logic
- New storage backends (Pinecone, Weaviate)
- Multi-hop graph traversal beyond 2 hops

---

## Pull Request Rules

1. **One change per PR** — keep it focused
2. **Tests required** — new endpoints or logic must have tests
3. **No breaking changes** to the 3 novel mechanisms without discussion
4. **Run tests locally** before pushing — `python -m pytest tests/ -v`
5. **Clear commit messages** — e.g. `fix: close Neo4j connection on error`

---

## Commit Message Format

\`\`\`
type: short description
\`\`\`

Types: feat | fix | test | docs | chore | refactor

Examples:
- `feat: add pagination to /v1/rules`
- `fix: handle empty rule_ids in reinforce`
- `test: add agent_id validation tests`
- `docs: update setup instructions`

---

## Novel Contributions — Handle With Care

This project contains 3 patent-targeted novel mechanisms:

| Tag | Mechanism | Core Files |
|-----|-----------|------------|
| NOVEL-1 | Hierarchical Concept Inheritance Retrieval | `memory_store.py`, `reflexion_engine.py` |
| NOVEL-2 | Cross-Agent Confidence Reinforcement | `memory_store.py` |
| NOVEL-3 | Temporal Decay on Distilled Behavioral Rules | `memory_store.py`, `api.py` |

Changes to these must be discussed in an issue first.

---

## Questions?

Open an issue with the `question` label.
Open an issue with the `question` label.