# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Agent Reflexion Memory, please **do not** open a public GitHub issue.

Instead, email the maintainer directly at the address on your GitHub profile, or use GitHub's private vulnerability reporting feature (Security → Report a vulnerability).

Please include:
- A description of the vulnerability
- Steps to reproduce
- Potential impact

You can expect a response within 48 hours.

## Scope

This project handles:
- API keys (GROQ_API_KEY, API_ACCESS_KEY) — stored in `.env`, never committed
- Neo4j credentials — configurable via `.env`
- Agent memory data — stored locally in ChromaDB and Neo4j

## Known Limitations

- ChromaDB has no native atomic increment; confidence updates are eventually consistent across Neo4j and ChromaDB under concurrent load
- `agent_id` is validated (alphanumeric, 1-64 chars) but not bound to auth identity — multi-tenant deployments should add agent-to-key binding at the API gateway layer