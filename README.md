## 📡 Key Endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/reflect` | Distill a failure into a corrective rule |
| `POST /v1/rules` | Retrieve relevant rules (hierarchical) for a task |
| `POST /v1/reinforce` | Update rule confidence after success/failure |
| `POST /v1/decay` | Trigger temporal decay on stale rules |
| `POST /v1/concepts/link` | Link a concept to a parent concept |
| `GET /v1/concepts/hierarchy` | View the concept hierarchy for an agent |
| `GET /health` | Health check |

## 🔧 Tech Stack

- **FastAPI** — REST API layer
- **ChromaDB** — vector embeddings + semantic search
- **Neo4j** — concept graph + hierarchy traversal
- **Groq (Llama)** — LLM-powered rule distillation and concept categorization
- **APScheduler** — autonomous temporal decay job
- **Docker / Docker Compose** — containerized deployment

## 🙋 Credits

Built by [YallaNuthan](https://github.com/YallaNuthan).