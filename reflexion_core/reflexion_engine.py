"""
ReflexionEngine
===============
Orchestrates the full Reflexion Loop:
  1. Evaluate failure
  2. Distill → imperative rule (LLM call 1)
  3. Categorize → CONCEPT + optional PARENT_CONCEPT (LLM call 2)  [NOVEL-1]
  4. Store → MemoryRepository (triggers cross-agent reinforcement)  [NOVEL-2]
  5. Retrieve → hierarchical graph traversal                        [NOVEL-1]
"""

from groq import Groq
from .config import settings
from .memory_store import MemoryRepository
from .logger import MarkdownLogger


class ReflexionEngine:
    def __init__(self, agent_id: str = "default"):
        self.client   = Groq(api_key=settings.groq_api_key)
        self.model    = settings.llm_model
        self.repo     = MemoryRepository(agent_id=agent_id)
        self.logger   = MarkdownLogger()

    def _llm(self, system: str, user: str, temperature: float = 0.2) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

    def reflect_on_failure(self, task_description: str, failure_reason: str) -> str:
        """
        Full reflexion pipeline:
          LLM call 1 → distill failure into an imperative rule
          LLM call 2 → extract CONCEPT and optional PARENT_CONCEPT  [NOVEL-1]
          Store rule with hierarchy metadata
        """
        # LLM Call 1: Rule Distillation
        rule = self._llm(
            system=(
                "You are an AI Agent Memory Optimizer. "
                "Analyze why an autonomous agent failed a task and extract a single, "
                "dense, actionable RULE. The rule must be imperative "
                "(e.g., 'Always do X', 'Never do Y'). "
                "Output ONLY the rule. No explanations. Max 2 sentences."
            ),
            user=(
                f"Task Attempted: '{task_description}'\n"
                f"Failure Reason: '{failure_reason}'\n"
                f"Extract the corrective rule:"
            ),
            temperature=0.1
        )

        # LLM Call 2: Hierarchical Concept Extraction [NOVEL-1]
        concept_raw = self._llm(
            system="You are a hierarchical concept categorizer for an AI agent memory system.",
            user=(
                f"Given this behavioral rule: '{rule}'\n\n"
                "Output TWO lines ONLY:\n"
                "LINE 1: A specific uppercase concept (e.g., ASYNC_HTTP_TIMEOUT)\n"
                "LINE 2: A broader parent concept it belongs to (e.g., HTTP_REQUEST_BEST_PRACTICES)\n"
                "If no sensible parent exists, repeat the same concept on LINE 2.\n"
                "No other text."
            ),
            temperature=0.0
        )

        lines          = [l.strip().replace(" ", "_").upper() for l in concept_raw.splitlines() if l.strip()]
        concept        = lines[0] if len(lines) >= 1 else "GENERAL_RULE"
        parent_concept = lines[1] if len(lines) >= 2 else concept

        # Avoid trivial self-parent
        parent_to_store = None if parent_concept == concept else parent_concept

        # Store with all 3 novel features active
        self.repo.store_rule(
            rule_text=rule,
            task=task_description,
            failure=failure_reason,
            concept=concept,
            parent_concept=parent_to_store
        )
        self.logger.log_rule(task_description, failure_reason, rule)
        return rule

    def get_relevant_rules_prompt(self, task_description: str) -> dict:
        """
        [NOVEL-1] Retrieves hierarchically-expanded rules and formats them
        as a system prompt injection string.
        Returns {"prompt": str, "rule_ids": list}
        """
        rules = self.repo.retrieve_rules(task_description)
        if not rules:
            return {"prompt": "No prior rules learned.", "rule_ids": []}

        lines = []
        for r in rules:
            src = r["metadata"].get("source_type", "direct")
            tag = "[inherited]" if src == "inherited" else ""
            lines.append(f"- {r['rule']} {tag}".strip())

        rule_ids = [r["id"] for r in rules]
        prompt   = (
            "CRITICAL RULES LEARNED FROM PAST FAILURES "
            "(strictly adhere — [inherited] rules come from parent concept hierarchy):\n"
            + "\n".join(lines)
        )
        return {"prompt": prompt, "rule_ids": rule_ids}

    def reinforce_rules(self, rule_ids: list, success: bool = True):
        """[NOVEL-3] Adjusts confidence + resets last_applied_at timestamps."""
        delta = 1 if success else -1
        for rid in rule_ids:
            self.repo.adjust_confidence(rid, delta)

    def run_temporal_decay(self) -> dict:
        """[NOVEL-3] Delegates temporal decay to MemoryRepository."""
        return self.repo.run_temporal_decay()