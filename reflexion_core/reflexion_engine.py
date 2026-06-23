from groq import Groq
from .config import settings
from .memory_store import MemoryRepository
from .logger import MarkdownLogger

class ReflexionEngine:
    def __init__(self, agent_id: str = "default"):
        self.client = Groq(api_key=settings.groq_api_key)
        self.model = settings.llm_model
        self.repo = MemoryRepository(agent_id=agent_id)
        self.logger = MarkdownLogger()

    def _generate_completion(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

    def reflect_on_failure(self, task_description: str, failure_reason: str) -> str:
        system_prompt = """You are an AI Agent Memory Optimizer. 
        Your job is to analyze why an autonomous agent failed a task and extract a single, dense, actionable RULE.
        The rule must be imperative (e.g., 'Always do X', 'Never do Y').
        Output ONLY the rule. No explanations. Max 2 sentences."""
        
        user_prompt = f"Task Attempted: '{task_description}'\nFailure Reason: '{failure_reason}'\nExtract the corrective rule:"
        rule = self._generate_completion(system_prompt, user_prompt, temperature=0.1)
        
        # NEW: Extract Concept Category for Graph DB
        concept_prompt = f"Categorize this rule into a single, uppercase concept node (e.g., HTTP_REQUEST_BEST_PRACTICES, ERROR_HANDLING, SECURITY).\nRule: '{rule}'\nOutput ONLY the category name:"
        concept = self._generate_completion("You are a categorizer.", concept_prompt, temperature=0.0).replace(" ", "_").upper()
        
        # Store rule with its concept
        self.repo.store_rule(rule_text=rule, task=task_description, failure=failure_reason, concept=concept)
        self.logger.log_rule(task_description, failure_reason, rule)
        return rule

    def get_relevant_rules_prompt(self, task_description: str):
        rules = self.repo.retrieve_rules(task_description)
        if not rules:
            return {"prompt": "No prior rules learned.", "rule_ids": []}
            
        rules_text = "\n".join([f"- {r['rule']}" for r in rules])
        rule_ids = [r['id'] for r in rules]
        prompt = f"CRITICAL RULES LEARNED FROM PAST FAILURES (Strictly adhere to these):\n{rules_text}"
        return {"prompt": prompt, "rule_ids": rule_ids}

    def reinforce_rules(self, rule_ids: list, success: bool = True):
        delta = 1 if success else -1
        for rid in rule_ids:
            self.repo.adjust_confidence(rid, delta)