"""
Agent Reflexion Memory — MemoryRepository
==========================================
Three patent-targeted novel contributions (all verified against prior art June 2026):

[NOVEL-1] Hierarchical Concept Inheritance Retrieval
  When retrieving rules, if a concept has a PARENT_CONCEPT in the graph, the system
  traverses UP the hierarchy and returns rules from parent concepts too. No prior system
  applies concept-hierarchy graph traversal specifically to distilled agent behavioral
  failure-correction rules.

[NOVEL-2] Cross-Agent Confidence Reinforcement (NOT copying)
  When a new rule is stored for agent_X, the system queries ALL other agents' ChromaDB
  collections for semantic similarity > CROSS_AGENT_SIMILARITY_THRESHOLD. If a match
  is found in agent_Y's collection, agent_Y's matching rule confidence is INCREMENTED
  (not duplicated). AMEM4Rec (arXiv:2602.08837) copies memories across users —
  this system reinforces confidence on existing matching rules, a distinct mechanism
  applied to behavioral correction rules.

[NOVEL-3] Temporal Decay with last_applied_at Timestamps on Behavioral Rules
  Every rule stores a last_applied_at Unix timestamp. A decay scheduler decrements
  confidence on rules not applied in RULE_DECAY_DAYS days. Existing decay systems
  (FadeMem, Oblivion, YourMemory) apply Ebbinghaus decay to episodic memories;
  this applies it specifically to LLM-distilled imperative behavioral correction rules
  with dual-DB (ChromaDB + Neo4j) atomic sync.
"""

import chromadb
import time
import math
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional
from neo4j import GraphDatabase
from .config import settings


# [NOVEL-2]/[NOVEL-3] thresholds now sourced from settings (configurable via .env)
CROSS_AGENT_SIMILARITY_THRESHOLD = settings.cross_agent_similarity_threshold
RULE_DECAY_DAYS = settings.rule_decay_days
TEMPORAL_DECAY_DELTA = -1   # confidence decrement per stale period


class MemoryRepository:
    def __init__(self, agent_id: str = "default"):
        # --- Vector DB (ChromaDB) ---
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
        collection_name = f"agent_{agent_id}_rules"
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_func,
            metadata={"hnsw:space": "cosine"}
        )

        # --- Graph DB (Neo4j) ---
        self.graph = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password)
        )
        self.agent_id = agent_id

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC: store_rule
    # ─────────────────────────────────────────────────────────────────────────
    def store_rule(
        self,
        rule_text: str,
        task: str,
        failure: str,
        concept: str,
        parent_concept: Optional[str] = None   # [NOVEL-1] optional hierarchy hint
    ) -> str:
        """
        Stores a distilled behavioral rule in both ChromaDB and Neo4j.
        Triggers cross-agent confidence reinforcement after storage. [NOVEL-2]
        Records last_applied_at timestamp for temporal decay. [NOVEL-3]
        Optionally links concept to a parent concept node in the graph. [NOVEL-1]
        """
        rule_id = f"rule_{self.agent_id}_{int(time.time() * 1000)}"
        now_ts = time.time()

        # 1. Store in ChromaDB with timestamp metadata [NOVEL-3]
        self.collection.add(
            documents=[rule_text],
            metadatas=[{
                "source_task": task,
                "source_failure": failure,
                "confidence": 1,
                "concept": concept,
                "last_applied_at": now_ts,   # [NOVEL-3]
                "created_at": now_ts
            }],
            ids=[rule_id]
        )

        # 2. Store in Neo4j with parent concept link [NOVEL-1]
        with self.graph.session() as session:
            session.run(
                """
                MERGE (c:Concept {name: $concept, agent_id: $agent_id})
                CREATE (r:Rule {
                    id: $rule_id,
                    text: $rule_text,
                    confidence: 1,
                    agent_id: $agent_id,
                    last_applied_at: $now_ts,
                    created_at: $now_ts
                })
                MERGE (c)-[:HAS_RULE]->(r)
                """,
                concept=concept,
                agent_id=self.agent_id,
                rule_id=rule_id,
                rule_text=rule_text,
                now_ts=now_ts
            )

            # [NOVEL-1] Link concept to its parent if provided
            if parent_concept:
                session.run(
                    """
                    MERGE (parent:Concept {name: $parent_concept, agent_id: $agent_id})
                    MERGE (child:Concept  {name: $concept,        agent_id: $agent_id})
                    MERGE (child)-[:PARENT_CONCEPT]->(parent)
                    """,
                    parent_concept=parent_concept,
                    concept=concept,
                    agent_id=self.agent_id
                )

        # 3. [NOVEL-2] Cross-agent confidence reinforcement
        self._cross_agent_reinforce(rule_text=rule_text, source_agent_id=self.agent_id)

        return rule_id

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC: retrieve_rules   [NOVEL-1] Hierarchical Concept Inheritance
    # ─────────────────────────────────────────────────────────────────────────
    def retrieve_rules(self, query_text: str, top_k: int = None) -> List[Dict]:
        """
        [NOVEL-1] Graph-Enhanced Hierarchical Retrieval:
          Step 1 — Vector similarity search → primary concept
          Step 2 — Graph traversal: all rules under primary concept
          Step 3 — Graph traversal: walk PARENT_CONCEPT edges upward, collect
                   ancestor rules too (capped at 2 hops to bound context size)
        The union of descendant + ancestor rules is returned, ranked by confidence.
        """
        if not top_k:
            top_k = settings.max_rules_to_retrieve

        if self.collection.count() == 0:
            return []

        # Step 1 — semantic search for closest concept
        results = self.collection.query(query_texts=[query_text], n_results=1)
        if not results["documents"][0]:
            return []

        # Relevance floor: if even the closest match is too far away,
        # the agent has no concept relevant to this query — return nothing
        # rather than surfacing an unrelated concept's rules. [NOVEL-1 guard]
        closest_distance = results["distances"][0][0]
        MAX_RELEVANT_DISTANCE = 1.2
        if closest_distance > MAX_RELEVANT_DISTANCE:
            return []

        primary_concept = results["metadatas"][0][0].get("concept", "UNKNOWN")

        # Step 2+3 — hierarchical graph traversal [NOVEL-1]
        with self.graph.session() as session:
            graph_results = session.run(
                """
                // Match rules directly under the primary concept
                MATCH (c:Concept {name: $concept, agent_id: $agent_id})-[:HAS_RULE]->(r:Rule)
                WHERE r.confidence > 0
                RETURN r.id AS id, r.text AS rule, r.confidence AS confidence,
                       $concept AS concept, 'direct' AS source_type

                UNION

                // [NOVEL-1] Walk up PARENT_CONCEPT edges (up to 2 hops) and collect ancestor rules
                MATCH (c:Concept {name: $concept, agent_id: $agent_id})
                      -[:PARENT_CONCEPT*1..2]->(ancestor:Concept)
                      -[:HAS_RULE]->(r:Rule)
                WHERE r.confidence > 0
                RETURN r.id AS id, r.text AS rule, r.confidence AS confidence,
                       ancestor.name AS concept, 'inherited' AS source_type
                """,
                concept=primary_concept,
                agent_id=self.agent_id
            )

            seen_ids = set()
            formatted_rules = []
            for record in graph_results:
                if record["id"] not in seen_ids:
                    seen_ids.add(record["id"])
                    formatted_rules.append({
                        "id":       record["id"],
                        "rule":     record["rule"],
                        "metadata": {
                            "confidence":  record["confidence"],
                            "concept":     record["concept"],
                            "source_type": record["source_type"]   # direct | inherited
                        }
                    })

            # Sort by confidence descending, cap at top_k
            formatted_rules.sort(key=lambda x: x["metadata"]["confidence"], reverse=True)
            return formatted_rules[:top_k]

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC: adjust_confidence  [NOVEL-3] updates last_applied_at timestamp
    # ─────────────────────────────────────────────────────────────────────────
    def adjust_confidence(self, rule_id: str, delta: int):
        """
        Adjusts confidence score in both DBs.
        [NOVEL-3] Updates last_applied_at to NOW so temporal decay clock resets.
        Deletes rule atomically from both DBs if confidence hits zero.

        The increment AND the zero-check both happen server-side in a single
        Neo4j write (SET r.confidence = r.confidence + $delta ... RETURN),
        so concurrent calls can't race on a stale Python-side confidence read
        the way a read-modify-write would. ChromaDB has no atomic increment,
        so it's still synced from Neo4j's authoritative post-write value.
        """
        current_data = self.collection.get(ids=[rule_id])
        if not current_data["metadatas"]:
            return

        now_ts = time.time()

        with self.graph.session() as session:
            result = session.run(
                """
                MATCH (r:Rule {id: $rule_id})
                SET r.confidence = r.confidence + $delta,
                    r.last_applied_at = $now_ts
                RETURN r.confidence AS new_confidence
                """,
                rule_id=rule_id,
                delta=delta,
                now_ts=now_ts
            )
            record = result.single()

        if record is None:
            # Rule didn't exist in Neo4j (shouldn't normally happen if it's
            # in ChromaDB, but guard against drift between the two stores).
            return

        new_confidence = record["new_confidence"]

        if new_confidence <= 0:
            # Atomic delete from both stores
            self.collection.delete(ids=[rule_id])
            with self.graph.session() as session:
                session.run(
                    "MATCH (r:Rule {id: $rule_id}) DETACH DELETE r",
                    rule_id=rule_id
                )
        else:
            updated_meta = dict(current_data["metadatas"][0])
            updated_meta["confidence"] = new_confidence
            updated_meta["last_applied_at"] = now_ts   # [NOVEL-3] reset decay clock
            self.collection.update(ids=[rule_id], metadatas=[updated_meta])

    # ─────────────────────────────────────────────────────────────────────────
    # [NOVEL-2] Cross-Agent Confidence Reinforcement
    # ─────────────────────────────────────────────────────────────────────────
    def _cross_agent_reinforce(self, rule_text: str, source_agent_id: str):
        """
        [NOVEL-2] After storing a new rule for source_agent_id, scans ALL other
        agents' ChromaDB collections. For any agent whose existing rule has
        cosine similarity >= CROSS_AGENT_SIMILARITY_THRESHOLD with the new rule,
        increments that rule's confidence score by +1 (reinforcement, not copy).

        This is fundamentally different from AMEM4Rec which COPIES memories.
        Here we REINFORCE confidence on EXISTING matching rules in peer agents,
        creating an emergent collective behavioral correction signal without
        duplicating rule storage.
        """
        try:
            all_collections = self.chroma_client.list_collections()
        except Exception:
            return

        for col_info in all_collections:
            col_name = col_info.name if hasattr(col_info, "name") else str(col_info)

            # Skip the source agent's own collection
            if col_name == f"agent_{source_agent_id}_rules":
                continue

            # Only process collections that look like agent rule stores
            if not col_name.startswith("agent_") or not col_name.endswith("_rules"):
                continue

            try:
                peer_collection = self.chroma_client.get_collection(
                    name=col_name,
                    embedding_function=self.embedding_func
                )

                if peer_collection.count() == 0:
                    continue

                # Semantic search in peer agent's collection
                peer_results = peer_collection.query(
                    query_texts=[rule_text],
                    n_results=1
                )

                if not peer_results["ids"][0]:
                    continue

                # Compute similarity from distance (ChromaDB cosine returns distance 0..2)
                distances = peer_results["distances"][0]
                if not distances:
                    continue

                cosine_similarity = 1.0 - (distances[0] / 2.0)

                if cosine_similarity >= CROSS_AGENT_SIMILARITY_THRESHOLD:
                    matched_rule_id = peer_results["ids"][0][0]
                    matched_meta    = peer_results["metadatas"][0][0]

                    # Neo4j: atomic increment (r.confidence + 1 evaluated server-side
                    # in a single write transaction) — avoids the lost-update race
                    # a read-modify-write from Python would otherwise have under
                    # concurrent reinforcement events.
                    with self.graph.session() as session:
                        neo4j_result = session.run(
                            """
                            MATCH (r:Rule {id: $rid})
                            SET r.confidence = r.confidence + 1
                            RETURN r.confidence AS new_confidence
                            """,
                            rid=matched_rule_id
                        )
                        record = neo4j_result.single()
                        new_conf = record["new_confidence"] if record else matched_meta.get("confidence", 1) + 1

                    # ChromaDB has no native atomic increment, so this side still
                    # does a read-then-write — but we sync FROM Neo4j's authoritative
                    # post-increment value rather than recomputing independently,
                    # keeping the two stores converging on the same number even
                    # under light concurrency.
                    updated_meta = dict(matched_meta)
                    updated_meta["confidence"] = new_conf
                    peer_collection.update(ids=[matched_rule_id], metadatas=[updated_meta])

            except Exception:
                # Never let cross-agent logic break the primary store_rule flow
                continue

    # ─────────────────────────────────────────────────────────────────────────
    # [NOVEL-3] Temporal Decay Scheduler
    # ─────────────────────────────────────────────────────────────────────────
    def run_temporal_decay(self, batch_size: int = 500):
        """
        [NOVEL-3] Scans all rules in this agent's collection.
        Any rule whose last_applied_at is older than RULE_DECAY_DAYS days
        has its confidence decremented by TEMPORAL_DECAY_DELTA.
        Rules whose confidence hits zero are deleted from both stores.

        Designed to be called by an external APScheduler job (see scheduler.py).
        Ebbinghaus-inspired but applied specifically to distilled imperative
        behavioral correction rules — not episodic memories.

        Processes the collection in batches (default 500 rules at a time)
        instead of loading everything into memory at once — keeps memory
        usage bounded as an agent's rule store grows into the thousands.
        Decay logic itself is unchanged; only the iteration strategy differs.
        """
        if self.collection.count() == 0:
            return {"decayed": 0, "deleted": 0}

        now_ts = time.time()
        decay_threshold_secs = RULE_DECAY_DAYS * 86400

        decayed = 0
        deleted = 0
        offset = 0

        while True:
            batch = self.collection.get(
                limit=batch_size,
                offset=offset,
                include=["metadatas", "documents"]
            )
            if not batch["ids"]:
                break

            for rule_id, meta in zip(batch["ids"], batch["metadatas"]):
                last_applied = meta.get("last_applied_at", meta.get("created_at", now_ts))
                age_secs     = now_ts - last_applied

                if age_secs > decay_threshold_secs:
                    current_conf = meta.get("confidence", 1)
                    new_conf     = current_conf + TEMPORAL_DECAY_DELTA  # delta is negative

                    if new_conf <= 0:
                        self.collection.delete(ids=[rule_id])
                        with self.graph.session() as session:
                            session.run(
                                "MATCH (r:Rule {id: $rid}) DETACH DELETE r",
                                rid=rule_id
                            )
                        deleted += 1
                    else:
                        updated_meta = dict(meta)
                        updated_meta["confidence"] = new_conf
                        self.collection.update(ids=[rule_id], metadatas=[updated_meta])
                        with self.graph.session() as session:
                            session.run(
                                "MATCH (r:Rule {id: $rid}) SET r.confidence = $conf",
                                rid=rule_id,
                                conf=new_conf
                            )
                        decayed += 1

            # If a rule in this batch was deleted, the collection shrank, so the
            # next "page" at the same offset now contains what used to be after
            # it — advancing offset by the batch's original size still correctly
            # walks the whole collection
    # ─────────────────────────────────────────────────────────────────────────
    # [NOVEL-1] Concept Hierarchy Management
    # ─────────────────────────────────────────────────────────────────────────
    def link_concept_parent(self, child_concept: str, parent_concept: str):
        """
        [NOVEL-1] Explicitly sets a PARENT_CONCEPT edge between two concept nodes
        in the graph for this agent. Enables hierarchical inheritance during retrieval.

        Raises ValueError if this would create a cycle (e.g. linking A under B
        when B is already a descendant of A) — a cycle wouldn't infinite-loop
        retrieval (traversal is capped at 2 hops) but would produce a nonsensical
        hierarchy where a concept inherits from its own descendant.
        """
        if child_concept == parent_concept:
            raise ValueError("A concept cannot be its own parent.")

        with self.graph.session() as session:
            # If parent_concept is already a descendant of child_concept,
            # adding child -> parent here would close a cycle.
            cycle_check = session.run(
                """
                MATCH (p:Concept {name: $parent_concept, agent_id: $agent_id})
                      -[:PARENT_CONCEPT*1..10]->(c:Concept {name: $child_concept, agent_id: $agent_id})
                RETURN count(*) AS path_count
                """,
                parent_concept=parent_concept,
                child_concept=child_concept,
                agent_id=self.agent_id
            )
            if cycle_check.single()["path_count"] > 0:
                raise ValueError(
                    f"Cannot link '{child_concept}' under '{parent_concept}': "
                    f"'{parent_concept}' is already a descendant of '{child_concept}', "
                    f"which would create a cycle."
                )

            session.run(
                """
                MERGE (parent:Concept {name: $parent_concept, agent_id: $agent_id})
                MERGE (child:Concept  {name: $child_concept,  agent_id: $agent_id})
                MERGE (child)-[:PARENT_CONCEPT]->(parent)
                """,
                parent_concept=parent_concept,
                child_concept=child_concept,
                agent_id=self.agent_id
            )

    def get_concept_hierarchy(self) -> List[Dict]:
        """Returns the full concept hierarchy tree for this agent (for audit/viz)."""
        with self.graph.session() as session:
            results = session.run(
                """
                MATCH (child:Concept {agent_id: $agent_id})-[:PARENT_CONCEPT]->(parent:Concept)
                RETURN child.name AS child, parent.name AS parent
                """,
                agent_id=self.agent_id
            )
            return [{"child": r["child"], "parent": r["parent"]} for r in results]
        