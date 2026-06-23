import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
from neo4j import GraphDatabase
from .config import settings

class MemoryRepository:
    def __init__(self, agent_id: str = "default"):
        # 1. Vector DB Setup (Chroma)
        self.client = chromadb.PersistentClient(path="./chroma_db")
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
        collection_name = f"agent_{agent_id}_rules"
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_func,
            metadata={"hnsw:space": "cosine"}
        )
        
        # 2. Graph DB Setup (Neo4j)
        self.graph = GraphDatabase.driver(
            settings.neo4j_uri, 
            auth=(settings.neo4j_user, settings.neo4j_password)
        )
        self.agent_id = agent_id

    def store_rule(self, rule_text: str, task: str, failure: str, concept: str) -> str:
        rule_id = f"rule_{self.collection.count() + 1}"
        
        # Store in Vector DB
        self.collection.add(
            documents=[rule_text],
            metadatas=[{"source_task": task, "source_failure": failure, "confidence": 1, "concept": concept}],
            ids=[rule_id]
        )
        
        # Store in Graph DB (Create Concept Node, Rule Node, and link them)
        with self.graph.session() as session:
            session.run(
                """
                MERGE (c:Concept {name: $concept, agent_id: $agent_id})
                CREATE (r:Rule {id: $rule_id, text: $rule_text, confidence: 1, agent_id: $agent_id})
                MERGE (c)-[:HAS_RULE]->(r)
                """,
                concept=concept, agent_id=self.agent_id, rule_id=rule_id, rule_text=rule_text
            )
        return rule_id

    def retrieve_rules(self, query_text: str, top_k: int = None) -> List[Dict]:
        """Graph-Enhanced Retrieval: Finds closest rule via Vector, then fetches the whole Concept Graph."""
        if not top_k:
            top_k = settings.max_rules_to_retrieve
            
        if self.collection.count() == 0:
            return []
            
        # 1. Semantic Search to find the primary concept
        results = self.collection.query(query_texts=[query_text], n_results=1)
        if not results['documents'][0]:
            return []
            
        primary_rule_id = results['ids'][0][0]
        primary_concept = results['metadatas'][0][0].get('concept', 'UNKNOWN')
        
        # 2. Graph Traversal: Get ALL rules that share this concept
        with self.graph.session() as session:
            graph_results = session.run(
                """
                MATCH (c:Concept {name: $concept, agent_id: $agent_id})-[:HAS_RULE]->(r:Rule)
                WHERE r.confidence > 0
                RETURN r.id as id, r.text as rule, r.confidence as confidence
                """,
                concept=primary_concept, agent_id=self.agent_id
            )
            
            formatted_rules = []
            for record in graph_results:
                formatted_rules.append({
                    "id": record["id"], 
                    "rule": record["rule"], 
                    "metadata": {"confidence": record["confidence"], "concept": primary_concept}
                })
                
            return formatted_rules

    def adjust_confidence(self, rule_id: str, delta: int):
        """Rule Decay mechanism in both Vector and Graph DBs."""
        # Update Chroma
        current_data = self.collection.get(ids=[rule_id])
        if not current_data['metadatas']:
            return
        current_confidence = current_data['metadatas'][0].get('confidence', 0)
        new_confidence = current_confidence + delta
        
        if new_confidence <= 0:
            self.collection.delete(ids=[rule_id])
            # Delete from Graph
            with self.graph.session() as session:
                session.run("MATCH (r:Rule {id: $rule_id}) DETACH DELETE r", rule_id=rule_id)
        else:
            updated_meta = current_data['metadatas'][0]
            updated_meta['confidence'] = new_confidence
            self.collection.update(ids=[rule_id], metadatas=[updated_meta])
            # Update Graph
            with self.graph.session() as session:
                session.run("MATCH (r:Rule {id: $rule_id}) SET r.confidence = $new_conf", rule_id=rule_id, new_conf=new_confidence)