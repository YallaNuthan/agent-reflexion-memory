import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class Settings(BaseSettings):
    groq_api_key: str
    llm_model: str = "llama-3.1-8b-instant"
    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_collection: str = "agent_reflexion_rules"
    max_rules_to_retrieve: int = 3
    
    # Neo4j Graph DB Settings
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password123"
    
    # API Authentication Settings
    api_access_key: str = "dev-secret-key"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()