import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))
    QDRANT_URL: str = os.getenv("QDRANT_URL", f"http://{QDRANT_HOST}:{QDRANT_PORT}")
    QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY", None)
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "documents_collection")

    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "384"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "40"))
    
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION: int = 384

    DOCUMENTS_DIR: str = os.getenv("DOCUMENTS_DIR", "documents")
    LOGS_DIR: str = os.getenv("LOGS_DIR", "logs")
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/query.log")

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
