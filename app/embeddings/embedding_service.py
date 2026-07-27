import time
from typing import List
from sentence_transformers import SentenceTransformer
from loguru import logger
from app.core.config import settings

class EmbeddingService:
    """
    Service for generating dense vector embeddings using BAAI/bge-small-en-v1.5.
    Dimensionality: 384
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL_NAME}")
        self.model_name = settings.EMBEDDING_MODEL_NAME
        self.dimension = settings.EMBEDDING_DIMENSION
        # Load SentenceTransformer model
        self.model = SentenceTransformer(self.model_name)
        logger.info(f"Embedding model loaded successfully. Dimension: {self.dimension}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of document chunks.
        """
        if not texts:
            return []
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> tuple[List[float], float]:
        """
        Embeds a single search query with latency tracking (in ms).
        BGE models perform best when prefixing queries for passage retrieval.
        """
        start_time = time.perf_counter()
        prefixed_query = f"Represent this sentence for searching relevant passages: {query}"
        embedding = self.model.encode(prefixed_query, normalize_embeddings=True, show_progress_bar=False)
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        return embedding.tolist(), elapsed_ms

# Module-level singleton helper
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()
