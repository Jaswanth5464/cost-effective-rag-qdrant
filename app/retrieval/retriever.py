import time
import re
from typing import List, Dict, Any, Optional
from app.embeddings.embedding_service import get_embedding_service
from app.retrieval.vector_store import get_vector_store

class VectorRetriever:
    """
    Retriever class for query vectorization, Qdrant vector similarity search, hybrid keyword reranking, and performance tracking.
    """
    def __init__(self):
        self.embedding_service = get_embedding_service()
        self.vector_store = get_vector_store()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> tuple[List[Dict[str, Any]], float, float]:
        """
        Retrieves top_k relevant chunks for a given query string with hybrid score boosting.
        """
        # Step 1: Embed query
        query_vector, embed_time_ms = self.embedding_service.embed_query(query)

        # Step 2: Search vector store with slightly expanded candidate pool for reranking
        start_search = time.perf_counter()
        candidate_k = max(top_k * 2, 10)
        retrieved_chunks = self.vector_store.search_vectors(
            query_vector=query_vector,
            top_k=candidate_k,
            metadata_filter=metadata_filter
        )

        # Step 3: Hybrid Keyword Reranking
        query_words = set(re.findall(r'\w+', query.lower())) - {
            "what", "is", "the", "how", "much", "many", "which", "are", "for", "a", "an", "in", "of", "and", "to", "was", "did"
        }

        if query_words and retrieved_chunks:
            for chunk in retrieved_chunks:
                chunk_text_lower = chunk.get("text", "").lower()
                chunk_words = set(re.findall(r'\w+', chunk_text_lower))
                keyword_matches = len(query_words & chunk_words)
                # Apply subtle keyword boost (0.05 per keyword match)
                chunk["score"] += keyword_matches * 0.05

            # Re-sort candidate chunks by boosted score
            retrieved_chunks.sort(key=lambda x: x["score"], reverse=True)

        # Trim to top_k
        final_chunks = retrieved_chunks[:top_k]
        retrieval_time_ms = (time.perf_counter() - start_search) * 1000

        return final_chunks, embed_time_ms, retrieval_time_ms
