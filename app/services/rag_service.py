import time
from typing import Dict, Any, List, Optional
from app.retrieval.retriever import VectorRetriever
from app.services.llm_service import get_llm_service
from app.core.logging_config import log_query_metrics
from app.core.config import settings

class RAGService:
    """
    End-to-end RAG orchestrator bringing together retrieval, grounded LLM generation, citation formatting, and audit logging.
    """
    def __init__(self):
        self.retriever = VectorRetriever()
        self.llm_service = get_llm_service()

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes complete RAG pipeline for a given user query.
        """
        total_start_time = time.perf_counter()
        effective_k = top_k if top_k is not None else settings.TOP_K

        # Step 1: Retrieve context chunks
        retrieved_chunks, embedding_time_ms, retrieval_time_ms = self.retriever.retrieve(
            query=query_text,
            top_k=effective_k,
            metadata_filter=metadata_filter
        )

        # Step 2: Generate grounded answer
        answer, prompt_tokens, completion_tokens, total_tokens, generation_time_ms = self.llm_service.generate_grounded_answer(
            query=query_text,
            retrieved_chunks=retrieved_chunks
        )

        total_latency_ms = (time.perf_counter() - total_start_time) * 1000

        # Step 3: Format explicit citations
        citations = []
        for chunk in retrieved_chunks:
            meta = chunk.get("metadata", {})
            citations.append({
                "document_name": meta.get("document_name"),
                "page": meta.get("page"),
                "chunk_id": chunk.get("chunk_id"),
                "category": meta.get("category"),
                "score": round(chunk.get("score", 0.0), 4)
            })

        # Step 4: Record audit log in logs/query.log
        log_query_metrics(
            query_text=query_text,
            embedding_time_ms=embedding_time_ms,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            total_latency_ms=total_latency_ms,
            chunks_retrieved=len(retrieved_chunks),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            metadata_filter=metadata_filter
        )

        return {
            "query": query_text,
            "answer": answer,
            "citations": citations,
            "retrieved_chunks": retrieved_chunks,
            "performance": {
                "embedding_time_ms": round(embedding_time_ms, 2),
                "retrieval_time_ms": round(retrieval_time_ms, 2),
                "generation_time_ms": round(generation_time_ms, 2),
                "total_latency_ms": round(total_latency_ms, 2),
                "chunks_retrieved": len(retrieved_chunks)
            },
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        }
