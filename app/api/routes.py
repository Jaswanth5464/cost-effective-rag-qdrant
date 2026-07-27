from fastapi import APIRouter, HTTPException, status
from loguru import logger
from app.api.schemas import IngestResponse, QueryRequest, QueryResponse, HealthResponse
from app.ingestion.ingestion_service import IngestionService
from app.services.rag_service import RAGService
from app.retrieval.vector_store import get_vector_store
from app.services.llm_service import get_llm_service
from app.core.config import settings

router = APIRouter()

ingestion_service = IngestionService()
rag_service = RAGService()

@router.post("/ingest", response_model=IngestResponse, summary="Ingest Documents", description="Scans documents directory, parses PDF/HTML/Markdown, generates embeddings, and inserts unique vectors into Qdrant idempotently.")
def ingest_documents():
    try:
        result = ingestion_service.ingest_directory()
        return IngestResponse(**result)
    except Exception as e:
        logger.error(f"Error during document ingestion: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )

@router.post("/query", response_model=QueryResponse, summary="Execute RAG Query", description="Retrieves top-k context chunks from Qdrant with optional metadata filtering and generates grounded answer with Gemini 2.5 Flash.")
def query_rag(request: QueryRequest):
    try:
        filter_dict = request.metadata_filter.model_dump(exclude_none=True) if request.metadata_filter else None
        response = rag_service.query(
            query_text=request.query,
            top_k=request.top_k,
            metadata_filter=filter_dict
        )
        return QueryResponse(**response)
    except Exception as e:
        logger.error(f"Error processing RAG query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}"
        )

@router.get("/health", response_model=HealthResponse, summary="Health Check", description="Checks Qdrant connection and Groq LLM API health.")
def health_check():
    vector_store = get_vector_store()
    collection_info = vector_store.get_collection_info()
    
    q_status = "connected" if "error" not in collection_info else "disconnected"
    llm_status = f"configured (Groq API: {settings.GROQ_MODEL})" if settings.GROQ_API_KEY else "unconfigured"

    return HealthResponse(
        status="healthy" if q_status == "connected" else "degraded",
        qdrant_status=q_status,
        qdrant_collection=collection_info,
        llm_status=llm_status,
        embedding_model=f"{settings.EMBEDDING_MODEL_NAME} ({settings.EMBEDDING_DIMENSION} dim)"
    )
