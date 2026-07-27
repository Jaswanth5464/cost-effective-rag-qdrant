from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class MetadataFilter(BaseModel):
    category: Optional[str] = Field(None, example="finance", description="Filter by category (e.g. finance, policy, engineering)")
    source: Optional[str] = Field(None, example="documents/financial_report.pdf", description="Filter by document file path")
    document_name: Optional[str] = Field(None, example="employee.pdf", description="Filter by exact document file name")

class IngestResponse(BaseModel):
    status: str = Field(..., example="success")
    processed_files: int = Field(..., example=3)
    total_chunks: int = Field(..., example=25)
    new_chunks_inserted: int = Field(..., example=25)
    skipped_chunks: int = Field(..., example=0)
    message: str = Field(..., example="Successfully ingested 25 new chunks.")

class QueryRequest(BaseModel):
    query: str = Field(..., example="What is the company policy regarding remote work reimbursement?", description="Search query")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Top-k chunks to retrieve")
    metadata_filter: Optional[MetadataFilter] = Field(None, description="Metadata filtering parameters")

class Citation(BaseModel):
    document_name: Optional[str] = Field(None, example="company_policy.html")
    page: Optional[int] = Field(None, example=1)
    chunk_id: Optional[str] = Field(None, example="e4d909c2-5555-5c1a-8888-abcdef123456")
    category: Optional[str] = Field(None, example="policy")
    score: float = Field(..., example=0.8842)

class PerformanceMetrics(BaseModel):
    embedding_time_ms: float
    retrieval_time_ms: float
    generation_time_ms: float
    total_latency_ms: float
    chunks_retrieved: int

class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[Citation]
    performance: PerformanceMetrics
    token_usage: TokenUsage

class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    qdrant_status: str = Field(..., example="connected")
    qdrant_collection: Dict[str, Any]
    llm_status: str = Field(..., example="configured (Groq API: llama-3.3-70b-versatile)")
    embedding_model: str = Field(..., example="BAAI/bge-small-en-v1.5 (384 dimensions)")
