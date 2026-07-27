import sys
import os
from loguru import logger
from app.core.config import settings

def setup_logger():
    """Configures Loguru logger with console output and structured JSON query.log output."""
    os.makedirs(settings.LOGS_DIR, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Add stdout handler with clean format
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # Add file handler specifically for query metrics
    logger.add(
        settings.LOG_FILE,
        rotation="10 MB",
        retention="10 days",
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    )

def log_query_metrics(
    query_text: str,
    embedding_time_ms: float,
    retrieval_time_ms: float,
    generation_time_ms: float,
    total_latency_ms: float,
    chunks_retrieved: int,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    metadata_filter: dict | None = None
):
    """
    Logs structured performance metrics for every query to logs/query.log.
    """
    log_entry = (
        f"[QUERY_METRICS] Query='{query_text}' | "
        f"EmbeddingTime={embedding_time_ms:.2f}ms | "
        f"RetrievalTime={retrieval_time_ms:.2f}ms | "
        f"GenerationTime={generation_time_ms:.2f}ms | "
        f"TotalLatency={total_latency_ms:.2f}ms | "
        f"ChunksRetrieved={chunks_retrieved} | "
        f"PromptTokens={prompt_tokens} | "
        f"CompletionTokens={completion_tokens} | "
        f"TotalTokens={total_tokens} | "
        f"Filter={metadata_filter}"
    )
    logger.info(log_entry)

# Initialize logger on module load
setup_logger()
