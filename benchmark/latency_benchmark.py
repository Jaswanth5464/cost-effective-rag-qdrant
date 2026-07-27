import time
import json
import numpy as np
from pathlib import Path
from loguru import logger
import sys

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.rag_service import RAGService
from app.ingestion.ingestion_service import IngestionService

SAMPLE_QUERIES = [
    "What was Acme Corp's total revenue in FY2025?",
    "What is the remote work monthly stipend?",
    "What embedding model is used in the architecture?",
    "How much annual cost reduction was achieved by switching to Qdrant?",
    "How many paid vacation days do employees get?",
    "What Large Language Model powers the RAG system?",
    "What is the setup budget for home office equipment?",
    "What percentage of dependent health insurance is covered?",
    "How does the system ensure idempotent chunk ingestion?",
    "What are the latency SLOs for vector search and generation?"
]

def run_latency_benchmark(num_queries: int = 100):
    logger.info("Verifying document ingestion prior to latency test...")
    ingest_svc = IngestionService()
    ingest_svc.ingest_directory()

    rag_svc = RAGService()

    embedding_latencies = []
    retrieval_latencies = []
    generation_latencies = []
    total_latencies = []

    logger.info(f"Executing {num_queries} queries for latency benchmarking (p50, p95, avg)...")

    for i in range(num_queries):
        query = SAMPLE_QUERIES[i % len(SAMPLE_QUERIES)]
        res = rag_svc.query(query_text=query, top_k=5)
        time.sleep(0.5)  # Rate limit pacing for Groq Free Tier TPM
        
        perf = res["performance"]
        embedding_latencies.append(perf["embedding_time_ms"])
        retrieval_latencies.append(perf["retrieval_time_ms"])
        generation_latencies.append(perf["generation_time_ms"])
        total_latencies.append(perf["total_latency_ms"])

        if (i + 1) % 20 == 0 or (i + 1) == num_queries:
            logger.info(f"Completed {i + 1}/{num_queries} queries...")

    # Calculate percentiles and averages
    stats = {
        "num_queries": num_queries,
        "embedding_latency_ms": {
            "p50": round(float(np.percentile(embedding_latencies, 50)), 2),
            "p95": round(float(np.percentile(embedding_latencies, 95)), 2),
            "avg": round(float(np.mean(embedding_latencies)), 2)
        },
        "retrieval_latency_ms": {
            "p50": round(float(np.percentile(retrieval_latencies, 50)), 2),
            "p95": round(float(np.percentile(retrieval_latencies, 95)), 2),
            "avg": round(float(np.mean(retrieval_latencies)), 2)
        },
        "generation_latency_ms": {
            "p50": round(float(np.percentile(generation_latencies, 50)), 2),
            "p95": round(float(np.percentile(generation_latencies, 95)), 2),
            "avg": round(float(np.mean(generation_latencies)), 2)
        },
        "total_latency_ms": {
            "p50": round(float(np.percentile(total_latencies, 50)), 2),
            "p95": round(float(np.percentile(total_latencies, 95)), 2),
            "avg": round(float(np.mean(total_latencies)), 2)
        }
    }

    out_file = Path(__file__).parent / "latency_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 65)
    print("         LATENCY BENCHMARK RESULTS (100 QUERIES)         ")
    print("=" * 65)
    print(f"{'Metric Stage':<25} | {'p50 (ms)':<10} | {'p95 (ms)':<10} | {'Average (ms)':<12}")
    print("-" * 65)
    print(f"{'Embedding Time':<25} | {stats['embedding_latency_ms']['p50']:<10} | {stats['embedding_latency_ms']['p95']:<10} | {stats['embedding_latency_ms']['avg']:<12}")
    print(f"{'Retrieval (Vector Store)':<25} | {stats['retrieval_latency_ms']['p50']:<10} | {stats['retrieval_latency_ms']['p95']:<10} | {stats['retrieval_latency_ms']['avg']:<12}")
    print(f"{'LLM Generation Time':<25} | {stats['generation_latency_ms']['p50']:<10} | {stats['generation_latency_ms']['p95']:<10} | {stats['generation_latency_ms']['avg']:<12}")
    print("-" * 65)
    print(f"{'Total End-to-End Latency':<25} | {stats['total_latency_ms']['p50']:<10} | {stats['total_latency_ms']['p95']:<10} | {stats['total_latency_ms']['avg']:<12}")
    print("=" * 65)
    print(f"Latency results saved to: {out_file}\n")

if __name__ == "__main__":
    run_latency_benchmark(100)
