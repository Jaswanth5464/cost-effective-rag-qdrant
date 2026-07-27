import json
import os
import sys
import time
import numpy as np
from pathlib import Path
from loguru import logger

# Add root directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.rag_service import RAGService
from app.ingestion.ingestion_service import IngestionService
from app.evaluation.metrics import RetrievalMetrics, AnswerMetrics

def run_evaluation():
    logger.info("Initializing document ingestion prior to benchmark run...")
    ingest_svc = IngestionService()
    ingest_result = ingest_svc.ingest_directory()
    logger.info(f"Ingestion result: {ingest_result}")

    dataset_path = Path(__file__).parent / "dataset.json"
    if not dataset_path.exists():
        logger.error(f"Benchmark dataset file not found at {dataset_path}")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        benchmark_items = json.load(f)

    logger.info(f"Running evaluation benchmark over {len(benchmark_items)} test questions...")

    rag_svc = RAGService()

    # Metric accumulators
    hit_rates = []
    recalls = []
    mrrs = []
    ndcgs = []
    context_precisions = []

    ems = []
    f1s = []
    faithfulness_scores = []
    relevance_scores = []

    detailed_results = []

    for item in benchmark_items:
        q_id = item["id"]
        question = item["question"]
        expected_answer = item["expected_answer"]
        target_doc = item.get("target_document")
        category = item.get("category")

        # Execute query through RAG pipeline
        response = rag_svc.query(query_text=question, top_k=5)
        time.sleep(1.0)  # Rate limit safety delay for Groq API TPM

        predicted_answer = response["answer"]
        retrieved_chunks = response["retrieved_chunks"]

        # Identify retrieved document names
        retrieved_doc_names = [c.get("metadata", {}).get("document_name", "") for c in retrieved_chunks]
        retrieved_chunk_ids = [c.get("chunk_id", "") for c in retrieved_chunks]
        retrieved_texts = [c.get("text", "") for c in retrieved_chunks]

        # Target relevance matching logic
        if target_doc and target_doc != "none":
            relevant_indices = {cid for cid, dname in zip(retrieved_chunk_ids, retrieved_doc_names) if target_doc in dname}
            if not relevant_indices:
                # If target doc chunk wasn't in top-k retrieved list, mark relevant_indices as hypothetical target
                relevant_indices = {f"target_doc_{target_doc}"}
        else:
            # Out of scope queries have no relevant chunks
            relevant_indices = set()

        # Compute Retrieval Metrics
        hr = RetrievalMetrics.hit_rate(retrieved_chunk_ids, relevant_indices)
        rec = RetrievalMetrics.recall_at_k(retrieved_chunk_ids, relevant_indices)
        mrr_val = RetrievalMetrics.mrr(retrieved_chunk_ids, relevant_indices)
        ndcg_val = RetrievalMetrics.ndcg_at_k(retrieved_chunk_ids, relevant_indices)
        cp_val = RetrievalMetrics.context_precision(retrieved_chunk_ids, relevant_indices)

        hit_rates.append(hr)
        recalls.append(rec)
        mrrs.append(mrr_val)
        ndcgs.append(ndcg_val)
        context_precisions.append(cp_val)

        # Compute Answer Metrics
        em_val = AnswerMetrics.exact_match(predicted_answer, expected_answer)
        f1_val = AnswerMetrics.f1_score(predicted_answer, expected_answer)
        faith_val = AnswerMetrics.faithfulness(predicted_answer, retrieved_texts)
        rel_val = AnswerMetrics.answer_relevance(predicted_answer, question)

        ems.append(em_val)
        f1s.append(f1_val)
        faithfulness_scores.append(faith_val)
        relevance_scores.append(rel_val)

        detailed_results.append({
            "id": q_id,
            "question": question,
            "expected_answer": expected_answer,
            "predicted_answer": predicted_answer,
            "target_document": target_doc,
            "retrieved_documents": list(set(retrieved_doc_names)),
            "retrieval_metrics": {
                "hit_rate": hr,
                "recall_at_k": rec,
                "mrr": mrr_val,
                "ndcg_at_k": ndcg_val,
                "context_precision": cp_val
            },
            "answer_metrics": {
                "exact_match": em_val,
                "f1_score": f1_val,
                "faithfulness": faith_val,
                "answer_relevance": rel_val
            }
        })

    # Summary Report
    summary = {
        "total_questions": len(benchmark_items),
        "retrieval_evaluation": {
            "mean_hit_rate_at_5": round(float(np.mean(hit_rates)), 4),
            "mean_recall_at_5": round(float(np.mean(recalls)), 4),
            "mean_mrr": round(float(np.mean(mrrs)), 4),
            "mean_ndcg_at_5": round(float(np.mean(ndcgs)), 4),
            "mean_context_precision": round(float(np.mean(context_precisions)), 4)
        },
        "answer_evaluation": {
            "mean_exact_match": round(float(np.mean(ems)), 4),
            "mean_f1_score": round(float(np.mean(f1s)), 4),
            "mean_faithfulness": round(float(np.mean(faithfulness_scores)), 4),
            "mean_answer_relevance": round(float(np.mean(relevance_scores)), 4)
        }
    }

    out_file = Path(__file__).parent / "benchmark_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "detailed_results": detailed_results}, f, indent=2)

    print("\n" + "=" * 60)
    print("         BENCHMARK EVALUATION RESULTS SUMMARY         ")
    print("=" * 60)
    print(f"Total Questions Evaluated : {summary['total_questions']}")
    print("\n--- RETRIEVAL METRICS (Top-k=5) ---")
    print(f"Hit Rate @ 5          : {summary['retrieval_evaluation']['mean_hit_rate_at_5']:.4f}")
    print(f"Recall @ 5            : {summary['retrieval_evaluation']['mean_recall_at_5']:.4f}")
    print(f"MRR                   : {summary['retrieval_evaluation']['mean_mrr']:.4f}")
    print(f"nDCG @ 5              : {summary['retrieval_evaluation']['mean_ndcg_at_5']:.4f}")
    print(f"Context Precision     : {summary['retrieval_evaluation']['mean_context_precision']:.4f}")
    print("\n--- ANSWER GENERATION METRICS ---")
    print(f"Faithfulness          : {summary['answer_evaluation']['mean_faithfulness']:.4f}")
    print(f"Answer Relevance      : {summary['answer_evaluation']['mean_answer_relevance']:.4f}")
    print(f"Exact Match (EM)      : {summary['answer_evaluation']['mean_exact_match']:.4f}")
    print(f"Token F1 Score        : {summary['answer_evaluation']['mean_f1_score']:.4f}")
    print("=" * 60)
    print(f"Detailed results written to: {out_file}\n")

if __name__ == "__main__":
    run_evaluation()
