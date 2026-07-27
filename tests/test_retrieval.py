import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from app.embeddings.embedding_service import get_embedding_service
from app.evaluation.metrics import RetrievalMetrics, AnswerMetrics

def test_embedding_service_dimension():
    service = get_embedding_service()
    vec, latency = service.embed_query("Sample search query")
    assert len(vec) == 384
    assert latency > 0.0

def test_retrieval_metrics_calculation():
    retrieved = ["id1", "id2", "id3", "id4", "id5"]
    relevant = {"id3"}
    
    assert RetrievalMetrics.hit_rate(retrieved, relevant) == 1.0
    assert RetrievalMetrics.recall_at_k(retrieved, relevant) == 1.0
    assert RetrievalMetrics.mrr(retrieved, relevant) == 1.0 / 3.0
    assert RetrievalMetrics.context_precision(retrieved, relevant) > 0.0

def test_answer_metrics_calculation():
    pred = "Acme Corp revenue in FY2025 was 42.5 million."
    gt = "Acme Corp revenue in FY2025 was 42.5 million."
    
    f1 = AnswerMetrics.f1_score(pred, gt)
    assert f1 == 1.0
    
    faith = AnswerMetrics.faithfulness(pred, ["Acme Corp recorded revenue in FY2025 of 42.5 million."])
    assert faith >= 0.7
