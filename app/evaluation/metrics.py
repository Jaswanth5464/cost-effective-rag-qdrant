import math
import re
from typing import List, Set, Dict, Any
import numpy as np

class RetrievalMetrics:
    """
    Computes Information Retrieval (IR) evaluation metrics for vector search performance.
    """
    @staticmethod
    def hit_rate(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """Hit Rate @ k: 1 if at least one relevant document is retrieved in top-k, else 0."""
        if not relevant_ids:
            return 0.0
        for doc_id in retrieved_ids:
            if doc_id in relevant_ids:
                return 1.0
        return 0.0

    @staticmethod
    def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """Recall @ k: Ratio of relevant documents retrieved in top-k over total relevant documents."""
        if not relevant_ids:
            return 0.0
        hits = sum(1 for doc_id in retrieved_ids if doc_id in relevant_ids)
        return hits / len(relevant_ids)

    @staticmethod
    def mrr(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """Mean Reciprocal Rank (MRR): 1 / rank of the first relevant document."""
        if not relevant_ids:
            return 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg_at_k(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """Normalized Discounted Cumulative Gain (nDCG @ k)."""
        if not relevant_ids:
            return 0.0
        
        dcg = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                dcg += 1.0 / math.log2(rank + 1)

        # Ideal DCG (all relevant items ranked at the top)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant_ids), len(retrieved_ids)) + 1))
        
        if idcg == 0.0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def context_precision(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
        """
        Context Precision: Proportion of relevant chunks among the retrieved chunks, weighted by position.
        Precision@k averaged across all positions where a relevant chunk is retrieved.
        """
        if not relevant_ids or not retrieved_ids:
            return 0.0
        
        hits = 0
        precision_sum = 0.0
        for rank, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_ids:
                hits += 1
                precision_sum += hits / rank
                
        if hits == 0:
            return 0.0
        return precision_sum / min(len(relevant_ids), len(retrieved_ids))


class AnswerMetrics:
    """
    Computes Generation Quality Metrics: Exact Match (EM), Token-level F1 score, Faithfulness, and Answer Relevance.
    """
    @staticmethod
    def _strip_citations(s: str) -> str:
        """Strips citation brackets [Document: ..., Page: ..., Chunk ID: ...] before evaluation."""
        s = re.sub(r'\[Document:[^\]]+\]', '', s)
        s = re.sub(r'\[Source[^\]]+\]', '', s)
        return s.strip()

    @classmethod
    def _normalize_answer(cls, s: str) -> str:
        """Lowercases text, strips citation brackets, expands common number formats, and removes punctuation/articles."""
        s = cls._strip_citations(s)
        s = s.lower()
        # Expand common short financial and percentage notations
        s = re.sub(r'\$([0-9\.]+)\s*m\b', r'\1 million dollars', s)
        s = re.sub(r'\$([0-9\.]+)\s*k\b', r'\1 thousand dollars', s)
        s = re.sub(r'([0-9\.]+)%', r'\1 percent', s)
        s = re.sub(r'\b(a|an|the)\b', ' ', s)
        s = re.sub(r'[^\w\s]', ' ', s)
        return ' '.join(s.split())

    @classmethod
    def exact_match(cls, prediction: str, ground_truth: str) -> float:
        """Exact Match (EM) score (1.0 if normalized strings match exactly or GT is contained, else 0.0)."""
        norm_pred = cls._normalize_answer(prediction)
        norm_gt = cls._normalize_answer(ground_truth)

        if not norm_pred or not norm_gt:
            return 0.0
            
        if norm_pred == norm_gt or norm_gt in norm_pred or norm_pred in norm_gt:
            return 1.0
        return 0.0

    @classmethod
    def f1_score(cls, prediction: str, ground_truth: str) -> float:
        """Token-level F1 score comparing prediction to ground truth."""
        pred_tokens = cls._normalize_answer(prediction).split()
        gt_tokens = cls._normalize_answer(ground_truth).split()

        if not pred_tokens or not gt_tokens:
            return 1.0 if pred_tokens == gt_tokens else 0.0

        common_tokens = set(pred_tokens) & set(gt_tokens)
        num_same = sum(min(pred_tokens.count(w), gt_tokens.count(w)) for w in common_tokens)

        if num_same == 0:
            return 0.0

        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        return (2 * precision * recall) / (precision + recall)

    @classmethod
    def faithfulness(cls, prediction: str, context_chunks: List[str]) -> float:
        """
        Faithfulness / Groundedness: Measures whether predicted answer claims are grounded in context chunks.
        Evaluated via token overlap / containment of prediction key terms inside context.
        """
        if "I don't know based on the provided documents" in prediction:
            return 1.0  # Properly refused ungrounded answer

        clean_pred = cls._strip_citations(prediction)
        pred_tokens = set(cls._normalize_answer(clean_pred).split())
        if not pred_tokens:
            return 1.0

        # Remove stop words for high-fidelity faithfulness check
        stop_words = {"is", "was", "are", "were", "to", "for", "in", "of", "and", "a", "an", "the", "by", "with", "at", "from"}
        meaningful_tokens = pred_tokens - stop_words
        if not meaningful_tokens:
            meaningful_tokens = pred_tokens

        all_context_text = cls._normalize_answer(" ".join(context_chunks))
        supported_tokens = sum(1 for token in meaningful_tokens if token in all_context_text)
        return supported_tokens / len(meaningful_tokens)

    @classmethod
    def answer_relevance(cls, prediction: str, question: str) -> float:
        """
        Answer Relevance: Measures alignment between generated answer and input question.
        Evaluated via keyword alignment and structural response validity.
        """
        if not prediction.strip():
            return 0.0

        if "I don't know based on the provided documents" in prediction:
            return 1.0

        clean_pred = cls._strip_citations(prediction)
        q_tokens = set(cls._normalize_answer(question).split()) - {"what", "is", "the", "how", "much", "many", "which", "are", "for"}
        a_tokens = set(cls._normalize_answer(clean_pred).split())

        if not q_tokens:
            return 1.0
        
        overlap = len(q_tokens & a_tokens)
        score = (overlap / len(q_tokens)) + 0.4
        return min(1.0, score)
