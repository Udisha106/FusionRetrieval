"""
metrics.py
----------
Core evaluation metrics for retrieval: NDCG@k, MRR, and latency tracking.
Used by evaluate.py to score any retriever against relevance judgments.
"""

import math
import time
from typing import List, Dict
from contextlib import contextmanager


def ndcg_at_k(ranked_ids: List[str], relevant_ids: Dict[str, int], k: int = 10) -> float:
    """
    ranked_ids: list of doc/chunk ids in ranked order (best first)
    relevant_ids: dict mapping doc/chunk id -> relevance grade (e.g. 1 = relevant, 0 = not)
    """
    def dcg(ids):
        return sum(
            relevant_ids.get(doc_id, 0) / math.log2(idx + 2)
            for idx, doc_id in enumerate(ids[:k])
        )

    actual_dcg = dcg(ranked_ids)
    ideal_order = sorted(relevant_ids, key=lambda d: relevant_ids[d], reverse=True)
    ideal_dcg = dcg(ideal_order)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def mrr(ranked_ids: List[str], relevant_ids: Dict[str, int]) -> float:
    """
    Mean Reciprocal Rank for a single query.
    Returns 1/rank of the first relevant result, or 0 if none found.
    """
    for idx, doc_id in enumerate(ranked_ids):
        if relevant_ids.get(doc_id, 0) > 0:
            return 1.0 / (idx + 1)
    return 0.0


class LatencyTracker:
    """
    Simple utility to time stages of the pipeline (query processing,
    BM25, dense, fusion, reranking) and aggregate results.
    """

    def __init__(self):
        self.records: Dict[str, List[float]] = {}

    @contextmanager
    def track(self, stage: str):
        start = time.perf_counter()
        yield
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.records.setdefault(stage, []).append(elapsed_ms)

    def summary(self) -> Dict[str, float]:
        """Returns average latency per stage in ms."""
        return {
            stage: sum(times) / len(times)
            for stage, times in self.records.items()
        }

    def total_avg_latency(self) -> float:
        return sum(self.summary().values())


def average_ndcg(all_results: List[List[str]], all_relevant: List[Dict[str, int]], k: int = 10) -> float:
    """Average NDCG@k across multiple queries."""
    scores = [ndcg_at_k(r, rel, k) for r, rel in zip(all_results, all_relevant)]
    return sum(scores) / len(scores) if scores else 0.0


def average_mrr(all_results: List[List[str]], all_relevant: List[Dict[str, int]]) -> float:
    """Average MRR across multiple queries."""
    scores = [mrr(r, rel) for r, rel in zip(all_results, all_relevant)]
    return sum(scores) / len(scores) if scores else 0.0