"""
evaluate.py
------------
Full evaluation harness: runs real queries through
QueryPipeline (Member 2) -> RetrievalPipeline (Member 3),
and scores the results using OUR metrics (NDCG@10, MRR, latency),
not just Recall@1/Recall@2 like the quick sanity script did.

This is the file the team should use going forward for anything
that needs NDCG@10 or per-stage latency numbers.
"""

import json
import time

from src.query.query_pipeline import QueryPipeline
from src.retrieval.pipeline import RetrievalPipeline
from src.evaluation.metrics import ndcg_at_k, mrr, average_ndcg, average_mrr, LatencyTracker


def load_chunks(path="data/chunks/chunks.jsonl"):
    chunks = []
    with open(path, "r") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


# Same 4 sanity queries Member 3 used, so results are directly comparable
EVALUATION_QUERIES = [
    {"query": "find code that calculates the total", "expected_function": "calculate_total"},
    {"query": "find code that verifies a token", "expected_function": "verify_token"},
    {"query": "find code that logs out a user", "expected_function": "logout"},
    {"query": "find code that normalizes text", "expected_function": "normalize_input"},
]


def run_evaluation(chunks_path="data/chunks/chunks.jsonl", queries=None, k=10):
    if queries is None:
        queries = EVALUATION_QUERIES

    chunks = load_chunks(chunks_path)
    documents = [c["augmented_text"] for c in chunks]

    # Map function_name -> its index in the documents/chunks list
    expected_indices = {
        c["function_name"]: idx for idx, c in enumerate(chunks)
    }

    query_pipeline = QueryPipeline()
    retrieval_pipeline = RetrievalPipeline(documents)

    tracker = LatencyTracker()

    all_ranked_ids = []
    all_relevant = []

    for item in queries:
        query_text = item["query"]
        expected_function = item["expected_function"]
        expected_index = expected_indices[expected_function]

        with tracker.track("query_processing"):
            mv_query = query_pipeline.process(query_text)

        with tracker.track("retrieval_total"):
            results = retrieval_pipeline.search(
                mv_query.views_for_retrieval(),
                rrf_top_k=50,
                rerank_top_k=20,
                final_top_k=k,
            )

        ranked_ids = [str(r["index"]) for r in results]
        relevant = {str(expected_index): 1}

        all_ranked_ids.append(ranked_ids)
        all_relevant.append(relevant)

        print("\nQuery:", query_text)
        print("Expected:", expected_function, "(index", expected_index, ")")
        print("Retrieved (ranked):", ranked_ids)
        print("NDCG@%d:" % k, round(ndcg_at_k(ranked_ids, relevant, k=k), 4))
        print("MRR:", round(mrr(ranked_ids, relevant), 4))

    overall_ndcg = average_ndcg(all_ranked_ids, all_relevant, k=k)
    overall_mrr = average_mrr(all_ranked_ids, all_relevant)
    latency_summary = tracker.summary()

    print("\n" + "=" * 50)
    print("FINAL EVALUATION RESULTS")
    print("=" * 50)
    print(f"NDCG@{k}: {overall_ndcg:.4f}")
    print(f"MRR:     {overall_mrr:.4f}")
    print("\nAverage latency per stage (ms):")
    for stage, avg_ms in latency_summary.items():
        print(f"  {stage}: {avg_ms:.2f} ms")
    print(f"  TOTAL: {tracker.total_avg_latency():.2f} ms")

    return {
        "ndcg_at_k": overall_ndcg,
        "mrr": overall_mrr,
        "latency_ms": latency_summary,
    }


if __name__ == "__main__":
    run_evaluation()