"""
experiments.py
----------------
Runs an ablation study across multiple retrieval configurations
(dense only, BM25 only, hybrid, multi-view, +reranker, +MMR) and
scores each with NDCG@10, MRR, and latency, using our own metrics.py.

Produces a comparison table so the team can decide what actually
improves retrieval vs what just adds latency.
"""

import json
import time

from src.query.query_pipeline import QueryPipeline
from src.retrieval.dense import DenseRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.fusion import RRFFusion
from src.retrieval.reranker import Reranker
from src.retrieval.mmr import MMR
from sentence_transformers import SentenceTransformer

from src.evaluation.metrics import ndcg_at_k, mrr, average_ndcg, average_mrr, LatencyTracker


EVALUATION_QUERIES = [
    {"query": "find code that calculates the total", "expected_function": "calculate_total"},
    {"query": "find code that verifies a token", "expected_function": "verify_token"},
    {"query": "find code that logs out a user", "expected_function": "logout"},
    {"query": "find code that normalizes text", "expected_function": "normalize_input"},
]


def load_chunks(path="data/chunks/chunks.jsonl"):
    chunks = []
    with open(path, "r") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


class ExperimentRunner:
    def __init__(self, chunks_path="data/chunks/chunks.jsonl", top_k=10):
        self.chunks = load_chunks(chunks_path)
        self.documents = [c["augmented_text"] for c in self.chunks]
        self.expected_indices = {
            c["function_name"]: idx for idx, c in enumerate(self.chunks)
        }
        self.top_k = top_k

        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.dense = DenseRetriever(self.embedding_model)
        self.bm25 = BM25Retriever(self.documents)
        self.fusion = RRFFusion()
        self.reranker = Reranker()
        self.mmr = MMR(self.embedding_model)

        self.query_pipeline = QueryPipeline()

    def _score_config(self, name, retrieve_fn):
        tracker = LatencyTracker()
        all_ranked_ids = []
        all_relevant = []

        for item in EVALUATION_QUERIES:
            query_text = item["query"]
            expected_index = self.expected_indices[item["expected_function"]]

            with tracker.track("total"):
                results = retrieve_fn(query_text)

            ranked_ids = [str(r["index"]) for r in results]
            relevant = {str(expected_index): 1}

            all_ranked_ids.append(ranked_ids)
            all_relevant.append(relevant)

        return {
            "name": name,
            "ndcg_at_k": average_ndcg(all_ranked_ids, all_relevant, k=self.top_k),
            "mrr": average_mrr(all_ranked_ids, all_relevant),
            "avg_latency_ms": tracker.total_avg_latency(),
        }

    # ---- Experiment configs ----

    def exp1_dense_only(self, query_text):
        return self.dense.search(query_text, self.documents, top_k=self.top_k)

    def exp2_bm25_only(self, query_text):
        return self.bm25.search(query_text, top_k=self.top_k)

    def exp3_dense_bm25_rrf(self, query_text):
        dense_results = self.dense.search(query_text, self.documents, top_k=50)
        bm25_results = self.bm25.search(query_text, top_k=50)
        return self.fusion.fuse([dense_results, bm25_results], top_k=self.top_k)

    def exp4_multiview_rrf(self, query_text):
        mv_query = self.query_pipeline.process(query_text)
        views = mv_query.views_for_retrieval()

        all_view_results = []
        for view in views:
            dense_results = self.dense.search(view, self.documents, top_k=50)
            bm25_results = self.bm25.search(view, top_k=50)
            view_results = self.fusion.fuse([dense_results, bm25_results], top_k=50)
            all_view_results.append(view_results)

        return self.fusion.fuse(all_view_results, top_k=self.top_k)

    def exp5_multiview_reranked(self, query_text):
        mv_query = self.query_pipeline.process(query_text)
        views = mv_query.views_for_retrieval()

        all_view_results = []
        for view in views:
            dense_results = self.dense.search(view, self.documents, top_k=50)
            bm25_results = self.bm25.search(view, top_k=50)
            view_results = self.fusion.fuse([dense_results, bm25_results], top_k=50)
            all_view_results.append(view_results)

        fused = self.fusion.fuse(all_view_results, top_k=50)
        return self.reranker.rerank(views[0], fused, top_k=self.top_k)

    def exp6_full_pipeline_with_mmr(self, query_text):
        mv_query = self.query_pipeline.process(query_text)
        views = mv_query.views_for_retrieval()

        all_view_results = []
        for view in views:
            dense_results = self.dense.search(view, self.documents, top_k=50)
            bm25_results = self.bm25.search(view, top_k=50)
            view_results = self.fusion.fuse([dense_results, bm25_results], top_k=50)
            all_view_results.append(view_results)

        fused = self.fusion.fuse(all_view_results, top_k=50)
        reranked = self.reranker.rerank(views[0], fused, top_k=20)
        return self.mmr.select(views[0], reranked, top_k=self.top_k)

    def run_all(self):
        experiments = [
            ("Exp1: Dense only", self.exp1_dense_only),
            ("Exp2: BM25 only", self.exp2_bm25_only),
            ("Exp3: Dense + BM25 (RRF)", self.exp3_dense_bm25_rrf),
            ("Exp4: Multi-view (Q1+Q2+Q3) + RRF", self.exp4_multiview_rrf),
            ("Exp5: + Reranker", self.exp5_multiview_reranked),
            ("Exp6: + MMR (full pipeline)", self.exp6_full_pipeline_with_mmr),
        ]

        results = []
        for name, fn in experiments:
            print(f"\nRunning {name}...")
            result = self._score_config(name, fn)
            results.append(result)
            print(f"  NDCG@{self.top_k}: {result['ndcg_at_k']:.4f}  "
                  f"MRR: {result['mrr']:.4f}  "
                  f"Latency: {result['avg_latency_ms']:.2f} ms")

        return results

    def print_table(self, results):
        print("\n" + "=" * 70)
        print(f"{'Experiment':<40}{'NDCG@10':>10}{'MRR':>10}{'Latency(ms)':>12}")
        print("=" * 70)
        for r in results:
            print(f"{r['name']:<40}{r['ndcg_at_k']:>10.4f}{r['mrr']:>10.4f}{r['avg_latency_ms']:>12.2f}")


if __name__ == "__main__":
    runner = ExperimentRunner()
    results = runner.run_all()
    runner.print_table(results)