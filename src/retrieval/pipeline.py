from src.retrieval.dense import DenseRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.fusion import RRFFusion
from src.retrieval.reranker import Reranker
from src.retrieval.mmr import MMR
from sentence_transformers import SentenceTransformer


class RetrievalPipeline:

    def __init__(self, documents):

        self.documents = documents

        embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        self.dense = DenseRetriever(embedding_model)

        self.dense.index(
            documents,
            batch_size=64
        )

        self.bm25 = BM25Retriever(documents)
        self.fusion = RRFFusion()
        self.reranker = Reranker()
        self.mmr = MMR()

    def search(
        self,
        query_views,
        rrf_top_k=50,
        rerank_top_k=20,
        final_top_k=10
    ):
        import time

        all_view_results = []

        dense_time = 0
        bm25_time = 0
        fusion_time = 0

        # Retrieve for every query view
        for query in query_views:

            t = time.perf_counter()

            dense_results = self.dense.search(
                query,
                self.documents,
                top_k=rrf_top_k
            )

            dense_time += time.perf_counter() - t

            t = time.perf_counter()

            bm25_results = self.bm25.search(
                query,
                top_k=rrf_top_k
            )

            bm25_time += time.perf_counter() - t

            t = time.perf_counter()

            view_results = self.fusion.fuse(
                [dense_results, bm25_results],
                top_k=rrf_top_k
            )

            fusion_time += time.perf_counter() - t

            all_view_results.append(view_results)

        # Combine all query views
        t = time.perf_counter()

        fused_results = self.fusion.fuse(
            all_view_results,
            top_k=rrf_top_k
        )

        fusion_time += time.perf_counter() - t

        # Rerank candidates
        t = time.perf_counter()

        reranked_results = self.reranker.rerank(
            query_views[0],
            fused_results,
            top_k=5
        )

        reranker_time = time.perf_counter() - t

        # Remove redundant results
        t = time.perf_counter()

        query_embedding = self.dense.encode(
            [query_views[0]]
        )[0]

        final_results = self.mmr.select(
            query_views[0],
            reranked_results,
            self.dense.document_embeddings,
            query_embedding,
            top_k=final_top_k
        )

        mmr_time = time.perf_counter() - t

        print("\n--- Retrieval Timing ---")
        print(f"Dense:     {dense_time * 1000:.2f} ms")
        print(f"BM25:      {bm25_time * 1000:.2f} ms")
        print(f"Fusion:    {fusion_time * 1000:.2f} ms")
        print(f"Reranker:  {reranker_time * 1000:.2f} ms")
        print(f"MMR:       {mmr_time * 1000:.2f} ms")
        print("------------------------\n")

        return final_results