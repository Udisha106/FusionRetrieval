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
        self.mmr = MMR(embedding_model)
    def search(
        self,
        query_views,
        rrf_top_k=50,
        rerank_top_k=20,
        final_top_k=10
    ):

        all_view_results = []

        # Retrieve for every query view
        for query in query_views:

            dense_results = self.dense.search(
                query,
                self.documents,
                top_k=rrf_top_k
            )

            bm25_results = self.bm25.search(
                query,
                top_k=rrf_top_k
            )

            # Combine Dense + BM25 for this view
            view_results = self.fusion.fuse(
                [dense_results, bm25_results],
                top_k=rrf_top_k
            )

            all_view_results.append(view_results)

        # Combine Original + Expanded + HyDE
        fused_results = self.fusion.fuse(
            all_view_results,
            top_k=rrf_top_k
        )

        # Rerank candidates
        reranked_results = self.reranker.rerank(
            query_views[0],
            fused_results,
            top_k=rerank_top_k
        )

        # Remove redundant results
        final_results = self.mmr.select(
            query_views[0],
            reranked_results,
            top_k=final_top_k
        )

        return final_results