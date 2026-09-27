import json

from src.retrieval.dense import DenseRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.fusion import RRFFusion
from src.retrieval.reranker import Reranker


with open("data/test_chunks.json", "r") as file:
    chunks = json.load(file)


documents = []

for chunk in chunks:
    documents.append(
        chunk["description"] + " " + chunk["code"]
    )


query = "calculate average numbers"


# 1. Dense retrieval
dense = DenseRetriever()

dense_results = dense.search(
    query,
    documents,
    top_k=5
)


# 2. BM25 retrieval
bm25 = BM25Retriever(documents)

bm25_results = bm25.search(
    query,
    top_k=5
)


# 3. RRF fusion
fusion = RRFFusion()

rrf_results = fusion.fuse(
    [dense_results, bm25_results],
    top_k=5
)


# 4. Reranking
reranker = Reranker()

final_results = reranker.rerank(
    query,
    rrf_results,
    top_k=3
)


print("\n===== RRF RESULTS =====")

for rank, result in enumerate(rrf_results, start=1):
    print(
        f"Rank {rank} | "
        f"RRF Score: {result['score']:.4f} | "
        f"Index: {result['index']}"
    )


print("\n===== RERANKED RESULTS =====")

for rank, result in enumerate(final_results, start=1):
    print(
        f"Rank {rank} | "
        f"Score: {result['score']:.4f} | "
        f"Index: {result['index']}"
    )
    print(f"Document: {result['document']}")