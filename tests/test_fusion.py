import json

from src.retrieval.dense import DenseRetriever
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.fusion import RRFFusion


with open("data/test_chunks.json", "r") as file:
    chunks = json.load(file)


documents = []

for chunk in chunks:
    documents.append(
        chunk["description"] + " " + chunk["code"]
    )


query = "calculate average numbers"


# Dense retrieval
dense = DenseRetriever()

dense_results = dense.search(
    query,
    documents,
    top_k=5
)


# BM25 retrieval
bm25 = BM25Retriever(documents)

bm25_results = bm25.search(
    query,
    top_k=5
)


# RRF fusion
fusion = RRFFusion()

fused_results = fusion.fuse(
    [dense_results, bm25_results],
    top_k=5
)


print("\n===== DENSE =====")

for rank, result in enumerate(dense_results, start=1):
    print(
        f"Rank {rank} | "
        f"Score: {result['score']:.4f} | "
        f"Index: {result['index']}"
    )


print("\n===== BM25 =====")

for rank, result in enumerate(bm25_results, start=1):
    print(
        f"Rank {rank} | "
        f"Score: {result['score']:.4f} | "
        f"Index: {result['index']}"
    )


print("\n===== RRF =====")

for rank, result in enumerate(fused_results, start=1):
    print(
        f"Rank {rank} | "
        f"RRF Score: {result['score']:.4f} | "
        f"Index: {result['index']}"
    )
    print(f"Document: {result['document']}")