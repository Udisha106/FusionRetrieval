import json

from src.query.query_pipeline import QueryPipeline
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


# Member 2: create query views
query_pipeline = QueryPipeline()

query = "find code that calculates the mean of numbers"

query_result = query_pipeline.process(query)

views = query_result.views_for_retrieval()


# Retrieval components
dense = DenseRetriever()
bm25 = BM25Retriever(documents)
fusion = RRFFusion()


all_results = []


for view in views:

    dense_results = dense.search(
        view,
        documents,
        top_k=5
    )

    bm25_results = bm25.search(
        view,
        top_k=5
    )

    view_results = fusion.fuse(
        [dense_results, bm25_results],
        top_k=5
    )

    all_results.append(view_results)


# Fuse Original + Expanded + HyDE
multiview_results = fusion.fuse(
    all_results,
    top_k=5
)


print("\n===== MULTI-VIEW RRF RESULTS =====")

for rank, result in enumerate(multiview_results, start=1):

    print(
        f"\nRank {rank}"
    )

    print(
        f"RRF Score: {result['score']:.4f}"
    )

    print(
        f"Index: {result['index']}"
    )

    print(
        f"Document: {result['document']}"
    )