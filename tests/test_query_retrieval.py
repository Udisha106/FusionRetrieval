import json

from src.query.query_pipeline import QueryPipeline
from src.retrieval.pipeline import RetrievalPipeline


# Load Member 1's real chunks
chunks = []

with open("data/chunks/chunks.jsonl", "r") as file:
    for line in file:
        chunks.append(json.loads(line))


# Use augmented_text as the retrieval document
documents = [
    chunk["augmented_text"]
    for chunk in chunks
]


# Query pipeline
query_pipeline = QueryPipeline()

# Retrieval pipeline
retrieval_pipeline = RetrievalPipeline(documents)


queries = [
    "find code that calculates the total",
    "find code that verifies a token",
    "find code that logs out a user",
    "find code that normalizes text"
]


for query in queries:

    print("\n" + "=" * 60)
    print(f"QUERY: {query}")
    print("=" * 60)

    query_result = query_pipeline.process(query)

    query_views = query_result.views_for_retrieval()

    results = retrieval_pipeline.search(
        query_views,
        rrf_top_k=5,
        rerank_top_k=3,
        final_top_k=2
    )

    for rank, result in enumerate(results, start=1):

        print(f"\nRank {rank}")
        print(f"Score: {result['score']:.4f}")
        print(f"Index: {result['index']}")
        print(f"Document: {result['document']}")